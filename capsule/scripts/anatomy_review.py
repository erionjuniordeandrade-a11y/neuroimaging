#!/usr/bin/env python3
"""Write per-label controls and an orthogonal colour-overlay montage for a capsule."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from capsule.pack import read_capsule


def _volume_data(manifest: dict, arrays: dict[str, np.ndarray], volume_id: str) -> np.ndarray:
    volume = next(item for item in manifest["volumes"] if item["id"] == volume_id)
    return arrays[volume["blob"]].astype(np.float32) * float(volume["slope"]) + float(volume["intercept"])


def _centroid_x(label_map: np.ndarray, value: int, affine: np.ndarray) -> float | None:
    positions = np.argwhere(label_map == value)
    if not len(positions):
        return None
    xyz = positions[:, ::-1].astype(float)
    return float((xyz @ affine[:3, :3].T + affine[:3, 3]).mean(axis=0)[0])


def anatomy_report(manifest: dict, arrays: dict[str, np.ndarray]) -> tuple[list[dict], dict]:
    """Independently recount grid voxels and calculate paired, laterality, and alignment controls."""
    spacing = np.asarray(manifest["grid"]["spacing_mm"], dtype=float)
    voxel_ml = float(np.prod(spacing)) / 1000.0
    affine = np.asarray(manifest["grid"]["affine_ras"], dtype=float)
    rows: list[dict] = []
    pair_checks: list[dict] = []
    ventricle_checks: dict = {}
    anatomy_by_id = {item["id"]: item for item in manifest.get("anatomy", [])}

    for item in manifest.get("anatomy", []):
        labels = arrays[item["blob"]]
        records = {record["key"]: record for record in item["labels"]}
        counts: dict[str, int] = {}
        for record in item["labels"]:
            count = int(np.count_nonzero(labels == int(record["value"])))
            volume_ml = count * voxel_ml
            counts[record["key"]] = count
            difference = abs(volume_ml - float(record["volume_ml"]))
            rows.append({"item": item["id"], "method": item["method"], "key": record["key"],
                         "name": record["name"], "value": int(record["value"]), "voxels": count,
                         "volume_ml": round(volume_ml, 6), "manifest_volume_ml": record["volume_ml"],
                         "volume_check": "PASS" if difference <= 0.000001 else "FAIL",
                         "pair_control": "not paired"})

        for key, record in records.items():
            if not key.endswith("_left"):
                continue
            other = key[:-5] + "_right"
            if other not in records:
                continue
            left, right = counts[key] * voxel_ml, counts[other] * voxel_ml
            if left == right == 0:
                ratio, status = None, "UNMEASURED (both empty)"
            elif left == 0 or right == 0:
                ratio, status = None, "FLAG (one side empty)"
            else:
                ratio = left / right
                asymmetry = max(left, right) / min(left, right)
                status = "FLAG (>1.5)" if asymmetry > 1.5 else "PASS"
            pair = {"item": item["id"], "left": key, "right": other,
                    "left_volume_ml": round(left, 6), "right_volume_ml": round(right, 6),
                    "left_right_ratio": None if ratio is None else round(ratio, 4),
                    "asymmetry_ratio": None if ratio is None else round(max(ratio, 1.0 / ratio), 4),
                    "control": "contralateral structure; expected near 1.0", "status": status}
            pair_checks.append(pair)
            for row in rows[-len(item["labels"]):]:
                if row["item"] == item["id"] and row["key"] in {key, other}:
                    row["pair_control"] = f"L/R={pair['left_right_ratio']}; {status}"

        if item["id"] == "anat_mr":
            for side, sign in (("left", -1), ("right", 1)):
                key = f"lateral_ventricle_{side}"
                record = records.get(key)
                centroid = _centroid_x(labels, int(record["value"]), affine) if record else None
                correct_sign = centroid is not None and centroid * sign > 0
                ventricle_checks[side] = {"centroid_x_ras_mm": None if centroid is None else round(centroid, 3),
                                          "expected_sign": "negative" if sign < 0 else "positive",
                                          "control": "RAS left is -x; right is +x",
                                          "status": "PASS" if correct_sign else "FAIL"}

    coverage = None
    mr_item = anatomy_by_id.get("anat_mr")
    if mr_item:
        brain = next((mask for mask in manifest.get("masks", [])
                      if mask.get("id") == "brain" and mask.get("for_volume") == mr_item["for_volume"]), None)
        if brain:
            synthseg = arrays[mr_item["blob"]] != 0
            brain_mask = arrays[brain["blob"]] != 0
            denominator = int(synthseg.sum())
            covered = int(np.count_nonzero(synthseg & brain_mask))
            fraction = covered / denominator if denominator else 0.0
            coverage = {"covered_voxels": covered, "synthseg_nonbackground_voxels": denominator,
                        "fraction": round(fraction, 6), "minimum": 0.95,
                        "control": "same-capsule brain render mask covers SynthSeg non-background",
                        "status": "PASS" if fraction >= 0.95 else "FAIL"}

    checks = {
        "grid_voxel_volume_ml": round(voxel_ml, 9),
        "tool_licences": [{"item": item["id"], "method": item["method"], "licence": item["licence"]}
                          for item in manifest.get("anatomy", [])],
        "label_volume_checks": {"count": len(rows), "failures": sum(row["volume_check"] != "PASS" for row in rows)},
        "paired_structures": pair_checks,
        "lateral_ventricle_centroids": ventricle_checks,
        "synthseg_grid_alignment": coverage,
    }
    return rows, checks


def _base_gray(values: np.ndarray, kind: str) -> np.ndarray:
    if kind == "CT":
        low, high = -450.0, 750.0
    else:
        low, high = np.percentile(values, [1, 99]) if values.size else (0.0, 1.0)
        if high <= low:
            high = low + 1.0
    gray = np.clip((values - low) / (high - low), 0, 1)
    return (gray * 255).astype(np.uint8)


def _plane(array: np.ndarray, axis: str, center: tuple[int, int, int]) -> np.ndarray:
    k, j, i = center
    if axis == "AXIAL":
        return array[k, :, :]
    if axis == "CORONAL":
        return array[:, j, :]
    return array[:, :, i]


def _crop_bounds(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int, int]]:
    points = np.argwhere(mask)
    shape = mask.shape
    if len(points):
        lower = points.min(axis=0)
        upper = points.max(axis=0) + 1
        center = tuple(int((a + b - 1) // 2) for a, b in zip(lower, upper))
    else:
        lower = np.array([0, 0, 0])
        upper = np.asarray(shape)
        center = tuple(int(size // 2) for size in shape)
    margin = 12
    cropped = tuple((max(0, int(a) - margin), min(size, int(b) + margin))
                    for a, b, size in zip(lower, upper, shape))
    return cropped[0], cropped[1], cropped[2], center


def _cropped_plane(array: np.ndarray, axis: str, bounds, center: tuple[int, int, int]) -> np.ndarray:
    (z0, z1), (y0, y1), (x0, x1) = bounds
    k, j, i = center
    if axis == "AXIAL":
        return array[k, y0:y1, x0:x1]
    if axis == "CORONAL":
        return array[z0:z1, j, x0:x1]
    return array[z0:z1, y0:y1, i]


def _color_overlays(gray: np.ndarray, labels: np.ndarray, item: dict) -> Image.Image:
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    for record in item["labels"]:
        value = int(record["value"])
        selected = labels == value
        if not selected.any():
            continue
        color = tuple(int(record["color"][i:i + 2], 16) for i in (1, 3, 5))
        rgb[selected] = rgb[selected] * 0.45 + np.asarray(color, dtype=np.float32) * 0.55
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def create_montage(manifest: dict, arrays: dict[str, np.ndarray], destination: Path) -> None:
    items = manifest.get("anatomy", [])
    if not items:
        raise ValueError("capsule has no anatomy items to review")
    panel_width, panel_height, header = 360, 300, 38
    width, row_height = panel_width * 3, panel_height + header
    canvas = Image.new("RGB", (width, row_height * len(items)), "#10151d")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for row_index, item in enumerate(items):
        volume = next(value for value in manifest["volumes"] if value["id"] == item["for_volume"])
        data = _volume_data(manifest, arrays, item["for_volume"])
        gray = _base_gray(data, volume["kind"])
        label_map = arrays[item["blob"]]
        bounds_z, bounds_y, bounds_x, center = _crop_bounds(label_map != 0)
        y = row_index * row_height
        draw.text((8, y + 3), item["id"], fill="#f4f4f4", font=font)
        for column, axis in enumerate(("AXIAL", "CORONAL", "SAGITTAL")):
            x = column * panel_width
            draw.text((8 + x, y + 19), axis, fill="#c3c9d1", font=font)
            image_plane = _cropped_plane(gray, axis, (bounds_z, bounds_y, bounds_x), center)
            label_plane = _cropped_plane(label_map, axis, (bounds_z, bounds_y, bounds_x), center)
            rendered = _color_overlays(image_plane, label_plane, item)
            rendered.thumbnail((panel_width - 12, panel_height - 8), Image.Resampling.NEAREST)
            canvas.paste(rendered, (x + (panel_width - rendered.width) // 2,
                                    y + header + (panel_height - rendered.height) // 2))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", optimize=True)


def _safe_stem(path: Path) -> str:
    stem = path.name.removesuffix(".capsule.html").removesuffix(".html")
    return re.sub(r"[^A-Za-z0-9._-]+", "-", stem)


def review_capsule(path: Path, output_dir: Path) -> dict[str, Path]:
    manifest, arrays = read_capsule(path)
    rows, checks = anatomy_report(manifest, arrays)
    stem = _safe_stem(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = output_dir / f"{stem}.anatomy.tsv"
    checks_path = output_dir / f"{stem}.checks.json"
    montage_path = output_dir / f"{stem}.montage.png"
    with tsv_path.open("w", newline="", encoding="utf-8") as file:
        fields = ["item", "method", "key", "name", "value", "voxels", "volume_ml",
                  "manifest_volume_ml", "volume_check", "pair_control"]
        writer = csv.DictWriter(file, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    checks_path.write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    create_montage(manifest, arrays, montage_path)
    return {"numbers": tsv_path, "checks": checks_path, "montage": montage_path}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capsule", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("out/anatomy-review"))
    args = parser.parse_args()
    outputs = review_capsule(args.capsule, args.output_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    checks = json.loads(outputs["checks"].read_text(encoding="utf-8"))
    print(json.dumps(checks, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
