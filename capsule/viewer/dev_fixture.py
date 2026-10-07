"""Write a fully synthetic capsule without using the imaging pipeline."""

from __future__ import annotations

import argparse
import base64
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).with_name("template.html")
PAYLOAD_MARKER = "<!--CAPSULE_PAYLOAD-->"


def _sphere(x: np.ndarray, y: np.ndarray, z: np.ndarray, center: tuple[float, float, float], radius: float) -> np.ndarray:
    cx, cy, cz = center
    return (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= radius**2


def _blob_script(blob_id: str, data: np.ndarray) -> str:
    packed = gzip.compress(data.tobytes(order="C"), compresslevel=6, mtime=0)
    encoded = base64.b64encode(packed).decode("ascii")
    return (
        f'<script id="capsule-blob-{blob_id}" type="application/octet-stream" '
        f'data-encoding="gzip+base64">{encoded}</script>'
    )


def synthetic_payload() -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Return the case-capsule/1 manifest and x-fastest binary arrays."""
    nx, ny, nz = 160, 192, 160
    shape = (nz, ny, nx)
    # Voxel indices map to RAS mm with the grid origin at (-80, -96, -80).
    x = np.arange(nx, dtype=np.float32)[None, None, :] - 80
    y = np.arange(ny, dtype=np.float32)[None, :, None] - 96
    z = np.arange(nz, dtype=np.float32)[:, None, None] - 80

    head_level = (x / 58.0) ** 2 + (y / 72.0) ** 2 + (z / 60.0) ** 2
    head = head_level <= 1.0
    bone_shell = head & (head_level >= 0.94)

    lesion = _sphere(x, y, z, (10.0, 0.0, 0.0), 10.0).astype(np.uint8)
    marker = _sphere(x, y, z, (-35.0, 0.0, 0.0), 4.0)
    unreviewed = _sphere(x, y, z, (-20.0, -20.0, 0.0), 5.0).astype(np.uint8)

    ct = np.full(shape, -1000, dtype=np.int16)
    ct[head] = 40
    ct[bone_shell] = 1000
    ct[lesion != 0] = 60
    ct[marker] = 2000

    # MR uses a distinct contrast, then a nontrivial linear rescale back to a.u.
    # Values -100, 0, 100, 200, and 300 map across the uint16 range.
    mr = np.zeros(shape, dtype=np.uint16)
    mr[head] = 16384
    mr[bone_shell] = 32768
    mr[lesion != 0] = 49151
    mr[marker] = 65535

    lesion_ml = float(lesion.sum()) / 1000.0
    unreviewed_ml = float(unreviewed.sum()) / 1000.0
    created = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    manifest: dict[str, object] = {
        "schema": "case-capsule/1",
        "version": 1,
        "created_utc": created,
        "generator": "case-capsule synthetic viewer fixture",
        "case": {"label": "Phantom sintético", "anonymized": True, "study_year": 2026},
        "locale": "pt-BR",
        "grid": {
            "dims": [nx, ny, nz],
            "spacing_mm": [1.0, 1.0, 1.0],
            "affine_ras": [
                [1.0, 0.0, 0.0, -80.0],
                [0.0, 1.0, 0.0, -96.0],
                [0.0, 0.0, 1.0, -80.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
        "volumes": [
            {
                "id": "ct",
                "blob": "ct",
                "kind": "CT",
                "label": "TC sintética",
                "dtype": "int16",
                "slope": 1.0,
                "intercept": 0.0,
                "units": "HU",
                "series": {"number": 1, "description": "TC sintética", "modality": "CT", "frames_used": "all"},
                "stats": {"min": -1000, "max": 2000, "p01": -1000, "p99": 40},
                "window_presets": [
                    {"name": "Cérebro", "center": 40, "width": 80},
                    {"name": "Subdural", "center": 75, "width": 215},
                    {"name": "AVC", "center": 35, "width": 40},
                    {"name": "Partes moles", "center": 40, "width": 400},
                    {"name": "Osso", "center": 600, "width": 2800},
                    {"name": "Osso temporal", "center": 700, "width": 4000},
                ],
                "registration": {"reference": True},
            },
            {
                "id": "mr",
                "blob": "mr",
                "kind": "MR",
                "label": "RM sintética",
                "dtype": "uint16",
                "slope": 400.0 / 65535.0,
                "intercept": -100.0,
                "units": "a.u.",
                "series": {"number": 2, "description": "RM sintética", "modality": "MR", "frames_used": "all"},
                "stats": {"min": -100, "max": 300, "p01": -100, "p99": 0},
                "registration": {"reference": False, "transform": "identity"},
            },
        ],
        "masks": [
            {
                "id": "lesion",
                "blob": "mask_lesion",
                "label": "Lesão sintética revisada",
                "color": "#E4572E",
                "volume_ml": lesion_ml,
                "source": "synthetic",
                "reviewed": True,
            },
            {
                "id": "unreviewed",
                "blob": "mask_unreviewed",
                "label": "Região sintética não revisada",
                "color": "#00FF00",
                "volume_ml": unreviewed_ml,
                "source": "synthetic",
                "reviewed": False,
            },
        ],
        "annotations": [],
        "tour": [
            {
                "id": "tour-lesion",
                "title": "Lesão sintética",
                "text": "Esta esfera sintética marca o centro da lesão usada no teste.",
                "view": {
                    "layout": "axial",
                    "crosshair_ras": [10.0, 0.0, 0.0],
                    "window": {"volume": "ct", "center": 60.0, "width": 160.0},
                    "camera": {"yaw": 0.45, "pitch": 0.2, "zoom": 1.0},
                    "visible_masks": ["lesion", "unreviewed"],
                    "visible_annotations": [],
                },
            },
            {
                "id": "tour-marker",
                "title": "Marcador sintético",
                "text": "O marcador sintético está à esquerda do paciente.",
                "view": {
                    "layout": "axial",
                    "crosshair_ras": [-35.0, 0.0, 0.0],
                    "window": {"volume": "ct", "center": 1600.0, "width": 400.0},
                    "camera": {"yaw": 0.45, "pitch": 0.2, "zoom": 1.0},
                    "visible_masks": ["lesion", "unreviewed"],
                    "visible_annotations": [],
                },
            },
        ],
        # Exercise preservation of application-specific manifest fields on save.
        "synthetic_extension": {"preserve_on_save": True, "kind": "fixture"},
    }
    return manifest, {"ct": ct, "mr": mr, "mask_lesion": lesion, "mask_unreviewed": unreviewed}


def write_fixture(output: Path, template_path: Path = TEMPLATE) -> Path:
    template = template_path.read_text(encoding="utf-8")
    if template.count(PAYLOAD_MARKER) != 1:
        raise ValueError("expected exactly one <!--CAPSULE_PAYLOAD--> marker in template")

    manifest, arrays = synthetic_payload()
    manifest_json = json.dumps(manifest, ensure_ascii=False, separators=(",", ":"))
    parts = [f'<script id="capsule-manifest" type="application/json">{manifest_json}</script>']
    parts.extend(_blob_script(blob_id, array) for blob_id, array in arrays.items())
    output_html = template.replace(PAYLOAD_MARKER, "\n".join(parts), 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(output_html, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a synthetic case capsule for viewer development.")
    parser.add_argument("-o", "--output", required=True, type=Path, help="output .capsule.html path")
    args = parser.parse_args()
    path = write_fixture(args.output)
    print(path)


if __name__ == "__main__":
    main()
