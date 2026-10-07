"""Build development capsules for the NiiVue viewer (viewer2).

Phantom mode (default): synthetic DICOM from ``capsule.phantom`` is built by the
real pipeline (``capsule.cli build``), then re-injected into
``viewer2/template.html`` with two masks, two synthetic tracts, a tour step and
an unknown manifest field. Round-2 contract additions: a synthetic table slab
(CT) and a fiducial plate with a diagonal rod (MR) outside the head, render masks
``head`` (CT), ``head_<mr>`` and ``brain_<mr>`` (MR), and an unreviewed dataset
lesion ``tumour``, and one automatic ``anatomy`` label map (``ANATOMY_LABELS``). ``TRUTH`` records where each synthetic object sits.

Real-data mode (``--nifti T1 --tck TRACT``): a public NIfTI volume plus one
MRtrix tract, subsampled to at most 1500 streamlines. Inputs are only read.

Outputs are opened with mode "x": an existing file is never overwritten.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capsule.pack import _read_payload, read_capsule, write_capsule  # noqa: E402
from capsule.phantom import write_phantom  # noqa: E402

TEMPLATE = ROOT / "viewer2" / "template.html"
MAX_STREAMLINES = 1500
# Phantom geometry (capsule.phantom): head ellipsoid semi-axes in mm, and the round-2 synthetic objects.
HEAD_SEMI_AXES = (31.0 * 2.2, 38.0 * 2.2, 29.0 * 2.2)
BRAIN_SCALE = 0.9  # phantom "core" = inside the 0.91 bone shell
TRUTH = {
    "table_hu": 400.0, "table_y_max_mm": -86.0, "table_x_halfwidth_mm": 50.0,
    "table_probe_ras": (0.0, -91.0, 0.0),
    "fiducial_of_max": 1.0, "fiducial_x_mm": (70.0, 73.0), "fiducial_yz_half_mm": 40.0,
    "fiducial_probe_ras": (71.5, 0.0, 0.0),
    "tumour_center_ras": (-30.0, -20.0, -18.0), "tumour_radius_mm": 9.0,
}


# --------------------------------------------------------------------- tck I/O
def tck_bytes(streamlines: list[np.ndarray]) -> bytes:
    """Minimal MRtrix .tck: Float32LE xyz, NaN triplet between streamlines, Inf at end."""
    body = []
    for line in streamlines:
        pts = np.asarray(line, dtype="<f4").reshape(-1, 3)
        if len(pts) < 2:
            continue
        body.append(pts.tobytes())
        body.append(np.full(3, np.nan, dtype="<f4").tobytes())
    body.append(np.full(3, np.inf, dtype="<f4").tobytes())
    count = sum(1 for line in streamlines if len(np.asarray(line).reshape(-1, 3)) >= 2)
    offset = 0
    while True:
        header = f"mrtrix tracks\ndatatype: Float32LE\ncount: {count}\nfile: . {offset}\nEND\n"
        if len(header.encode("ascii")) == offset:
            break
        offset = len(header.encode("ascii"))
    return header.encode("ascii") + b"".join(body)


def read_tck(data: bytes) -> list[np.ndarray]:
    """Independent reader used to check the writer (and by tests)."""
    head_end = data.index(b"END\n") + 4
    header = data[:head_end].decode("ascii")
    offset = int(next(line for line in header.splitlines() if line.startswith("file:")).split()[-1])
    assert "Float32LE" in header
    pts = np.frombuffer(data[offset:], dtype="<f4").reshape(-1, 3)
    lines, start = [], 0
    for index, row in enumerate(pts):
        if np.isinf(row).all():
            break
        if np.isnan(row).all():
            lines.append(pts[start:index].copy())
            start = index + 1
    return lines


def _as_blob(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.uint8).copy()


def _tract_meta(tract_id: str, label: str, color: str, n: int, n_source: int, source: str, reviewed: bool) -> dict:
    return {"id": tract_id, "blob": f"tract_{tract_id}", "label": label, "color": color, "format": "tck",
            "n_streamlines": n, "n_streamlines_source": n_source, "source": source, "reviewed": reviewed}


# ---------------------------------------------------------------- phantom mode
def _sphere_mask(manifest: dict, center: tuple[float, float, float], radius: float) -> np.ndarray:
    nx, ny, nz = manifest["grid"]["dims"]
    affine = np.asarray(manifest["grid"]["affine_ras"], dtype=float)
    k, j, i = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    ras = [affine[r, 0] * i + affine[r, 1] * j + affine[r, 2] * k + affine[r, 3] for r in range(3)]
    dist2 = sum((ras[r] - center[r]) ** 2 for r in range(3))
    return (dist2 <= radius * radius).astype(np.uint8)


def _grid_ras(manifest: dict) -> list[np.ndarray]:
    nx, ny, nz = manifest["grid"]["dims"]
    affine = np.asarray(manifest["grid"]["affine_ras"], dtype=float)
    k, j, i = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    return [affine[r, 0] * i + affine[r, 1] * j + affine[r, 2] * k + affine[r, 3] for r in range(3)]


def _head_masks(manifest: dict) -> tuple[np.ndarray, np.ndarray]:
    """Analytic phantom head (ellipsoid + nose, 3 mm margin) and brain (inside the bone shell)."""
    x, y, z = _grid_ras(manifest)
    a, b, c = HEAD_SEMI_AXES
    r = np.sqrt((x / a) ** 2 + (y / b) ** 2 + (z / c) ** 2)
    xu, yu, zu = x / 2.2, y / 2.2, z / 2.2
    nose = ((xu - 2.0) / 5.0) ** 2 + ((yu - 37.0) / 7.0) ** 2 + ((zu + 6.0) / 8.0) ** 2 <= 1.2
    head = (r <= 1.0 + 3.0 / min(HEAD_SEMI_AXES)) | nose
    brain = r <= BRAIN_SCALE
    return head.astype(np.uint8), brain.astype(np.uint8)


def _set_values(meta: dict, packed: np.ndarray, where: np.ndarray, value: float) -> None:
    raw = (value - float(meta.get("intercept", 0) or 0)) / float(meta.get("slope", 1) or 1)
    info = np.iinfo(packed.dtype)
    packed[where] = np.clip(np.round(raw), info.min, info.max).astype(packed.dtype)


def add_render_scene(manifest: dict, arrays: dict) -> None:
    """Round-2 contract: table/fiducials outside the head, render masks, unreviewed dataset tumour."""
    x, y, z = _grid_ras(manifest)
    head, brain = _head_masks(manifest)
    ml = _voxel_ml(manifest)
    table = (y <= TRUTH["table_y_max_mm"]) & (np.abs(x) <= TRUTH["table_x_halfwidth_mm"]) & (head == 0)
    x0, x1 = TRUTH["fiducial_x_mm"]
    half = TRUTH["fiducial_yz_half_mm"]
    plate_box = (x >= x0) & (x <= x1) & (np.abs(y) <= half) & (np.abs(z) <= half)
    # Leksell-style N: two vertical rods and a diagonal joining them.
    rods = (np.abs(np.abs(y) - half + 1.0) <= 1.5) | (np.abs(y - z) <= 1.5)
    fiducial = plate_box & rods & (head == 0)
    masks = manifest.setdefault("masks", [])
    for meta in manifest["volumes"]:
        packed = arrays[meta["blob"]]
        if meta["kind"] == "CT":
            _set_values(meta, packed, table, TRUTH["table_hu"])
            masks.append({"id": "head", "blob": "mask_head", "label": "Cabeça (renderização)", "color": "#FFFFFF",
                          "volume_ml": round(float(head.sum()) * ml, 3), "source": "auto", "reviewed": False,
                          "role": "render", "for_volume": meta["id"]})
            arrays["mask_head"] = head
        else:
            # As bright as the brightest tissue (the real Leksell plates sit above p99).
            top = float(packed.max()) * float(meta.get("slope", 1) or 1) + float(meta.get("intercept", 0) or 0)
            _set_values(meta, packed, fiducial, top * TRUTH["fiducial_of_max"])
            for name, data in (("head", head), ("brain", brain)):
                mid = f"{name}_{meta['id']}"
                masks.append({"id": mid, "blob": f"mask_{mid}", "label": f"{name} (renderização)", "color": "#FFFFFF",
                              "volume_ml": round(float(data.sum()) * ml, 3), "source": "auto", "reviewed": False,
                              "role": "render", "for_volume": meta["id"]})
                arrays[f"mask_{mid}"] = data
    tumour = _sphere_mask(manifest, TRUTH["tumour_center_ras"], TRUTH["tumour_radius_mm"])
    masks.append({"id": "tumour", "blob": "mask_tumour", "label": "Schwannoma vestibular (dataset)", "color": "#E4572E",
                  "volume_ml": round(float(tumour.sum()) * ml, 3), "source": "dataset", "reviewed": False, "role": "lesion"})
    arrays["mask_tumour"] = tumour


def add_corridor_vessel(manifest: dict, arrays: dict) -> None:
    """Add a compact synthetic MR render vessel for corridor acceptance coverage."""
    x, y, z = _grid_ras(manifest)
    vessel = (x >= -28.0) & (x <= 28.0) & ((y - 45.0) ** 2 + (z - 20.0) ** 2 <= 3.2 ** 2)
    mr = next(meta for meta in manifest["volumes"] if meta["kind"] != "CT")
    ident = f"vessels_{mr['id']}"
    data = vessel.astype(np.uint8)
    manifest.setdefault("masks", []).append({
        "id": ident, "blob": f"mask_{ident}", "label": "Vasos sintéticos", "color": "#3A60D6",
        "volume_ml": round(float(data.sum()) * _voxel_ml(manifest), 3), "source": "auto", "reviewed": False,
        "role": "render", "for_volume": mr["id"],
    })
    arrays[f"mask_{ident}"] = data


# Synthetic anatomy (contract: top-level ``anatomy``): labelled ellipsoids clear of the lesion, marker and tumour.
# (value, key, name, group, colour, centre RAS mm, semi-axes mm)
ANATOMY_LABELS = (
    (1, "lateral_ventricle_left", "Ventrículo lateral esquerdo", "Ventrículos", "#4FD1E8", (-10.0, 18.0, 16.0), (4.0, 10.0, 5.0)),
    (2, "lateral_ventricle_right", "Ventrículo lateral direito", "Ventrículos", "#7FE3F2", (10.0, 18.0, 16.0), (4.0, 10.0, 5.0)),
    (3, "brainstem", "Tronco encefálico", "Tronco e cerebelo", "#D9A66B", (0.0, -28.0, -26.0), (7.0, 8.0, 12.0)),
    (4, "internal_carotid_right", "Artéria carótida interna direita", "Vasos", "#D32F2F", (20.0, 8.0, -30.0), (2.5, 2.5, 8.0)),
    (5, "optic_nerve_left", "Nervo óptico esquerdo", "Nervos e órbita", "#F2D22E", (-15.0, 52.0, -8.0), (2.0, 10.0, 2.0)),
)


def add_anatomy(manifest: dict, arrays: dict) -> None:
    """One unreviewed automatic label map for the MR volume (a paired L/R structure, four groups)."""
    x, y, z = _grid_ras(manifest)
    labels = np.zeros(x.shape, dtype=np.uint8)
    for value, _key, _name, _group, _color, (cx, cy, cz), (a, b, c) in ANATOMY_LABELS:
        labels[((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 + ((z - cz) / c) ** 2 <= 1.0] = value
    ml = _voxel_ml(manifest)
    mr = next(v for v in manifest["volumes"] if v["kind"] != "CT")
    manifest["anatomy"] = [{
        "id": "anat_mr", "blob": "anatomy_anat_mr", "for_volume": mr["id"], "source": "auto",
        "method": "elipsoides sintéticos (fixture)", "licence": "CC0", "reviewed": False,
        "labels": [{"value": v, "key": k, "name": n, "group": g, "color": col,
                    "volume_ml": round(float((labels == v).sum()) * ml, 3)}
                   for v, k, n, g, col, _c, _s in ANATOMY_LABELS],
    }]
    arrays["anatomy_anat_mr"] = labels


def _voxel_ml(manifest: dict) -> float:
    affine = np.asarray(manifest["grid"]["affine_ras"], dtype=float)
    return abs(float(np.linalg.det(affine[:3, :3]))) / 1000.0


def synthetic_tracts(lesion: tuple[float, float, float], seed: int = 11) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Bundle A: 300 S-I streamlines through the lesion. Bundle B: 200 A-P streamlines at patient left (x<0)."""
    rng = np.random.default_rng(seed)
    through = []
    for _ in range(300):
        r, theta = 7.0 * np.sqrt(rng.random()), rng.random() * 2 * np.pi
        dx, dy = r * np.cos(theta), r * np.sin(theta)
        z = np.linspace(lesion[2] - 45.0, lesion[2] + 45.0, 61)
        bend = 4.0 * np.sin((z - z[0]) / 90.0 * np.pi)
        through.append(np.stack([lesion[0] + dx + bend, lesion[1] + dy + 0 * z, z], axis=1))
    left = []
    for _ in range(200):
        r, theta = 5.0 * np.sqrt(rng.random()), rng.random() * 2 * np.pi
        y = np.linspace(-50.0, 50.0, 51)
        left.append(np.stack([-40.0 + r * np.cos(theta) + 0 * y, y, 20.0 + r * np.sin(theta) + 0.08 * y], axis=1))
    return through, left


def build_phantom(output: Path, seed: int = 5171, corridor_vessel: bool = False, crop: bool = False) -> Path:
    with tempfile.TemporaryDirectory(prefix="capsule-v2-") as tmp:
        tmp_path = Path(tmp)
        truth = write_phantom(tmp_path / "dicom", seed=seed)
        v1 = tmp_path / "phantom.capsule.html"
        subprocess.run([sys.executable, "-m", "capsule.cli", "build", str(tmp_path / "dicom"), "--series", "1,2",
                        "--label", "Fantoma sintetico", "-o", str(v1)], cwd=ROOT, check=True,
                       capture_output=True, text=True)
        manifest, arrays = read_capsule(v1)

    if crop:
        nx, ny, nz = manifest["grid"]["dims"]
        affine = np.asarray(manifest["grid"]["affine_ras"], dtype=float)
        corners = np.asarray([
            affine @ [i, j, k, 1.0]
            for i in (0, nx - 1) for j in (0, ny - 1) for k in (0, nz - 1)
        ])[:, :3]
        manifest["grid"]["crop"] = {
            "ras_mm": [*corners.min(axis=0).tolist(), *corners.max(axis=0).tolist()],
            "source": "ras", "margin_mm": 0.0,
        }

    lesion_center = tuple(float(v) for v in truth["lesion_center_ras_mm"])
    marker_center = tuple(float(v) for v in truth["marker_center_ras_mm"])
    lesion = _sphere_mask(manifest, lesion_center, 10.0)
    unreviewed = _sphere_mask(manifest, marker_center, 6.0)
    ml = _voxel_ml(manifest)
    manifest["masks"] = [
        {"id": "lesion", "blob": "mask_lesion", "label": "Lesão sintética revisada", "color": "#E4572E",
         "volume_ml": round(float(lesion.sum()) * ml, 3), "source": "synthetic", "reviewed": True},
        {"id": "unreviewed", "blob": "mask_unreviewed", "label": "Região sintética não revisada", "color": "#00FF00",
         "volume_ml": round(float(unreviewed.sum()) * ml, 3), "source": "synthetic", "reviewed": False},
    ]
    through, left = synthetic_tracts(lesion_center)
    manifest["tracts"] = [
        _tract_meta("through", "Feixe sintético pela lesão", "#4FC3F7", len(through), len(through), "synthetic", True),
        _tract_meta("left", "Feixe sintético à esquerda (não revisado)", "#7CFC00", len(left), len(left), "synthetic", False),
    ]
    # r8: the "left" bundle carries 20 short strays as filtered fibres (outlier_blob), hidden by default.
    strays = [line[20:28] + np.array([0.0, 0.0, 6.0]) for line in left[:20]]
    left_meta = manifest["tracts"][1]
    left_meta.update({"n_streamlines_source": len(left) + len(strays), "n_streamlines_outliers": len(strays),
                      "outlier_filter": "comprimento < 0,5 × mediana (fixture)", "outlier_blob": "tract_left_outliers",
                      "n_streamlines_outlier_blob": len(strays)})
    arrays["tract_left_outliers"] = _as_blob(tck_bytes(strays))
    # An unknown per-tract field: save must carry it through untouched.
    manifest["tracts"][0]["x_fixture_note"] = "campo desconhecido preservado"
    manifest["tour"] = [{
        "id": "tour-lesion", "title": "Lesão e feixe",
        "text": "A lesão sintética e o feixe revisado que passa por ela.",
        "view": {"layout": "3d", "crosshair_ras": list(lesion_center),
                 "window": {"volume": "ct", "center": 40, "width": 400},
                 "camera": {"azimuth": 110, "elevation": 15, "zoom": 1.0},
                 "visible_volumes": ["ct"], "visible_masks": ["lesion", "unreviewed"],
                 "visible_tracts": ["through", "left"], "visible_annotations": [], "clip": "sagittal"},
    }]
    manifest["synthetic_extension"] = {"preserve_on_save": True, "kind": "fixture-v2"}
    arrays["mask_lesion"] = lesion
    arrays["mask_unreviewed"] = unreviewed
    arrays["tract_through"] = _as_blob(tck_bytes(through))
    arrays["tract_left"] = _as_blob(tck_bytes(left))
    add_render_scene(manifest, arrays)
    if corridor_vessel:
        add_corridor_vessel(manifest, arrays)
    add_anatomy(manifest, arrays)
    write_capsule(TEMPLATE, output, manifest, arrays)
    return output


# ------------------------------------------------------------- large-grid mode
LARGE_DIMS = (340, 400, 320)
LARGE_SPACING_MM = 0.5


def build_large(output: Path, source: Path | None = None, dims: tuple[int, int, int] = LARGE_DIMS,
                spacing_mm: float = LARGE_SPACING_MM, template: Path = TEMPLATE) -> Path:
    """Memory fixture: the synthetic phantom capsule (CT + MR, head/brain/vessel render masks, anatomy)
    resampled by nearest neighbour onto a ~340x400x320 grid. Every voxel array is synthetic; tracts and
    manifest fields are carried over as they are. ``source`` is a phantom capsule built with the corridor
    vessel (built here when omitted); ``template`` picks the viewer to embed (baseline runs pass another)."""
    if source is None:
        with tempfile.TemporaryDirectory(prefix="capsule-large-") as tmp:
            return build_large(output, build_phantom(Path(tmp) / "src.capsule.html", corridor_vessel=True),
                               dims, spacing_mm, template)
    manifest, arrays = read_capsule(source)
    _, raw = _read_payload(source)
    for tract in manifest.get("tracts", []):  # read_capsule leaves tract blobs out: carry the bytes as they are
        for key in ("blob", "outlier_blob"):
            if tract.get(key):
                arrays[tract[key]] = _as_blob(raw[tract[key]])
    grid = manifest["grid"]
    old = np.asarray(grid["affine_ras"], dtype=float)
    assert np.allclose(old[:3, :3], np.diag(np.diag(old[:3, :3]))), "phantom grid is expected to be axis-aligned"
    old_dims = grid["dims"]
    affine = old.copy()
    for axis in range(3):
        sign = 1.0 if old[axis, axis] >= 0 else -1.0
        affine[axis, axis] = sign * spacing_mm
        centre = old[axis, axis] * (old_dims[axis] - 1) / 2 + old[axis, 3]  # the phantom's centre stays the centre
        affine[axis, 3] = centre - affine[axis, axis] * (dims[axis] - 1) / 2
    idx = []
    for axis in range(3):
        ras = affine[axis, axis] * np.arange(dims[axis]) + affine[axis, 3]
        idx.append(np.clip(np.rint((ras - old[axis, 3]) / old[axis, axis]), 0, old_dims[axis] - 1).astype(np.intp))
    grid["dims"] = list(dims)
    grid["affine_ras"] = affine.tolist()
    if "spacing_mm" in grid:
        grid["spacing_mm"] = [spacing_mm] * 3
    if grid.get("crop"):
        del grid["crop"]
    ml = spacing_mm ** 3 / 1000.0
    # A large real capsule carries head, brain and vessel render masks for the CT base and nothing else on the
    # grid: keep exactly those (remapped to the CT), drop the lesion/marker/tumour masks, the anatomy and the tour.
    ct = next(v["id"] for v in manifest["volumes"] if v["kind"] == "CT")
    keep = {"head": "head", "brain_mr": "brain_ct", "vessels_mr": "vessels_ct"}
    masks = []
    for meta in manifest["masks"]:
        if meta["id"] in keep:
            arrays[f"mask_{keep[meta['id']]}"] = arrays.pop(meta["blob"]) if meta["id"] != "head" else arrays[meta["blob"]]
            meta.update(id=keep[meta["id"]], blob=f"mask_{keep[meta['id']]}", for_volume=ct)
            masks.append(meta)
    manifest["masks"] = masks
    for meta in manifest.pop("anatomy", []):
        arrays.pop(meta["blob"], None)
    for blob in [b for b in arrays if b.startswith("mask_") and b not in {m["blob"] for m in masks}]:
        del arrays[blob]
    manifest["tour"] = []
    big = {}
    for blob, array in arrays.items():
        if array.shape == (old_dims[2], old_dims[1], old_dims[0]):
            big[blob] = np.ascontiguousarray(array[np.ix_(idx[2], idx[1], idx[0])])
        else:
            big[blob] = array
    for meta in manifest.get("masks", []) + manifest.get("anatomy", []):
        if "volume_ml" in meta and meta.get("blob") in big:
            meta["volume_ml"] = round(float(np.count_nonzero(big[meta["blob"]])) * ml, 3)
    write_capsule(template, output, manifest, big)
    return output


# -------------------------------------------------------------- real-data mode
def build_nifti(output: Path, nifti: Path, tck: Path, label: str, seed: int = 3) -> Path:
    import nibabel as nib
    from capsule.resample import pack_scalar, scalar_stats

    image = nib.load(str(nifti))
    data = np.asarray(image.get_fdata(dtype=np.float32))
    if data.ndim == 4:
        data = data[..., 0]
    nx, ny, nz = data.shape
    zyx = np.ascontiguousarray(np.transpose(data, (2, 1, 0)))
    packed, slope, intercept = pack_scalar(zyx, "MR")
    # Skull-stripped T1: ignore background and interpolation dust (<0.1% of max) for the auto window.
    tissue = zyx > 1e-3 * float(np.nanmax(zyx))
    stats = scalar_stats(zyx[tissue] if np.any(tissue) else zyx)
    affine = image.affine.astype(float)
    tract = nib.streamlines.load(str(tck))
    lines = list(tract.streamlines)
    n_source = len(lines)
    if n_source > MAX_STREAMLINES:
        pick = np.sort(np.random.default_rng(seed).choice(n_source, MAX_STREAMLINES, replace=False))
        lines = [lines[i] for i in pick]
    center = ((p01 := stats["p01"]) + stats["p99"]) / 2
    manifest = {
        "schema": "case-capsule/1", "version": 1, "generator": "viewer2 dev_fixture", "locale": "pt-BR",
        "case": {"label": label, "anonymized": True},
        "grid": {"dims": [nx, ny, nz], "spacing_mm": [float(v) for v in image.header.get_zooms()[:3]],
                 "affine_ras": affine.tolist()},
        "volumes": [{"id": "t1", "blob": "t1", "kind": "MR", "label": "T1", "dtype": "uint16",
                     "slope": slope, "intercept": intercept, "units": "a.u.",
                     "series": {"description": "T1 (dados públicos)", "modality": "MR", "frames_used": "all"},
                     "stats": stats,
                     "window_presets": [{"name": "Auto", "center": center, "width": max(1.0, stats["p99"] - p01)}],
                     "registration": {"reference": True}}],
        "masks": [], "annotations": [],
        "tracts": [_tract_meta("cst_l", "Trato corticoespinal esquerdo", "#E8C547", len(lines), n_source,
                               "tractlab:" + tck.name, False)],
        "tour": [],
    }
    write_capsule(TEMPLATE, output, manifest, {"t1": packed, "tract_cst_l": _as_blob(tck_bytes(lines))})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--nifti", type=Path)
    parser.add_argument("--tck", type=Path)
    parser.add_argument("--label", default="Dados públicos")
    parser.add_argument("--seed", type=int, default=5171)
    parser.add_argument("--corridor-vessel", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--large", action="store_true", help="resample the phantom onto a 340x400x320 grid (memory fixture)")
    parser.add_argument("--template", type=Path, default=TEMPLATE, help="viewer template to embed (default: current)")
    parser.add_argument("--crop", action="store_true", help="record the synthetic grid bounds as a crop")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    if args.large:
        build_large(args.output, template=args.template)
        print(args.output)
        return
    if bool(args.nifti) != bool(args.tck):
        parser.error("--nifti and --tck go together")
    if args.nifti:
        build_nifti(args.output, args.nifti.expanduser(), args.tck.expanduser(), args.label)
    else:
        build_phantom(args.output, args.seed, args.corridor_vessel, args.crop)
    print(args.output)


if __name__ == "__main__":
    main()
