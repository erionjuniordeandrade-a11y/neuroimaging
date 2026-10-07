"""Render and lesion masks on the real public demo inputs, through the production build path.

Data-gated: skipped when the public inputs are absent. Every derived number has a control:
the CT table check runs on the plain threshold (which must contain the table), the MR head check
needs detached foreground (the Leksell fiducials) to exist, and tumour enhancement is compared with
both a 3 mm ring and the mirrored contralateral ROI.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pytest
import SimpleITK as sitk

from capsule import cli, masks
from capsule.ingest import convert_series, scan_series
from capsule.pack import read_capsule
from capsule.register import register
from capsule.resample import common_grid

DATA = Path(__file__).resolve().parent.parent / "data" / "public"
CPTAC = DATA / "CPTAC-AML" / "head-ct"
CQ500 = DATA / "CQ500" / "head-ct-thin"
VS = DATA / "Vestibular-Schwannoma-SEG"
RT_T1 = VS / "rtstruct" / "rtstruct_T1.dcm"
RT_T2 = VS / "rtstruct" / "rtstruct_T2.dcm"


def _build(tmp: Path, source: Path, series: str, *extra: str) -> tuple[dict, dict, sitk.Image]:
    output = tmp / "x.capsule.html"
    assert cli.main(["build", str(source), "--series", series, "--label", "t", "--viewer", "v2",
                     "--brain-mask", "none", "-o", str(output), *extra]) == 0
    manifest, arrays = read_capsule(output)
    return manifest, arrays, masks.grid_from_manifest(manifest)


def _values(manifest: dict, arrays: dict, volume_id: str) -> np.ndarray:
    volume = next(v for v in manifest["volumes"] if v["id"] == volume_id)
    return arrays[volume["blob"]].astype(np.float64) * volume["slope"] + volume["intercept"]


def _mask(manifest: dict, arrays: dict, mask_id: str) -> tuple[dict, np.ndarray]:
    entry = next(m for m in manifest["masks"] if m["id"] == mask_id)
    return entry, arrays[entry["blob"]].astype(bool)


@pytest.mark.parametrize("source,series", [(CPTAC, "2"), (CQ500, "3")], ids=["cptac-aml", "cq500-thin"])
def test_ct_head_render_mask_excludes_table(tmp_path, source, series):
    if not source.is_dir():
        pytest.skip(f"{source} not present")
    manifest, arrays, grid = _build(tmp_path, source, series)
    entry, head = _mask(manifest, arrays, "head")
    assert entry["role"] == "render" and entry["for_volume"] == "ct" and entry["source"] == "auto"
    assert [m["id"] for m in manifest["masks"]] == ["head"]
    hu = _values(manifest, arrays, "ct")
    render = masks.posterior_disconnected_fraction(head, hu, grid)
    plain = hu > -500
    control = masks.posterior_disconnected_fraction(plain, hu, grid)
    y = masks.ras_axis_grid(grid, 1)
    slab = y < float(y[plain].min()) + 15.0
    table = plain & slab & ~head
    print(f"{source.parent.name}: render slab {render}; plain-threshold control {control}; "
          f"plain voxels in slab excluded by the render mask {int(table.sum())}; head {entry['volume_ml']} mL")
    assert render["fraction"] == 0.0
    # Control: the plain threshold does include the table in the posterior slab.
    assert table.sum() > 1000
    assert 1500 < entry["volume_ml"] < 6000
    # The render mask never removes bone of the skull component.
    assert (head & (hu > 300)).sum() > 0.9 * ((hu > 300) & ~slab).sum()


def test_ct_table_control_is_detached_on_cptac(tmp_path):
    """Known positive: on CPTAC-AML the table is a separate component in the plain threshold."""
    if not CPTAC.is_dir():
        pytest.skip("CPTAC-AML head CT not present")
    manifest, arrays, grid = _build(tmp_path, CPTAC, "2")
    hu = _values(manifest, arrays, "ct")
    assert masks.posterior_disconnected_fraction(hu > -500, hu, grid)["fraction"] > 0.3


@pytest.fixture(scope="module")
def vs_capsule(tmp_path_factory):
    if not (VS / "brain-mri").is_dir() or not RT_T1.is_file():
        pytest.skip("Vestibular-Schwannoma-SEG MR/RTSTRUCT not present (scripts/fetch_vs_rtstruct.py)")
    return _build(tmp_path_factory.mktemp("vs"), VS / "brain-mri", "2,3", "--rtstruct", str(RT_T1))


def test_mr_head_masks_exclude_detached_fiducials(vs_capsule):
    manifest, arrays, grid = vs_capsule
    ids = [m["id"] for m in manifest["masks"]]
    assert ids == ["tumour", "head", "head_mr_3"]  # brain masks disabled in this build (--brain-mask none)
    values = _values(manifest, arrays, "mr")
    entry, head = _mask(manifest, arrays, "head")
    assert entry["for_volume"] == "mr" and entry["role"] == "render"
    foreground = values > masks.mr_head_threshold(values)
    labels = sitk.GetArrayFromImage(sitk.ConnectedComponent(masks._image(foreground, grid)))
    near = sitk.GetArrayFromImage(sitk.BinaryDilate(masks._image(head, grid), [10] * 3)).astype(bool)
    outside = np.setdiff1d(np.unique(labels[foreground & ~near]), np.unique(labels[near & foreground]))
    sizes = np.bincount(labels.ravel())[outside]
    print(f"MR head {entry['volume_ml']} mL; foreground components >10 mm from the head: "
          f"{len(outside)} (>=20 voxels: {int((sizes >= 20).sum())})")
    # Control: detached foreground (frame fiducials) exists on this input ...
    assert (sizes >= 20).sum() >= 2
    # ... and none of it is in the head mask.
    assert not head[np.isin(labels, outside)].any()
    assert 2500 < entry["volume_ml"] < 5500
    _, head_t2 = _mask(manifest, arrays, "head_mr_3")
    assert _mask(manifest, arrays, "head_mr_3")[0]["for_volume"] == "mr_3"
    # T2 is a 60 mm slab; its head mask must cover that slab's tissue, not only bright CSF.
    t2 = _values(manifest, arrays, "mr_3")
    assert head_t2.sum() > 0.5 * (t2 > 0).sum() * 0.5


def _tumour(vs_capsule):
    manifest, arrays, grid = vs_capsule
    entry, tumour = _mask(manifest, arrays, "tumour")
    return manifest, arrays, grid, entry, tumour


def test_tumour_mask_contract_and_volume(vs_capsule):
    manifest, arrays, grid, entry, tumour = _tumour(vs_capsule)
    assert entry["label"] == "Schwannoma vestibular (dataset)" and entry["color"] == "#E4572E"
    assert entry["source"] == "dataset" and entry["reviewed"] is False and entry["role"] == "lesion"
    assert entry["for_volume"] == "mr" and entry["roi"] == "AN"
    print(f"tumour: grid {entry['volume_ml']} mL, native raster {entry['native_volume_ml']} mL, "
          f"planar contour (shoelace x slice spacing) {entry['planar_volume_ml']} mL")
    # The dataset reports no volume; the independent reference is the planar contour volume.
    assert abs(entry["volume_ml"] - entry["planar_volume_ml"]) <= 0.10 * entry["planar_volume_ml"]
    assert abs(entry["native_volume_ml"] - entry["planar_volume_ml"]) <= 0.10 * entry["planar_volume_ml"]


def test_tumour_centroid_in_cpa_on_dataset_side(vs_capsule):
    manifest, arrays, grid, entry, tumour = _tumour(vs_capsule)
    _, head = _mask(manifest, arrays, "head")
    centroid = masks.voxel_ras(grid, np.argwhere(tumour)).mean(axis=0)
    structure = masks.read_rtstruct(RT_T1)
    contour = masks.contour_centroid_lps(structure) * np.array([-1, -1, 1])  # LPS -> RAS
    head_centre = masks.voxel_ras(grid, np.argwhere(head)).mean(axis=0)
    lateral = centroid[0] - head_centre[0]
    print(f"tumour centroid RAS {np.round(centroid, 1)}; contour-point centroid {np.round(contour, 1)} "
          f"(rasterizer check only); lateral offset from head midline {lateral:.1f} mm; "
          f"offset below head centre {head_centre[2] - centroid[2]:.1f} mm")
    assert np.linalg.norm(centroid - contour) < 3.0
    assert np.sign(lateral) == np.sign(contour[0] - head_centre[0])
    assert 10.0 < abs(lateral) < 40.0  # cerebellopontine angle / internal auditory canal, not midline
    assert head_centre[2] - centroid[2] > 10.0  # infratentorial


def test_tumour_enhances_versus_ring_and_mirror(vs_capsule):
    manifest, arrays, grid, entry, tumour = _tumour(vs_capsule)
    t1 = _values(manifest, arrays, "mr")
    _, head = _mask(manifest, arrays, "head")
    dilated = sitk.GetArrayFromImage(sitk.BinaryDilate(masks._image(tumour, grid), [3] * 3)).astype(bool)
    ring = dilated & ~tumour
    points = masks.voxel_ras(grid, np.argwhere(tumour))
    midline = masks.voxel_ras(grid, np.argwhere(head)).mean(axis=0)[0]
    points[:, 0] = 2 * midline - points[:, 0]
    affine = np.asarray(manifest["grid"]["affine_ras"])
    ijk = np.rint((points - affine[:3, 3]) @ np.linalg.inv(affine[:3, :3]).T).astype(int)
    mirror = t1[ijk[:, 2], ijk[:, 1], ijk[:, 0]]
    inside, around = t1[tumour].mean(), t1[ring].mean()
    print(f"T1+Gd mean: tumour {inside:.1f}, 3 mm ring {around:.1f}, mirrored contralateral ROI {mirror.mean():.1f}")
    assert inside > around and inside > mirror.mean()


def test_t2_rtstruct_agrees_after_registration():
    """The T2 delineation, rasterized on T2 and carried by the production registration, overlaps T1's."""
    if not RT_T2.is_file() or not RT_T1.is_file():
        pytest.skip("RTSTRUCTs not present")
    series = {s.number: s for s in scan_series(VS / "brain-mri")[0]}
    with tempfile.TemporaryDirectory() as temporary:
        t1 = sitk.ReadImage(str(convert_series(series[2], Path(temporary))))
        t2 = sitk.ReadImage(str(convert_series(series[3], Path(temporary))))
    grid = common_grid(t1, "MR", 1.0).image
    transform, _, _ = register(t1, t2)
    a = masks.to_grid(masks.rasterize(masks.read_rtstruct(RT_T1), t1), t1, grid, sitk.Transform(3, sitk.sitkIdentity))
    b = masks.to_grid(masks.rasterize(masks.read_rtstruct(RT_T2), t2), t2, grid, transform)
    dice = 2 * (a & b).sum() / (a.sum() + b.sum())
    print(f"T1 vs T2 RTSTRUCT on the grid: {a.sum()} / {b.sum()} voxels, Dice {dice:.3f}")
    assert dice > 0.5
