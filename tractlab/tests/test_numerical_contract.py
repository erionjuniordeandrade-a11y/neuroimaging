"""Generated-data regressions for the audited numerical grid contracts.

These tests never read a case tree or invoke tractography.  The native MIF
oracle is generated from a tiny NIfTI only when an MRtrix installation is
available; the NIfTI affine remains the reference for that oracle.
"""

from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from tractlab.delta_qc import compute_delta
from tractlab.fidelity import FidelityRefusal, build_sidecar, segment_ratios, summaries_for_bank
from tractlab.fidelity_sweep import _marked_frac
from tractlab.check_registration import auto_sanity
from tractlab.grid import Grid, grid_id, load_grid
from tractlab.seed import rasterize_points, write_seed_nifti
from tractlab.serve import TrackService


AFF = np.diag([2.0, 2.0, 2.0, 1.0])
Y00 = 1.0 / np.sqrt(4.0 * np.pi)


def _write_nii(
    path: Path,
    data: np.ndarray,
    affine: np.ndarray = AFF,
    *,
    spatial_unit: str | None = "mm",
    affine_codes: bool = True,
) -> None:
    image = nib.Nifti1Image(np.asarray(data), np.asarray(affine))
    image.header.set_xyzt_units(spatial_unit, "sec")
    if not affine_codes:
        image.set_sform(np.asarray(affine), code=0)
        image.set_qform(np.asarray(affine), code=0)
    nib.save(image, str(path))


def _manifest_case(
    tmp_path: Path,
    *,
    shifted: str | None = None,
    spatial_unit: str | None = "mm",
    assume_unknown_spatial_units_mm: object = None,
    affine_codes: bool = True,
) -> Path:
    shape = (8, 8, 8)
    b0 = np.linspace(0.0, 1.0, int(np.prod(shape)), dtype=np.float32).reshape(shape)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[1:7, 1:7, 1:7] = 1
    fod = np.zeros(shape + (45,), dtype=np.float32)
    fod[..., 0] = 2.0
    lesion = np.zeros(shape, dtype=np.uint8)
    lesion[2:6, 2:6, 2:6] = 1
    t1 = b0.copy()
    paths = {"b0": tmp_path / "b0.nii.gz", "mask": tmp_path / "mask.nii.gz",
             "fod": tmp_path / "fod.nii.gz", "lesion": tmp_path / "lesion.nii.gz",
             "t1": tmp_path / "t1.nii.gz"}
    for key, data in (("b0", b0), ("mask", mask), ("fod", fod),
                      ("lesion", lesion), ("t1", t1)):
        affine = AFF.copy()
        if shifted == key:
            affine[0, 3] += 2.0
        _write_nii(
            paths[key], data, affine, spatial_unit=spatial_unit,
            affine_codes=affine_codes,
        )
    manifest = {
        "case_id": "generated-numerical-contract",
        "case_root": str(tmp_path),
        "inputs": {key: {"path": path.name} for key, path in paths.items()},
    }
    if assume_unknown_spatial_units_mm is not None:
        manifest["acquisition"] = {
            "assume_unknown_spatial_units_mm": assume_unknown_spatial_units_mm,
        }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


@pytest.mark.parametrize("shifted", ["mask", "fod", "lesion"])
def test_track_service_rejects_shifted_core_grid(tmp_path, shifted):
    with pytest.raises(ValueError, match="grid|affine"):
        TrackService(str(_manifest_case(tmp_path, shifted=shifted)))


def test_track_service_accepts_aligned_generated_grid_and_seed_reference(tmp_path):
    service = TrackService(str(_manifest_case(tmp_path)))
    assert service.grid.shape == (8, 8, 8)
    assert np.allclose(service.grid.affine, AFF)
    assert grid_id(service.grid) == grid_id(load_grid(str(tmp_path / "b0.nii.gz")))
    seed = rasterize_points(service.grid, np.asarray([[6.0, 6.0, 6.0]]), radius_mm=0.25)
    seed_path = tmp_path / "painted-seed.nii.gz"
    write_seed_nifti(seed, service.mask, str(seed_path))
    seed_img = nib.load(str(seed_path))
    b0_img = nib.load(str(tmp_path / "b0.nii.gz"))
    np.testing.assert_allclose(seed_img.affine, b0_img.affine)
    np.testing.assert_allclose(seed_img.get_qform(), b0_img.get_qform())
    np.testing.assert_allclose(seed_img.get_sform(), b0_img.get_sform())


def test_track_service_rejects_unknown_units_without_manifest_opt_in(tmp_path):
    manifest_path = _manifest_case(tmp_path, spatial_unit=None)
    with pytest.raises(ValueError, match="assume_unknown_spatial_units_mm"):
        TrackService(str(manifest_path))


def test_track_service_requires_strict_boolean_unit_opt_in(tmp_path):
    manifest_path = _manifest_case(
        tmp_path, spatial_unit=None, assume_unknown_spatial_units_mm="true",
    )
    with pytest.raises(ValueError, match="strict boolean"):
        TrackService(str(manifest_path))


def test_track_service_exposes_legacy_mm_assumption(tmp_path):
    manifest_path = _manifest_case(
        tmp_path, spatial_unit=None, assume_unknown_spatial_units_mm=True,
    )
    service = TrackService(str(manifest_path))
    assert service.grid.spatial_unit == "unknown"
    assert service.grid.unit_semantics == "legacy_assumed_mm"
    assert service.grid.numeric_world_unit == "mm"
    assert service.grid_provenance["assume_unknown_spatial_units_mm"] is True
    assert service.grid_provenance["unit_status"] == "legacy_assumed_mm"


def test_track_service_rejects_missing_sform_and_qform_codes(tmp_path):
    manifest_path = _manifest_case(
        tmp_path, spatial_unit=None, assume_unknown_spatial_units_mm=True,
        affine_codes=False,
    )
    with pytest.raises(ValueError, match="sform_code|qform_code|authoritative"):
        TrackService(str(manifest_path))


def test_track_service_rejects_shifted_signed_t1(tmp_path):
    manifest_path = _manifest_case(tmp_path, shifted="t1")
    manifest = json.loads(manifest_path.read_text())
    manifest["t1_qc"] = {"approved_by": "synthetic-owner", "date": "2026-09-06",
                           "sheet_sha": "synthetic"}
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="grid|affine"):
        TrackService(str(manifest_path))


def test_auto_sanity_rejects_shifted_mask(tmp_path):
    _manifest_case(tmp_path, shifted="mask")
    result = auto_sanity(
        b0_path=str(tmp_path / "b0.nii.gz"),
        t1_path=str(tmp_path / "t1.nii.gz"),
        mask_path=str(tmp_path / "mask.nii.gz"),
        expected_grid_id=None,
    )
    assert result.ok is False
    assert any("grid" in note or "affine" in note for note in result.notes)


@pytest.mark.parametrize("bad_affine", [
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
    np.array([[np.nan, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
])
def test_grid_rejects_singular_or_nonfinite_affine(bad_affine):
    with pytest.raises(ValueError, match="affine"):
        Grid(shape=(2, 2, 2), affine=bad_affine, axcodes=("R", "A", "S"))


def test_delta_qc_rejects_shifted_same_shape_mask():
    field = nib.Nifti1Image(np.full((4, 4, 4), 20.0, dtype=np.float32), AFF)
    field.header.set_xyzt_units("mm", "sec")
    shifted = AFF.copy()
    shifted[0, 3] += 2.0
    mask = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.uint8), shifted)
    mask.header.set_xyzt_units("mm", "sec")
    with pytest.raises(ValueError, match="grid|affine"):
        compute_delta(field, mask, readout_time_s=0.1, pe_axis=0)


def test_delta_qc_aligned_control_retains_physical_shift():
    field = nib.Nifti1Image(np.full((4, 4, 4), 20.0, dtype=np.float32), AFF)
    mask = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.uint8), AFF)
    field.header.set_xyzt_units("mm", "sec")
    mask.header.set_xyzt_units("mm", "sec")
    stats = compute_delta(field, mask, readout_time_s=0.1, pe_axis=0)
    assert stats.n_vox == 4 * 4 * 4
    assert stats.median_mm == pytest.approx(4.0)


def test_delta_qc_unknown_units_require_explicit_assumption():
    field = nib.Nifti1Image(np.full((4, 4, 4), 20.0, dtype=np.float32), AFF)
    mask = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.uint8), AFF)
    with pytest.raises(ValueError, match="assume_unknown_spatial_units_mm"):
        compute_delta(field, mask, readout_time_s=0.1, pe_axis=0)
    stats = compute_delta(
        field,
        mask,
        readout_time_s=0.1,
        pe_axis=0,
        assume_unknown_spatial_units_mm=True,
    )
    assert stats.median_mm == pytest.approx(4.0)


def test_delta_qc_rejects_known_spatial_unit_mismatch():
    field = nib.Nifti1Image(np.full((4, 4, 4), 20.0, dtype=np.float32), AFF)
    mask = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.uint8), AFF)
    field.header.set_xyzt_units("mm", "sec")
    mask.header.set_xyzt_units("meter", "sec")
    with pytest.raises(ValueError, match="spatial unit"):
        compute_delta(field, mask, readout_time_s=0.1, pe_axis=0)


def test_grid_id_binds_unit_semantics(tmp_path):
    known = tmp_path / "known.nii.gz"
    unknown = tmp_path / "unknown.nii.gz"
    _write_nii(known, np.zeros((2, 2, 2)), spatial_unit="mm")
    _write_nii(unknown, np.zeros((2, 2, 2)), spatial_unit=None)
    from tractlab.grid import grid_from_image

    known_grid = grid_from_image(nib.load(str(known)))
    unknown_grid = grid_from_image(nib.load(str(unknown)))
    assert known_grid.numeric_world_unit == unknown_grid.numeric_world_unit == "mm"
    assert known_grid.unit_semantics == "explicit_mm"
    assert unknown_grid.unit_semantics == "legacy_assumed_mm"
    assert grid_id(known_grid) != grid_id(unknown_grid)


@pytest.mark.parametrize("unit", ["meter", "micron"])
def test_explicit_non_mm_units_are_rejected_even_when_declared(unit, tmp_path):
    path = tmp_path / f"{unit}.nii.gz"
    _write_nii(path, np.zeros((2, 2, 2)), spatial_unit=unit)
    with pytest.raises(ValueError, match="millimetres|spatial unit"):
        load_grid(str(path))


def test_mrinfo_number_parser_rejects_trailing_junk():
    from tractlab.grid import _parse_mrinfo_numbers

    with pytest.raises(ValueError, match="malformed"):
        _parse_mrinfo_numbers("1 2 trailing", option="-size", path="generated.mif")


def test_fidelity_sweep_excludes_unmeasurable_rows_from_low_support_fraction():
    col = np.asarray([[np.nan], [0.25]], dtype=np.float32)
    assert _marked_frac(col, 0, 0.70) == pytest.approx(1.0)


def test_fidelity_summaries_keep_partial_invalid_streamlines_unmeasurable(tmp_path):
    sh = _iso_sh((4, 4, 4))
    peak = np.ones((4, 4, 4), dtype=np.float64)
    sh_path = tmp_path / "fod.nii.gz"
    peak_path = tmp_path / "peak.nii.gz"
    _write_nii(sh_path, sh, affine=np.eye(4))
    _write_nii(peak_path, peak, affine=np.eye(4))
    lines = [
        np.asarray([[100.0, 1.0, 1.0], [102.0, 1.0, 1.0]], dtype=np.float32),
        np.asarray([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]], dtype=np.float32),
    ]
    tck_path = tmp_path / "bank.tck"
    nib.streamlines.save(
        nib.streamlines.Tractogram(lines, affine_to_rasmm=np.eye(4)),
        str(tck_path),
    )
    bank_sha = hashlib.sha256(tck_path.read_bytes()).hexdigest()
    fod_sha = hashlib.sha256(sh_path.read_bytes()).hexdigest()
    provenance = {"bank_sha256": bank_sha, "fod_sha256": fod_sha}
    summaries = summaries_for_bank(
        tck_path, sh_path, peak_path, provenance=provenance,
    )
    assert np.isnan(summaries["p5_ratio"][0])
    assert np.isfinite(summaries["p5_ratio"][1])
    sidecar = build_sidecar(
        tck_path, sh_path, peak_path, provenance=provenance,
    )
    assert np.isnan(sidecar["frac_ge"][0]).all()
    assert np.isfinite(sidecar["frac_ge"][1]).all()


def _iso_sh(shape=(4, 4, 4)):
    sh = np.zeros(tuple(shape) + (1,), dtype=np.float64)
    sh[..., 0] = 1.0 / Y00
    return sh


def test_fidelity_out_of_field_is_unmeasurable_not_edge_scored():
    sh = _iso_sh()
    peak = np.ones((4, 4, 4), dtype=np.float64)
    line = np.array([[100.0, 1.0, 1.0], [102.0, 1.0, 1.0]])
    ratios = segment_ratios(line, sh, np.eye(4), peak, np.eye(4))
    assert ratios.shape == (1,)
    assert np.isnan(ratios[0])


def test_fidelity_rejects_sh_peak_affine_mismatch():
    sh = _iso_sh()
    peak = np.ones((4, 4, 4), dtype=np.float64)
    shifted = np.eye(4)
    shifted[0, 3] = 1.0
    with pytest.raises(FidelityRefusal, match="grid|affine"):
        segment_ratios(
            np.array([[0.5, 1.0, 1.0], [1.5, 1.0, 1.0]]),
            sh, np.eye(4), peak, shifted,
        )


def test_fidelity_unrelated_peak_nan_does_not_zero_valid_segment():
    sh = _iso_sh()
    peak = np.ones((4, 4, 4), dtype=np.float64)
    peak[0, 0, 0] = np.nan
    line = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]])
    ratios = segment_ratios(line, sh, np.eye(4), peak, np.eye(4))
    assert ratios.shape == (1,)
    assert ratios[0] == pytest.approx(1.0)


def test_fidelity_sampled_nan_is_unmeasurable():
    sh = _iso_sh()
    peak = np.ones((4, 4, 4), dtype=np.float64)
    peak[1, 1, 1] = np.nan
    line = np.array([[0.5, 1.0, 1.0], [1.5, 1.0, 1.0]])
    ratios = segment_ratios(line, sh, np.eye(4), peak, np.eye(4))
    assert np.isnan(ratios[0])


@pytest.mark.skipif(
    shutil.which("mrconvert") is None
    and not (Path.home() / "mrtrix3/bin/mrconvert").is_file(),
    reason="MRtrix mrconvert is required for the native MIF oracle",
)
def test_native_mif_grid_oracle_preserves_orientation_and_strides(tmp_path):
    """MIF transform+spacing is the logical grid; strides describe storage."""
    from tractlab.grid import load_mif_grid

    mrconvert = shutil.which("mrconvert") or str(Path.home() / "mrtrix3/bin/mrconvert")
    source = tmp_path / "source.nii.gz"
    data = np.zeros((3, 4, 5, 2), dtype=np.float32)
    theta = np.deg2rad(25.0)
    rotation = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    affine = np.array([
        [0.0, 0.0, 0.0, 12.0],
        [0.0, 0.0, 0.0, -7.0],
        [0.0, 0.0, 0.0, 22.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    affine[:3, :3] = rotation @ np.diag([2.0, 3.0, 5.0])
    img = nib.Nifti1Image(data, affine)
    img.header.set_xyzt_units("mm", "sec")
    nib.save(img, str(source))
    for tag, strides in {
        "positive": "1,2,3,4",
        "negative": "-1,2,3,4",
        "permuted": "2,3,1,4",
        "negative_permuted": "-3,-1,-2,4",
    }.items():
        mif = tmp_path / f"source-{tag}.mif"
        subprocess.run(
            [mrconvert, str(source), str(mif), "-strides", strides,
             "-force", "-quiet"],
            check=True,
            timeout=20,
        )
        back = tmp_path / f"source-{tag}.nii.gz"
        subprocess.run(
            [mrconvert, str(mif), str(back), "-force", "-quiet"],
            check=True,
            timeout=20,
        )
        expected = load_grid(str(back))
        actual = load_mif_grid(str(mif))
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual.affine, expected.affine, atol=1e-5)

    # A negative source orientation is the production-shaped case: MRtrix
    # realigns the transform and reports a negative symbolic stride. The
    # canonical reconstruction must still agree with a generated NIfTI oracle.
    negative_source = tmp_path / "negative-source.nii.gz"
    negative_affine = affine.copy()
    negative_affine[:3, :3] = rotation @ np.diag([-2.0, 3.0, 5.0])
    negative_img = nib.Nifti1Image(data, negative_affine)
    negative_img.header.set_xyzt_units("mm", "sec")
    nib.save(negative_img, str(negative_source))
    negative_mif = tmp_path / "negative-source.mif"
    negative_back = tmp_path / "negative-source-back.nii.gz"
    subprocess.run(
        [mrconvert, str(negative_source), str(negative_mif), "-force", "-quiet"],
        check=True,
        timeout=20,
    )
    subprocess.run(
        [mrconvert, str(negative_mif), str(negative_back), "-force", "-quiet"],
        check=True,
        timeout=20,
    )
    negative_expected = load_grid(str(negative_back))
    negative_actual = load_mif_grid(str(negative_mif))
    assert negative_actual.shape == negative_expected.shape
    np.testing.assert_allclose(negative_actual.affine, negative_expected.affine, atol=1e-5)

    # The same native header path is exercised at the startup boundary.  The
    # generated case is disposable and contains no private image data.
    case = tmp_path / "mif-service"
    case.mkdir()
    manifest_path = _manifest_case(case)
    case_mif = case / "fod.mif"
    subprocess.run(
        [mrconvert, str(case / "fod.nii.gz"), str(case_mif), "-force", "-quiet"],
        check=True,
        timeout=20,
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["inputs"]["fod"]["path"] = case_mif.name
    manifest_path.write_text(json.dumps(manifest))
    service = TrackService(str(manifest_path))
    assert service.grid.shape == (8, 8, 8)
