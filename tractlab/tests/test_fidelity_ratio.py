"""E1 — scale-invariant support ratios on SOURCE geometry (never packed)."""
import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from tractlab.fidelity import (
    FidelityRefusal,
    segment_ratios,
    summaries_for_bank,
)

Y00 = 0.28209479177387814  # 1/sqrt(4*pi)
AFF = np.diag([2.0, 2.0, 2.0, 1.0])
NCOEF = 45  # lmax 8


def _iso_sh(c0: float = 2.0) -> np.ndarray:
    sh = np.zeros((8, 8, 8, NCOEF))
    sh[..., 0] = c0
    return sh


def _peak_of(sh: np.ndarray) -> np.ndarray:
    # l=0-only field: amplitude is isotropic, peak = c0 * Y00 per voxel
    return sh[..., 0] * Y00


def test_isotropic_fod_gives_ratio_one():
    sh = _iso_sh()
    peak = _peak_of(sh)
    line = np.array([[3.0, 7.0, 7.0], [11.0, 7.0, 7.0], [12.0, 8.0, 7.0]])
    ratios = segment_ratios(line, sh, AFF, peak, AFF)
    assert ratios.shape == (2,)
    assert np.all(ratios > 0.95)


def test_zeroed_slab_drops_ratio():
    sh = _iso_sh()
    sh[3:5, :, :, :] = 0.0  # slab with no fODF support at all
    peak = _peak_of(sh)
    inside = np.array([[7.0, 7.0, 7.0], [8.5, 7.0, 7.0]])  # midpoint in slab
    ratios = segment_ratios(inside, sh, AFF, peak, AFF)
    assert ratios.min() < 0.05


def test_degenerate_line_gives_empty_ratios():
    sh = _iso_sh()
    peak = _peak_of(sh)
    assert segment_ratios(np.array([[3.0, 3.0, 3.0]]), sh, AFF, peak, AFF).shape == (0,)


def _write_tck(path: Path, lines) -> None:
    t = nib.streamlines.Tractogram(lines, affine_to_rasmm=np.eye(4))
    nib.streamlines.save(t, str(path))


def _write_nii(path: Path, arr: np.ndarray) -> None:
    image = nib.Nifti1Image(arr.astype(np.float32), AFF)
    image.header.set_xyzt_units("mm", "sec")
    nib.save(image, str(path))


@pytest.fixture()
def bank_case(tmp_path):
    sh = _iso_sh()
    _write_nii(tmp_path / "fod.nii.gz", sh)
    _write_nii(tmp_path / "peak.nii.gz", _peak_of(sh))
    _write_tck(tmp_path / "bank.tck",
               [np.array([[3.0, 7.0, 7.0], [11.0, 7.0, 7.0]], dtype=np.float32)])
    fod_sha = hashlib.sha256((tmp_path / "fod.nii.gz").read_bytes()).hexdigest()
    return tmp_path, fod_sha


def test_refuses_without_provenance(bank_case):
    case, _sha = bank_case
    with pytest.raises(FidelityRefusal, match="provenance"):
        summaries_for_bank(
            case / "bank.tck", case / "fod.nii.gz", case / "peak.nii.gz",
            provenance=None,
        )


def test_refuses_without_fod_sha(bank_case):
    case, _sha = bank_case
    with pytest.raises(FidelityRefusal, match="fod_sha256"):
        summaries_for_bank(
            case / "bank.tck", case / "fod.nii.gz", case / "peak.nii.gz",
            provenance={"bank_sha256": "ab" * 32},
        )


def test_refuses_on_sha_mismatch(bank_case):
    case, _sha = bank_case
    with pytest.raises(FidelityRefusal, match="mismatch"):
        summaries_for_bank(
            case / "bank.tck", case / "fod.nii.gz", case / "peak.nii.gz",
            provenance={"fod_sha256": "00" * 32},
        )


def test_summaries_on_matching_sha(bank_case):
    case, sha = bank_case
    out = summaries_for_bank(
        case / "bank.tck", case / "fod.nii.gz", case / "peak.nii.gz",
        provenance={"fod_sha256": sha},
    )
    assert out["n"] == 1
    assert out["p5_ratio"].dtype == np.float32
    assert out["p5_ratio"].shape == (1,)
    assert float(out["p5_ratio"][0]) > 0.95  # isotropic support everywhere
    assert float(out["mean_ratio"][0]) > 0.95


def test_source_vs_packed_divergence_documented():
    """Spec success criterion 4: quantify how much the packed display K-grid
    diverges from source geometry — the reason evidence values are computed
    on source .tck only, never on packed values."""
    from tractlab.pack import pack_streamlines

    x = np.linspace(0.0, 100.0, 201)  # source step 0.5 mm, sinuous
    line = np.column_stack([x, 20.0 * np.sin(x / 2.0), np.zeros_like(x)])
    src_step = float(np.linalg.norm(np.diff(line, axis=0), axis=1).mean())
    buf, header = pack_streamlines([line], k=64, space_id="t", minlength_mm=50.0)
    k = int(header["pointsPerLine"])
    import struct

    pts = np.frombuffer(buf, dtype=np.float32)[: k * 3].reshape(k, 3)
    packed_step = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).mean())
    assert packed_step > src_step, (
        f"packed segment length {packed_step:.3f} mm vs source {src_step:.3f} mm "
        f"(ratio {packed_step / src_step:.2f}x) — packed geometry is coarser; "
        "evidence must be computed on source"
    )


def test_degenerate_streamline_is_nan_not_zero(tmp_path):
    """Grok+Sol batch-4 HIGH: unmeasurable geometry must be NaN — a measured
    0.0 would inflate low-support marking."""
    sh = _iso_sh()
    _write_nii(tmp_path / "fod.nii.gz", sh)
    _write_nii(tmp_path / "peak.nii.gz", _peak_of(sh))
    pt = np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]], dtype=np.float32)  # 0-length
    ok = np.array([[3.0, 7.0, 7.0], [11.0, 7.0, 7.0]], dtype=np.float32)
    _write_tck(tmp_path / "bank.tck", [ok, pt])
    sha = hashlib.sha256((tmp_path / "fod.nii.gz").read_bytes()).hexdigest()
    out = summaries_for_bank(
        tmp_path / "bank.tck", tmp_path / "fod.nii.gz", tmp_path / "peak.nii.gz",
        provenance={"fod_sha256": sha},
    )
    assert float(out["p5_ratio"][0]) > 0.95
    assert np.isnan(out["p5_ratio"][1]) and np.isnan(out["mean_ratio"][1])


def test_ratio_scale_invariance():
    """Sol batch-4: a globally rescaled FOD must give identical ratios (the
    peak floor is relative, not absolute)."""
    sh = _iso_sh(c0=2.0)
    line = np.array([[3.0, 7.0, 7.0], [11.0, 7.0, 7.0]])
    base = segment_ratios(line, sh, AFF, _peak_of(sh), AFF)
    tiny = sh * 1e-8
    scaled = segment_ratios(line, tiny, AFF, _peak_of(tiny), AFF)
    assert np.allclose(base, scaled, atol=1e-9)
    assert np.all(scaled > 0.95)
