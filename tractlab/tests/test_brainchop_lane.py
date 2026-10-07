"""brainchop DKatlas lane — synthetic, offline (no imaging, no network).

Covers: remap correctness + fail-closed on unknown/malformed ids, LUT
shape/uniqueness/location, run_dkatlas's argv + typed-failure handling
(stale-output guard, complete-provenance gate, OSError/timeout/engine-error)
with subprocess mocked out, and the brainchop_compare.py comparison script's
fail-open refusals (mismatched affine, missing/mismatched ROI). Real
brainchop/uvx/mrtrix are never invoked here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest

from tractlab.brainchop_lane import (
    LUT_PATH,
    InvalidLabelVolumeError,
    Outcome,
    UnknownLabelError,
    load_lut,
    remap_to_freesurfer,
    run_dkatlas,
)

SCRIPTS_DIR = str(Path(__file__).resolve().parents[1] / "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
import brainchop_compare  # noqa: E402


# ── LUT shape + location ─────────────────────────────────────────────────

def test_lut_has_104_entries_with_unique_freesurfer_ids():
    lut = load_lut()
    assert len(lut) == 104
    assert len(set(lut.values())) == 104, "freesurfer ids must be unique"


def test_lut_file_has_provenance_header():
    with open(LUT_PATH) as f:
        doc = json.load(f)
    prov = doc["_provenance"]
    assert prov["source_url"].startswith("https://")
    assert len(prov["source_sha256"]) == 64
    assert prov["model_folder"] == "model24chan104cls_synth"


def test_lut_lives_under_lut_dir_not_the_gitignored_data_dir():
    """src/tractlab/data/ is caught by the repo's blanket PHI .gitignore rule
    `data/`; the LUT must live somewhere `git check-ignore` does not flag."""
    assert os.path.normpath(LUT_PATH).endswith(os.path.join("tractlab", "lut", "brainchop_dkatlas_lut.json"))
    result = subprocess.run(
        ["git", "check-ignore", "-v", LUT_PATH],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True, text=True,
    )
    assert result.returncode == 1, f"LUT_PATH is gitignored: {result.stdout}"


def test_lut_known_ids_match_the_morning_dice_run():
    """Cross-check against the compact ids the orchestrator reported this morning."""
    lut = load_lut()
    expected = {
        1: 1001,   # ctx-lh-bankssts (start of L cortex block)
        34: 1035,  # ctx-lh-insula (end of L cortex block)
        35: 2001,  # ctx-rh-bankssts (start of R cortex block)
        68: 2035,  # ctx-rh-insula (end of R cortex block)
        69: 10,    # thalamus L
        70: 49,    # thalamus R
        71: 11,    # caudate L
        72: 50,    # caudate R
        73: 12,    # putamen L
        74: 51,    # putamen R
        75: 13,    # pallidum L
        76: 52,    # pallidum R
        77: 17,    # hippocampus L
        78: 53,    # hippocampus R
        79: 18,    # amygdala L
        80: 54,    # amygdala R
        85: 2,     # WM L
        86: 41,    # WM R
        94: 16,    # brainstem
    }
    for bc_id, fs_id in expected.items():
        assert lut[bc_id] == fs_id, f"brainchop id {bc_id} -> expected fs {fs_id}, got {lut[bc_id]}"


# ── remap_to_freesurfer ──────────────────────────────────────────────────

def test_remap_translates_every_id_via_the_lut():
    lut = {0: 0, 1: 1001, 85: 2}
    vol = np.array([[[0, 1], [85, 1]]], dtype=np.uint8)
    out = remap_to_freesurfer(vol, lut)
    assert out.tolist() == [[[0, 1001], [2, 1001]]]
    assert out.dtype == np.int32


def test_remap_fails_closed_on_unknown_id():
    lut = {0: 0, 1: 1001}
    vol = np.array([[[0, 1, 99]]], dtype=np.uint8)  # 99 is not in the LUT
    with pytest.raises(UnknownLabelError) as exc:
        remap_to_freesurfer(vol, lut)
    assert "99" in str(exc.value)


def test_remap_never_writes_output_when_it_fails():
    """A single unknown id anywhere in the volume voids the whole remap."""
    lut = {0: 0}
    vol = np.zeros((4, 4, 4), dtype=np.uint8)
    vol[0, 0, 0] = 7  # unknown
    with pytest.raises(UnknownLabelError):
        remap_to_freesurfer(vol, lut)


def test_remap_with_the_real_vendored_lut_round_trips_known_ids():
    lut = load_lut()
    vol = np.array([1, 34, 35, 68, 85, 86, 94], dtype=np.int32).reshape(1, 1, -1)
    out = remap_to_freesurfer(vol, lut)
    assert out.flatten().tolist() == [1001, 1035, 2001, 2035, 2, 41, 16]


def test_remap_refuses_fractional_nan_and_negative_labels():
    lut = {0: 0, 1: 1001}

    fractional = np.array([[[0.0, 1.5]]], dtype=np.float32)
    with pytest.raises(InvalidLabelVolumeError, match="fractional"):
        remap_to_freesurfer(fractional, lut)

    nan_vol = np.array([[[0.0, np.nan]]], dtype=np.float32)
    with pytest.raises(InvalidLabelVolumeError, match="NaN"):
        remap_to_freesurfer(nan_vol, lut)

    negative_int = np.array([[[0, -1]]], dtype=np.int32)
    with pytest.raises(InvalidLabelVolumeError, match="negative"):
        remap_to_freesurfer(negative_int, lut)

    negative_float = np.array([[[0.0, -1.0]]], dtype=np.float32)
    with pytest.raises(InvalidLabelVolumeError, match="negative"):
        remap_to_freesurfer(negative_float, lut)


def test_remap_refuses_non_integer_dtype_even_when_values_are_whole():
    """Refusing the dtype itself (not truncating float->int) is the point: a
    float array that happens to hold only whole numbers must still be refused."""
    lut = {0: 0, 1: 1001}
    whole_valued_float = np.array([[[0.0, 1.0]]], dtype=np.float32)
    with pytest.raises(InvalidLabelVolumeError, match="non-integer dtype"):
        remap_to_freesurfer(whole_valued_float, lut)


# ── run_dkatlas: argv + typed outcomes, subprocess mocked ───────────────

class _FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _fake_models_cache(tmp_path, folder="model24chan104cls_synth", weights=b"fake-weights"):
    """A synthetic, fully offline stand-in for ~/.cache/brainchop/{models.json,models/}."""
    models_json = tmp_path / "models.json"
    models_json.write_text(json.dumps({"DKatlas": {"folder": folder}}))
    models_dir = tmp_path / "models_cache"
    weights_dir = models_dir / folder
    weights_dir.mkdir(parents=True)
    (weights_dir / "model.pth").write_bytes(weights)
    return str(models_json), str(models_dir)


def _fake_run_writing(write_fn):
    """subprocess.run replacement that simulates brainchop actually writing its
    output (argv[-1] is the -o path), instead of a test pre-seeding the file."""
    def _fake_run(argv, capture_output, text, timeout=None):
        del capture_output, text, timeout
        write_fn(argv[-1])
        return _FakeCompleted(returncode=0)
    return _fake_run


def _write_label_volume(path, vol, affine=None):
    nib.save(nib.Nifti1Image(vol, affine if affine is not None else np.eye(4)), path)


def test_run_dkatlas_builds_expected_argv(tmp_path):
    t1 = tmp_path / "t1.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), str(t1))
    out_dir = tmp_path / "out"

    with patch("tractlab.brainchop_lane.subprocess.run") as mock_run:
        mock_run.return_value = _FakeCompleted(returncode=1, stderr="boom")
        result = run_dkatlas(str(t1), str(out_dir), timeout_s=5)

    assert result.outcome is Outcome.ENGINE_ERROR
    argv = mock_run.call_args_list[0].args[0]
    assert argv[0].endswith("uvx")
    assert argv[1:8] == [
        "brainchop", str(t1), "-m", "DKatlas", "--no-optimize", "--inverse-conform", "-o",
    ]
    assert argv[8] == str(out_dir / "brainchop_raw.nii.gz")


def test_run_dkatlas_nonzero_exit_is_typed_engine_error_not_silent(tmp_path):
    t1 = tmp_path / "t1.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), str(t1))
    out_dir = tmp_path / "out"

    with patch("tractlab.brainchop_lane.subprocess.run") as mock_run:
        mock_run.return_value = _FakeCompleted(returncode=2, stderr="model download failed")
        result = run_dkatlas(str(t1), str(out_dir), timeout_s=5)

    assert result.outcome is Outcome.ENGINE_ERROR
    assert result.aparc_path is None
    assert "model download failed" in (result.warning or "")
    assert not (out_dir / "aparc_brainchop.nii.gz").exists()


def test_run_dkatlas_timeout_is_typed_not_zero(tmp_path):
    t1 = tmp_path / "t1.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), str(t1))
    out_dir = tmp_path / "out"

    with patch("tractlab.brainchop_lane.subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["uvx"], timeout=5)
        result = run_dkatlas(str(t1), str(out_dir), timeout_s=5)

    assert result.outcome is Outcome.TIMEOUT
    assert result.aparc_path is None
    assert "5" in (result.warning or "")


def test_run_dkatlas_oserror_is_typed(tmp_path):
    """subprocess.run raising OSError (e.g. uvx binary missing/not executable)
    must come back as a typed OSERROR outcome, never an unhandled crash."""
    t1 = tmp_path / "t1.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), str(t1))
    out_dir = tmp_path / "out"

    with patch("tractlab.brainchop_lane.subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError("[Errno 2] No such file or directory: 'uvx'")
        result = run_dkatlas(str(t1), str(out_dir), timeout_s=5)

    assert result.outcome is Outcome.OSERROR
    assert result.aparc_path is None
    assert "uvx" in (result.warning or "")


def test_run_dkatlas_rc0_but_missing_output_is_engine_error(tmp_path):
    t1 = tmp_path / "t1.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), str(t1))
    out_dir = tmp_path / "out"

    with patch("tractlab.brainchop_lane.subprocess.run") as mock_run:
        mock_run.return_value = _FakeCompleted(returncode=0)  # no file actually written
        result = run_dkatlas(str(t1), str(out_dir), timeout_s=5)

    assert result.outcome is Outcome.ENGINE_ERROR
    assert "missing" in (result.warning or "")


def test_run_dkatlas_refuses_stale_preexisting_outputs(tmp_path):
    """A leftover brainchop_raw.nii.gz from a previous run, sitting in out_dir
    with an old mtime, must never be mistaken for this run's output — even if
    the pre-invocation cleanup step is somehow ineffective (simulated here by
    making os.remove a no-op) and brainchop reports rc==0 without writing."""
    t1 = tmp_path / "t1.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), str(t1))
    out_dir = tmp_path / "out"
    os.makedirs(out_dir, exist_ok=True)
    raw_out = out_dir / "brainchop_raw.nii.gz"
    stale = np.zeros((4, 4, 4), dtype=np.uint8)
    stale[0, 0, 0] = 85
    _write_label_volume(str(raw_out), stale)
    old = time.time() - 3600
    os.utime(raw_out, (old, old))

    with patch("tractlab.brainchop_lane.subprocess.run") as mock_run, \
         patch("tractlab.brainchop_lane.os.remove") as mock_remove:
        mock_remove.return_value = None  # simulate cleanup being ineffective
        mock_run.return_value = _FakeCompleted(returncode=0)  # "succeeds" without writing
        result = run_dkatlas(str(t1), str(out_dir), timeout_s=5)

    assert result.outcome is Outcome.ENGINE_ERROR
    assert "stale" in (result.warning or "").lower()
    assert not (out_dir / "aparc_brainchop.nii.gz").exists()


def test_run_dkatlas_unmapped_label_in_raw_output_is_engine_error_not_silent(tmp_path):
    """A raw brainchop output with an id our LUT doesn't cover must fail closed,
    not silently produce a corrupted aparc volume."""
    t1 = tmp_path / "t1.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), str(t1))
    out_dir = tmp_path / "out"
    bad = np.zeros((4, 4, 4), dtype=np.uint8)
    bad[0, 0, 0] = 250  # not a valid DKatlas class id

    with patch("tractlab.brainchop_lane.subprocess.run",
               side_effect=_fake_run_writing(lambda p: _write_label_volume(p, bad))):
        result = run_dkatlas(str(t1), str(out_dir), timeout_s=5)

    assert result.outcome is Outcome.ENGINE_ERROR
    assert "remap failed" in (result.warning or "")
    assert not (out_dir / "aparc_brainchop.nii.gz").exists()


def test_run_dkatlas_incomplete_provenance_is_not_ok(tmp_path):
    """Outcome.OK requires brainchop_version AND tinygrad_device to be known —
    a run that can't attest either is a failure, even though the remap itself
    succeeded and looks scientifically fine."""
    t1 = tmp_path / "t1.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), str(t1))
    out_dir = tmp_path / "out"
    vol = np.zeros((4, 4, 4), dtype=np.uint8)
    vol[1, 1, 1] = 85
    models_json, models_dir = _fake_models_cache(tmp_path)

    with patch("tractlab.brainchop_lane.subprocess.run",
               side_effect=_fake_run_writing(lambda p: _write_label_volume(p, vol))), \
         patch("tractlab.brainchop_lane._brainchop_version", return_value=None), \
         patch("tractlab.brainchop_lane._tinygrad_device", return_value="METAL"):
        result = run_dkatlas(
            str(t1), str(out_dir), timeout_s=5,
            models_json_path=models_json, models_cache_dir=models_dir,
        )

    assert result.outcome is Outcome.ENGINE_ERROR
    assert "provenance" in (result.warning or "").lower()
    assert "brainchop_version" in (result.warning or "")
    assert not (out_dir / "aparc_brainchop.nii.gz").exists()


def test_failure_writes_sidecar(tmp_path):
    """Every non-OK outcome (timeout/engine/oserror) leaves a typed sidecar
    record on disk — a failure is never silent."""
    t1 = tmp_path / "t1.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), str(t1))
    out_dir = tmp_path / "out"

    with patch("tractlab.brainchop_lane.subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["uvx"], timeout=5)
        result = run_dkatlas(str(t1), str(out_dir), timeout_s=5)

    assert result.outcome is Outcome.TIMEOUT
    sidecar_path = out_dir / "brainchop_run.json"
    assert sidecar_path.exists()
    with open(sidecar_path) as f:
        sidecar = json.load(f)
    assert sidecar["outcome"] == "timeout"
    assert "reason" in sidecar and sidecar["reason"]
    assert sidecar["argv"][0].endswith("uvx")
    assert "started_at" in sidecar and "finished_at" in sidecar


def test_run_dkatlas_ok_writes_remapped_output_and_sidecar(tmp_path):
    t1 = tmp_path / "t1.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), str(t1))
    out_dir = tmp_path / "out"
    vol = np.zeros((4, 4, 4), dtype=np.uint8)
    vol[1, 1, 1] = 85  # WM L
    models_json, models_dir = _fake_models_cache(tmp_path)

    with patch("tractlab.brainchop_lane.subprocess.run",
               side_effect=_fake_run_writing(lambda p: _write_label_volume(p, vol))), \
         patch("tractlab.brainchop_lane._brainchop_version", return_value="0.2.5"), \
         patch("tractlab.brainchop_lane._tinygrad_device", return_value="METAL"):
        result = run_dkatlas(
            str(t1), str(out_dir), timeout_s=5,
            models_json_path=models_json, models_cache_dir=models_dir,
        )

    assert result.outcome is Outcome.OK
    assert result.aparc_path == str(out_dir / "aparc_brainchop.nii.gz")
    out_img = nib.load(result.aparc_path)
    out_data = np.asanyarray(out_img.dataobj)
    assert out_data[1, 1, 1] == 2  # FreeSurfer Left-Cerebral-White-Matter

    sidecar_path = out_dir / "brainchop_run.json"
    assert sidecar_path.exists()
    with open(sidecar_path) as f:
        sidecar = json.load(f)
    assert sidecar["outcome"] == "ok"
    assert sidecar["model"] == "DKatlas"
    assert sidecar["model_folder"] == "model24chan104cls_synth"
    assert sidecar["brainchop_version"] == "0.2.5"
    assert sidecar["tinygrad_device"] == "METAL"
    assert len(sidecar["input_sha256"]) == 64
    assert len(sidecar["output_sha256"]) == 64
    # Required provenance fields added in the fix round — all non-null.
    assert sidecar["lut_path"] and len(sidecar["lut_sha256"]) == 64
    assert sidecar["model_weights_path"].endswith("model.pth")
    assert len(sidecar["model_weights_sha256"]) == 64
    assert sidecar["started_at"] and sidecar["finished_at"]


def test_success_sidecar_has_complete_nonnull_provenance(tmp_path):
    """Every REQUIRED provenance field is present and non-null on Outcome.OK."""
    t1 = tmp_path / "t1.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), str(t1))
    out_dir = tmp_path / "out"
    vol = np.zeros((4, 4, 4), dtype=np.uint8)
    vol[1, 1, 1] = 85
    models_json, models_dir = _fake_models_cache(tmp_path)

    with patch("tractlab.brainchop_lane.subprocess.run",
               side_effect=_fake_run_writing(lambda p: _write_label_volume(p, vol))), \
         patch("tractlab.brainchop_lane._brainchop_version", return_value="0.2.5"), \
         patch("tractlab.brainchop_lane._tinygrad_device", return_value="METAL"):
        result = run_dkatlas(
            str(t1), str(out_dir), timeout_s=5,
            models_json_path=models_json, models_cache_dir=models_dir,
        )

    assert result.outcome is Outcome.OK
    required_fields = [
        "lut_path", "lut_sha256",
        "model_weights_path", "model_weights_sha256",
        "brainchop_version", "tinygrad_device",
        "model_folder", "input_sha256", "output_sha256", "raw_output_sha256",
        "started_at", "finished_at", "argv",
    ]
    for key in required_fields:
        assert result.sidecar.get(key), f"missing/null required provenance field: {key}"
    assert result.sidecar["model_folder"] == "model24chan104cls_synth"  # from the fake models.json


# ── brainchop_compare.py: fail-open refusals ─────────────────────────────

def _save(path, arr, affine):
    nib.save(nib.Nifti1Image(arr, affine), str(path))


def test_dice_refuses_equal_shape_different_affine(tmp_path):
    """Same voxel shape but a different affine is a FALSE-shared grid — a
    voxel-wise Dice on that pair is meaningless and must be refused, not
    silently computed."""
    shape = (4, 4, 4)
    a_path = tmp_path / "a.nii.gz"
    b_path = tmp_path / "b.nii.gz"
    arr = np.zeros(shape + (5,), dtype=np.float32)
    arr[1, 1, 1, 0] = 1.0
    _save(a_path, arr, np.eye(4))
    _save(b_path, arr, np.diag([2.0, 2.0, 2.0, 1.0]))  # same shape, different voxel size

    with pytest.raises(brainchop_compare.ComparisonError, match="affine"):
        brainchop_compare.five_tt_dice(str(a_path), str(b_path))


def test_main_refuses_missing_or_mismatched_roi(tmp_path, capsys):
    shape = (4, 4, 4)
    aff = np.eye(4)
    five_tt = np.zeros(shape + (5,), dtype=np.float32)
    (tmp_path / "5tt_bc.nii.gz").parent.mkdir(parents=True, exist_ok=True)
    _save(tmp_path / "5tt_bc.nii.gz", five_tt, aff)
    _save(tmp_path / "5tt_fs.nii.gz", five_tt, aff)
    aparc = np.zeros(shape, dtype=np.int32)
    _save(tmp_path / "aparc_bc.nii.gz", aparc, aff)
    _save(tmp_path / "aparc_fs.nii.gz", aparc, aff)
    _save(tmp_path / "aparc_dwi.nii.gz", aparc, aff)
    roi_dir = tmp_path / "roi"
    roi_dir.mkdir()
    # Deliberately do NOT create any of the ROI files brainchop_compare expects.

    out_path = tmp_path / "report.json"
    argv = [
        "brainchop_compare.py",
        "--five-tt-brainchop", str(tmp_path / "5tt_bc.nii.gz"),
        "--five-tt-freesurfer", str(tmp_path / "5tt_fs.nii.gz"),
        "--aparc-brainchop-fs-grid", str(tmp_path / "aparc_bc.nii.gz"),
        "--aparc-freesurfer", str(tmp_path / "aparc_fs.nii.gz"),
        "--aparc-brainchop-dwi", str(tmp_path / "aparc_dwi.nii.gz"),
        "--roi-dir", str(roi_dir),
        "--out", str(out_path),
    ]
    with patch.object(sys, "argv", argv):
        with pytest.raises(SystemExit) as exc:
            brainchop_compare.main()

    assert exc.value.code != 0
    assert not out_path.exists(), "a refusal must not leave a partial report on disk"


def test_roi_dice_refuses_shape_mismatched_roi(tmp_path):
    shape = (4, 4, 4)
    aff = np.eye(4)
    aparc = np.zeros(shape, dtype=np.int32)
    aparc_path = tmp_path / "aparc_dwi.nii.gz"
    _save(aparc_path, aparc, aff)
    roi_dir = tmp_path / "roi"
    roi_dir.mkdir()
    # Wrong shape for the first ROI brainchop_compare looks up.
    first_roi = next(iter(brainchop_compare.ROI_TO_LABEL))
    _save(roi_dir / first_roi, np.zeros((3, 3, 3), dtype=np.uint8), aff)

    with pytest.raises(brainchop_compare.ComparisonError, match="shape"):
        brainchop_compare.roi_dice(str(aparc_path), str(roi_dir))
