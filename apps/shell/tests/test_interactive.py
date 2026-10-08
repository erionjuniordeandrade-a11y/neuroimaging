"""nnInteractive click-to-segment on a synthetic volume, with the model process replaced by a stub."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from neuro_workbench import interactive


@pytest.fixture()
def volume(tmp_path: Path) -> Path:
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = [-10.0, -12.0, -14.0]
    path = tmp_path / "vol.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((10, 12, 14), np.int16), affine), str(path))
    return path


@pytest.fixture()
def available(monkeypatch, tmp_path):
    py = tmp_path / "python"
    py.write_text("")
    monkeypatch.setenv("CASE_CAPSULE_NNINTERACTIVE_PYTHON", str(py))
    monkeypatch.setattr(interactive, "runner_path", lambda: tmp_path / "runner.py")


def test_status_says_why_the_model_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setenv("CASE_CAPSULE_NNINTERACTIVE_PYTHON", str(tmp_path / "missing"))
    assert interactive.status() == {"available": False, "reason": "the nnInteractive environment is not installed"}
    with pytest.raises(RuntimeError, match="not installed"):
        interactive.segment(tmp_path / "x.nii.gz", [{"ras": [0, 0, 0]}])


def test_segment_maps_ras_points_and_returns_the_mask_in_niivue_order(volume, available):
    seen = {}

    def fake_run(cmd, **kw):
        arg = lambda name: Path(cmd[cmd.index(name) + 1])
        seen["request"] = json.loads(arg("--request").read_text())
        with np.load(arg("--input")) as d:
            seen["shape"] = d["image"].shape
        mask = np.zeros((10, 12, 14), np.uint8)
        mask[3, 4, 5] = mask[6, 7, 8] = 1
        np.savez(arg("--output-dir") / "mask_0000.npz", mask=mask)
        arg("--result").write_text(json.dumps({"model_id": "nnInteractive_v1.0", "version": "1.1", "weights_licence": "CC-BY-NC-SA-4.0"}))
        return SimpleNamespace(returncode=0, stderr="")

    out = interactive.segment(volume, [{"ras": [-4.0, -4.0, -4.0], "positive": True},
                                       {"ras": [0.0, 0.0, 0.0], "positive": False}], run=fake_run)
    pts = seen["request"]["prompts"][0]["points_kji"]
    assert pts == [{"kji": [3, 4, 5], "positive": True}, {"kji": [5, 6, 7], "positive": False}]
    assert seen["shape"] == (10, 12, 14)
    flat = np.frombuffer(base64.b64decode(out["indices"]), np.uint32)
    assert flat.tolist() == [3 + 10 * (4 + 12 * 5), 6 + 10 * (7 + 12 * 8)]
    assert out["count"] == 2 and out["volume_ml"] == 0.016 and out["dims"] == [10, 12, 14]
    assert "nnInteractive 1.1" in out["source"] and "non-commercial" in out["licence"]


@pytest.mark.parametrize("points,message", [
    ([], "at least one point"),
    ([{"ras": [500, 0, 0]}], "outside the image"),
    ([{"ras": [0, 0, 0], "positive": False}], "inside the structure"),
    ([{"ras": "x"}], "no RAS position"),
])
def test_segment_refuses_bad_points(volume, available, points, message):
    with pytest.raises(ValueError, match=message):
        interactive.segment(volume, points, run=lambda *a, **k: pytest.fail("model must not run"))


def test_model_failure_is_one_line_without_paths(volume, available):
    fail = lambda cmd, **kw: SimpleNamespace(returncode=1, stderr="Traceback\n  File \"/secret/path.py\"\nRuntimeError: MPS out of memory")
    with pytest.raises(RuntimeError) as err:
        interactive.segment(volume, [{"ras": [0, 0, 0]}], run=fail)
    assert str(err.value) == "nnInteractive stopped (exit 1, RuntimeError)"
