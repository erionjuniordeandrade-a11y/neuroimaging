from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk

from capsule import cli, masks, tracts
from capsule.pack import read_capsule
from capsule.tracts import encode_tck


def _bundle(x: float, count: int = 60, *, y_fn=None, length_mm: float = 100.0) -> list[np.ndarray]:
    points = np.linspace(-length_mm / 2, length_mm / 2, 201)
    curve = np.zeros_like(points) if y_fn is None else y_fn(points)
    return [np.column_stack((np.full_like(points, x + index * 0.001), curve, points)).astype(np.float32)
            for index in range(count)]


def _compute(bundles: dict[str, list[np.ndarray]], **kwargs) -> dict[str, dict]:
    compute = getattr(tracts, "compute_tract_trust", None)
    assert callable(compute), "capsule.tracts.compute_tract_trust is required"
    return compute(bundles, **kwargs)


def test_smooth_contralateral_bundle_pair_passes_and_uses_contract_thresholds():
    blocks = _compute({"slf1_l": _bundle(-20), "slf1_r": _bundle(20)})

    assert blocks["slf1_l"]["verdict"] == "PASS"
    assert blocks["slf1_r"]["verdict"] == "PASS"
    assert blocks["slf1_l"]["metrics"]["contralateral"] == "slf1_r"
    assert blocks["slf1_l"]["metrics"]["tortuosity_ratio"] == pytest.approx(1.0)
    assert blocks["slf1_l"]["thresholds"] == {
        "tortuosity_ratio_max": 1.5,
        "median_tortuosity_max": 3.5,
        "cap_fraction_max": 0.10,
        "min_streamlines": 50,
        "rim_mm": 5.0,
        "rim_fraction_flag": 0.30,
        "yield_ratio_flag": 1 / 3,
    }
    assert blocks["slf1_l"]["calibration"] == "pilot: 1 case, 10 contralateral pairs"


def test_low_yield_side_is_flagged_but_keeps_its_verdict():
    blocks = _compute({"cst_l": _bundle(-20, count=60), "cst_r": _bundle(20, count=600)})
    left, right = blocks["cst_l"], blocks["cst_r"]

    assert left["verdict"] == "PASS" and left["failing"] == []
    assert left["yield_flag"] is True
    assert left["metrics"]["yield_ratio"] == pytest.approx(0.1)
    assert right["yield_flag"] is False
    assert right["metrics"]["yield_ratio"] == pytest.approx(10.0)


def test_yield_ratio_at_the_flag_boundary_is_not_flagged():
    blocks = _compute({"af_l": _bundle(-20, count=100), "af_r": _bundle(20, count=300)})

    assert blocks["af_l"]["metrics"]["yield_ratio"] == pytest.approx(1 / 3)
    assert blocks["af_l"]["yield_flag"] is False


def test_unpaired_or_empty_contralateral_has_no_yield_ratio():
    blocks = _compute({"fx_l": _bundle(-20), "slf1_l": _bundle(-20), "slf1_r": []})

    assert blocks["fx_l"]["metrics"]["yield_ratio"] is None and blocks["fx_l"]["yield_flag"] is False
    assert blocks["slf1_l"]["metrics"]["yield_ratio"] is None and blocks["slf1_l"]["yield_flag"] is False


def test_sparse_contralateral_gives_no_yield_ratio():
    blocks = _compute({"slf2_l": _bundle(-20, count=5), "slf2_r": _bundle(20, count=20)})

    assert blocks["slf2_l"]["metrics"]["yield_ratio"] is None and blocks["slf2_l"]["yield_flag"] is False
    assert blocks["slf2_r"]["metrics"]["yield_ratio"] is None and blocks["slf2_r"]["yield_flag"] is False


def test_sparse_bundle_below_minimum_is_unavailable_and_still_yield_flagged():
    blocks = _compute({"slf2_l": _bundle(-20, count=5), "slf2_r": _bundle(20, count=608)})

    assert blocks["slf2_l"]["verdict"] == "UNAVAILABLE"
    assert blocks["slf2_l"]["yield_flag"] is True


def test_looping_left_bundle_fails_by_contralateral_tortuosity_ratio():
    loop = lambda z: 18.0 * np.sin(4.0 * np.pi * (z + 50.0) / 100.0)
    blocks = _compute({"slf1_l": _bundle(-20, y_fn=loop), "slf1_r": _bundle(20)})
    left = blocks["slf1_l"]

    assert left["verdict"] == "FAIL"
    assert left["failing"] == ["tortuosity_ratio"]
    assert left["metrics"]["tortuosity_ratio"] > 1.5
    assert left["metrics"]["median_tortuosity"] < 3.5


def test_symmetric_naturally_curved_pair_passes_the_absolute_tortuosity_guard():
    # Keep the path below the default 250 mm cap while preserving the same ~2.8 tortuosity.
    curve = lambda z: 16.8 * np.sin(6.0 * np.pi * (z + 40.0) / 80.0)
    blocks = _compute({"uf_l": _bundle(-20, y_fn=curve, length_mm=80.0),
                       "uf_r": _bundle(20, y_fn=curve, length_mm=80.0)})

    assert blocks["uf_l"]["metrics"]["median_tortuosity"] == pytest.approx(2.8, abs=0.1)
    assert blocks["uf_r"]["metrics"]["median_tortuosity"] == pytest.approx(2.8, abs=0.1)
    assert blocks["uf_l"]["verdict"] == blocks["uf_r"]["verdict"] == "PASS"


def test_bundle_hugging_a_lesion_sphere_is_flagged_but_still_passes():
    shape = (80, 80, 80)
    affine = np.eye(4)
    affine[:3, 3] = -40
    k, j, i = np.indices(shape)
    lesion = ((i - 25) ** 2 + (j - 45) ** 2 + (k - 40) ** 2) <= 6 ** 2
    geometry_type = getattr(tracts, "LesionGrid", None)
    assert geometry_type is not None, "capsule.tracts.LesionGrid must pair the lesion array with its RAS affine"
    geometry = geometry_type(mask_kji=lesion, affine_ras=affine)
    points = np.linspace(-30.0, 0.0, 101)
    left = [np.column_stack((points, np.full_like(points, 12.0), np.zeros_like(points))).astype(np.float32)
            for _ in range(60)]
    right_points = np.linspace(0.0, 30.0, 101)
    right = [np.column_stack((right_points, np.full_like(right_points, 12.0), np.zeros_like(right_points))).astype(np.float32)
             for _ in range(60)]
    blocks = _compute({"slf1_l": left, "slf1_r": right}, lesion=geometry)

    assert blocks["slf1_l"]["metrics"]["rim_fraction"] > 0.30
    assert blocks["slf1_l"]["rim_flag"] is True
    assert blocks["slf1_l"]["verdict"] == "PASS"


def test_cap_fraction_above_contract_limit_fails():
    capped = _bundle(-20, length_mm=249.0)
    block = _compute({"cst_l": capped})["cst_l"]

    assert block["metrics"]["cap_fraction"] == 1.0
    assert block["failing"] == ["cap_fraction"]
    assert block["verdict"] == "FAIL"


def test_fewer_than_minimum_source_streamlines_is_unavailable():
    block = _compute({"cst_l": _bundle(-20, count=49)})["cst_l"]

    assert block["metrics"]["n_source"] == 49
    assert block["verdict"] == "UNAVAILABLE"


def _mask_grid(shape=(30, 30, 30), spacing=(10.0, 10.0, 10.0)) -> sitk.Image:
    grid = sitk.GetImageFromArray(np.zeros(shape, dtype=np.float32))
    grid.SetSpacing(spacing)
    return grid


def _fake_mask_commands(monkeypatch) -> None:
    monkeypatch.setattr(masks, "_synthstrip_command",
                        lambda source, target: ["fake-synthstrip", str(source), str(target)])
    monkeypatch.setattr(masks, "_bet_command",
                        lambda source, target: ["fake-bet", str(source), str(target.with_name("bet")), "-m"])


def _mask_runner(volumes_ml: list[int]):
    calls = []

    def run(command):
        index = len(calls)
        calls.append(command[0])
        source = sitk.ReadImage(command[1])
        data = np.zeros(sitk.GetArrayFromImage(source).shape, dtype=np.uint8)
        data.flat[:min(volumes_ml[index], data.size)] = 1
        output = Path(command[2]) if command[0] == "fake-synthstrip" else Path(command[2]).with_name("bet_mask.nii.gz")
        mask = sitk.GetImageFromArray(data)
        mask.CopyInformation(source)
        sitk.WriteImage(mask, str(output))
        return subprocess.CompletedProcess(command, 0, "", "")

    return run, calls


def test_brain_volume_gate_uses_bet_when_synthstrip_volume_fails(monkeypatch):
    _fake_mask_commands(monkeypatch)
    grid = _mask_grid()
    runner, calls = _mask_runner([2000, 1200])

    brain, record = masks.brain_mask(np.zeros((30, 30, 30), dtype=np.float32), grid,
                                     command_runner=runner)

    assert calls == ["fake-synthstrip", "fake-bet"]
    assert brain.sum() == 1200
    assert record["method"] == "bet"
    assert record["qc"]["attempts"] == [
        {"method": "synthstrip", "volume_ml": 2000.0, "verdict": "FAIL"},
        {"method": "bet", "volume_ml": 1200.0, "verdict": "PASS"},
    ]
    assert record["qc"]["verdict"] == "PASS"


def test_every_brain_method_failing_raises_qc_error_with_attempts(monkeypatch):
    _fake_mask_commands(monkeypatch)
    grid = _mask_grid()
    runner, calls = _mask_runner([300, 200])

    try:
        masks.brain_mask(np.zeros((30, 30, 30), dtype=np.float32), grid, command_runner=runner)
    except RuntimeError as error:
        assert type(error).__name__ == "BrainMaskQCError"
        assert len(error.attempts) == 2
        assert [item["verdict"] for item in error.attempts] == ["FAIL", "FAIL"]
    else:
        pytest.fail("a nonempty out-of-range mask must not pass the default brain-volume gate")
    assert calls == ["fake-synthstrip", "fake-bet"]


def test_brain_volume_gate_off_records_not_gated(monkeypatch):
    _fake_mask_commands(monkeypatch)
    grid = _mask_grid()
    runner, calls = _mask_runner([2000])

    brain, record = masks.brain_mask(np.zeros((30, 30, 30), dtype=np.float32), grid,
                                     volume_gate=None, command_runner=runner)

    assert brain.any()
    assert calls == ["fake-synthstrip"]
    assert record["qc"]["gate_ml"] is None
    assert record["qc"]["verdict"] == "NOT_GATED"


def _write_volume(path: Path, size: int = 32) -> None:
    k, j, i = np.indices((size, size, size))
    inside = ((i - size / 2) ** 2 + (j - size / 2) ** 2 + (k - size / 2) ** 2) <= (size * 0.42) ** 2
    affine = np.eye(4)
    affine[:3, 3] = -size // 2
    image = nib.Nifti1Image(np.where(inside, 100.0, 0.0).astype(np.float32), affine)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    nib.save(image, path)


def _write_lesion(path: Path, size: int = 32) -> None:
    k, j, i = np.indices((size, size, size))
    lesion = ((i - (size // 2 - 8)) ** 2 + (j - size // 2) ** 2 + (k - size // 2) ** 2) <= 2 ** 2
    affine = np.eye(4)
    affine[:3, 3] = -size // 2
    nib.save(nib.Nifti1Image(lesion.astype(np.uint8), affine), path)


def _write_tract(path: Path, x: float, count: int = 60) -> None:
    z = np.linspace(-10.0, 10.0, 41)
    streamlines = [np.column_stack((np.full_like(z, x), np.full_like(z, index * 0.001), z)).astype(np.float32)
                   for index in range(count)]
    path.write_bytes(encode_tck(streamlines))


def test_build_nifti_packs_lesion_and_trust_from_full_source_streamlines(tmp_path):
    volume = tmp_path / "synthetic.nii.gz"
    lesion = tmp_path / "lesion.nii.gz"
    left = tmp_path / "left.tck"
    right = tmp_path / "right.tck"
    output = tmp_path / "synthetic.capsule.html"
    _write_volume(volume)
    _write_lesion(lesion)
    _write_tract(left, -8.0)
    _write_tract(right, 8.0)

    run = subprocess.run(
        [sys.executable, "-m", "capsule.cli", "build-nifti",
         "--volume", f"{volume}:MR:RM T1+C", "--tract", f"{left}:slf1_l",
         "--tract", f"{right}:slf1_r", "--max-streamlines", "10",
         "--tract-max-length-mm", "250", "--lesion-mask", str(lesion),
         "--brain-volume-gate", "off", "--brain-mask", "none",
         "--label", "synthetic", "-o", str(output), "--viewer", "v1"],
        text=True, capture_output=True, check=False)

    assert run.returncode == 0, run.stderr
    manifest, arrays = read_capsule(output)
    trust = {item["label"]: item["trust"] for item in manifest["tracts"]}
    assert trust["slf1_l"]["verdict"] == trust["slf1_r"]["verdict"] == "PASS"
    assert trust["slf1_l"]["metrics"]["n_source"] == 60
    assert trust["slf1_l"]["metrics"]["contralateral"] == "slf1_r"
    assert manifest["tracts"][0]["n_streamlines"] == 10
    assert trust["slf1_l"]["thresholds"] == tracts.TRUST_THRESHOLDS
    (packed_lesion,) = [item for item in manifest["masks"] if item["role"] == "lesion"]
    assert packed_lesion["id"].startswith("lesion")
    assert packed_lesion["for_volume"] == manifest["volumes"][0]["id"]
    assert packed_lesion["source"] == "pipeline" and packed_lesion["reviewed"] is False
    assert packed_lesion["label"] == "Lesão (ROI de rastreamento)"
    assert arrays[packed_lesion["blob"]].sum() > 0
    assert "trust PASS" in run.stdout


def test_all_brain_methods_fail_manifest_qc_and_omit_brain_mask(tmp_path, monkeypatch):
    _fake_mask_commands(monkeypatch)
    volume = tmp_path / "synthetic.nii.gz"
    output = tmp_path / "failed.capsule.html"
    _write_volume(volume)

    result = cli.main(
        ["build-nifti", "--volume", f"{volume}:MR:FLAIR", "--label", "synthetic",
         "-o", str(output), "--viewer", "v2", "--brain-mask", "synthstrip"],
        command_runner=_mask_runner([10, 20])[0])

    assert result == 0
    manifest, _ = read_capsule(output)
    assert manifest["brain_mask_qc"]["verdict"] == "FAIL"
    assert [attempt["method"] for attempt in manifest["brain_mask_qc"]["attempts"]] == ["synthstrip", "bet"]
    assert not any(mask["id"].startswith("brain") for mask in manifest["masks"])


def test_flair_label_has_no_vessel_mask_and_postcontrast_vessels_stay_in_brain(tmp_path, monkeypatch):
    volume = tmp_path / "synthetic.nii.gz"
    output = tmp_path / "vessels.capsule.html"
    _write_volume(volume)
    calls = []

    def brain_mask(values, grid, method="synthstrip", *, volume_gate=(900.0, 1800.0), command_runner=None):
        brain = np.zeros_like(values, dtype=bool)
        brain[8:24, 8:24, 8:24] = True
        return brain, {"method": "synthetic", "seconds": 0,
                       "qc": {"gate_ml": [900.0, 1800.0], "attempts": [], "verdict": "PASS"}}

    def vessel_mask(values, brain, head, grid):
        calls.append(True)
        return np.ones_like(brain, dtype=bool), {"components": 1}

    monkeypatch.setattr(cli.maskops, "brain_mask", brain_mask)
    monkeypatch.setattr(cli.maskops, "vessel_mask", vessel_mask)
    monkeypatch.setattr(cli, "vessels_plausible", lambda *_: True)

    result = cli.main(
        ["build-nifti", "--volume", f"{volume}:MR:T1+C", "--volume", f"{volume}:MR:FLAIR",
         "--label", "synthetic", "-o", str(output), "--viewer", "v2", "--brain-mask", "bet"])

    assert result == 0
    manifest, arrays = read_capsule(output)
    vessels = [mask for mask in manifest["masks"] if mask.get("role") == "render" and mask["id"].startswith("vessels")]
    assert calls == [True]
    assert len(vessels) == 1
    assert vessels[0]["for_volume"] == manifest["volumes"][0]["id"]
    brain = arrays[next(mask["blob"] for mask in manifest["masks"] if mask["id"] == "brain")].astype(bool)
    vessel = arrays[vessels[0]["blob"]].astype(bool)
    assert np.all(~vessel | brain)


def _write_brain_file(path: Path, size: int = 32) -> int:
    k, j, i = np.indices((size, size, size))
    inside = ((i - size / 2) ** 2 + (j - size / 2) ** 2 + (k - size / 2) ** 2) <= (size * 0.35) ** 2
    affine = np.eye(4)
    affine[:3, 3] = -size // 2
    nib.save(nib.Nifti1Image(inside.astype(np.uint8), affine), path)
    return int(inside.sum())


def _no_brain_tools(monkeypatch) -> None:
    def refuse(*_):
        raise AssertionError("a precomputed brain mask must not re-run SynthStrip or BET")

    monkeypatch.setattr(masks, "_synthstrip_command", refuse)
    monkeypatch.setattr(masks, "_bet_command", refuse)


def test_brain_mask_file_is_gated_without_running_a_tool(tmp_path, monkeypatch):
    _no_brain_tools(monkeypatch)
    path = tmp_path / "act_synthstrip_mask.nii.gz"
    voxels = _write_brain_file(path)
    grid = sitk.ReadImage(str(path))

    brain, record = masks.brain_mask_from_file(path, grid, volume_gate=(1.0, 100.0))

    assert int(brain.sum()) == voxels
    assert record["method"] == "synthstrip"
    assert record["qc"] == {"gate_ml": [1.0, 100.0], "verdict": "PASS", "attempts": [
        {"method": "synthstrip", "source": "file", "volume_ml": voxels / 1000.0, "verdict": "PASS"}]}


def test_brain_mask_file_outside_the_gate_raises_qc_error(tmp_path, monkeypatch):
    _no_brain_tools(monkeypatch)
    path = tmp_path / "act_synthstrip_mask.nii.gz"
    _write_brain_file(path)

    with pytest.raises(masks.BrainMaskQCError) as error:
        masks.brain_mask_from_file(path, sitk.ReadImage(str(path)))

    assert error.value.qc["verdict"] == "FAIL"
    assert error.value.attempts[0]["source"] == "file"


def test_build_nifti_packs_brain_mask_file_and_defaces_without_synthstrip(tmp_path, monkeypatch):
    _no_brain_tools(monkeypatch)
    volume = tmp_path / "synthetic.nii.gz"
    brain_file = tmp_path / "act_synthstrip_mask.nii.gz"
    output = tmp_path / "file-mask.capsule.html"
    _write_volume(volume)
    _write_brain_file(brain_file)

    result = cli.main(
        ["build-nifti", "--volume", f"{volume}:MR:T1+C", "--label", "synthetic", "-o", str(output),
         "--viewer", "v2", "--brain-mask", "synthstrip", "--brain-volume-gate", "1,100", "--brain-mask-file", f"{brain_file}:synthstrip",
         "--deface"])

    assert result == 0
    manifest, _ = read_capsule(output)
    brain = [mask for mask in manifest["masks"] if mask["id"].startswith("brain") and mask.get("role") == "render"]
    assert len(brain) == 1 and brain[0]["method"] == "synthstrip"
    assert brain[0]["qc"]["attempts"][0]["source"] == "file"
    assert "brain_mask_qc" not in manifest
    assert manifest["case"]["deidentification"]["face_removed"] is True


@pytest.mark.parametrize("spec", ["", "mask.nii.gz:watershed"])
def test_brain_mask_file_spec_rejects_bad_method(spec):
    with pytest.raises(ValueError, match="--brain-mask-file"):
        cli._parse_brain_mask_file(spec)
