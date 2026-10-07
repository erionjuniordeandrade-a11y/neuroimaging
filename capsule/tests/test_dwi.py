from __future__ import annotations

import json
import os
from dataclasses import replace
from inspect import Parameter, signature
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from capsule import dwi, dwi_sentinel


def _input_files(tmp_path):
    files = {}
    for name in ("dwi.nii.gz", "dwi.bvec", "dwi.bval", "rpe.nii.gz", "t1.nii.gz"):
        path = tmp_path / name
        path.write_bytes(b"synthetic")
        files[name] = path
    return files


def _fake_toolchain(tmp_path, monkeypatch, profile="full", *, empty=False):
    mrtrix = tmp_path / "mrtrix"
    fsldir = tmp_path / "fsl"
    ants = tmp_path / "ants"
    ss3t = tmp_path / "ss3t" / "ss3t_csd_beta1"
    python = tmp_path / "python3.11"
    freesurfer = tmp_path / "freesurfer"
    fake_path = tmp_path / "fake-bin"
    for directory in (mrtrix, fsldir / "bin", ants, fsldir / "data" / "xtract_data" / "Human"):
        directory.mkdir(parents=True, exist_ok=True)
    (fsldir / "data" / "standard").mkdir(parents=True, exist_ok=True)
    (fsldir / "etc" / "flirtsch").mkdir(parents=True, exist_ok=True)
    if not empty:
        for path in dwi.required_tool_paths(profile, env={
            "MRTRIX_BIN": str(mrtrix),
            "FSLDIR": str(fsldir),
            "ANTS_BIN": str(ants),
            "SS3T": str(ss3t),
            "PY311": str(python),
        }):
            if path.parent.name in {"Human", "standard", "flirtsch"}:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                path.chmod(0o755)
        for bundle in dwi.DEFAULT_BUNDLES:
            source = fsldir / "data" / "xtract_data" / "Human" / bundle / "seed.nii.gz"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.touch()
        eddy = fsldir / "bin" / "eddy_openmp"
        eddy.touch()
        eddy.chmod(0o755)
        for path in (mrtrix / "5ttgen", mrtrix / "5ttedit", mrtrix / "5ttcheck",
                     fsldir / "bin" / "run_first_all", fsldir / "bin" / "fast"):
            path.touch()
            path.chmod(0o755)
        synthstrip = freesurfer / "python" / "scripts" / "mri_synthstrip"
        synthstrip.parent.mkdir(parents=True, exist_ok=True)
        synthstrip.touch()
        model = freesurfer / "models" / "synthstrip.1.pt"
        model.parent.mkdir(parents=True, exist_ok=True)
        model.touch()
        fake_path.mkdir(parents=True, exist_ok=True)
        uv = fake_path / "uv"
        uv.touch()
        uv.chmod(0o755)
    env = {
        "MRTRIX_BIN": str(mrtrix),
        "FSLDIR": str(fsldir),
        "ANTS_BIN": str(ants),
        "SS3T": str(ss3t),
        "PY311": str(python),
        "FREESURFER_HOME": str(freesurfer),
    }
    if monkeypatch is not None:
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        if not empty:
            monkeypatch.setenv("PATH", str(fake_path) + os.pathsep + os.environ.get("PATH", ""))
    return env


def _dwi_args(tmp_path, files, profile, *, output=None, work=None):
    return [
        "dwi",
        "--dwi", str(files["dwi.nii.gz"]),
        "--bvec", str(files["dwi.bvec"]),
        "--bval", str(files["dwi.bval"]),
        "--rpe", str(files["rpe.nii.gz"]),
        "--t1", str(files["t1.nii.gz"]),
        "--work", str(work or tmp_path / "work"),
        "--label", "Synthetic",
        "--profile", profile,
        "-o", str(output or tmp_path / "capsule.html"),
        "--dry-run",
    ]


def test_dry_run_fast_and_full_change_csd_and_seed_budget(tmp_path, monkeypatch, capsys):
    files = _input_files(tmp_path)
    _fake_toolchain(tmp_path, monkeypatch)
    monkeypatch.setattr(dwi.subprocess, "run",
                        lambda *args, **kwargs: pytest.fail("dry-run invoked an external command"))

    reports = {}
    for profile in ("fast", "full"):
        assert dwi.main(_dwi_args(tmp_path, files, profile)) == 0
        reports[profile] = json.loads(capsys.readouterr().out)

    fast, full = reports["fast"], reports["full"]
    assert fast["settings"]["fod_algorithm"] == "msmt_csd"
    assert full["settings"]["fod_algorithm"] == "SS3T"
    assert fast["settings"]["seeds"] == 750_000
    assert full["settings"]["seeds"] == 1_500_000
    assert len(dwi.DEFAULT_BUNDLES) == 20
    assert fast["shortcut"] == "2-tissue CSD instead of SS3T; 750k seeds instead of 1.5M"
    fast_fod = _report_stages(fast)["fod"]["commands"]
    full_fod = _report_stages(full)["fod"]["commands"]
    assert any("msmt_csd" in command for command in fast_fod)
    assert any(any("ss3t_csd_beta1" in part for part in command) for command in full_fod)
    assert [stage["stage"] for stage in fast["stages"]] == [
        "preproc", "t1", "register", "act", "fod", "track", "qc", "capsule"]
    assert len(full["stages"]) == 8
    register_commands = fast["stages"][2]["commands"]
    sentinel_end = max(index for index, command in enumerate(register_commands)
                       if any("sent_true_" in part or "sent_mirrored_control_" in part for part in command))
    header_index = next(index for index, command in enumerate(register_commands)
                        if command[0].endswith("transformconvert"))
    regcheck_index = next(index for index, command in enumerate(register_commands)
                          if command[0].endswith("antsRegistration"))
    assert sentinel_end < header_index < regcheck_index
    normalize = lambda command, profile: [part.replace(f"/work/{profile}/", "/work/RUN/")
                                          for part in command]
    fast_stages, full_stages = _report_stages(fast), _report_stages(full)
    for stage_name in ("preproc", "t1", "register", "act", "track", "qc", "capsule"):
        fast_commands = fast_stages[stage_name]["commands"]
        full_commands = full_stages[stage_name]["commands"]
        assert len(fast_commands) == len(full_commands)
        for fast_command, full_command in zip(fast_commands, full_commands):
            fast_command = normalize(fast_command, "fast")
            full_command = normalize(full_command, "full")
            if stage_name == "track" and "-seeds" in fast_command:
                seed_index = fast_command.index("-seeds") + 1
                assert fast_command[:seed_index] == full_command[:seed_index]
                assert fast_command[seed_index] == "750000"
                assert full_command[seed_index] == "1500000"
                assert fast_command[seed_index + 1:] == full_command[seed_index + 1:]
            else:
                assert fast_command == full_command
    assert all(not Path(path).exists() for path in (tmp_path / "work", tmp_path / "capsule.html"))


def test_capsule_cli_wires_dwi_tool_check(tmp_path, monkeypatch, capsys):
    _fake_toolchain(tmp_path, monkeypatch, profile="fast")

    from capsule.cli import main as capsule_main

    assert capsule_main(["dwi", "--check-tools", "--profile", "fast"]) == 0
    assert capsys.readouterr().out.strip() == "DWI tool check passed for profile=fast"


def test_dry_run_seed_override_reaches_every_bundle_command(tmp_path, monkeypatch, capsys):
    files = _input_files(tmp_path)
    _fake_toolchain(tmp_path, monkeypatch)

    assert dwi.main(_dwi_args(tmp_path, files, "fast") + ["--seeds", "1234"]) == 0
    report = json.loads(capsys.readouterr().out)

    assert report["settings"]["seeds"] == 1234
    tracking = [command for command in _report_stages(report)["track"]["commands"]
                if "-algorithm" in command]
    assert len(tracking) == 20
    assert all(command[command.index("-seeds") + 1] == "1234" for command in tracking)


def test_dry_run_includes_reverse_pe_gradient_metadata(tmp_path, monkeypatch, capsys):
    files = _input_files(tmp_path)
    files["rpe.bvec"] = tmp_path / "rpe.bvec"
    files["rpe.bval"] = tmp_path / "rpe.bval"
    files["rpe.bvec"].write_text("0 0 0\n0 0 0\n0 0 0\n")
    files["rpe.bval"].write_text("0 0 0\n")
    _fake_toolchain(tmp_path, monkeypatch)

    argv = _dwi_args(tmp_path, files, "fast") + [
        "--rpe-bvec", str(files["rpe.bvec"]), "--rpe-bval", str(files["rpe.bval"]),
    ]
    assert dwi.main(argv) == 0
    report = json.loads(capsys.readouterr().out)
    reverse = report["stages"][0]["commands"][1]

    assert reverse[0].endswith("mrconvert")
    metadata = reverse.index("-fslgrad")
    assert reverse[metadata + 1:metadata + 3] == [
        str(tmp_path / "work" / "nii" / "s8.bvec"),
        str(tmp_path / "work" / "nii" / "s8.bval"),
    ]
    assert not (tmp_path / "work").exists()


def test_existing_copied_input_reuses_identical_and_names_changed_file(tmp_path):
    files = _input_files(tmp_path)
    work = tmp_path / "work"
    legacy = work / "nii" / dwi.INPUT_LAYOUT["dwi"]
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(files["dwi.nii.gz"].read_bytes())
    parser = __import__("argparse").ArgumentParser()
    dwi.configure_parser(parser)
    args = parser.parse_args(["--dwi", str(files["dwi.nii.gz"]), "--bvec", str(files["dwi.bvec"]),
                              "--bval", str(files["dwi.bval"]), "--rpe", str(files["rpe.nii.gz"]),
                              "--t1", str(files["t1.nii.gz"])])

    dwi._materialize_inputs(args, work, "preproc")
    files["dwi.nii.gz"].write_bytes(b"changed synthetic input")
    with pytest.raises(ValueError, match=r"stage preproc: existing input file differs.*s7.nii.gz"):
        dwi._materialize_inputs(args, work, "preproc")


def test_prepopulated_legacy_flair_dry_run_is_repeatable(tmp_path, monkeypatch, capsys):
    files = _input_files(tmp_path)
    files["flair.nii.gz"] = tmp_path / "flair.nii.gz"
    files["flair.nii.gz"].write_bytes(b"synthetic flair")
    _fake_toolchain(tmp_path, monkeypatch)
    work = tmp_path / "legacy-work"
    (work / "pp").mkdir(parents=True)
    (work / "pp" / "dwi_pp.mif").write_bytes(b"legacy dwi")
    (work / "pp" / "mask.mif").write_bytes(b"legacy mask")
    t1 = work / "t1"
    t1.mkdir()
    for name in ("t1c_1mm.nii.gz", "t1c_brain.nii.gz", "mni2t1_warp.nii.gz",
                 "t12mni_warp.nii.gz", "flair.nii.gz", "flair_t1.nii.gz"):
        (t1 / name).write_bytes(b"legacy " + name.encode())
    argv = _dwi_args(tmp_path, files, "fast", work=work) + ["--flair", str(files["flair.nii.gz"]),
                                                               "--adopt-existing"]

    reports = []
    for _ in range(2):
        assert dwi.main(argv) == 0
        reports.append(json.loads(capsys.readouterr().out))

    for report in reports:
        assert [stage["status"] for stage in report["stages"][:2]] == ["adopted", "adopted"]
        register = report["stages"][2]
        assert register["status"] == "planned"
        assert "FLIRT dof6 normmi" in register["params"]["flair_registration"]
        assert any(any("flair_t1.nii.gz" in part for part in command)
                   for command in register["commands"])
        capsule = report["stages"][-1]["commands"]
        assert "--volume-registration" in capsule[0]
        assert "none-shared-world" not in " ".join(capsule[0])
    assert (work / "t1" / "flair.nii.gz").is_file()
    assert not (work / "fast" / "nifti").exists()
    parser = __import__("argparse").ArgumentParser()
    dwi.configure_parser(parser)
    args = parser.parse_args(argv[1:])
    t1_spec = dwi._stage_specs(args, work, tmp_path / "capsule.html",
                               dwi.Toolchain.from_env())[1]
    assert t1 / "flair.nii.gz" in t1_spec.key_outputs
    assert t1 / "flair_t1.nii.gz" in t1_spec.key_outputs
    adopted = dwi.run_stage(t1_spec, profile="fast", work_dir=work, tool_versions={},
                            adopt_existing=True, action=lambda: pytest.fail("legacy T1 ran"))
    record = json.loads(adopted.marker.read_text())
    assert adopted.status == "adopted"
    assert {item["path"] for item in record["outputs"]} >= {"t1/flair.nii.gz", "t1/flair_t1.nii.gz"}
    skipped = dwi.run_stage(t1_spec, profile="fast", work_dir=work, tool_versions={},
                            adopt_existing=False, action=lambda: pytest.fail("adopted T1 reran"))
    assert skipped.status == "skipped"


def test_capsule_brain_mask_receives_dwi_command_logger(monkeypatch):
    from types import SimpleNamespace

    from capsule import cli

    command_runner = object()
    received = {}

    def brain_mask(values, grid, method, *, command_runner=None):
        received.update(values=values, grid=grid, method=method, runner=command_runner)
        return np.ones((1,), dtype=bool), {"method": "bet", "seconds": 0}

    monkeypatch.setattr(cli.maskops, "brain_mask", brain_mask)
    values, grid = np.asarray([1]), object()

    result = cli._select_brain(
        SimpleNamespace(brain_mask="bet", _command_runner=command_runner),
        [("mr", "MR", values, True)], grid)

    assert received["values"] is values
    assert received["grid"] is grid
    assert received["method"] == "bet"
    assert received["runner"] is command_runner
    assert result[0] is not None


def test_stage_marker_skips_identical_params_and_reruns_when_params_change(tmp_path):
    work = tmp_path / "work"
    directory = work / "pp"
    output = directory / "dwi_pp.mif"
    spec = dwi.StageSpec("preproc", directory, {"readout_time": 0.05}, (output,), (output,))
    calls = []

    def action():
        calls.append(1)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(str(len(calls)).encode())

    first = dwi.run_stage(spec, profile="fast", work_dir=work, tool_versions={},
                          adopt_existing=False, action=action)
    second = dwi.run_stage(spec, profile="fast", work_dir=work, tool_versions={},
                           adopt_existing=False, action=action)
    changed = replace(spec, params={"readout_time": 0.06})
    third = dwi.run_stage(changed, profile="fast", work_dir=work, tool_versions={},
                          adopt_existing=False, action=action)

    assert [first.status, second.status, third.status] == ["ran", "skipped", "ran"]
    assert len(calls) == 2
    marker = json.loads((directory / ".stage-preproc.json").read_text())
    assert set(marker) == {"stage", "profile", "params", "tool_versions",
                           "started_utc", "finished_utc", "outputs"}
    assert marker["outputs"] == [{"path": "pp/dwi_pp.mif", "bytes": 1}]


@pytest.mark.parametrize("stage", ["preproc", "t1"])
def test_adopt_existing_is_limited_to_preproc_and_t1(tmp_path, stage):
    work = tmp_path / "work"
    directory = work / stage
    output = directory / "key-output"
    other_output = directory / "optional-intermediate"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"legacy")
    spec = dwi.StageSpec(stage, directory, {"input": "synthetic"},
                         (output, other_output), (output,))

    result = dwi.run_stage(spec, profile="full", work_dir=work, tool_versions={},
                           adopt_existing=True, action=lambda: pytest.fail("adopted stage ran"))
    rerun = dwi.run_stage(spec, profile="full", work_dir=work, tool_versions={},
                          adopt_existing=False, action=lambda: pytest.fail("adopted stage did not skip"))

    marker = json.loads((directory / f".stage-{stage}.json").read_text())
    assert result.status == "adopted"
    assert rerun.status == "skipped"
    assert marker["adopted"] is True
    assert marker["outputs"] == [{"path": f"{stage}/key-output", "bytes": 6}]


def test_later_stage_is_not_adopted(tmp_path):
    work = tmp_path / "work"
    directory = work / "full" / "nifti"
    output = directory / "wmfod_norm.mif"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"legacy")
    spec = dwi.StageSpec("fod", directory, {"algorithm": "SS3T"}, (output,), (output,))
    calls = []

    def action():
        calls.append(1)
        output.write_bytes(b"recomputed")

    result = dwi.run_stage(spec, profile="full", work_dir=work, tool_versions={},
                           adopt_existing=True, action=action)

    assert result.status == "ran"
    assert calls == [1]
    assert "adopted" not in json.loads((directory / ".stage-fod.json").read_text())


def test_tool_check_reports_all_missing_paths_from_fake_roots(tmp_path):
    env = _fake_toolchain(tmp_path, monkeypatch=None, empty=True)

    missing = dwi.check_tools("full", env=env)

    joined = "\n".join(missing)
    assert len(missing) > 20
    assert str(tmp_path / "mrtrix" / "mrconvert") in joined
    assert str(tmp_path / "fsl" / "data" / "standard" / "MNI152_T1_2mm_brain.nii.gz") in joined
    assert str(tmp_path / "ants" / "N4BiasFieldCorrection") in joined
    assert str(tmp_path / "fsl" / "bin" / "topup") in joined
    # dwifslpreproc -rpe_pair -eddyqc_all also calls these three.
    for tool in (("fsl", "bin", "applytopup"), ("fsl", "bin", "eddy_quad"), ("mrtrix", "mrcalc")):
        assert str(tmp_path.joinpath(*tool)) in joined
    assert str(tmp_path / "fsl" / "bin" / "eddy_openmp") in joined
    assert str(tmp_path / "ss3t" / "ss3t_csd_beta1") in joined
    assert str(tmp_path / "python3.11") in joined
    for name in ("5ttgen", "5ttedit", "5ttcheck"):
        assert str(tmp_path / "mrtrix" / name) in joined
    for name in ("run_first_all", "fast"):
        assert str(tmp_path / "fsl" / "bin" / name) in joined
    assert str(tmp_path / "freesurfer" / "python" / "scripts" / "mri_synthstrip") in joined
    assert str(tmp_path / "freesurfer" / "models" / "synthstrip.1.pt") in joined


def test_cli_tool_check_prints_all_missing_paths_before_stages(tmp_path, monkeypatch, capsys):
    env = _fake_toolchain(tmp_path, monkeypatch=None, empty=True)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    assert dwi.main(["--check-tools", "--profile", "full"]) == 1

    reported = capsys.readouterr().err
    assert "Missing DWI dependencies:" in reported
    assert str(tmp_path / "mrtrix" / "mrconvert") in reported
    assert str(tmp_path / "fsl" / "bin" / "flirt") in reported
    assert str(tmp_path / "ants" / "N4BiasFieldCorrection") in reported
    assert str(tmp_path / "fsl" / "bin" / "topup") in reported
    assert str(tmp_path / "fsl" / "bin" / "eddy_openmp") in reported
    assert str(tmp_path / "ss3t" / "ss3t_csd_beta1") in reported
    assert str(tmp_path / "python3.11") in reported


def test_refuses_output_overwrite_and_work_inside_repo(tmp_path):
    output = tmp_path / "existing.html"
    output.write_text("existing")
    with pytest.raises(FileExistsError):
        dwi.validate_output_path(output)
    with pytest.raises(ValueError, match="inside the Git repository"):
        dwi.validate_work_path(dwi.REPO_ROOT / "dwi-work")


def test_cli_refuses_existing_output_and_repository_work_directory(tmp_path, monkeypatch, capsys):
    files = _input_files(tmp_path)
    _fake_toolchain(tmp_path, monkeypatch, profile="fast")
    existing = tmp_path / "existing.html"
    existing.write_text("keep")

    assert dwi.main(_dwi_args(tmp_path, files, "fast", output=existing)) == 1
    assert "refusing to overwrite existing output" in capsys.readouterr().err
    assert existing.read_text() == "keep"

    repo_work = dwi.REPO_ROOT / "dwi-work-must-not-be-created"
    assert dwi.main(_dwi_args(tmp_path, files, "fast", work=repo_work)) == 1
    assert "inside the Git repository" in capsys.readouterr().err
    assert not repo_work.exists()


def _save_tck(path, streamlines):
    tractogram = nib.streamlines.Tractogram(streamlines, affine_to_rasmm=np.eye(4))
    nib.streamlines.save(tractogram, str(path))


def _line(x0, x1):
    return np.asarray([[x0, 0.0, 0.0], [x1, 1.0, 0.0]], dtype=np.float32)


def _qc_mask(path):
    data = np.ones((24, 20, 20), dtype=np.uint8)
    affine = np.eye(4)
    affine[0, 3] = -12.0
    affine[1, 3] = -10.0
    affine[2, 3] = -10.0
    nib.save(nib.Nifti1Image(data, affine), str(path))


def test_bundle_qc_passes_ipsilateral_bundle(tmp_path):
    mask = tmp_path / "mask.nii.gz"
    _qc_mask(mask)
    tracts = tmp_path / "tracts"
    tracts.mkdir()
    _save_tck(tracts / "fx_cst_l.tck", [_line(-10.0, -8.0) for _ in range(500)])

    report = dwi.bundle_qc(mask, tracts, seeds_per_bundle=750_000)

    assert report["seeds_per_bundle"] == 750_000
    assert report["scope"] == "descriptive"
    assert report["tract_validity"] == "not assessed"
    assert report["bundles"][0]["verdict"] == "PASS"


def test_bundle_qc_warns_for_midline_crossing(tmp_path):
    mask = tmp_path / "mask.nii.gz"
    _qc_mask(mask)
    tracts = tmp_path / "tracts"
    tracts.mkdir()
    _save_tck(tracts / "fx_cst_l.tck", [_line(-8.0, 8.0) for _ in range(500)])

    row = dwi.bundle_qc(mask, tracts, seeds_per_bundle=750_000)["bundles"][0]

    assert row["midline_cross"] == 500
    assert row["verdict"] == "WARN"


def test_bundle_qc_warns_for_wrong_hemisphere_label(tmp_path):
    mask = tmp_path / "mask.nii.gz"
    _qc_mask(mask)
    tracts = tmp_path / "tracts"
    tracts.mkdir()
    _save_tck(tracts / "fx_cst_l.tck", [_line(8.0, 10.0) for _ in range(500)])

    row = dwi.bundle_qc(mask, tracts, seeds_per_bundle=750_000)["bundles"][0]

    assert row["centroid_wrong_hemi"] == 500
    assert row["verdict"] == "WARN"


def test_bundle_qc_fails_empty_bundle(tmp_path):
    mask = tmp_path / "mask.nii.gz"
    _qc_mask(mask)
    tracts = tmp_path / "tracts"
    tracts.mkdir()
    _save_tck(tracts / "fx_cst_l.tck", [])

    row = dwi.bundle_qc(mask, tracts, seeds_per_bundle=1_500_000)["bundles"][0]

    assert row["streamlines"] == 0
    assert row["verdict"] == "FAIL"


def test_tracking_execution_uses_the_dry_run_argv_plan(tmp_path, monkeypatch):
    env = _fake_toolchain(tmp_path, monkeypatch, profile="fast")
    tools = dwi.Toolchain.from_env(env)
    seed = tools.xtract / "cst_l" / "seed.nii.gz"
    seed.parent.mkdir(parents=True, exist_ok=True)
    seed.write_bytes(b"synthetic ROI")
    args = type("Args", (), {"profile": "fast", "bundles": ["cst_l"],
                              "seeds": 750_000, "threads": 1})()
    plan, roi_outputs, tck_outputs = dwi._tracking_plan(args, tmp_path / "work", tools)
    tracts = tmp_path / "work" / "fast" / "tracts"
    spec = dwi.StageSpec("track", tracts, {}, tuple([*roi_outputs, *tck_outputs]),
                         tuple(tck_outputs), plan.commands, track_plan=plan)

    class RecordingRunner:
        def __init__(self):
            self.commands = []

        def run(self, argv, *, stage, cwd):
            argv = tuple(map(str, argv))
            self.commands.append(argv)
            if argv[0].endswith("tckgen"):
                Path(argv[2]).parent.mkdir(parents=True, exist_ok=True)
                Path(argv[2]).write_bytes(b"synthetic tract")
                return dwi.subprocess.CompletedProcess(argv, 0, "", "")
            if argv[0].endswith("tckinfo"):
                return dwi.subprocess.CompletedProcess(argv, 0, "count: 1\\n", "")
            return dwi.subprocess.CompletedProcess(argv, 0, "10 10\\n", "")

    runner = RecordingRunner()
    counts = dwi._run_track(spec, args, tmp_path / "work", runner)

    assert counts == {"cst_l": 1}
    assert runner.commands == list(plan.commands)
    assert (tracts / "fx_cst_l.tck").is_file()


def _report_stages(report):
    return {stage["stage"]: stage for stage in report["stages"]}


def test_act_stage_commands_and_tckgen_include_lesion_in_pathological_tissue(tmp_path, monkeypatch, capsys):
    files = _input_files(tmp_path)
    files["lesion.nii.gz"] = tmp_path / "lesion.nii.gz"
    files["lesion.nii.gz"].write_bytes(b"synthetic lesion mask")
    _fake_toolchain(tmp_path, monkeypatch)

    assert dwi.main(_dwi_args(tmp_path, files, "fast") + ["--lesion", str(files["lesion.nii.gz"])]) == 0
    stages = _report_stages(json.loads(capsys.readouterr().out))
    act = stages["act"]["commands"]
    tools = [Path(command[0]).name for command in act]
    synthstrip_index = next(i for i, command in enumerate(act)
                            if any(str(part).endswith("mri_synthstrip") for part in command))
    fslmaths_index = next(i for i, tool in enumerate(tools) if tool == "fslmaths")
    five_ttgen_index = next(i for i, tool in enumerate(tools) if tool == "5ttgen")
    lesion_flirt_index = next(i for i, command in enumerate(act)
                              if Path(command[0]).name == "flirt" and "lesion1_dwi.nii.gz" in command[-1])
    transform_index = next(i for i, tool in enumerate(tools) if tool == "mrtransform")
    edit_index = next(i for i, tool in enumerate(tools) if tool == "5ttedit")
    check_index = next(i for i, tool in enumerate(tools) if tool == "5ttcheck")
    assert synthstrip_index < fslmaths_index < five_ttgen_index < lesion_flirt_index < transform_index < edit_index < check_index

    raw = str(tmp_path / "work" / "fast" / "nifti" / "5tt_raw.mif")
    final = str(tmp_path / "work" / "fast" / "nifti" / "5tt.mif")
    lesion_dwi = str(tmp_path / "work" / "fast" / "tracts" / "roi" / "lesion1_dwi.nii.gz")
    lesion_on_5tt = str(tmp_path / "work" / "fast" / "nifti" / "lesion_on5tt.mif")
    assert act[five_ttgen_index][1] == "fsl"
    assert act[five_ttgen_index][2].endswith("t1c_dwiworld_brainmasked.nii.gz")
    assert act[five_ttgen_index][3] == raw
    assert act[five_ttgen_index][-2:] == ["-premasked", "-nocrop"]
    assert act[transform_index][1:] == [lesion_dwi, "-template", raw, "-interp", "nearest",
                                        lesion_on_5tt, "-force"]
    assert act[edit_index][1:] == [raw, "-path", lesion_on_5tt, final, "-force"]
    assert act[check_index][1] == final
    tracking = next(command for command in stages["track"]["commands"] if "-algorithm" in command)
    assert tracking[tracking.index("-act") + 1] == final
    assert "-backtrack" in tracking
    assert "-crop_at_gmwmi" in tracking


def test_act_stage_without_lesion_promotes_raw_5tt_and_checks_final(tmp_path, monkeypatch, capsys):
    files = _input_files(tmp_path)
    _fake_toolchain(tmp_path, monkeypatch)

    assert dwi.main(_dwi_args(tmp_path, files, "fast")) == 0
    stages = _report_stages(json.loads(capsys.readouterr().out))
    act = stages["act"]["commands"]
    tools = [Path(command[0]).name for command in act]
    raw = str(tmp_path / "work" / "fast" / "nifti" / "5tt_raw.mif")
    final = str(tmp_path / "work" / "fast" / "nifti" / "5tt.mif")

    assert "flirt" not in tools
    assert "mrtransform" not in tools
    assert "5ttedit" not in tools
    promote = next(command for command in act if Path(command[0]).name == "mrconvert")
    check = next(command for command in act if Path(command[0]).name == "5ttcheck")
    assert promote[1:] == [raw, final, "-force"]
    assert check[1] == final


def test_act_off_keeps_tracking_argv_unchanged_and_omits_act_stage(tmp_path, monkeypatch, capsys):
    files = _input_files(tmp_path)
    _fake_toolchain(tmp_path, monkeypatch)

    assert dwi.main(_dwi_args(tmp_path, files, "fast") + ["--act", "off"]) == 0
    stages = _report_stages(json.loads(capsys.readouterr().out))
    assert "act" not in stages
    tracking = [command for command in stages["track"]["commands"] if "-algorithm" in command]
    assert len(tracking) == 20
    assert all(not {"-act", "-backtrack", "-crop_at_gmwmi"}.intersection(command)
               for command in tracking)


def test_capsule_argv_uses_synthstrip_deface_lesion_and_tracking_length(tmp_path):
    nifti = tmp_path / "fast" / "nifti"
    tracts = tmp_path / "fast" / "tracts"
    args = SimpleNamespace(flair=None, bundles=["cst_l"], lesion="lesion.nii.gz", no_deface=False)

    argv = dwi._capsule_argv(args, tmp_path / "capsule.html", nifti, tracts, "Synthetic")

    assert argv[argv.index("--brain-mask") + 1] == "synthstrip"
    assert "--deface" in argv
    assert argv[argv.index("--lesion-mask") + 1] == (
        f"{tracts / 'roi' / 'lesion1_dwi.nii.gz'}:Lesão (ROI de rastreamento)")
    assert argv[argv.index("--tract-max-length-mm") + 1] == str(dwi.TRACKING["max_length_mm"])

    args.no_deface = True
    argv_without_deface = dwi._capsule_argv(args, tmp_path / "capsule.html", nifti, tracts, "Synthetic")
    assert "--deface" not in argv_without_deface


def test_capsule_argv_reuses_the_act_synthstrip_mask_only_when_act_ran(tmp_path):
    nifti = tmp_path / "fast" / "nifti"
    tracts = tmp_path / "fast" / "tracts"
    args = SimpleNamespace(flair=None, bundles=["cst_l"], lesion=None, no_deface=False, act="auto")

    argv = dwi._capsule_argv(args, tmp_path / "capsule.html", nifti, tracts, "Synthetic")
    assert argv[argv.index("--brain-mask-file") + 1] == (
        f"{nifti / 't1c_dwiworld_synthstrip_mask.nii.gz'}:synthstrip")

    args.act = "off"
    argv_without_act = dwi._capsule_argv(args, tmp_path / "capsule.html", nifti, tracts, "Synthetic")
    assert "--brain-mask-file" not in argv_without_act
    assert argv_without_act[argv_without_act.index("--brain-mask") + 1] == "synthstrip"


def test_capsule_provenance_records_act_and_per_bundle_roi_names(tmp_path):
    path = tmp_path / "capsule-dwi-provenance.json"
    act = {"enabled": True, "method": "5ttgen fsl", "pathological_tissue": "lesion.nii.gz",
           "backtrack": True, "crop_at_gmwmi": True}
    bundle_rois = {"cst_l": {"seed": "roi/cst_l_seed.nii.gz",
                              "include": ["roi/cst_l_target.nii.gz"], "exclude": []}}

    dwi._write_capsule_provenance(path, "fast", 750_000, act=act, bundle_rois=bundle_rois)

    record = json.loads(path.read_text())
    assert record["act"] == act
    assert record["bundle_rois"] == bundle_rois


def test_tracking_reports_every_failed_bundle_and_message(tmp_path):
    jobs = tuple(dwi._TrackBundlePlan(
        bundle, ("tckgen", "input.mif", f"{bundle}.tmp.tck"), ("tckinfo", f"{bundle}.tmp.tck"),
        tmp_path / f"{bundle}.tmp.tck", tmp_path / f"{bundle}.tck")
        for bundle in ("cst_l", "af_l", "uf_l"))
    plan = dwi._TrackPlan((), jobs, concurrent_jobs=3)
    spec = dwi.StageSpec("track", tmp_path / "tracts", {}, (), (), track_plan=plan)
    args = SimpleNamespace(profile="fast")

    class FailingRunner:
        def run(self, argv, *, stage, cwd):
            raise RuntimeError(f"synthetic failure for {argv[2]}")

    with pytest.raises(RuntimeError) as raised:
        dwi._run_track(spec, args, tmp_path, FailingRunner())

    message = str(raised.value)
    for bundle in ("cst_l", "af_l", "uf_l"):
        assert bundle in message
        assert f"synthetic failure for {bundle}.tmp.tck" in message


def test_act_stage_resume_skips_matching_outputs_and_reruns_changed_lesion_signature(tmp_path, monkeypatch):
    files = _input_files(tmp_path)
    files["lesion.nii.gz"] = tmp_path / "lesion.nii.gz"
    files["lesion.nii.gz"].write_bytes(b"synthetic lesion v1")
    env = _fake_toolchain(tmp_path, monkeypatch, profile="fast")
    parser = __import__("argparse").ArgumentParser()
    dwi.configure_parser(parser)
    argv = _dwi_args(tmp_path, files, "fast") + ["--lesion", str(files["lesion.nii.gz"])]
    args = parser.parse_args(argv[1:])
    work = tmp_path / "work"
    tools = dwi.Toolchain.from_env(env)
    act = next(spec for spec in dwi._stage_specs(args, work, tmp_path / "capsule.html", tools)
               if spec.name == "act")

    def produce_outputs():
        for output in act.outputs:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"synthetic ACT output")

    first = dwi.run_stage(act, profile="fast", work_dir=work, tool_versions={},
                          adopt_existing=False, action=produce_outputs)
    skipped = dwi.run_stage(act, profile="fast", work_dir=work, tool_versions={},
                            adopt_existing=False, action=lambda: pytest.fail("matching ACT stage reran"))
    assert first.status == "ran"
    assert skipped.status == "skipped"

    files["lesion.nii.gz"].write_bytes(b"synthetic lesion v2")
    changed_args = parser.parse_args(argv[1:])
    changed_act = next(spec for spec in dwi._stage_specs(
        changed_args, work, tmp_path / "capsule.html", tools) if spec.name == "act")
    rerun = dwi.run_stage(changed_act, profile="fast", work_dir=work, tool_versions={},
                          adopt_existing=False, action=produce_outputs)
    assert rerun.status == "ran"


def _sentinel_found(*, mirrored=False):
    found = {}
    for name, spec in dwi_sentinel.MARKERS.items():
        world = [0.0, 0.0, 0.0]
        world[spec["axis"]] = 10.0 if spec["expect"] == "+" else -10.0
        if mirrored and spec["axis"] == 0:
            world[0] *= -1
        found[name] = {"world": world}
    return found


def test_orientation_sentinel_judge_keeps_mirrored_positive_control():
    real_ok, real_rows = dwi_sentinel.judge(_sentinel_found())
    mirror_ok, mirror_rows = dwi_sentinel.judge(_sentinel_found(mirrored=True))

    assert real_ok is True
    assert mirror_ok is False
    assert any(row[2] == "WRONG SIDE" for row in mirror_rows if row[0] in {"RIGHT", "LEFT"})
    assert len(real_rows) == len(dwi_sentinel.MARKERS)
    assert signature(dwi_sentinel.main).parameters["out_json"].default is Parameter.empty
