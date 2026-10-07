from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "pipeline" / "run_case.sh"


def make_case(
    root: Path,
    *,
    rpe_mode: str,
    header_metadata: bool = True,
    regrid_voxel_mm: list[float] | None = None,
) -> str:
    case_id = "case-1a2b3c4d"
    raw = root / "raw" / "dcm2niix"
    raw.mkdir(parents=True)
    for name in ("dwi.nii.gz", "dwi.bval", "dwi.bvec", "dwi.json", "rpe.nii.gz", "rpe.json", "t1.nii.gz"):
        (raw / name).write_text("synthetic placeholder\n")

    dwi_sidecar = {
        "PhaseEncodingDirection": "j-",
        "TotalReadoutTime": 0.049,
    }
    rpe_sidecar = {"PhaseEncodingDirection": "j", "TotalReadoutTime": 0.049}
    if not header_metadata:
        dwi_sidecar.pop("PhaseEncodingDirection")
        rpe_sidecar.pop("TotalReadoutTime")
    (raw / "dwi.json").write_text(json.dumps(dwi_sidecar))
    (raw / "rpe.json").write_text(json.dumps(rpe_sidecar))

    case = {
        "schema": "tractlab.case/1",
        "case_id": case_id,
        "label": "Synthetic runner test",
        "created_utc": "2026-09-26T12:00:00Z",
        "source": {"n_files": 7},
        "inputs": {
            "dwi": {
                "nifti": "raw/dcm2niix/dwi.nii.gz",
                "bval": "raw/dcm2niix/dwi.bval",
                "bvec": "raw/dcm2niix/dwi.bvec",
                "json": "raw/dcm2niix/dwi.json",
                "pe_dir": "j-",
                "total_readout_time": 0.049,
                "regrid_voxel_mm": regrid_voxel_mm,
            },
            "rpe": (
                {"mode": "pair", "nifti": "raw/dcm2niix/rpe.nii.gz", "json": "raw/dcm2niix/rpe.json"}
                if rpe_mode == "pair"
                else {"mode": "none"}
            ),
            "t1": {"nifti": "raw/dcm2niix/t1.nii.gz", "json": "raw/dcm2niix/t1.json", "post_contrast": False},
        },
        "warnings": [],
    }
    (root / "case.json").write_text(json.dumps(case, indent=2) + "\n")
    manifest = {
        "case_id": case_id,
        "case_root": ".",
        "disclaimer": "Research/preview only — not navigation.",
        "status": "ingested",
        "inputs": {
            "b0": {"path": "nifti/b0.nii.gz"},
            "fod": {"path": "nifti/wmfod_norm.mif"},
            "mask": {"path": "nifti/mask_up.nii.gz"},
            "t1": {"path": "nifti/t1_brain_dwi.nii.gz"},
        },
        "t1_qc": {"approved_by": None, "date": None},
        "atlas_prior_qc": {"approved_by": None, "date": None},
        "parcellation_qc": {"approved_by": None, "date": None},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return case_id


def run_dry_case(
    root: Path,
    *,
    fail_at: str | None = None,
    tmpdir: Path | None = None,
    fs_recon: str | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("TRACTLAB_FS_RECON", None)
    if fs_recon is not None:
        env["TRACTLAB_FS_RECON"] = fs_recon
    if tmpdir is not None:
        env["TMPDIR"] = str(tmpdir)
    env["TRACTLAB_DRYRUN"] = "1"
    env["TRACTLAB_PYTHON"] = sys.executable
    env["HOME"] = str(root / "missing-toolchain-home")
    env["FSLDIR"] = str(root / "missing-fsl")
    env["ANTSPATH"] = str(root / "missing-ants")
    env["FREESURFER_HOME"] = str(root / "missing-freesurfer")
    env["PATH"] = "/usr/bin:/bin"
    env.pop("TRACTLAB_DRYRUN_FAIL_AT", None)
    if fail_at:
        env["TRACTLAB_DRYRUN_FAIL_AT"] = fail_at
    return subprocess.run(
        ["/bin/bash", str(RUNNER), str(root)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def status_for(root: Path) -> dict:
    return json.loads((root / "status.json").read_text())


def test_pair_dry_run_matches_house_commands_and_writes_done_status(tmp_path: Path) -> None:
    case_id = make_case(tmp_path, rpe_mode="pair")

    result = run_dry_case(tmp_path)

    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert re.findall(r"^DRY STAGE (\w+)$", output, flags=re.MULTILINE) == [
        "ss3t",
        "t1reg",
        "banks",
        "scalar_maps",
    ]
    assert "DRY check command -v ss3t_csd_beta1" in output
    assert "DRY check command -v flirt" in output
    assert f"DRY check command -v {tmp_path}/missing-toolchain-home/mrtrix3/bin/dwi2tensor" in output
    preproc = next(line for line in output.splitlines() if line.startswith("DRY dwifslpreproc "))
    assert " -rpe_pair " in preproc
    assert " -se_epi " in preproc
    preproc_args = shlex.split(preproc.removeprefix("DRY "))
    assert preproc_args[0] == "dwifslpreproc"
    assert preproc_args[3:6] == ["-rpe_pair", "-se_epi", str(tmp_path / "work/ss3t/se_pair.mif")]
    assert sum(arg.startswith("-rpe_") for arg in preproc_args) == 1
    assert "-pe_dir" not in preproc_args and "-readout_time" not in preproc_args
    eddy_options = preproc_args[preproc_args.index("-eddy_options") + 1]
    assert "--nthr=8" in eddy_options.split()
    assert preproc_args[-5:] == ["-eddyqc_all", str(tmp_path / "work/ss3t/eddyqc"), "-nthreads", "8", "-quiet"]
    assert "DRY mrcat " in output and "work/ss3t/dwi_b0_pe.mif" in output and "work/ss3t/rpe.mif" in output

    recon = next(line for line in output.splitlines() if line.startswith("DRY recon-all "))
    assert f"-s {case_id}" in recon
    tracking = next(
        line for line in output.splitlines()
        if "tckgen" in line and line.startswith("DRY ") and not line.startswith("DRY check ")
    )
    assert " -algorithm iFOD2 " in tracking
    assert " -seed_dynamic " in tracking
    assert " -act " in tracking
    assert " -backtrack -crop_at_gmwmi -select 10000000 -seeds 0 -cutoff 0.06 -minlength 10 -maxlength 250 " in tracking

    status = status_for(tmp_path)
    assert status["schema"] == "tractlab.status/1"
    assert status["state"] == "done"
    assert status["stage"] == "scalar_maps"
    assert status["error"] is None
    assert [stage["name"] for stage in status["stages"]] == ["ss3t", "t1reg", "banks", "scalar_maps"]
    assert all(stage["state"] == "done" for stage in status["stages"])
    assert all("started_utc" in stage and "ended_utc" in stage for stage in status["stages"])

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["case_root"] == "."
    assert "filter_bank" not in manifest["inputs"]
    assert manifest["t1_qc"]["approved_by"] is None
    assert "/Users/" not in RUNNER.read_text()


def test_none_dry_run_uses_case_pe_metadata(tmp_path: Path) -> None:
    make_case(tmp_path, rpe_mode="none")

    result = run_dry_case(tmp_path)

    assert result.returncode == 0, result.stderr
    preproc = next(line for line in result.stdout.splitlines() if line.startswith("DRY dwifslpreproc "))
    assert " -rpe_none -pe_dir j- -readout_time 0.049 " in preproc
    assert " -se_epi " not in preproc
    assert " -align_seepi " not in preproc
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert "filter_bank" not in manifest["inputs"]


def test_manifest_bank_update_adds_filter_record_and_keeps_qc_unsigned(tmp_path: Path) -> None:
    make_case(tmp_path, rpe_mode="none")
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["inputs"]["bank_demo"] = {
        "path": "tracts/bank/demo.tck",
        "note": "Research only — no reverse-PE, not navigation.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    helper = RUNNER.parent / "runner_state.py"

    root_result = subprocess.run(
        [sys.executable, str(helper), "manifest-root", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    bank_result = subprocess.run(
        [sys.executable, str(helper), "manifest-banks", str(tmp_path), "none"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert root_result.returncode == 0, root_result.stderr
    assert bank_result.returncode == 0, bank_result.stderr
    result = json.loads(manifest_path.read_text())
    assert result["case_root"] == str(tmp_path.resolve())
    assert result["inputs"]["filter_bank"]["path"].endswith("wholebrain_act_ifod2_100000.tck")
    assert "no reverse-PE: residual EPI distortion uncorrected" in result["inputs"]["filter_bank"]["note"]
    assert "no reverse-PE: residual EPI distortion uncorrected" in result["inputs"]["bank_demo"]["note"]
    assert result["phase0"] == "SS3T FODs and ACT 10M banks built; atlas/parcellation QC unsigned."
    assert result["status"] == "banks-built-unsigned-priors"
    assert all(result[key]["approved_by"] is None for key in ("t1_qc", "atlas_prior_qc", "parcellation_qc"))


def test_pair_without_complete_sidecars_uses_explicit_case_metadata(tmp_path: Path) -> None:
    make_case(tmp_path, rpe_mode="pair", header_metadata=False)

    result = run_dry_case(tmp_path)

    assert result.returncode == 0, result.stderr
    preproc = next(line for line in result.stdout.splitlines() if line.startswith("DRY dwifslpreproc "))
    assert " -rpe_pair " in preproc and " -se_epi " in preproc
    assert " -pe_dir j- -readout_time 0.049 " in preproc
    assert " -rpe_header " not in preproc


def test_injected_failure_marks_stage_failed_and_exits_nonzero(tmp_path: Path) -> None:
    make_case(tmp_path, rpe_mode="none")

    result = run_dry_case(tmp_path, fail_at="t1reg")

    assert result.returncode != 0
    status = status_for(tmp_path)
    assert status["schema"] == "tractlab.status/1"
    assert status["state"] == "failed"
    assert status["stage"] == "t1reg"
    assert status["stages"][0]["state"] == "done"
    assert status["stages"][1]["state"] == "failed"
    assert status["stages"][2]["state"] == "pending"
    assert status["error"] and "Injected dry-run failure at stage t1reg" in status["error"]


def test_runner_accepts_the_manifest_skeleton_written_by_ingest(tmp_path: Path) -> None:
    from tractlab import ingest

    case_id = make_case(tmp_path, rpe_mode="none")
    ingest.write_manifest_skeleton(tmp_path, case_id)

    result = run_dry_case(tmp_path)

    assert result.returncode == 0, result.stderr


def test_pair_without_complete_sidecars_strips_partial_pe_headers(tmp_path: Path) -> None:
    make_case(tmp_path, rpe_mode="pair", header_metadata=False)

    result = run_dry_case(tmp_path)

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    raw = next(line for line in lines if line.startswith("DRY mrconvert ") and "dwi_raw.mif" in line)
    rpe = next(line for line in lines if line.startswith("DRY mrconvert ") and "/rpe.mif" in line)
    for line in (raw, rpe):
        assert "-clear_property PhaseEncodingDirection" in line
        assert "-clear_property TotalReadoutTime" in line
    preproc = shlex.split(next(line for line in lines if line.startswith("DRY dwifslpreproc ")).removeprefix("DRY "))
    assert sum(arg.startswith("-rpe_") for arg in preproc) == 1


def test_se_pair_is_capped_to_three_b0_per_polarity(tmp_path: Path) -> None:
    make_case(tmp_path, rpe_mode="pair")

    result = run_dry_case(tmp_path)

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    caps = [line for line in lines if line.startswith("DRY mrconvert ") and " -coord 3 0:2 " in line]
    assert any("dwi_b0_pe.mif" in line for line in caps)
    assert any("/rpe_acq.mif" in line or "/rpe.mif" in line for line in caps)
    mrcat = next(line for line in lines if line.startswith("DRY mrcat "))
    assert "dwi_b0_cap.mif" in mrcat and "rpe_b0_cap.mif" in mrcat


def test_regrid_to_acquired_voxel_runs_before_denoise(tmp_path: Path) -> None:
    make_case(tmp_path, rpe_mode="pair", regrid_voxel_mm=[1.846, 1.846, 2.0])

    result = run_dry_case(tmp_path)

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    regrids = [line for line in lines if line.startswith("DRY mrgrid ") and " -interp sinc " in line]
    assert len(regrids) == 2
    for line in regrids:
        args = shlex.split(line.removeprefix("DRY "))
        assert args[args.index("-voxel") + 1] == "1.846,1.846,2"
    assert any("dwi_acq.mif" in line for line in regrids)
    assert any("rpe_acq.mif" in line for line in regrids)
    denoise = next(line for line in lines if line.startswith("DRY dwidenoise "))
    assert "dwi_acq.mif" in denoise
    assert lines.index(next(l for l in regrids if "dwi_acq.mif" in l)) < lines.index(denoise)


def test_no_regrid_when_not_configured(tmp_path: Path) -> None:
    make_case(tmp_path, rpe_mode="none")

    result = run_dry_case(tmp_path)

    assert result.returncode == 0, result.stderr
    assert not any(" -interp sinc " in line for line in result.stdout.splitlines())
    denoise = next(line for line in result.stdout.splitlines() if line.startswith("DRY dwidenoise "))
    assert "dwi_raw.mif" in denoise


def test_runner_never_leaves_scratch_in_caller_cwd(tmp_path: Path) -> None:
    make_case(tmp_path, rpe_mode="none")

    result = run_dry_case(tmp_path)

    assert result.returncode == 0, result.stderr
    assert f"DRY cd {tmp_path}/work" in result.stdout
    gradcheck = next(line for line in result.stdout.splitlines() if line.startswith("DRY dwigradcheck "))
    assert " -mask " in gradcheck and "mask_up.mif" in gradcheck


def test_case_root_with_space_runs_through_space_free_alias(tmp_path: Path) -> None:
    root = tmp_path / "Application Support" / "case-abc"
    root.mkdir(parents=True)
    links = tmp_path / "links"
    links.mkdir()
    make_case(root, rpe_mode="pair", regrid_voxel_mm=[2.0, 2.0, 2.0])

    result = run_dry_case(root, tmpdir=links)

    assert result.returncode == 0, result.stderr
    alias = links / "tractlab-root-case-abc"
    assert alias.is_symlink() and alias.resolve() == root.resolve()
    tool_lines = [line for line in result.stdout.splitlines() if line.startswith("DRY ") and "STAGE" not in line]
    assert tool_lines
    # The fake toolchain HOME lives under the case root in these tests; only case data must be aliased.
    real_data = re.compile(r"Application\\ Support/case-abc/(work|nifti|raw|tracts)\b")
    assert not [line for line in tool_lines if real_data.search(line)]
    for tool in ("dwibiascorrect ants", "dwi2response dhollander", "mrconvert"):
        assert any(line.startswith(f"DRY {tool} {alias}/") for line in tool_lines), tool
    assert status_for(root)["state"] == "done"


GRADCHECK_IDENTITY = """dwigradcheck: Testing gradient table alterations (0 of 48)... [====]
Mean length     Axis flipped    Axis permutations    Axis basis
67.17         none                (0, 1, 2)           scanner
66.71         none                (0, 1, 2)           image
45.28            1                (0, 1, 2)           image
44.71            1                (0, 1, 2)           scanner
"""

GRADCHECK_FLIPPED = """Mean length     Axis flipped    Axis permutations    Axis basis
61.02            0                (0, 1, 2)           scanner
40.10         none                (0, 1, 2)           scanner
"""


def _gradcheck(tmp_path: Path, text: str, env_extra: dict | None = None) -> subprocess.CompletedProcess[str]:
    make_case(tmp_path, rpe_mode="none")
    report = tmp_path / "gradcheck.txt"
    report.write_text(text)
    env = os.environ.copy()
    env.pop("TRACTLAB_ALLOW_GRAD_FLIP", None)
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, str(RUNNER.parent / "runner_state.py"), "gradcheck", str(tmp_path), str(report)],
        capture_output=True, text=True, check=False, env=env,
    )


def test_gradcheck_identity_passes_and_writes_top_rows(tmp_path: Path) -> None:
    result = _gradcheck(tmp_path, GRADCHECK_IDENTITY)

    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "work/ss3t/gradcheck.json").read_text())
    assert report["verdict"] == "pass"
    assert report["top"][0] == {"mean_length": 67.17, "flip": "none", "permutation": [0, 1, 2], "basis": "scanner"}
    assert len(report["top"]) == 3


def test_gradcheck_flip_fails_closed_and_names_override(tmp_path: Path) -> None:
    result = _gradcheck(tmp_path, GRADCHECK_FLIPPED)

    assert result.returncode != 0
    assert "flip 0" in result.stderr and "TRACTLAB_ALLOW_GRAD_FLIP=1" in result.stderr
    assert json.loads((tmp_path / "work/ss3t/gradcheck.json").read_text())["verdict"] == "fail"


def test_gradcheck_flip_override_warns_in_case(tmp_path: Path) -> None:
    result = _gradcheck(tmp_path, GRADCHECK_FLIPPED, {"TRACTLAB_ALLOW_GRAD_FLIP": "1"})

    assert result.returncode == 0, result.stderr
    case = json.loads((tmp_path / "case.json").read_text())
    assert any("dwigradcheck" in warning for warning in case["warnings"])
    assert json.loads((tmp_path / "work/ss3t/gradcheck.json").read_text())["verdict"] == "override"


def test_gradcheck_unparseable_report_fails(tmp_path: Path) -> None:
    result = _gradcheck(tmp_path, "dwigradcheck: crashed\n")

    assert result.returncode != 0


def test_clinical_recon_uses_recon_all_clinical_and_synthsr(tmp_path: Path) -> None:
    case_id = make_case(tmp_path, rpe_mode="pair")
    result = run_dry_case(tmp_path, fs_recon="clinical")
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert not any(line.startswith("DRY recon-all ") for line in lines)
    recon = shlex.split(next(line for line in lines if line.startswith("DRY recon-all-clinical.sh ")).removeprefix("DRY "))
    fs_sub = tmp_path / "work/freesurfer" / case_id
    assert recon[recon.index("-subjid") + 1] == case_id
    assert recon[recon.index("-sdir") + 1] == str(tmp_path / "work/freesurfer")
    assert any(line == f"DRY ln -sfn norm.mgz {fs_sub}/mri/orig.mgz" for line in lines)
    t1_convert = next(line for line in lines if "anat/t1_fs.mif" in line and "mrconvert" in line)
    assert f"{fs_sub}/mri/synthSR.mgz" in t1_convert
    # bbregister init uses FreeSurfer's native mri_coreg; FreeSurfer 8.2 arm64 still ships an x86_64 flirt.fsl.
    bbregister = shlex.split(next(line for line in lines if line.startswith("DRY bbregister ")).removeprefix("DRY "))
    assert "--init-coreg" in bbregister and "--init-fsl" not in bbregister
    # FreeSurfer volumes are LIA on disk; the DWI-space parcellation must take the b0's strides too.
    aparc = shlex.split(next(line for line in lines if line.startswith("DRY ") and "aparc_dwi.nii.gz" in line and "mrtransform" in line).removeprefix("DRY "))
    assert aparc[aparc.index("-strides") + 1] == aparc[aparc.index("-template") + 1]
    assert status_for(tmp_path)["state"] == "done"


def test_unknown_fs_recon_mode_is_refused(tmp_path: Path) -> None:
    make_case(tmp_path, rpe_mode="pair")
    result = run_dry_case(tmp_path, fs_recon="fast")
    assert result.returncode == 2
    assert "TRACTLAB_FS_RECON must be standard or clinical" in result.stderr
