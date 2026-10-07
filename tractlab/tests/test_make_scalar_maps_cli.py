"""Argument-handling + active-lineage-routing checks for
scripts/make_scalar_maps.sh --case-root.

Shell syntax, --help, refusal paths, and one full synthetic run (with the
scalar_maps sidecar pre-populated so it short-circuits to "reuse" and never
invokes the real MRtrix tensor pipeline) — never touches a real case, per
the task's dry-run/synthetic-only rule.
"""
import hashlib
import json
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_scalar_maps.sh"
SUBPROCESS_TIMEOUT = 60


def _run(*args, timeout=SUBPROCESS_TIMEOUT):
    return subprocess.run(
        ["bash", str(SCRIPT), *args], capture_output=True, text=True, timeout=timeout,
    )


def _write_manifest(case_dir: Path, manifest: dict) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "manifest.json").write_text(json.dumps(manifest))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_bash_syntax_is_valid():
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)], capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT,
    )
    assert result.returncode == 0, result.stderr


def test_help_flag_prints_usage_and_exits_zero():
    result = _run("--help")
    assert result.returncode == 0
    assert "--case-root" in result.stdout
    assert "--force" in result.stdout


def test_short_help_flag_also_works():
    result = _run("-h")
    assert result.returncode == 0
    assert "Usage:" in result.stdout


def test_unknown_argument_refuses():
    result = _run("--bogus")
    assert result.returncode == 2
    assert "unknown argument" in result.stderr


def test_case_root_without_manifest_refuses(tmp_path):
    empty = tmp_path / "no-manifest"
    empty.mkdir()
    result = _run("--case-root", str(empty))
    assert result.returncode == 2
    assert "missing manifest.json" in result.stderr


def test_nonexistent_case_root_dir_refuses(tmp_path):
    missing = tmp_path / "does-not-exist"
    result = _run("--case-root", str(missing))
    assert result.returncode == 2
    assert "no such case-root directory" in result.stderr


# ── ADR-0007: case_id/case_root validity, active-lineage routing, escape guard ──


def test_manifest_without_case_root_refuses(tmp_path):
    case = tmp_path / "case"
    _write_manifest(case, {"case_id": "demo"})  # no case_root at all
    result = _run("--case-root", str(case))
    assert result.returncode == 2
    assert "REFUSE" in result.stderr
    assert "case_root" in result.stderr
    assert "<unknown>" not in result.stdout


def test_invalid_case_id_refuses(tmp_path):
    case = tmp_path / "case"
    _write_manifest(case, {"case_id": "", "case_root": str(case)})  # empty case_id
    result = _run("--case-root", str(case))
    assert result.returncode == 2
    assert "REFUSE" in result.stderr
    assert "case_id" in result.stderr
    assert "<unknown>" not in result.stdout


def test_case_root_pointing_at_missing_data_dir_refuses(tmp_path):
    """--case-root resolves (through a symlink) to a manifest whose recorded
    case_root does not exist on disk: refuse before any MRtrix call. A
    recorded case_root that exists elsewhere is the house indirection and is
    honoured (see the build_connectome tests)."""
    real_case = tmp_path / "real_case"
    real_case.mkdir()
    _write_manifest(real_case, {"case_id": "missing-data", "case_root": str(tmp_path / "nope")})
    symlink_case = tmp_path / "symlink_case"
    symlink_case.symlink_to(real_case, target_is_directory=True)

    result = _run("--case-root", str(symlink_case))
    assert result.returncode == 2
    assert "REFUSE" in result.stderr
    assert "does not exist" in result.stderr


def test_active_second_derivation_uses_scoped_inputs_and_outputs(tmp_path):
    """With an active SECOND lineage, mask/DWI must come from that lineage's
    own inputs (never d1's), and outputs must land under
    derivations/d2/profiles/ — never the flat work/profiles/ used by the
    first/legacy lineage. Pre-populates the scalar_maps sidecar so the run
    short-circuits to "reuse" — this proves ROUTING, not the MRtrix tensor
    pipeline (unrelated product code)."""
    case = tmp_path / "case"
    case.mkdir()

    # d1 (first, inactive) — present but must never be read by this run.
    (case / "d1" / "nifti").mkdir(parents=True)
    (case / "d1" / "nifti" / "mask.nii.gz").write_bytes(b"d1 mask bytes")
    (case / "d1" / "work" / "ss3t").mkdir(parents=True)
    (case / "d1" / "work" / "ss3t" / "dwi_preproc.mif").write_bytes(b"d1 dwi bytes")

    # d2 (active, second lineage) — its own mask + DWI, in its own subtree.
    (case / "d2" / "nifti").mkdir(parents=True)
    d2_mask = case / "d2" / "nifti" / "mask.nii.gz"
    d2_mask.write_bytes(b"d2 mask bytes")
    (case / "derivations" / "d2" / "work" / "ss3t").mkdir(parents=True)
    d2_dwi = case / "derivations" / "d2" / "work" / "ss3t" / "dwi_preproc.mif"
    d2_dwi.write_bytes(b"d2 dwi bytes")

    manifest = {
        "case_id": "scoped-demo",
        "case_root": str(case),
        "active_derivation": "d2",
        "derivations": {
            "d1": {"kind": "uncorrected", "inputs": {"mask": {"path": "d1/nifti/mask.nii.gz"}}},
            "d2": {
                "kind": "rpe_pair",
                "inputs": {
                    "mask": {
                        "path": "d2/nifti/mask.nii.gz",
                        "sha256_by_field": {"path": _sha256_bytes(d2_mask.read_bytes())},
                        "bytes_by_field": {"path": d2_mask.stat().st_size},
                    },
                },
            },
        },
    }
    (case / "manifest.json").write_text(json.dumps(manifest))

    out_dir = case / "derivations" / "d2" / "profiles"
    out_dir.mkdir(parents=True)
    for name in ("mask_casemask.mif", "dt_casemask.mif", "fa_casemask.nii.gz", "md_casemask.nii.gz"):
        (out_dir / name).write_bytes(b"fake product bytes")
    sidecar = {
        "dwi_sha256": _sha256_bytes(d2_dwi.read_bytes()),
        "mask_sha256": _sha256_bytes(d2_mask.read_bytes()),
        "dwi_path": str(d2_dwi),
        "mask_path": str(d2_mask),
    }
    (out_dir / "casemask.sources.json").write_text(json.dumps(sidecar))

    result = _run("--case-root", str(case))

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "case_id: scoped-demo" in result.stdout
    assert "active_derivation: d2" in result.stdout
    assert str(d2_mask) in result.stdout  # active lineage's OWN mask, never d1's
    assert "d1/nifti/mask.nii.gz" not in result.stdout
    assert "SKIP products match sidecar" in result.stdout
    assert (out_dir / "mask_casemask.mif").is_file()
    assert not (case / "work" / "profiles").exists()  # never the flat/first-lineage location


def test_helper_out_dir_comes_from_shared_derivation_rule(tmp_path):
    """The helper must not carry its own copy of the profiles layout rule:
    its out_dir equals derivation.artifact_dir(manifest, "profiles")."""
    import importlib.util
    import sys as _sys

    helper_path = SCRIPT.parent / "_case_scalar_maps_inputs.py"
    src = helper_path.read_text()
    assert '"profiles"' not in src.replace('artifact_dir(manifest, "profiles")', ""), (
        "helper still spells its own profiles layout"
    )
    spec = importlib.util.spec_from_file_location("_csmi", helper_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from tractlab import derivation as dv

    case = tmp_path / "case"
    (case / "nifti").mkdir(parents=True)
    (case / "nifti" / "mask.nii.gz").write_bytes(b"mask")
    manifest = {"case_id": "demo", "case_root": str(case),
                "inputs": {"mask": {"path": "nifti/mask.nii.gz"}}}
    (case / "manifest.json").write_text(json.dumps(manifest))
    info = mod.resolve(str(case))
    assert info["out_dir"] == str(dv.artifact_dir(manifest, "profiles"))
