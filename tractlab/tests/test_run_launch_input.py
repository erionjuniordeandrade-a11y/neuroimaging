"""run.py's launch-input check reads manifest inputs through the active
derivation lineage (ADR-0007 / derivation.active_inputs), not a flat
top-level ``inputs`` read — a second (active) derivation's own paths must be
what gets validated, never a first-derivation sibling's."""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from run import _missing_launch_input  # noqa: E402


def _viewer_dir(tmp_path: Path) -> Path:
    viewer = tmp_path / "viewer"
    viewer.mkdir()
    (viewer / "index.html").write_text("<html></html>")
    return viewer


def _write_flat_case(tmp_path: Path) -> Path:
    case = tmp_path / "case"
    case.mkdir()
    for name in ("b0.nii.gz", "mask.nii.gz", "fod.mif"):
        (case / name).write_bytes(b"x")
    manifest = {
        "case_id": "demo",
        "case_root": str(case),
        "inputs": {
            "b0": {"path": "b0.nii.gz"},
            "mask": {"path": "mask.nii.gz"},
            "fod": {"path": "fod.mif"},
        },
    }
    (case / "manifest.json").write_text(json.dumps(manifest))
    return case


def _two_derivation_manifest(case: Path, *, second_b0_present: bool) -> dict:
    (case / "d1").mkdir()
    (case / "d2").mkdir()
    for name in ("b0.nii.gz", "mask.nii.gz", "fod.mif"):
        (case / "d1" / name).write_bytes(b"first-lineage")
        if name != "b0.nii.gz" or second_b0_present:
            (case / "d2" / name).write_bytes(b"second-lineage")
    return {
        "case_id": "demo",
        "case_root": str(case),
        "active_derivation": "d2",
        "derivations": {
            "d1": {
                "kind": "uncorrected",
                "inputs": {
                    "b0": {"path": "d1/b0.nii.gz"},
                    "mask": {"path": "d1/mask.nii.gz"},
                    "fod": {"path": "d1/fod.mif"},
                },
            },
            "d2": {
                "kind": "rpe_pair",
                "inputs": {
                    "b0": {"path": "d2/b0.nii.gz"},
                    "mask": {"path": "d2/mask.nii.gz"},
                    "fod": {"path": "d2/fod.mif"},
                },
            },
        },
    }


def test_legacy_flat_manifest_still_resolves(tmp_path):
    case = _write_flat_case(tmp_path)
    viewer = _viewer_dir(tmp_path)
    assert _missing_launch_input(str(case / "manifest.json"), str(viewer)) is None


def test_active_second_derivation_inputs_are_the_ones_checked(tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    manifest = _two_derivation_manifest(case, second_b0_present=True)
    (case / "manifest.json").write_text(json.dumps(manifest))
    viewer = _viewer_dir(tmp_path)
    # d2 (active) is fully on disk even though it is not the first lineage.
    assert _missing_launch_input(str(case / "manifest.json"), str(viewer)) is None


def test_missing_active_lineage_input_fails_closed_even_though_sibling_has_it(tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    manifest = _two_derivation_manifest(case, second_b0_present=False)
    (case / "manifest.json").write_text(json.dumps(manifest))
    viewer = _viewer_dir(tmp_path)
    # d1 (first lineage) has b0 on disk; the active lineage d2 does not — a
    # top-level-only ``inputs`` read would wrongly pass this case.
    result = _missing_launch_input(str(case / "manifest.json"), str(viewer))
    assert result == "required input unavailable: b0"


def test_unresolvable_active_derivation_is_invalid_inputs(tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    manifest = _two_derivation_manifest(case, second_b0_present=True)
    manifest["active_derivation"] = "missing-id"
    (case / "manifest.json").write_text(json.dumps(manifest))
    viewer = _viewer_dir(tmp_path)
    assert _missing_launch_input(str(case / "manifest.json"), str(viewer)) == (
        "manifest inputs are invalid"
    )
