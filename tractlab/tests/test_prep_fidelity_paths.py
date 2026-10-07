"""Path-containment refusals for the offline fidelity-prep script."""

import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prep_fidelity import main  # noqa: E402

MRTRIX_AVAILABLE = all(
    shutil.which(tool) or Path(f"~/mrtrix3/bin/{tool}").expanduser().is_file()
    for tool in ("dirgen", "sh2amp", "mrmath")
)


def _write_manifest(case: Path, fod_path: str) -> None:
    case.mkdir()
    (case / "manifest.json").write_text(
        json.dumps({"inputs": {"fod": {"path": fod_path}}})
    )


def _d2_meta(path: str) -> dict:
    return {
        "path": path,
        "sha256_by_field": {"path": "aa" * 32},
        "bytes_by_field": {"path": 1},
    }


def test_prep_refuses_absolute_fod_path(tmp_path, capsys):
    case = tmp_path / "case"
    _write_manifest(case, "/etc/hosts")

    rc = main(["--case-root", str(case)])

    captured = capsys.readouterr()
    assert rc == 1
    assert "REFUSE" in captured.err


def test_prep_refuses_traversal_fod_path(tmp_path, capsys):
    case = tmp_path / "case"
    outside = tmp_path / "outside.fod"
    outside.write_bytes(b"outside fod")
    _write_manifest(case, "../outside.fod")

    rc = main(["--case-root", str(case)])

    captured = capsys.readouterr()
    assert rc == 1
    assert "REFUSE" in captured.err


# ── ADR-0007: routed through derivation.active_inputs ───────────────────────


def test_prep_selects_active_second_derivation_fod_not_first_lineage(tmp_path, capsys):
    """d1 (first lineage) has a real FOD on disk; d2 (active) does not. A
    top-level/first-lineage-only read would silently succeed on d1's file —
    the script must refuse on d2's declared (missing) path instead."""
    case = tmp_path / "case"
    case.mkdir()
    (case / "d1").mkdir()
    (case / "d1" / "fod.mif").write_bytes(b"first lineage fod")
    manifest = {
        "active_derivation": "d2",
        "derivations": {
            "d1": {"kind": "uncorrected", "inputs": {"fod": {"path": "d1/fod.mif"}}},
            "d2": {"kind": "rpe_pair", "inputs": {"fod": _d2_meta("d2/fod.mif")}},
        },
    }
    (case / "manifest.json").write_text(json.dumps(manifest))

    rc = main(["--case-root", str(case)])

    captured = capsys.readouterr()
    assert rc == 1
    assert "REFUSE: FOD missing on disk (d2/fod.mif)" in captured.err


def test_prep_refuses_unresolvable_active_derivation(tmp_path, capsys):
    case = tmp_path / "case"
    case.mkdir()
    manifest = {
        "active_derivation": "missing-id",
        "derivations": {
            "d1": {"kind": "uncorrected", "inputs": {"fod": {"path": "fod.mif"}}},
        },
    }
    (case / "manifest.json").write_text(json.dumps(manifest))

    rc = main(["--case-root", str(case)])

    captured = capsys.readouterr()
    assert rc == 1
    assert "REFUSE" in captured.err


def test_prep_refuses_manifest_that_fails_derivation_validate(tmp_path, capsys):
    """d2 (active, non-first) declares a fod path but carries none of the
    required sha256_by_field/bytes_by_field content metadata — validate()
    must catch this before any fidelity work is attempted."""
    case = tmp_path / "case"
    case.mkdir()
    (case / "d1").mkdir()
    (case / "d1" / "fod.mif").write_bytes(b"first lineage fod")
    manifest = {
        "active_derivation": "d2",
        "derivations": {
            "d1": {"kind": "uncorrected", "inputs": {"fod": {"path": "d1/fod.mif"}}},
            "d2": {"kind": "rpe_pair", "inputs": {"fod": {"path": "d2/fod.mif"}}},  # no content metadata
        },
    }
    (case / "manifest.json").write_text(json.dumps(manifest))

    rc = main(["--case-root", str(case)])

    captured = capsys.readouterr()
    assert rc == 1
    assert "REFUSE: manifest failed derivation.validate()" in captured.err


@pytest.mark.skipif(not MRTRIX_AVAILABLE, reason="MRtrix3 (dirgen/sh2amp/mrmath) not on PATH/~/mrtrix3/bin")
def test_apply_second_derivation_writes_only_scoped_fidelity(tmp_path):
    """With an active SECOND lineage, --apply must write fod_peak.nii.gz
    under derivations/d2/fidelity/ — never the flat case_root/fidelity/ used
    by the first/legacy lineage (ADR-0007 lineage isolation)."""
    nib = pytest.importorskip("nibabel")
    import numpy as np

    case = tmp_path / "case"
    case.mkdir()
    (case / "d1").mkdir()
    (case / "d2").mkdir()
    # Trivial lmax=0 FOD (single SH coefficient) — enough for a real
    # dirgen/sh2amp/mrmath run without needing a genuine acquisition.
    fod_img = nib.Nifti1Image(np.ones((2, 2, 2, 1), dtype=np.float32), affine=np.eye(4))
    nib.save(fod_img, str(case / "d1" / "fod.nii.gz"))
    nib.save(fod_img, str(case / "d2" / "fod.nii.gz"))
    manifest = {
        "active_derivation": "d2",
        "derivations": {
            "d1": {"kind": "uncorrected", "inputs": {"fod": {"path": "d1/fod.nii.gz"}}},
            "d2": {"kind": "rpe_pair", "inputs": {"fod": _d2_meta("d2/fod.nii.gz")}},
        },
    }
    (case / "manifest.json").write_text(json.dumps(manifest))

    rc = main(["--case-root", str(case), "--apply"])

    assert rc == 0
    scoped_peak = case / "derivations" / "d2" / "fidelity" / "fod_peak.nii.gz"
    assert scoped_peak.is_file(), f"expected scoped output at {scoped_peak}"
    assert not (case / "fidelity").exists(), "must never write the flat/first-lineage fidelity dir"
