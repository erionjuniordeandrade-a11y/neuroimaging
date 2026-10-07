"""E0 — provenance backfill: real hashes, honest absence, dry-run default."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from backfill_bank_provenance import compute_backfill, main  # noqa: E402

BANK_BYTES = b"fake tck bytes for hashing"
FOD_BYTES = b"fake fod mif bytes"


def _make_case(tmp_path: Path) -> Path:
    case = tmp_path / "case"
    (case / "tracts/bank").mkdir(parents=True)
    (case / "nifti").mkdir()
    (case / "tracts/bank/bank.tck").write_bytes(BANK_BYTES)
    (case / "nifti/fod.mif").write_bytes(FOD_BYTES)
    manifest = {
        "case_id": "test",
        "inputs": {
            "fod": {"path": "nifti/fod.mif"},
            "bank_test": {"path": "tracts/bank/bank.tck", "label": "Test",
                          "engine": "ACT iFOD2 whole-brain 10M | strict multi-ROI"},
        },
    }
    (case / "manifest.json").write_text(json.dumps(manifest))
    return case


def test_backfill_hashes_and_honest_absence(tmp_path):
    case = _make_case(tmp_path)
    out = compute_backfill(str(case), recipe_commit="4885ff3")
    p = out["bank_test"]
    assert p["bank_sha256"] == hashlib.sha256(BANK_BYTES).hexdigest()
    assert p["fod_sha256"] == hashlib.sha256(FOD_BYTES).hexdigest()
    assert p["algorithm"] == "iFOD2" and p["act"] is True
    assert p["cutoff"] == 0.06
    assert p["sources"]["cutoff"] == "recipe-script@4885ff3"
    assert p["sources"]["bank_sha256"] == "recorded"
    assert p["sources"]["step_mm"].startswith("absent:")
    assert p["sources"]["mrtrix_version"].startswith("absent:")


def test_backfill_output_parses_with_task1_schema(tmp_path):
    from tractlab.bank import provenance_from_manifest

    case = _make_case(tmp_path)
    out = compute_backfill(str(case), recipe_commit="4885ff3")
    parsed = provenance_from_manifest({"provenance": out["bank_test"]})
    assert parsed is not None and parsed.cutoff == 0.06


def test_dry_run_never_writes(tmp_path):
    case = _make_case(tmp_path)
    before = (case / "manifest.json").read_bytes()
    main(["--case-root", str(case)])  # no --apply
    assert (case / "manifest.json").read_bytes() == before


def test_unknown_engine_gets_absent_params_never_guessed(tmp_path):
    case = _make_case(tmp_path)
    m = json.loads((case / "manifest.json").read_text())
    m["inputs"]["bank_test"]["engine"] = "DSI Studio deterministic"
    (case / "manifest.json").write_text(json.dumps(m))
    out = compute_backfill(str(case), recipe_commit="4885ff3")
    p = out["bank_test"]
    assert p["cutoff"] is None and p["algorithm"] is None and p["act"] is None
    assert p["sources"]["cutoff"].startswith("absent:engine")
    assert p["bank_sha256"] is not None  # hashes are still recorded facts


def test_existing_provenance_kept_unless_refresh(tmp_path, capsys):
    case = _make_case(tmp_path)
    m = json.loads((case / "manifest.json").read_text())
    m["inputs"]["bank_test"]["provenance"] = {
        "bank_sha256": "ee" * 32, "sources": {"bank_sha256": "recorded"}}
    (case / "manifest.json").write_text(json.dumps(m))
    out = compute_backfill(str(case), recipe_commit="4885ff3")
    assert out == {}
    assert "KEPT bank_test" in capsys.readouterr().err
    out = compute_backfill(str(case), recipe_commit="4885ff3", refresh=True)
    assert out["bank_test"]["bank_sha256"] == hashlib.sha256(BANK_BYTES).hexdigest()


def test_malformed_entry_reported_not_silent(tmp_path, capsys):
    case = _make_case(tmp_path)
    m = json.loads((case / "manifest.json").read_text())
    m["inputs"]["bank_broken"] = {"path": 123}
    (case / "manifest.json").write_text(json.dumps(m))
    out = compute_backfill(str(case), recipe_commit="4885ff3")
    assert "bank_broken" not in out and "bank_test" in out
    assert "MALFORMED bank_broken" in capsys.readouterr().err


def test_backfill_refuses_bank_path_escaping_root(tmp_path):
    case = _make_case(tmp_path)
    outside = tmp_path / "outside.tck"
    outside.write_bytes(b"outside bank")
    m = json.loads((case / "manifest.json").read_text())
    m["inputs"]["bank_test"]["path"] = "../outside.tck"
    (case / "manifest.json").write_text(json.dumps(m))

    with pytest.raises(ValueError, match="escapes case_root"):
        compute_backfill(str(case), recipe_commit="4885ff3")


def test_backfill_absolute_fod_path_is_honest_absence(tmp_path):
    case = _make_case(tmp_path)
    outside = tmp_path / "outside.fod"
    outside.write_bytes(b"outside fod")
    m = json.loads((case / "manifest.json").read_text())
    m["inputs"]["fod"]["path"] = str(outside)
    (case / "manifest.json").write_text(json.dumps(m))

    out = compute_backfill(str(case), recipe_commit="4885ff3")
    p = out["bank_test"]
    assert p["fod_sha256"] is None
    assert p["sources"]["fod_sha256"].startswith("absent:")


def test_apply_is_atomic_no_tmp_left_and_valid_json(tmp_path):
    case = _make_case(tmp_path)
    main(["--case-root", str(case), "--apply"])
    m = json.loads((case / "manifest.json").read_text())  # parses = not truncated
    assert m["inputs"]["bank_test"]["provenance"]["cutoff"] == 0.06
    leftovers = [p.name for p in case.iterdir() if p.name.startswith(".manifest-")]
    assert leftovers == []


def test_missing_bank_file_raises(tmp_path):
    case = _make_case(tmp_path)
    (case / "tracts/bank/bank.tck").unlink()
    import pytest

    with pytest.raises(FileNotFoundError, match="bank_test"):
        compute_backfill(str(case), recipe_commit="4885ff3")


def test_no_fod_declared_is_honest_absence_not_a_crash(tmp_path):
    case = _make_case(tmp_path)
    m = json.loads((case / "manifest.json").read_text())
    del m["inputs"]["fod"]
    (case / "manifest.json").write_text(json.dumps(m))
    out = compute_backfill(str(case), recipe_commit="4885ff3")
    p = out["bank_test"]
    assert p["fod_sha256"] is None
    assert p["sources"]["fod_sha256"].startswith("absent:")


def test_apply_writes_provenance_blocks(tmp_path):
    case = _make_case(tmp_path)
    main(["--case-root", str(case), "--apply"])
    m = json.loads((case / "manifest.json").read_text())
    p = m["inputs"]["bank_test"]["provenance"]
    assert p["bank_sha256"] == hashlib.sha256(BANK_BYTES).hexdigest()
    # non-bank entries untouched
    assert "provenance" not in m["inputs"]["fod"]


# ── ADR-0007: routed through derivation.active_inputs ───────────────────────
D1_BANK_BYTES = b"first lineage bank bytes"
D2_BANK_BYTES = b"second lineage bank bytes"
D1_FOD_BYTES = b"first lineage fod bytes"
D2_FOD_BYTES = b"second lineage fod bytes"


def _content_meta(path: Path, rel: str) -> dict:
    """A non-first derivation's declared path field needs real
    sha256_by_field/bytes_by_field content metadata — derivation.validate()
    requires the fields be well-formed, and validate_active_artifact_metadata
    additionally verifies them against the real on-disk bytes."""
    data = path.read_bytes()
    return {
        "path": rel,
        "sha256_by_field": {"path": hashlib.sha256(data).hexdigest()},
        "bytes_by_field": {"path": len(data)},
    }


def _make_two_derivation_case(tmp_path: Path, *, drift_d2_bank: bool = False) -> Path:
    """A migrated, two-derivation case where d2 (active) and d1 (first) each
    declare their own on-disk bank + FOD, so a top-level ``inputs`` read
    would silently pick the wrong (first) lineage instead of refusing.
    ``drift_d2_bank=True`` declares a bank_test sha256 that does not match
    the real on-disk bytes, for validate_active_artifact_metadata drift
    refusal tests."""
    case = tmp_path / "case2"
    (case / "d1/tracts").mkdir(parents=True)
    (case / "d2/tracts").mkdir(parents=True)
    (case / "d1/nifti").mkdir()
    (case / "d2/nifti").mkdir()
    (case / "d1/tracts/bank.tck").write_bytes(D1_BANK_BYTES)
    (case / "d2/tracts/bank.tck").write_bytes(D2_BANK_BYTES)
    (case / "d1/nifti/fod.mif").write_bytes(D1_FOD_BYTES)
    (case / "d2/nifti/fod.mif").write_bytes(D2_FOD_BYTES)
    engine = "ACT iFOD2 whole-brain 10M | strict multi-ROI"

    d2_bank_meta = _content_meta(case / "d2/tracts/bank.tck", "d2/tracts/bank.tck")
    d2_bank_meta["label"] = "Test"
    d2_bank_meta["engine"] = engine
    if drift_d2_bank:
        # Syntactically valid but wrong — the file on disk was NOT rehashed.
        d2_bank_meta["sha256_by_field"] = {"path": "ff" * 32}

    d2_fod_meta = _content_meta(case / "d2/nifti/fod.mif", "d2/nifti/fod.mif")

    manifest = {
        "case_id": "test2",
        "case_root": str(case),
        "active_derivation": "d2",
        "derivations": {
            "d1": {
                "kind": "uncorrected",
                "inputs": {
                    "fod": {"path": "d1/nifti/fod.mif"},
                    "bank_test": {"path": "d1/tracts/bank.tck", "label": "Test", "engine": engine},
                },
            },
            "d2": {
                "kind": "rpe_pair",
                "inputs": {
                    "fod": d2_fod_meta,
                    "bank_test": d2_bank_meta,
                },
            },
        },
    }
    (case / "manifest.json").write_text(json.dumps(manifest))
    return case


def test_compute_backfill_uses_active_lineage_bank_and_fod_not_first(tmp_path):
    case = _make_two_derivation_case(tmp_path)
    out = compute_backfill(str(case), recipe_commit="4885ff3")
    p = out["bank_test"]
    assert p["bank_sha256"] == hashlib.sha256(D2_BANK_BYTES).hexdigest()
    assert p["fod_sha256"] == hashlib.sha256(D2_FOD_BYTES).hexdigest()
    assert p["bank_sha256"] != hashlib.sha256(D1_BANK_BYTES).hexdigest()


def test_apply_writes_into_active_derivation_never_a_top_level_inputs_block(tmp_path):
    case = _make_two_derivation_case(tmp_path)
    main(["--case-root", str(case), "--apply"])
    m = json.loads((case / "manifest.json").read_text())
    assert "inputs" not in m  # a migrated manifest never grows a legacy top-level block
    prov = m["derivations"]["d2"]["inputs"]["bank_test"]["provenance"]
    assert prov["bank_sha256"] == hashlib.sha256(D2_BANK_BYTES).hexdigest()
    assert "provenance" not in m["derivations"]["d1"]["inputs"]["bank_test"]


def test_apply_refuses_invalid_or_hash_drifted_active_lineage(tmp_path, capsys):
    """--apply must run derivation.validate() (format) and
    validate_active_artifact_metadata() (content, for a non-first active
    lineage) before writing anything — never partially trust an active
    lineage whose declared metadata is malformed or has drifted from the
    real on-disk bytes."""
    # (a) active (non-first) lineage entry carries NO content metadata at
    # all — derivation.validate() must refuse before any hashing happens.
    case_invalid = tmp_path / "invalid"
    case_invalid.mkdir()
    (case_invalid / "bank.tck").write_bytes(b"some bank bytes")
    manifest_invalid = {
        "case_id": "invalid",
        "case_root": str(case_invalid),
        "active_derivation": "d2",
        "derivations": {
            "d1": {"kind": "uncorrected", "inputs": {"bank_test": {"path": "bank.tck"}}},
            "d2": {"kind": "rpe_pair", "inputs": {"bank_test": {"path": "bank.tck"}}},
        },
    }
    (case_invalid / "manifest.json").write_text(json.dumps(manifest_invalid))

    rc = main(["--case-root", str(case_invalid), "--apply"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "REFUSE: manifest failed validation" in captured.err
    assert not (case_invalid / "manifest.json").read_text().find('"provenance"') >= 0

    # (b) well-formed metadata, but the declared sha256 does not match the
    # real on-disk bytes — validate_active_artifact_metadata must catch the
    # drift that a format-only check (derivation.validate) cannot see.
    case_drift = _make_two_derivation_case(tmp_path, drift_d2_bank=True)
    before = (case_drift / "manifest.json").read_bytes()

    rc = main(["--case-root", str(case_drift), "--apply"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "REFUSE: manifest failed validation" in captured.err
    assert "sha256 mismatch" in captured.err
    assert (case_drift / "manifest.json").read_bytes() == before  # never written on refusal
