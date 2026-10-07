"""E0 — bank provenance: parse, refuse-friendly access, honest absence."""
import pytest

from tractlab.bank import BankProvenance, provenance_from_manifest


def test_parses_full_provenance():
    entry = {"provenance": {
        "bank_sha256": "ab" * 32, "fod_sha256": "cd" * 32,
        "cutoff": 0.05, "step_mm": 0.6, "downsample": 3,
        "algorithm": "iFOD2", "act": True, "mrtrix_version": "3.0.4",
        "sources": {"cutoff": "recipe-script@4885ff3"},
    }}
    p = provenance_from_manifest(entry)
    assert p.cutoff == 0.05 and p.fod_sha256 == "cd" * 32
    assert p.sources["cutoff"] == "recipe-script@4885ff3"


def test_absent_provenance_returns_none():
    assert provenance_from_manifest({}) is None


def test_absent_field_is_none_with_reason():
    entry = {"provenance": {"bank_sha256": "ab" * 32,
                            "sources": {"cutoff": "absent:not recorded at build"}}}
    p = provenance_from_manifest(entry)
    assert p.cutoff is None
    assert p.sources["cutoff"].startswith("absent:")


def test_unknown_key_fails_loud():
    entry = {"provenance": {"bank_sha256": "ab" * 32, "seed_count": 5,
                            "sources": {"bank_sha256": "recorded"}}}
    with pytest.raises(ValueError, match="seed_count"):
        provenance_from_manifest(entry)


# --- Sol T1 review conditions (REVIEW-sol-t1.log) ---------------------------

_GOOD_SOURCES = {"bank_sha256": "recorded"}


def test_explicit_null_is_rejected_not_absent():
    with pytest.raises(ValueError, match="null"):
        provenance_from_manifest({"provenance": None})


def test_sources_is_required():
    with pytest.raises(ValueError, match="sources"):
        provenance_from_manifest({"provenance": {"bank_sha256": "ab" * 32}})


@pytest.mark.parametrize("field,bad", [
    ("bank_sha256", 123),
    ("cutoff", "0.05"),
    ("step_mm", True),
    ("downsample", 2.5),
    ("downsample", True),
    ("act", "yes"),
    ("mrtrix_version", 3.0),
])
def test_wrong_runtime_types_fail_loud(field, bad):
    entry = {"provenance": {field: bad, "sources": dict(_GOOD_SOURCES)}}
    with pytest.raises(ValueError, match=field):
        provenance_from_manifest(entry)


def test_source_vocabulary_enforced():
    entry = {"provenance": {"bank_sha256": "ab" * 32,
                            "sources": {"bank_sha256": "trust me"}}}
    with pytest.raises(ValueError, match="bank_sha256"):
        provenance_from_manifest(entry)


def test_source_keys_must_be_real_fields():
    entry = {"provenance": {"bank_sha256": "ab" * 32,
                            "sources": {"seed_count": "recorded"}}}
    with pytest.raises(ValueError, match="seed_count"):
        provenance_from_manifest(entry)


@pytest.mark.parametrize("token", ["recipe-script@", "absent:"])
def test_empty_source_suffix_rejected(token):
    entry = {"provenance": {"bank_sha256": "ab" * 32,
                            "sources": {"bank_sha256": token}}}
    with pytest.raises(ValueError, match="non-empty suffix"):
        provenance_from_manifest(entry)


def test_absent_field_cannot_claim_recorded():
    entry = {"provenance": {"bank_sha256": "ab" * 32,
                            "sources": {"bank_sha256": "recorded",
                                        "cutoff": "recorded"}}}
    with pytest.raises(ValueError, match="cutoff"):
        provenance_from_manifest(entry)


def test_populated_field_cannot_claim_absent():
    entry = {"provenance": {"bank_sha256": "ab" * 32,
                            "sources": {"bank_sha256": "absent:but it is here"}}}
    with pytest.raises(ValueError, match="populated"):
        provenance_from_manifest(entry)


def test_sources_not_aliased_and_read_only():
    src = {"bank_sha256": "recorded"}
    p = provenance_from_manifest(
        {"provenance": {"bank_sha256": "ab" * 32, "sources": src}})
    src["bank_sha256"] = "absent:mutated after parse"
    assert p.sources["bank_sha256"] == "recorded"  # no aliasing
    with pytest.raises(TypeError):
        p.sources["bank_sha256"] = "tampered"  # read-only view


def test_one_malformed_bank_does_not_take_down_the_rest(tmp_path, capsys):
    from tractlab.bank import discover_banks

    root = tmp_path / "case"
    (root / "tracts/bank").mkdir(parents=True)
    for name in ("good.tck", "bad.tck"):
        (root / "tracts/bank" / name).write_bytes(b"x")
    inputs = {
        "bank_good": {"path": "tracts/bank/good.tck",
                      "provenance": {"bank_sha256": "ab" * 32,
                                     "sources": {"bank_sha256": "recorded"}}},
        "bank_bad": {"path": "tracts/bank/bad.tck",
                     "provenance": {"bank_sha256": 123,
                                    "sources": {"bank_sha256": "recorded"}}},
    }
    banks = discover_banks(str(root), inputs)
    assert "bank_good" in banks and "bank_bad" not in banks
    assert "REFUSED bank_bad" in capsys.readouterr().err
