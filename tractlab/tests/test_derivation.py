import json

import nibabel as nib
import numpy as np
import pytest

from tractlab import atlas_prep
from tractlab import delta_qc
from tractlab import derivation as dv


def _d2_meta(path):
    return {
        "path": path,
        "sha256_by_field": {"path": "aa" * 32},
        "bytes_by_field": {"path": 1},
    }


def _legacy_manifest():
    return {"case_id": "demo", "inputs": {"b0": {"path": "nifti/b0.nii.gz"}},
            "atlas_prior_qc": {"auto": {}, "approved_by": "erion",
                               "date": "2026-08-16", "sheet_sha": "abc",
                               "sheet_path": "qc/x.png"}}


def test_migrate_stamps_kind_active_and_qc_scope():
    m = dv.migrate_manifest(_legacy_manifest(), dv.KIND_RPE_PAIR)
    assert m["active_derivation"] == dv.DEFAULT_ID
    assert m["derivations"][dv.DEFAULT_ID]["kind"] == dv.KIND_RPE_PAIR
    # existing signatures described this (only) lineage — stamped, not voided
    assert m["atlas_prior_qc"]["derivation"] == dv.DEFAULT_ID


def test_migrate_does_not_mutate_input():
    src = _legacy_manifest()
    dv.migrate_manifest(src, dv.KIND_UNCORRECTED)
    assert "derivations" not in src


def test_migrate_rejects_unknown_kind_and_double_migration():
    with pytest.raises(ValueError, match="kind"):
        dv.migrate_manifest(_legacy_manifest(), "topup")
    once = dv.migrate_manifest(_legacy_manifest(), dv.KIND_RPE_PAIR)
    with pytest.raises(ValueError, match="already"):
        dv.migrate_manifest(once, dv.KIND_RPE_PAIR)


def test_active_derivation_none_for_unmigrated():
    assert dv.active_derivation(_legacy_manifest()) is None


def test_validate_passes_migrated_single_lineage():
    m = dv.migrate_manifest(_legacy_manifest(), dv.KIND_RPE_PAIR)
    dv.validate(m)  # no raise


def test_validate_fails_on_cross_derivation_qc():
    m = dv.migrate_manifest(_legacy_manifest(), dv.KIND_RPE_PAIR)
    m["atlas_prior_qc"]["derivation"] = "d2"
    with pytest.raises(dv.DerivationError, match="atlas_prior_qc"):
        dv.validate(m)


def test_validate_accepts_two_derivations_with_scoped_inputs(tmp_path):
    m = dv.migrate_manifest(_legacy_manifest(), dv.KIND_RPE_PAIR)
    m["case_root"] = str(tmp_path)
    m["derivations"]["d1"]["inputs"] = m.pop("inputs")
    m["derivations"]["d2"] = {
        "kind": dv.KIND_UNCORRECTED,
        "inputs": {"b0": _d2_meta("d2/b0.nii.gz")},
    }
    dv.validate(m)


def test_validate_requires_inputs_for_every_derivation(tmp_path):
    m = dv.migrate_manifest(_legacy_manifest(), dv.KIND_RPE_PAIR)
    m["case_root"] = str(tmp_path)
    m["derivations"]["d2"] = {"kind": dv.KIND_UNCORRECTED}
    with pytest.raises(dv.DerivationError, match="d2.*inputs"):
        dv.validate(m)


def test_active_inputs_legacy_flat_manifest_resolves_to_d1():
    legacy = _legacy_manifest()
    assert dv.active_inputs(legacy) is legacy["inputs"]
    m = dv.migrate_manifest(legacy, dv.KIND_RPE_PAIR)
    assert dv.active_inputs(m) is m["inputs"]


def test_validate_refuses_shared_artifact_path(tmp_path):
    m = dv.migrate_manifest(_legacy_manifest(), dv.KIND_RPE_PAIR)
    m["case_root"] = str(tmp_path)
    m["derivations"]["d2"] = {
        "kind": dv.KIND_UNCORRECTED,
        "inputs": {"other": _d2_meta("nifti/b0.nii.gz")},
    }
    with pytest.raises(dv.DerivationError, match=r"d1.*d2|d2.*d1") as exc:
        dv.validate(m)
    assert "b0.nii.gz" in str(exc.value)


def test_validate_preserves_first_lineage_qc_while_d2_active(tmp_path):
    m = dv.migrate_manifest(_legacy_manifest(), dv.KIND_RPE_PAIR)
    m["case_root"] = str(tmp_path)
    m["derivations"]["d2"] = {
        "kind": dv.KIND_UNCORRECTED,
        "inputs": {"b0": _d2_meta("d2/b0.nii.gz")},
    }
    m["active_derivation"] = "d2"
    m["atlas_prior_qc"]["derivation"] = "d1"
    dv.validate(m)
    assert dv.qc_block_for(m, "atlas_prior_qc") is None


def test_add_derivation_refuses_duplicate_id(tmp_path):
    m = dv.migrate_manifest(_legacy_manifest(), dv.KIND_RPE_PAIR)
    m["case_root"] = str(tmp_path)
    with pytest.raises(ValueError, match="already exists"):
        dv.add_derivation(
            m, "d1", dv.KIND_UNCORRECTED, {"b0": {"path": "d2/b0.nii.gz"}}
        )


def test_add_derivation_refuses_shared_paths_and_does_not_mutate(tmp_path):
    m = dv.migrate_manifest(_legacy_manifest(), dv.KIND_RPE_PAIR)
    m["case_root"] = str(tmp_path)
    with pytest.raises(dv.DerivationError, match=r"d1.*d2|d2.*d1"):
        dv.add_derivation(
            m, "d2", dv.KIND_UNCORRECTED,
            {"other": _d2_meta("nifti/b0.nii.gz")},
            note="comparison",
        )
    assert "d2" not in m["derivations"]


def test_add_derivation_copies_inputs_and_starts_without_qc(tmp_path):
    m = dv.migrate_manifest(_legacy_manifest(), dv.KIND_RPE_PAIR)
    m["case_root"] = str(tmp_path)
    inputs = {"b0": _d2_meta("d2/b0.nii.gz")}
    out = dv.add_derivation(m, "d2", dv.KIND_UNCORRECTED, inputs, note="comparison")
    assert out is not m
    assert out["derivations"]["d2"] == {
        "kind": dv.KIND_UNCORRECTED,
        "inputs": inputs,
        "note": "comparison",
    }
    assert all(name not in out["derivations"]["d2"] for name in (
        "t1_qc", "atlas_prior_qc", "parcellation_qc", "delta_qc",
    ))
    assert "d2" not in m["derivations"]


def test_cli_add_derivation_roundtrip(tmp_path):
    root = tmp_path / "case"
    root.mkdir()
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps({
        **dv.migrate_manifest(_legacy_manifest(), dv.KIND_RPE_PAIR),
        "case_root": str(root),
    }))
    inputs_path = tmp_path / "d2-inputs.json"
    inputs_path.write_text(json.dumps({"b0": {
        "path": "d2/b0.nii.gz",
        "sha256_by_field": {"path": "aa" * 32},
        "bytes_by_field": {"path": 1},
    }}))

    assert dv.main([
        "add", str(root), "--id", "d2", "--kind", "uncorrected",
        "--inputs-json", str(inputs_path), "--note", "comparison",
    ]) == 0
    saved = json.loads(manifest_path.read_text())
    assert saved["active_derivation"] == "d1"
    assert saved["derivations"]["d2"]["inputs"] == {
        "b0": _d2_meta("d2/b0.nii.gz")
    }


def test_validate_unmigrated_manifest_is_a_noop():
    dv.validate(_legacy_manifest())  # legacy manifests stay valid


def test_cli_migrate_roundtrip(tmp_path):
    import json
    root = tmp_path / "case"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(_legacy_manifest()))
    rc = dv.main(["migrate", str(root), "--kind", "rpe_pair"])
    assert rc == 0
    m = json.loads((root / "manifest.json").read_text())
    assert m["active_derivation"] == "d1"


# --- Task 7: floor_label (owner-ruling strings, verbatim) -------------------

def test_floor_label_unmigrated_is_conservative_default():
    assert dv.floor_label(_legacy_manifest()) == "no reverse-PE — ~3 mm geometric floor"


def test_floor_label_uncorrected_kind():
    m = dv.migrate_manifest(_legacy_manifest(), dv.KIND_UNCORRECTED)
    assert dv.floor_label(m) == "no reverse-PE — ~3 mm geometric floor"


def test_floor_label_rpe_pair_unsigned_delta():
    m = dv.migrate_manifest(_legacy_manifest(), dv.KIND_RPE_PAIR)
    assert dv.floor_label(m) == (
        "reverse-PE corrected — residual uncertainty unquantified (delta QC unsigned)"
    )


def test_floor_label_rpe_pair_present_but_unsigned_approved_by():
    m = dv.migrate_manifest(_legacy_manifest(), dv.KIND_RPE_PAIR)
    m["delta_qc"] = {"auto": {"median_mm": 1.7}, "approved_by": None, "derivation": "d1"}
    assert dv.floor_label(m) == (
        "reverse-PE corrected — residual uncertainty unquantified (delta QC unsigned)"
    )


def test_floor_label_rpe_pair_signed_delta_includes_median():
    m = dv.migrate_manifest(_legacy_manifest(), dv.KIND_RPE_PAIR)
    m["delta_qc"] = {"auto": {"median_mm": 1.73}, "approved_by": "erion", "derivation": "d1"}
    assert dv.floor_label(m) == "reverse-PE corrected — median shift 1.7 mm (signed delta QC)"


def test_floor_label_rpe_pair_signed_but_nan_median_is_treated_as_unsigned():
    """NaN is a float (isinstance passes) but is not a usable measurement —
    the disclaimer must never render 'median shift nan mm'."""
    m = dv.migrate_manifest(_legacy_manifest(), dv.KIND_RPE_PAIR)
    m["delta_qc"] = {
        "auto": {"median_mm": float("nan")}, "approved_by": "erion", "derivation": "d1",
    }
    assert dv.floor_label(m) == (
        "reverse-PE corrected — residual uncertainty unquantified (delta QC unsigned)"
    )


# --- Task 7 fix round 1: single delta_qc_signed predicate, NaN-safe --------

def test_delta_qc_signed_true_for_signed_finite_median():
    m = dv.migrate_manifest(_legacy_manifest(), dv.KIND_RPE_PAIR)
    m["delta_qc"] = {"auto": {"median_mm": 1.73}, "approved_by": "erion", "derivation": "d1"}
    assert dv.delta_qc_signed(m) is True


def test_delta_qc_signed_false_when_approved_by_missing():
    m = dv.migrate_manifest(_legacy_manifest(), dv.KIND_RPE_PAIR)
    m["delta_qc"] = {"auto": {"median_mm": 1.73}, "approved_by": None, "derivation": "d1"}
    assert dv.delta_qc_signed(m) is False


def test_delta_qc_signed_false_when_median_mm_missing():
    m = dv.migrate_manifest(_legacy_manifest(), dv.KIND_RPE_PAIR)
    m["delta_qc"] = {"auto": {}, "approved_by": "erion", "derivation": "d1"}
    assert dv.delta_qc_signed(m) is False


def test_delta_qc_signed_false_for_nan_median():
    m = dv.migrate_manifest(_legacy_manifest(), dv.KIND_RPE_PAIR)
    m["delta_qc"] = {
        "auto": {"median_mm": float("nan")}, "approved_by": "erion", "derivation": "d1",
    }
    assert dv.delta_qc_signed(m) is False


def test_delta_qc_signed_false_when_no_delta_qc_block():
    m = dv.migrate_manifest(_legacy_manifest(), dv.KIND_RPE_PAIR)
    assert dv.delta_qc_signed(m) is False


@pytest.mark.parametrize("delta_qc, expect_signed", [
    ({"auto": {"median_mm": 1.73}, "approved_by": "erion"}, True),
    ({"auto": {"median_mm": None}, "approved_by": "erion"}, False),
    ({"auto": {"median_mm": float("nan")}, "approved_by": "erion"}, False),
    ({"auto": {"median_mm": 1.73}, "approved_by": None}, False),
])
def test_floor_label_and_delta_qc_signed_never_disagree(delta_qc, expect_signed):
    """The endpoint's delta_qc_signed field and floor_label's signed/unsigned
    wording must be derived from the SAME predicate — they can never diverge."""
    m = dv.migrate_manifest(_legacy_manifest(), dv.KIND_RPE_PAIR)
    m["delta_qc"] = {**delta_qc, "derivation": "d1"}

    signed = dv.delta_qc_signed(m)
    label = dv.floor_label(m)
    assert signed is expect_signed
    assert ("median shift" in label) is signed
    assert ("unquantified" in label) is (not signed)


# --- Task 8: integration — legacy manifest through the whole S2 chain -----

def test_end_to_end_legacy_manifest_migrate_sheet_sign_delta(tmp_path):
    """One case, real functions, no HTTP/subprocess:

    legacy manifest -> CLI migrate -> validate -> atlas sheet-write carries
    derivation -> sign-with-mismatch raises -> delta sheet block unsigned.
    """
    root = tmp_path / "case"
    (root / "qc").mkdir(parents=True)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps({
        "case_id": "demo-int",
        "case_root": str(root),
        "inputs": {"b0": {"path": "nifti/b0.nii.gz"}},
    }))

    # 1. legacy manifest -> CLI migrate
    rc = dv.main(["migrate", str(root), "--kind", "rpe_pair"])
    assert rc == 0
    migrated = json.loads(manifest_path.read_text())
    assert migrated["active_derivation"] == "d1"
    assert migrated["derivations"]["d1"]["kind"] == dv.KIND_RPE_PAIR

    # 2. validate
    dv.validate(migrated)  # no raise

    # 3. atlas sheet-write carries derivation
    (root / "qc" / "atlas_prior_contours.png").write_bytes(b"not-a-real-png")
    auto = atlas_prep.AutoQC(
        ok=True, cst_symmetry_ok=True, hull_containment_ok=True,
        cst_overlap=0.9, hull_frac=0.9, notes=(),
    )
    atlas_prep.update_case_manifest_priors(
        manifest_path, tract_index={}, auto=auto,
        sheet_sha="deadbeef", reg_note="test-reg",
    )
    with_atlas = json.loads(manifest_path.read_text())
    assert with_atlas["atlas_prior_qc"]["derivation"] == "d1"

    # 4. sign-with-mismatch raises (sheet describes a lineage that moved on)
    with_atlas["atlas_prior_qc"]["derivation"] = "d0-stale"
    manifest_path.write_text(json.dumps(with_atlas))
    with pytest.raises(ValueError, match="derivation mismatch"):
        atlas_prep.approve_atlas_qc(str(manifest_path), approved_by="tester")
    assert json.loads(manifest_path.read_text())["atlas_prior_qc"]["approved_by"] is None
    # Writers now validate before output; restore the active scope after the
    # deliberate mismatch test before generating the next QC artifact.
    with_atlas["atlas_prior_qc"]["derivation"] = "d1"
    manifest_path.write_text(json.dumps(with_atlas))

    # 5. delta sheet block stays unsigned until a human approves
    field = nib.Nifti1Image(np.full((2, 2, 2), 5.0), np.diag([2.0, 2.0, 2.0, 1.0]))
    mask = nib.Nifti1Image(np.ones((2, 2, 2)), np.diag([2.0, 2.0, 2.0, 1.0]))
    field.header.set_xyzt_units("mm", "sec")
    mask.header.set_xyzt_units("mm", "sec")
    stats = delta_qc.compute_delta(field, mask, readout_time_s=0.01, pe_axis=1)
    delta_block = delta_qc.write_sheet(root, stats, root / "qc" / "delta_qc.png")
    assert delta_block["approved_by"] is None
    assert delta_block["derivation"] == "d1"

    final = json.loads(manifest_path.read_text())
    final["delta_qc"] = delta_block
    assert dv.delta_qc_signed(final) is False
    assert "unquantified" in dv.floor_label(final)


def test_cli_migrate_twice_fails_loud(tmp_path):
    import json
    root = tmp_path / "case"
    root.mkdir()
    manifest_path = root / "manifest.json"
    manifest_bytes = json.dumps(_legacy_manifest()).encode()
    manifest_path.write_bytes(manifest_bytes)
    # First migration should succeed
    rc = dv.main(["migrate", str(root), "--kind", "rpe_pair"])
    assert rc == 0
    # Snapshot the successful migration's bytes
    snapshot = manifest_path.read_bytes()
    # Second migration should fail
    rc = dv.main(["migrate", str(root), "--kind", "rpe_pair"])
    assert rc == 1
    # Manifest must be byte-identical (failure precedes any write)
    assert manifest_path.read_bytes() == snapshot


def test_delta_qc_signed_false_when_scoped_to_stale_derivation():
    """Read-time scope guard (final-review fix): a delta_qc block whose
    derivation no longer matches the active one counts as unsigned, matching
    atlas_prior_qc_ok/parcellation_qc_ok — serve never runs validate()."""
    m = dv.migrate_manifest(_legacy_manifest(), dv.KIND_RPE_PAIR)
    m["active_derivation"] = "d2-hand-edited"
    m["delta_qc"] = {"auto": {"median_mm": 1.73}, "approved_by": "erion", "derivation": "d1"}
    assert dv.delta_qc_signed(m) is False
    assert "unsigned" in dv.floor_label(m) or "no reverse-PE" in dv.floor_label(m)


def test_t1_qc_ok_requires_full_signature():
    man = {"t1_qc": {"approved_by": "erion"}}
    assert dv.t1_qc_ok(man) is False
    man["t1_qc"]["date"] = "2026-08-09"
    assert dv.t1_qc_ok(man) is False
    man["t1_qc"]["sheet_sha"] = "abc"
    assert dv.t1_qc_ok(man) is True


def test_t1_qc_ok_refuses_stale_derivation_scope():
    signed = {
        "t1_qc": {
            "approved_by": "erion",
            "date": "2026-08-09",
            "sheet_sha": "abc",
            "derivation": "d1",
        },
        "active_derivation": "d2",
    }
    assert dv.t1_qc_ok(signed) is False
    signed["active_derivation"] = "d1"
    assert dv.t1_qc_ok(signed) is True
    del signed["active_derivation"]
    assert dv.t1_qc_ok(signed) is True


def test_t1_qc_ok_absent_or_malformed():
    assert dv.t1_qc_ok({}) is False
    assert dv.t1_qc_ok({"t1_qc": "yes"}) is False


# ── Wave 3: "profiles" is a shared artifact kind (first lineage keeps work/profiles) ──


def _two_lineage_manifest(case, active):
    return {
        "case_id": "profiles-demo",
        "case_root": str(case),
        "active_derivation": active,
        "derivations": {
            "d1": {"kind": "uncorrected", "inputs": {}},
            "d2": {"kind": "rpe_pair", "inputs": {}},
        },
    }


def test_artifact_dir_profiles_first_lineage_keeps_work_profiles(tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    legacy = {"case_id": "x", "case_root": str(case), "inputs": {}}
    assert dv.artifact_dir(legacy, "profiles") == case.resolve() / "work" / "profiles"
    first = _two_lineage_manifest(case, "d1")
    assert dv.artifact_dir(first, "profiles") == case.resolve() / "work" / "profiles"


def test_artifact_dir_profiles_later_lineage_is_scoped(tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    later = _two_lineage_manifest(case, "d2")
    assert dv.artifact_dir(later, "profiles") == case.resolve() / "derivations" / "d2" / "profiles"


def test_artifact_path_profiles_refuses_first_lineage_alias(tmp_path):
    """A later lineage must not write into the first lineage's work/profiles."""
    case = tmp_path / "case"
    (case / "work").mkdir(parents=True)
    (case / "derivations" / "d2").mkdir(parents=True)
    # symlink the d2 profiles dir onto d1's historical location
    (case / "work" / "profiles").mkdir()
    (case / "derivations" / "d2" / "profiles").symlink_to(case / "work" / "profiles")
    later = _two_lineage_manifest(case, "d2")
    with pytest.raises(dv.DerivationError, match="would overwrite derivation 'd1'"):
        dv.artifact_path(later, "profiles", "fa_casemask.nii.gz")
