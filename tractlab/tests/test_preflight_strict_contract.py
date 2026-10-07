"""Generated fixtures for the strict preflight receipt/path contract."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tractlab import preflight
from tractlab.preflight import (
    CRITERION_IDS,
    build_receipt,
    compute_drift,
    load_receipt,
    run_preflight,
)


def _write_gradients(
    root: Path,
    *,
    bvec: str | None = None,
    bval: str = "0 1000 2000 3000\n",
) -> None:
    (root / "dwi.bvec").write_text(
        bvec
        if bvec is not None
        else "0 0 0 0\n0 0 0 0\n1 1 1 1\n"
    )
    (root / "dwi.bval").write_text(bval)


def _manifest(root: Path) -> dict:
    return {
        "case_id": "generated-strict-contract",
        "case_root": str(root),
        "acquisition_evidence": {"gradients": {"path": "dwi.bvec"}},
    }


def _receipt(root: Path) -> dict:
    _write_gradients(root)
    return build_receipt(str(root), _manifest(root))


def _write_receipt(root: Path, receipt: dict) -> None:
    (root / "preflight.json").write_text(json.dumps(receipt))


def test_schema2_receipt_has_exact_criterion_and_fingerprint_coverage(tmp_path: Path):
    receipt = _receipt(tmp_path)

    assert receipt["schema"] == 2
    assert [criterion["id"] for criterion in receipt["criteria"]] == list(
        CRITERION_IDS
    )
    assert set(receipt["fingerprints"]) == set(CRITERION_IDS)
    assert all(
        isinstance(value, str) and len(value) == 64 and value == value.lower()
        for value in receipt["fingerprints"].values()
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda receipt: receipt.update(schema=3), "schema"),
        (lambda receipt: receipt["fingerprints"].pop("gradients"), "fingerprint"),
        (lambda receipt: receipt["fingerprints"].update(extra="0" * 64), "fingerprint"),
        (
            lambda receipt: receipt["fingerprints"].update(gradients="not-a-sha"),
            "fingerprint",
        ),
        (
            lambda receipt: receipt["criteria"].append(
                {"id": "future", "verdict": "pass", "evidence": "x"}
            ),
            "criterion",
        ),
        (lambda receipt: receipt["criteria"][0].update(verdict="maybe"), "verdict"),
    ],
)
def test_compute_drift_rejects_noncanonical_receipts(
    tmp_path: Path, mutation, message: str
):
    stored = _receipt(tmp_path)
    verified = run_preflight(str(tmp_path), _manifest(tmp_path)).to_json()
    mutation(stored)

    with pytest.raises(ValueError, match=message):
        compute_drift(stored, verified, active_derivation=stored.get("derivation"))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda receipt: receipt.pop("schema"),
        lambda receipt: receipt.update(schema=1),
        lambda receipt: receipt.update(schema=999),
        lambda receipt: receipt["fingerprints"].pop("gradients"),
        lambda receipt: receipt["fingerprints"].update(extra="0" * 64),
    ],
)
def test_load_receipt_rejects_missing_old_future_and_incomplete_schema(
    tmp_path: Path, mutation
):
    receipt = _receipt(tmp_path)
    mutation(receipt)
    _write_receipt(tmp_path, receipt)

    with pytest.raises(ValueError):
        load_receipt(str(tmp_path))


def test_approve_receipt_does_not_sign_an_incomplete_receipt(tmp_path: Path):
    receipt = _receipt(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps(_manifest(tmp_path)))
    receipt["fingerprints"].pop("gradients")
    _write_receipt(tmp_path, receipt)

    with pytest.raises(ValueError, match="fingerprint"):
        preflight.approve_receipt(str(tmp_path), who="owner")
    persisted = json.loads((tmp_path / "preflight.json").read_text())
    assert persisted.get("approved_by") is None


def test_declared_symlink_outside_case_root_fails_before_hash(tmp_path: Path):
    case_root = tmp_path / "case"
    outside = tmp_path / "outside"
    case_root.mkdir()
    outside.mkdir()
    _write_gradients(outside)
    try:
        (case_root / "escape.bvec").symlink_to(outside / "dwi.bvec")
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    manifest = _manifest(case_root)
    manifest["acquisition_evidence"]["gradients"] = {"path": "escape.bvec"}
    report = run_preflight(str(case_root), manifest).to_json()
    gradient = next(item for item in report["criteria"] if item["id"] == "gradients")

    assert gradient["verdict"] == "fail"
    assert "case root" in gradient["evidence"].lower()


def test_gradient_companion_symlink_outside_case_root_is_rejected_and_unhashed(
    tmp_path: Path, monkeypatch
):
    case_root = tmp_path / "case"
    outside = tmp_path / "outside"
    case_root.mkdir()
    outside.mkdir()
    _write_gradients(case_root)
    (outside / "dwi.bval").write_text("0 1000 2000 3000\n")
    try:
        (case_root / "dwi.bval").unlink()
        (case_root / "dwi.bval").symlink_to(outside / "dwi.bval")
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    seen: list[Path] = []
    original = preflight._sha256_file

    def record(path: Path) -> str:
        seen.append(Path(path).resolve())
        return original(path)

    monkeypatch.setattr(preflight, "_sha256_file", record)
    report = run_preflight(str(case_root), _manifest(case_root)).to_json()
    gradient = next(item for item in report["criteria"] if item["id"] == "gradients")

    assert gradient["verdict"] == "fail"
    assert outside not in seen


def test_one_dimensional_six_value_bvec_is_malformed(tmp_path: Path):
    _write_gradients(tmp_path, bvec="0 0 1 0 0 1\n", bval="1000\n")

    report = run_preflight(str(tmp_path), _manifest(tmp_path)).to_json()
    gradient = next(item for item in report["criteria"] if item["id"] == "gradients")

    assert gradient["verdict"] == "fail"
    assert "shape" in gradient["evidence"].lower()
