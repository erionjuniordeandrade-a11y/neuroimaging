"""T13 — operating-point sweep over sidecar R_grid × minFrac. No bank re-walk."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tractlab.fidelity import R_GRID, save_sidecar
from tractlab.fidelity_sweep import MIN_FRAC_GRID, run_sweep, sheet_template, sweep_table


def _sidecar(tmp: Path, frac_ge: np.ndarray, name="bank_x.fidelity.npz") -> Path:
    n, n_r = frac_ge.shape
    assert n_r == len(R_GRID)
    data = {
        "schema": 2,
        "bank_sha256": "ab" * 32,
        "fod_sha256": "cd" * 32,
        "sh_load_sha256": "cd" * 32,
        "peak_sha256": "ef" * 32,
        "R_grid": R_GRID.copy(),
        "frac_ge": frac_ge.astype(np.float32),
        "p5_ratio": np.zeros((n,), dtype=np.float32),
        "n_segments": np.full((n,), 10, dtype=np.uint16),
        "flags": np.zeros((n,), dtype=np.uint8),
        "ratios_present": True,
    }
    out = tmp / "fidelity"
    out.mkdir(exist_ok=True)
    manifest = tmp / "manifest.json"
    if not manifest.exists():
        manifest.write_text(json.dumps({
            "case_id": "synthetic-fidelity-sweep",
            "case_root": str(tmp),
            "inputs": {},
        }))
    path = out / name
    save_sidecar(path, data)
    return path


def test_sweep_row_count_is_r_grid_times_minfrac(tmp_path):
    frac = np.ones((4, len(R_GRID)), dtype=np.float32)
    _sidecar(tmp_path, frac)
    rows = sweep_table(tmp_path)
    assert len(rows) == len(R_GRID) * len(MIN_FRAC_GRID)


def test_higher_R_never_decreases_marked_fraction(tmp_path):
    # all streamlines have every-segment ratio 0.40
    frac = (R_GRID[None, :] <= 0.40).astype(np.float32)
    frac = np.repeat(frac, 5, axis=0)
    _sidecar(tmp_path, frac)
    rows = [r for r in sweep_table(tmp_path) if r["min_frac"] == 0.7]
    rows.sort(key=lambda r: r["R"])
    marked = [r["marked_frac"] for r in rows]
    assert marked == sorted(marked)


def test_sheet_template_has_empty_approved_by():
    text = sheet_template(csv_sha256="deadbeef")
    assert "approved_by:" in text
    assert "approved_by: " in text or "approved_by:\n" in text or "approved_by: \n" in text
    assert "erion" not in text
    assert "deadbeef" in text


def test_run_sweep_writes_csv_png_and_unsigned_sheet(tmp_path):
    frac = np.ones((2, len(R_GRID)), dtype=np.float32)
    _sidecar(tmp_path, frac)
    (tmp_path / "docs" / "qc").mkdir(parents=True)
    summary = run_sweep(tmp_path, sheet_path=tmp_path / "docs/qc/OPERATING-POINT-fidelity.md")
    csv_path = tmp_path / "fidelity" / "sweep.csv"
    png_path = tmp_path / "fidelity" / "sweep.png"
    assert csv_path.is_file() and png_path.is_file()
    sheet = Path(summary["sheet"]).read_text()
    assert "approved_by:" in sheet
    assert summary["n_rows"] == len(R_GRID) * len(MIN_FRAC_GRID)


# ── signed operating-point sheet → read_operating_point ──────────────────────
import hashlib

import pytest

from tractlab.fidelity import (
    DEFAULT_MIN_FRAC,
    DEFAULT_R,
    FidelityRefusal,
    parse_sheet,
    read_operating_point,
)


def _case_with_sheet(tmp_path, body: str, *, csv_bytes: bytes = b"bank_id,R\n"):
    (tmp_path / "fidelity").mkdir(exist_ok=True)
    (tmp_path / "fidelity" / "sweep.csv").write_bytes(csv_bytes)
    sheet = tmp_path / "docs" / "qc" / "OPERATING-POINT-fidelity.md"
    sheet.parent.mkdir(parents=True, exist_ok=True)
    sheet.write_text(body)
    return hashlib.sha256(csv_bytes).hexdigest()


def _signed(sha, *, R="0.45", min_frac="0.9", who="erion", date="2026-09-01"):
    return (
        "# Fidelity operating point\n\n"
        f"R: {R}\nmin_frac: {min_frac}\napproved_by: {who}\ndate: {date}\n"
        f"sweep_csv_sha256: {sha}\n"
    )


def test_template_round_trips_to_unsigned_pilot(tmp_path):
    """The exact writer output parses as unsigned → pilot fallback."""
    text = sheet_template(csv_sha256="deadbeef")
    fields = parse_sheet(text)
    assert fields["approved_by"] == "" and fields["R"] == ""
    _case_with_sheet(tmp_path, text)
    op = read_operating_point(tmp_path)
    assert op.source == "pilot" and not op.signed
    assert op.R == DEFAULT_R and op.min_frac == DEFAULT_MIN_FRAC


def test_no_sheet_is_pilot(tmp_path):
    op = read_operating_point(tmp_path)
    assert op.source == "pilot"


def test_signed_sheet_is_read_and_bites(tmp_path):
    sha = _case_with_sheet(tmp_path, "")
    (tmp_path / "docs" / "qc" / "OPERATING-POINT-fidelity.md").write_text(_signed(sha))
    op = read_operating_point(tmp_path)
    assert op.signed and op.source == "signed"
    assert abs(op.R - 0.45) < 1e-6 and abs(op.min_frac - 0.9) < 1e-9
    assert op.approved_by == "erion" and op.date == "2026-09-01"
    assert (op.R, op.min_frac) != (DEFAULT_R, DEFAULT_MIN_FRAC)


@pytest.mark.parametrize(
    "kw, needle",
    [
        ({"R": "0.33"}, "not on R_GRID"),
        ({"min_frac": "0.75"}, "not on MIN_FRAC_GRID"),
        ({"date": ""}, "date blank"),
        ({"R": ""}, "numeric"),
    ],
)
def test_signed_sheet_refuses_when_it_cannot_be_honoured(tmp_path, kw, needle):
    sha = _case_with_sheet(tmp_path, "")
    (tmp_path / "docs" / "qc" / "OPERATING-POINT-fidelity.md").write_text(_signed(sha, **kw))
    with pytest.raises(FidelityRefusal, match=needle):
        read_operating_point(tmp_path)


def test_signed_sheet_refuses_on_sweep_drift(tmp_path):
    sha = _case_with_sheet(tmp_path, "")
    sheet = tmp_path / "docs" / "qc" / "OPERATING-POINT-fidelity.md"
    sheet.write_text(_signed(sha))
    (tmp_path / "fidelity" / "sweep.csv").write_bytes(b"bank_id,R\nchanged\n")
    with pytest.raises(FidelityRefusal, match="drifted"):
        read_operating_point(tmp_path)
    (tmp_path / "fidelity" / "sweep.csv").unlink()
    with pytest.raises(FidelityRefusal, match="absent"):
        read_operating_point(tmp_path)


def test_signed_sheet_never_downgrades_to_pilot_silently(tmp_path):
    """A refused signed sheet must raise, not return the pilot point."""
    sha = _case_with_sheet(tmp_path, "")
    (tmp_path / "docs" / "qc" / "OPERATING-POINT-fidelity.md").write_text(
        _signed("0" * 64)
    )
    with pytest.raises(FidelityRefusal):
        read_operating_point(tmp_path)
