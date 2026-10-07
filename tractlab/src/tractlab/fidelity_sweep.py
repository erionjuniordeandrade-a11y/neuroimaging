"""T13 — fidelity operating-point sweep from existing sidecars. No bank re-walk."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from neuro_core.hashing import sha256_file

from . import derivation as dv
from .fidelity import MIN_FRAC_GRID, R_GRID, load_sidecar  # noqa: F401 (re-export)


_sha256 = sha256_file


def _marked_frac(frac_ge: np.ndarray, r_index: int, min_frac: float) -> float:
    col = frac_ge[:, r_index]
    ok = np.isfinite(col)
    if not np.any(ok):
        return float("nan")
    return float(np.mean(col[ok] < min_frac))


def _layout_manifest(case_root: Path) -> dict:
    manifest_path = case_root / "manifest.json"
    if not manifest_path.is_file():
        raise dv.DerivationError(f"manifest missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise dv.DerivationError(f"invalid manifest: {manifest_path}") from e
    if not isinstance(manifest, dict):
        raise dv.DerivationError(f"manifest must be a JSON object: {manifest_path}")
    return {**manifest, "case_root": str(case_root)}


def sweep_table(case_root: Path, manifest: dict | None = None) -> list[dict]:
    """One row per (bank, R, min_frac). Sidecars only; missing dir → empty.

    ``manifest`` is the caller's validated snapshot; ``run_sweep`` passes its
    own so the table and the sheet always describe one lineage (ADR-0004).
    """
    case_root = Path(case_root)
    if manifest is None:
        manifest = _layout_manifest(case_root)
    folder = dv.artifact_dir(manifest, "fidelity")
    rows: list[dict] = []
    if not folder.is_dir():
        return rows
    for path in sorted(folder.glob("*.fidelity.npz")):
        bank_id = path.name[: -len(".fidelity.npz")]
        # SHA gate needs the sidecar's own recorded bank sha (no re-walk).
        with np.load(str(path), allow_pickle=False) as z:
            bank_sha = bytes(z["bank_sha256"]).decode("ascii")
            fod_sha = bytes(z["fod_sha256"]).decode("ascii")
        sc = load_sidecar(path, expected_bank_sha=bank_sha, expected_fod_sha=fod_sha or None)
        if not sc.ratios_present:
            continue
        for ri, r in enumerate(sc.R_grid):
            for mf in MIN_FRAC_GRID:
                rows.append({
                    "bank_id": bank_id,
                    "R": float(r),
                    "min_frac": float(mf),
                    "marked_frac": _marked_frac(sc.frac_ge, ri, mf),
                    "n": int(sc.n),
                })
    return rows


def sheet_template(*, csv_sha256: str) -> str:
    return (
        "# Fidelity operating point\n\n"
        "Unsigned until the owner fills `approved_by`. Automation never signs.\n\n"
        "R:\n"
        "min_frac:\n"
        "approved_by:\n"
        "date:\n"
        f"sweep_csv_sha256: {csv_sha256}\n"
    )


def _write_png(rows: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0d0f12")
        ax.text(0.5, 0.5, "no sidecars", ha="center", va="center", color="#ccc")
        fig.savefig(out, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return
    banks = sorted({r["bank_id"] for r in rows})
    # mean marked_frac across banks at each (R, min_frac)
    grid = np.full((len(MIN_FRAC_GRID), len(R_GRID)), np.nan)
    for i, mf in enumerate(MIN_FRAC_GRID):
        for j, r in enumerate(R_GRID):
            vals = [row["marked_frac"] for row in rows
                    if row["min_frac"] == mf and abs(row["R"] - float(r)) < 1e-6]
            if vals:
                grid[i, j] = float(np.nanmean(vals))
    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor="#0d0f12")
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="inferno", vmin=0, vmax=1)
    ax.set_xticks(range(len(R_GRID)))
    ax.set_xticklabels([f"{x:.2f}" for x in R_GRID], fontsize=7, color="#ccc")
    ax.set_yticks(range(len(MIN_FRAC_GRID)))
    ax.set_yticklabels([str(x) for x in MIN_FRAC_GRID], fontsize=8, color="#ccc")
    ax.set_xlabel("R", color="#ccc")
    ax.set_ylabel("min_frac", color="#ccc")
    ax.set_title(f"marked fraction (mean over {len(banks)} bank(s))", color="#e8c56a", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def run_sweep(case_root: Path, *, sheet_path: Path | None = None) -> dict:
    case_root = Path(case_root)
    manifest = _layout_manifest(case_root)
    dv.validate(manifest)
    expected_sheet = dv.operating_point_path(manifest)
    if sheet_path is not None and Path(sheet_path).resolve() != expected_sheet.resolve():
        raise dv.DerivationError(
            f"fidelity operating-point sheet must use active derivation layout: {expected_sheet}"
        )
    rows = sweep_table(case_root, manifest)
    out_dir = dv.artifact_dir(manifest, "fidelity")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dv.artifact_path(manifest, "fidelity", "sweep.csv")
    png_path = dv.artifact_path(manifest, "fidelity", "sweep.png")
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bank_id", "R", "min_frac", "marked_frac", "n"])
        w.writeheader()
        w.writerows(rows)
    _write_png(rows, png_path)
    if sheet_path is None:
        sheet_path = expected_sheet
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet_path.write_text(sheet_template(csv_sha256=_sha256(csv_path)))
    return {
        "csv": str(csv_path),
        "png": str(png_path),
        "sheet": str(sheet_path),
        "n_rows": len(rows),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tractlab.fidelity_sweep")
    ap.add_argument("--case-root", required=True)
    ap.add_argument("--sheet", default=None)
    args = ap.parse_args(argv)
    summary = run_sweep(Path(args.case_root), sheet_path=Path(args.sheet) if args.sheet else None)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
