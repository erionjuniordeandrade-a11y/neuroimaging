"""Geometry-delta QC: topup field(Hz) -> measured per-case displacement (mm).

Converts a topup fieldmap into the actual geometric distortion a case's own
readout time and voxel size imply, instead of trusting a generic "a few mm"
disclaimer. Unsigned QC block: ``approved_by``/``date`` are always None here —
only the owner signs (see ``check_registration.py``'s ``approve_t1_qc``
pattern; this repo voided an agent-stamped QC once, never repeat it).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import nibabel as nib
import numpy as np

from . import derivation as dv
from .grid import (
    assert_grids_match,
    grid_from_image,
    unknown_units_assumption_from_manifest,
)


@dataclass(frozen=True)
class DeltaStats:
    median_mm: float
    p95_mm: float
    max_mm: float
    n_vox: int
    shift_map: np.ndarray  # nan outside mask / at nan field voxels — sheet-only


def compute_delta(
    field_nii,
    mask_nii,
    readout_time_s: float,
    pe_axis: int = 1,
    *,
    assume_unknown_spatial_units_mm: bool = False,
) -> DeltaStats:
    """shift_mm = |field_Hz| * readout_time_s * zoom_pe_mm, nan-safe masking.

    zoom_pe_mm is read from the field image's own header (nibabel authority),
    never assumed. Voxels outside the mask, or NaN in the field, are excluded
    from the stats rather than fabricated as zero (honest nulls).
    """
    field_grid = grid_from_image(
        field_nii,
        source="field",
        require_mm=True,
        assume_unknown_spatial_units_mm=assume_unknown_spatial_units_mm,
        require_authoritative_affine=True,
    )
    mask_grid = grid_from_image(
        mask_nii,
        source="mask",
        require_mm=True,
        assume_unknown_spatial_units_mm=assume_unknown_spatial_units_mm,
        require_authoritative_affine=True,
    )
    assert_grids_match(field_grid, mask_grid, context="field/mask grid")

    field = np.asanyarray(field_nii.dataobj, dtype=np.float64)
    mask = np.asanyarray(mask_nii.dataobj) > 0
    if field.ndim != 3 or mask.ndim != 3:
        raise ValueError(f"field/mask must be 3-D: {field.shape} vs {mask.shape}")
    if field.shape != mask.shape:
        raise ValueError(f"field/mask shape mismatch: {field.shape} vs {mask.shape}")

    zoom_pe_mm = float(field_nii.header.get_zooms()[pe_axis])
    shift = np.abs(field) * readout_time_s * zoom_pe_mm

    shift_map = np.full(field.shape, np.nan, dtype=np.float64)
    shift_map[mask] = shift[mask]
    vals = shift_map[mask]
    vals = vals[~np.isnan(vals)]
    n_vox = int(vals.size)

    if n_vox == 0:
        return DeltaStats(median_mm=float("nan"), p95_mm=float("nan"),
                           max_mm=float("nan"), n_vox=0, shift_map=shift_map)
    return DeltaStats(
        median_mm=float(np.median(vals)),
        p95_mm=float(np.percentile(vals, 95)),
        max_mm=float(np.max(vals)),
        n_vox=n_vox,
        shift_map=shift_map,
    )


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _render_sheet(stats: DeltaStats, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shift_map = stats.shift_map
    valid = ~np.isnan(shift_map)
    vals = shift_map[valid]

    fig, (ax_hist, ax_slice) = plt.subplots(1, 2, figsize=(9, 4), facecolor="#0d0f12")
    if vals.size:
        ax_hist.hist(vals, bins=40, color="#e8c56a")
        k = int(np.unravel_index(np.nanargmax(shift_map), shift_map.shape)[2])
        im = ax_slice.imshow(shift_map[:, :, k].T, origin="lower", cmap="inferno")
        fig.colorbar(im, ax=ax_slice, fraction=0.046)
        ax_slice.set_title(f"max-shift slice (z={k})", color="#ccc", fontsize=9)
    else:
        ax_hist.text(0.5, 0.5, "no voxels in mask", ha="center", va="center", color="#ccc")
        ax_slice.set_title("no voxels in mask", color="#ccc", fontsize=9)
    ax_hist.set_title("shift_mm histogram", color="#ccc", fontsize=9)
    ax_hist.tick_params(colors="#ccc")
    ax_slice.axis("off")
    fig.suptitle("Geometry-delta QC — field(Hz) to displacement(mm), unsigned",
                 color="#e8c56a", fontsize=11)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def write_sheet(case_root, stats: DeltaStats, png_path=None) -> dict:
    """Render the QC sheet, sha256 it, and return the unsigned delta_qc block.

    Reads case_root/manifest.json to resolve the active derivation and its
    output layout — never writes it (persistence is the CLI's job, via the
    atomic idiom in derivation.py's main()).
    """
    case_root = Path(case_root).resolve()
    manifest = json.loads((case_root / "manifest.json").read_text())
    dv.validate(manifest)
    layout = {**manifest, "case_root": str(case_root)}
    expected = dv.artifact_path(layout, "qc", "delta_qc.png")
    if png_path is None:
        png_path = expected
    else:
        png_path = Path(png_path)
        if png_path.resolve() != expected.resolve():
            raise dv.DerivationError(
                f"delta QC output must use active derivation layout: {expected}"
            )
    _render_sheet(stats, png_path)
    sheet_sha = sha256_file(png_path)

    active = dv.active_derivation(manifest)
    try:
        sheet_rel = png_path.relative_to(case_root).as_posix()
    except ValueError:
        sheet_rel = png_path.as_posix()

    return {
        "auto": {
            "median_mm": stats.median_mm,
            "p95_mm": stats.p95_mm,
            "max_mm": stats.max_mm,
            "n_vox": stats.n_vox,
        },
        "approved_by": None,
        "date": None,
        "sheet_sha": sheet_sha,
        "sheet_path": sheet_rel,
        "derivation": active,
    }


def approve_delta_qc(
    case_root,
    *,
    approved_by: str,
    sheet_sha: str | None = None,
) -> dict:
    """Human sign-off for the geometry-delta QC sheet (mirrors atlas_prep's
    ``approve_atlas_qc``/``approve_parcellation_qc`` pattern exactly).

    Fail-closed: refuses (``ValueError``) when there is no ``delta_qc`` block
    to sign, when the sheet PNG is missing, when a supplied ``sheet_sha``
    doesn't match the live file, or when the block's derivation no longer
    matches the manifest's active derivation (ADR-0004). ``approved_by`` is a
    required keyword argument — it must come from the caller (the owner, at
    CLI invocation) and is never defaulted or auto-filled; an agent-stamped
    QC was voided once in this repo's history and that must never repeat.
    """
    case_root = Path(case_root).resolve()
    manifest_path = case_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    layout = {**manifest, "case_root": str(case_root)}
    dv.validate(manifest)
    qc = dv.qc_block_for(layout, "delta_qc")
    if qc is None:
        raise ValueError("no delta_qc block to sign — run delta_qc first")

    expected = dv.artifact_path(layout, "qc", "delta_qc.png")
    expected_rel = expected.relative_to(case_root).as_posix()
    if qc.get("sheet_path") != expected_rel:
        raise ValueError(
            f"delta_qc sheet_path is not the active derivation artifact: {expected_rel}"
        )
    sheet = expected
    if not sheet.is_file():
        raise ValueError(f"QC sheet missing: {sheet}")

    live = sha256_file(sheet)
    if sheet_sha and sheet_sha != live:
        raise ValueError("sheet_sha mismatch")

    result = dv.set_qc_block(manifest, "delta_qc", {
        **qc,
        "approved_by": approved_by,
        "date": date.today().isoformat(),
        "sheet_sha": live,
    })
    tmp_path = manifest_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2))
    os.replace(str(tmp_path), str(manifest_path))
    return result


def _sidecar_json(dwi_path: Path) -> Path:
    name = dwi_path.name
    base = dwi_path.with_name(name[: -len(".nii.gz")]) if name.endswith(".nii.gz") \
        else dwi_path.with_suffix("")
    return base.with_suffix(".json")


def main(argv: list[str] | None = None) -> int:
    """CLI: two modes over one case_root, mutually exclusive.

    Compute (unsigned): python -m tractlab.delta_qc <case_root> --field <path> --pe-axis 1
      Computes stats from the given topup field against the case's own DWI
      readout time + mask, writes qc/delta_qc.png, and persists the resulting
      unsigned delta_qc block into manifest.json atomically (.tmp + os.replace,
      matching derivation.py's CLI idiom).

    Sign (owner only): python -m tractlab.delta_qc <case_root> --approve <name> [--sheet-sha <sha>]
      Calls approve_delta_qc — see that function's docstring for the
      fail-closed guards. Never invoked by an agent; the owner runs this from
      the terminal.
    """
    ap = argparse.ArgumentParser(
        prog="python -m tractlab.delta_qc",
        description="Geometry-delta QC: topup field(Hz) -> measured mm, unsigned; owner signs separately")
    ap.add_argument("case_root", type=str)
    ap.add_argument("--field", default=None,
                     help="topup field NIfTI (Hz), absolute or relative to case_root")
    ap.add_argument("--mask", default=None,
                     help="mask NIfTI on the SAME grid as --field (default: manifest inputs.mask)")
    ap.add_argument("--pe-axis", type=int, default=1)
    ap.add_argument("--approve", metavar="NAME", default=None,
                     help="sign the existing delta_qc sheet as NAME (owner only)")
    ap.add_argument("--sheet-sha", default=None,
                     help="optional expected sheet sha256 to verify before signing")
    args = ap.parse_args(argv)

    case_root = Path(args.case_root)

    if args.approve is not None:
        try:
            block = approve_delta_qc(case_root, approved_by=args.approve, sheet_sha=args.sheet_sha)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        print(json.dumps(block, indent=2))
        return 0

    if args.field is None:
        ap.error("either --field (compute) or --approve (sign) is required")

    manifest_path = case_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    dv.validate(manifest)
    inputs = dv.active_inputs(manifest)

    field_path = Path(args.field)
    if not field_path.is_absolute():
        field_path = case_root / field_path
    if args.mask:
        mask_path = Path(args.mask)
        if not mask_path.is_absolute():
            mask_path = case_root / mask_path
    else:
        mask_path = case_root / inputs["mask"]["path"]
    dwi_path = case_root / manifest["raw"]["dwi"]["path"]
    readout_time_s = json.loads(_sidecar_json(dwi_path).read_text())["TotalReadoutTime"]

    try:
        assume_unknown_units_mm = unknown_units_assumption_from_manifest(manifest)
        stats = compute_delta(
            nib.load(str(field_path)), nib.load(str(mask_path)),
            readout_time_s,
            pe_axis=args.pe_axis,
            assume_unknown_spatial_units_mm=assume_unknown_units_mm,
        )
    except (OSError, ValueError) as exc:
        print(f"UNIT/GRID POLICY FAILED: {exc}", file=sys.stderr)
        return 2

    # Clear any prior signature before replacing the QC artefact. Persist the
    # unsigned state atomically so an interrupted render cannot leave an old
    # signature over new bytes.
    if dv.qc_block_for(manifest, "delta_qc") is not None:
        dv.set_qc_block(manifest, "delta_qc", {
            "auto": {},
            "approved_by": None,
            "date": None,
            "sheet_sha": None,
            "voided": True,
            "note": "voided by delta QC regeneration",
        })
        dv.write_manifest_atomic(manifest_path, manifest)
    block = write_sheet(case_root, stats)

    dv.set_qc_block(manifest, "delta_qc", block)
    dv.write_manifest_atomic(manifest_path, manifest)

    print(json.dumps(block, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
