#!/usr/bin/env python3
"""Offline fidelity prep: build the per-case fODF peak volume (E1, Task 5).

Peak = max amplitude over a dense 300-direction set:
    sh2amp <fod> <dirs300> <tmp_amp> ; mrmath <tmp_amp> max <peak> -axis 3

Dry-run by default (house rule): prints the exact argv it WOULD run.
``--apply`` executes. Argv-only subprocess, no shell (white-box style,
same as track.py). Writes <case>/fidelity/fod_peak.nii.gz and, per bank,
<case>/fidelity/<bank_id>.fidelity.npz (Task 6): provenance-complete banks
get ratio arrays; incomplete banks get a bit2-flagged refusal sidecar.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from tractlab import derivation as dv
from tractlab.casepath import resolve_case_path

MRTRIX = os.path.expanduser("~/mrtrix3/bin")
N_DIRS = 300


def build_argvs(out_dir: Path, fod_path: Path) -> tuple[list[list[str]], Path, Path]:
    fod = fod_path
    peak = out_dir / "fod_peak.nii.gz"
    tmp = out_dir / ".amp300.tmp.nii.gz"
    dirs = out_dir / ".dirs300.txt"
    argvs = [
        [f"{MRTRIX}/dirgen", str(N_DIRS), str(dirs), "-cartesian", "-force", "-quiet"],
        [f"{MRTRIX}/sh2amp", str(fod), str(dirs), str(tmp), "-force", "-quiet"],
        [f"{MRTRIX}/mrmath", str(tmp), "max", str(peak), "-axis", "3", "-force", "-quiet"],
    ]
    # nibabel cannot read .mif: the sidecar step reads a converted scratch copy,
    # while the sha pin stays on the manifest FOD itself (fidelity.py refuses on
    # mismatch there, so the conversion can never stand in for identity).
    fod_load = fod
    if fod.suffix == ".mif":
        fod_load = out_dir / ".fod.tmp.nii.gz"
        argvs.append(
            [f"{MRTRIX}/mrconvert", str(fod), str(fod_load), "-force", "-quiet"]
        )
    return argvs, peak, fod_load


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case-root", required=True)
    ap.add_argument("--apply", action="store_true",
                    help="execute (default: print the argv plan)")
    args = ap.parse_args(argv)

    root = Path(args.case_root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    # Anchor derivation.py's path helpers (artifact_dir, active_inputs) at the
    # directory this invocation actually operates on. The manifest's own
    # recorded case_root may name a different (canonical, symlink-target)
    # worktree location — that cross-worktree redirect concern is handled by
    # the containment checks in build_connectome.py / make_scalar_maps.sh,
    # not here; this script has always resolved every path against its own
    # --case-root, and outputs must land under that same tree.
    manifest["case_root"] = str(root)
    try:
        dv.validate(manifest)
    except dv.DerivationError as e:
        print(f"REFUSE: manifest failed derivation.validate(): {e}", file=sys.stderr)
        return 1
    try:
        inputs = dv.active_inputs(manifest)
    except dv.DerivationError as e:
        print(f"REFUSE: {e}", file=sys.stderr)
        return 1
    fod_entry = inputs.get("fod")
    if not isinstance(fod_entry, dict) or not isinstance(fod_entry.get("path"), str):
        print("REFUSE: manifest has no inputs.fod path", file=sys.stderr)
        return 1
    fod_rel = fod_entry["path"]
    try:
        fod_path = Path(resolve_case_path(str(root), fod_rel, what="fod"))
    except ValueError as e:
        print(f"REFUSE: {e}", file=sys.stderr)
        return 1
    if not fod_path.is_file():
        print(f"REFUSE: FOD missing on disk ({fod_rel})", file=sys.stderr)
        return 1

    fod_is_mif = fod_path.suffix == ".mif"
    # ADR-0007: the active lineage's own fidelity subtree — root/fidelity/
    # only for the first (or legacy single) lineage; derivations/<id>/fidelity/
    # for any other active lineage, never a flat root/fidelity/ write.
    out_dir = dv.artifact_dir(manifest, "fidelity")
    argvs, peak, fod_load = build_argvs(out_dir, fod_path)
    if not args.apply:
        for a in argvs:
            print(" ".join(a))
        print(f"\nDRY-RUN: would write {peak}. Re-run with --apply.", file=sys.stderr)
        return 0

    peak.parent.mkdir(parents=True, exist_ok=True)
    try:
        for a in argvs:
            subprocess.run(a, check=True)
    finally:
        for scratch in (peak.parent / ".amp300.tmp.nii.gz",
                        peak.parent / ".dirs300.txt"):
            if scratch.exists():
                scratch.unlink()
    print(f"WROTE {peak}")

    # ── Task 6: per-bank sidecars ────────────────────────────────────────────
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tractlab.bank import provenance_from_manifest
    from tractlab.fidelity import build_sidecar, save_sidecar
    from tractlab.grid import load_grid, unknown_units_assumption_from_manifest
    from tractlab.recovery import load_lesion_bool

    grid = None
    assume_unknown = unknown_units_assumption_from_manifest(manifest)
    lesion = None
    b0 = root / "nifti/b0.nii.gz"
    if b0.is_file():
        grid = load_grid(str(b0), require_mm=True, require_authoritative_affine=True,
                         assume_unknown_spatial_units_mm=assume_unknown)
        les_entry = inputs.get("lesion")
        if isinstance(les_entry, dict) and isinstance(les_entry.get("path"), str):
            try:
                les_path = Path(
                    resolve_case_path(str(root), les_entry["path"], what="lesion")
                )
            except ValueError:
                print("REFUSE: lesion path escapes case_root", file=sys.stderr)
                return 1
            if les_path.is_file():
                lesion = load_lesion_bool(str(les_path), grid)

    wrote = 0
    try:
        for key, meta in inputs.items():
            if not key.startswith("bank_") or not isinstance(meta, dict):
                continue
            rel = meta.get("path")
            if not isinstance(rel, str):
                print(f"SKIP {key}: no bank file", file=sys.stderr)
                continue
            try:
                bank_path = Path(resolve_case_path(str(root), rel, what=key))
            except ValueError as e:
                print(f"SKIP {key}: {e}", file=sys.stderr)
                continue
            if not bank_path.is_file():
                print(f"SKIP {key}: no bank file", file=sys.stderr)
                continue
            try:
                prov = provenance_from_manifest(meta)
            except ValueError as e:
                # Malformed provenance still gets a bit2 refusal sidecar — a
                # missing sidecar would read downstream as "untested", not
                # "provenance-incomplete" (Grok batch-4 #4).
                print(f"REFUSED {key}: malformed provenance — {e}", file=sys.stderr)
                prov = None
            data = build_sidecar(
                bank_path, fod_path, peak,
                provenance=prov, lesion_mask=lesion, grid=grid,
                sh_load_path=fod_load if fod_is_mif else None,
                assume_unknown_spatial_units_mm=assume_unknown,
            )
            out = peak.parent / f"{key}.fidelity.npz"
            save_sidecar(out, data)
            state = "ratios" if data["ratios_present"] else "REFUSAL-FLAGGED (bit2)"
            print(f"WROTE {out.name} [{state}] n={data['flags'].shape[0]}")
            wrote += 1
    finally:
        if fod_is_mif and fod_load.exists():
            fod_load.unlink()
    print(f"sidecars written: {wrote}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
