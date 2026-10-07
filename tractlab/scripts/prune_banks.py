#!/usr/bin/env python3
"""Prune all manifest bank_*.tck files: length + spatial + SIFT2 floor.

Usage (from repo root):
  PYTHONPATH=src python scripts/prune_banks.py \\
    --case-root ~/tractlab-data/cases/local-case \\
    --manifest cases/local-case/manifest.json

Backs up pre-prune .tck to tracts/bank/raw/ once. Overwrites bank paths with
pruned extracts. Regenerates sibling .sift2.txt when FOD is available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

# repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tractlab.prune import PruneParams, prune_tck_file  # noqa: E402
from tractlab.sift2_util import ensure_sift2, sift2_path_for_tck  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--fod", default=None, help="wmfod for SIFT2 regen")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--length-lo", type=float, default=5.0)
    ap.add_argument("--length-hi", type=float, default=95.0)
    ap.add_argument("--spatial-k", type=float, default=3.0)
    ap.add_argument("--sift2-rel", type=float, default=0.05,
                    help="Drop SIFT2 weight < this × median (outlier junk only)")
    ap.add_argument(
        "--sift2-regen",
        action="store_true",
        help="Regenerate tcksift2 weights on pruned banks (slow on large n)",
    )
    ap.add_argument(
        "--sift2-regen-max-n",
        type=int,
        default=15000,
        help="Auto-regen SIFT2 only when pruned n ≤ this (default 15000)",
    )
    args = ap.parse_args()

    case = Path(os.path.expanduser(args.case_root)).resolve()
    man_path = Path(args.manifest).resolve()
    with open(man_path) as f:
        man = json.load(f)

    fod = args.fod
    if fod is None:
        cand = case / "nifti/connectome/model/pre/wmfod_lmax8_norm.mif"
        if cand.is_file():
            fod = str(cand)

    params = PruneParams(
        length_lo_pct=args.length_lo,
        length_hi_pct=args.length_hi,
        spatial_mad_k=args.spatial_k,
        spatial_max_pct=99.0,
        sift2_rel_floor=args.sift2_rel,
        sift2_drop_pct=0.0,
        min_keep=32,
    )
    raw_dir = case / "tracts/bank/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    inputs = man.get("inputs") or {}
    results = []
    for key, meta in sorted(inputs.items()):
        if not key.startswith("bank_") or not isinstance(meta, dict):
            continue
        rel = meta.get("path")
        if not isinstance(rel, str):
            continue
        tck = case / rel
        if not tck.is_file():
            print(f"SKIP missing {key}: {tck}")
            continue

        backup = raw_dir / tck.name
        if not backup.is_file():
            if args.dry_run:
                print(f"would backup {tck.name} → raw/")
            else:
                shutil.copy2(tck, backup)
                print(f"backup {tck.name} → raw/")

        # always prune from raw backup so re-runs are idempotent
        src = backup if backup.is_file() else tck
        w_src = Path(sift2_path_for_tck(str(src)))
        if not w_src.is_file():
            # sibling next to live bank
            alt = Path(sift2_path_for_tck(str(tck)))
            w_src = alt if alt.is_file() else w_src

        out_w = sift2_path_for_tck(str(tck))
        print(f"=== {key} ({tck.name}) ===")
        if args.dry_run:
            print(f"  would prune {src} → {tck}")
            continue

        rep = prune_tck_file(
            str(src),
            str(tck),
            weights_path=str(w_src) if w_src.is_file() else None,
            out_weights_path=out_w if w_src.is_file() else None,
            params=params,
        )
        print(
            f"  n {rep.n_in} → {rep.n_out} "
            f"(len {rep.n_length}, spat {rep.n_spatial}, sift {rep.n_sift2})"
            + (" REFUSED" if rep.refused else "")
        )
        print(f"  L=[{rep.length_lo_mm:.1f},{rep.length_hi_mm:.1f}] mm"
              + (f" spat≤{rep.spatial_cut_mm:.1f}mm" if rep.spatial_cut_mm else "")
              + (f" sift≥{rep.sift2_floor:.4g}" if rep.sift2_floor is not None else ""))

        # Stale weight files must not outlive a different streamline set
        if Path(out_w).is_file() and not rep.refused:
            Path(out_w).unlink()
            print("  removed stale .sift2.txt")

        do_sift = (
            fod
            and not rep.refused
            and (args.sift2_regen or rep.n_out <= args.sift2_regen_max_n)
        )
        if do_sift:
            got = ensure_sift2(str(tck), fod, force=True, timeout_s=900)
            print(f"  SIFT2 {'ok' if got else 'skip/fail'}")
        elif fod and not rep.refused:
            print(
                f"  SIFT2 deferred (n={rep.n_out} > {args.sift2_regen_max_n}; "
                f"pass --sift2-regen to force)"
            )

        meta["n_streamlines"] = int(rep.n_out if not rep.refused else rep.n_in)
        meta["bytes"] = int(tck.stat().st_size)
        meta["sha256"] = _sha256(tck)
        note = meta.get("note") or ""
        # rewrite prune note (idempotent re-runs from raw/)
        base_note = note.split(" Pruned P")[0].rstrip()
        prune_tag = (
            f" Pruned P{params.length_lo_pct:g}–P{params.length_hi_pct:g} length + "
            f"spatial MAD×{params.spatial_mad_k:g}"
            + (f" + SIFT2≥{params.sift2_rel_floor:g}×med" if params.sift2_rel_floor else "")
            + f" (n={meta['n_streamlines']})."
        )
        meta["note"] = (base_note + prune_tag).strip()
        eng = meta.get("engine") or ""
        eng = eng.replace(" | pruned outliers", "").rstrip()
        meta["engine"] = eng + " | pruned outliers"
        results.append((key, rep))

    if not args.dry_run:
        with open(man_path, "w") as f:
            json.dump(man, f, indent=2)
            f.write("\n")
        print(f"\nUpdated {man_path}")

    print("\nSummary:")
    for key, rep in results:
        pct = 100.0 * (1 - rep.n_out / rep.n_in) if rep.n_in else 0
        print(f"  {key:28s} {rep.n_in:6d} → {rep.n_out:6d}  (-{pct:.0f}%)"
              + (" REFUSED" if rep.refused else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
