#!/usr/bin/env python3
"""Backfill bank provenance blocks into a case manifest (E0, Task 2).

Dry-run by default (house rule): prints the provenance blocks it WOULD write.
``--apply`` rewrites manifest.json ATOMICALLY (temp file + os.replace in the
same directory); the document is re-serialized (indent=2), but the only keys
that change semantically are ``provenance`` under ``bank_*`` entries.

Honesty rules (Sol+Grok batch-2 review):
- Recipe params are stamped ONLY when the bank's ``engine`` string matches the
  house ACT iFOD2 recipe — never guessed onto an unknown recipe.
- Banks that already carry provenance are left alone unless ``--refresh``.
- A manifest-declared bank whose file is missing ABORTS (partial truth is
  worse than no backfill); malformed entries are reported, never silent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

from tractlab import derivation as dv
from tractlab.casepath import resolve_case_path

# House tckgen flags — scripts/run_demo_d0_banks.sh §4 (10M ACT iFOD2 corpus).
# Applied per-bank only when the engine string proves the recipe lineage.
HOUSE_ENGINE_SIGNATURE = "ACT iFOD2"
RECIPE_PARAMS: dict[str, object] = {
    "algorithm": "iFOD2",
    "act": True,
    "cutoff": 0.06,
}

# Fields the recipe never recorded — honest absence, with the reason.
ABSENT_FIELDS: dict[str, str] = {
    "step_mm": "absent:not set in recipe; iFOD2 tool default (0.5x voxel)",
    "downsample": "absent:not set in recipe; iFOD2 tool default",
    "mrtrix_version": "absent:not recorded at build",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_backfill(
    case_root: str,
    recipe_commit: str,
    manifest: dict | None = None,
    refresh: bool = False,
) -> dict[str, dict]:
    """Return {bank_id: provenance_dict} for every on-disk bank in the manifest."""
    root = Path(case_root).resolve()
    if manifest is None:
        manifest = json.loads((root / "manifest.json").read_text())
    inputs = dv.active_inputs(manifest)

    fod_sha: str | None = None
    fod_source = "absent:manifest has no inputs.fod path"
    fod_entry = inputs.get("fod")
    if isinstance(fod_entry, dict) and isinstance(fod_entry.get("path"), str):
        try:
            fod_path = Path(
                resolve_case_path(str(root), fod_entry["path"], what="fod")
            )
        except ValueError as e:
            fod_source = f"absent:{e}"
        else:
            if fod_path.is_file():
                fod_sha = _sha256(fod_path)
                fod_source = "recorded"
            else:
                fod_source = f"absent:fod file missing on disk ({fod_entry['path']})"

    out: dict[str, dict] = {}
    for key, meta in inputs.items():
        if not key.startswith("bank_"):
            continue
        if not isinstance(meta, dict) or not isinstance(meta.get("path"), str):
            print(f"MALFORMED {key}: entry is not a dict with a string path — "
                  f"not backfilled", file=sys.stderr)
            continue
        if "provenance" in meta and not refresh:
            print(f"KEPT {key}: existing provenance (use --refresh to recompute)",
                  file=sys.stderr)
            continue
        bank_path = Path(
            resolve_case_path(str(root), meta["path"], what=key)
        )
        if not bank_path.is_file():
            # Manifest says the bank exists; a missing file is an anomaly, not
            # a skippable row — abort rather than backfill a partial truth.
            raise FileNotFoundError(f"{key}: bank file missing ({meta['path']})")

        engine = str(meta.get("engine") or "")
        if HOUSE_ENGINE_SIGNATURE in engine:
            params: dict[str, object] = dict(RECIPE_PARAMS)
            param_sources = {
                k: f"recipe-script@{recipe_commit}" for k in RECIPE_PARAMS
            }
        else:
            params = {k: None for k in RECIPE_PARAMS}
            param_sources = {
                k: f"absent:engine {engine!r} does not match house recipe "
                   f"({HOUSE_ENGINE_SIGNATURE})"
                for k in RECIPE_PARAMS
            }

        prov: dict = {
            "bank_sha256": _sha256(bank_path),
            "fod_sha256": fod_sha,
            **params,
            **{k: None for k in ABSENT_FIELDS},
            "sources": {
                "bank_sha256": "recorded",
                "fod_sha256": fod_source,
                **param_sources,
                **ABSENT_FIELDS,
            },
        }
        out[key] = prov
    return out


def _atomic_write_json(path: Path, obj: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".manifest-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case-root", required=True)
    ap.add_argument(
        "--recipe-commit",
        default="4885ff3",
        help="commit id of the recipe scripts the params were read from",
    )
    ap.add_argument(
        "--refresh",
        action="store_true",
        help="recompute banks that already carry provenance (default: keep)",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="write provenance blocks into manifest.json (default: dry-run print)",
    )
    args = ap.parse_args(argv)

    root = Path(args.case_root).resolve()
    man_path = root / "manifest.json"
    manifest = json.loads(man_path.read_text())  # single snapshot, hashed AND written
    # Anchor derivation.py's path helpers at this invocation's own case
    # directory — see prep_fidelity.py for the same convention/rationale.
    manifest["case_root"] = str(root)
    try:
        dv.validate(manifest)
        dv.validate_active_artifact_metadata(manifest)
    except dv.DerivationError as e:
        print(f"REFUSE: manifest failed validation: {e}", file=sys.stderr)
        return 1
    try:
        blocks = compute_backfill(
            args.case_root,
            recipe_commit=args.recipe_commit,
            manifest=manifest,
            refresh=args.refresh,
        )
    except dv.DerivationError as e:
        print(f"REFUSE: {e}", file=sys.stderr)
        return 1

    # Conformance gate: everything we emit must parse under the Task 1 schema.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tractlab.bank import provenance_from_manifest

    for bank_id, prov in blocks.items():
        provenance_from_manifest({"provenance": prov})  # raises on drift

    if not args.apply:
        print(json.dumps(blocks, indent=2))
        print(f"\nDRY-RUN: {len(blocks)} bank(s) would gain provenance. "
              f"Re-run with --apply to write.", file=sys.stderr)
        return 0

    active_inputs = dv.active_inputs(manifest)  # live reference — write lands in the active lineage
    for bank_id, prov in blocks.items():
        active_inputs[bank_id]["provenance"] = prov
    _atomic_write_json(man_path, manifest)
    print(f"APPLIED provenance to {len(blocks)} bank(s) in {man_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
