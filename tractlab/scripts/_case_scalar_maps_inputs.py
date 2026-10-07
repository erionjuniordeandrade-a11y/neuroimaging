#!/usr/bin/env python3
"""Small helper for scripts/make_scalar_maps.sh (ADR-0007).

Resolves the ACTIVE lineage's mask + DWI-preproc paths and the profiles
output directory for a case directory — never guesses, never silently
trusts a manifest to redirect execution somewhere the caller never named.

Refuses (REFUSE: ... on stderr, exit 2) when:
  - the case directory has no manifest.json
  - the manifest has no valid case_id or case_root
  - the manifest's own case_root does not identify the given case
    directory (or a location inside it) — the manifest is untrusted input
    and must not be able to redirect execution elsewhere
  - derivation.validate() fails
  - the active lineage's inputs.mask is missing/invalid, or its file is not
    on disk under the case root

On success prints ``key=value`` lines to stdout (one per line, in a fixed
order): case_id, case_root, active_derivation (empty string for the first/
legacy lineage), mask_path, dwi_path, out_dir — all absolute paths, meant
to be read by make_scalar_maps.sh, not humans.

DWI-preproc location is NOT a manifest-declared input (no such key exists
in the schema) — it has always been a fixed-convention path,
work/ss3t/dwi_preproc.mif under the case root, for the first/legacy
lineage. For a non-first active lineage this helper nests that same
convention under its private subtree (derivations/<id>/work/ss3t/...),
so a second lineage's raw preprocessing can never be mistaken for (or
silently overwrite) the first lineage's. The profiles output directory is
derivation.artifact_dir(manifest, "profiles") — the shared layout rule.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tractlab import derivation as dv  # noqa: E402
from tractlab.casepath import resolve_case_path  # noqa: E402


class CaseLoadError(ValueError):
    pass


def _is_first_active(manifest: dict) -> tuple[bool, str | None]:
    """Return (is_first_or_legacy_lineage, active_derivation_id_or_None).

    Mirrors derivation.py's own _is_first_derivation branching (kept local,
    not imported, since that helper is a private module member) — never
    touches derivation.py itself.
    """
    derivs = manifest.get("derivations")
    if not isinstance(derivs, dict) or len(derivs) <= 1:
        return True, None
    active = manifest.get("active_derivation")
    first = next(iter(derivs), None)
    return (active == first), active


def resolve(case_dir_arg: str) -> dict:
    case_dir = Path(case_dir_arg).resolve()
    manifest_path = case_dir / "manifest.json"
    if not manifest_path.is_file():
        raise CaseLoadError(f"no manifest.json under case root {case_dir}")
    manifest = json.loads(manifest_path.read_text())

    case_id = manifest.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise CaseLoadError(f"manifest at {manifest_path} has no case_id")
    case_root = manifest.get("case_root")
    if not isinstance(case_root, str) or not case_root:
        raise CaseLoadError(f"manifest at {manifest_path} has no case_root")

    case_dir_real = Path(os.path.realpath(str(case_dir)))
    case_root_real = Path(os.path.realpath(case_root))
    # House convention (same trust rule as serve.py): the --case-root argument
    # selects the manifest; the manifest's recorded case_root is where the data
    # lives (the private case keeps its manifest in the repo and its imaging
    # under ~/tractlab-data). The recorded root must exist; both are printed.
    if not case_root_real.is_dir():
        raise CaseLoadError(
            f"manifest case_root {case_root!r} (resolved {case_root_real}) does not "
            f"exist as a directory; selected --case-root was {case_dir!r} "
            f"(resolved {case_dir_real})"
        )
    if case_root_real != case_dir_real:
        print(f"note: manifest case_root {case_root_real} differs from selected {case_dir_real} (indirection honoured)", file=sys.stderr)

    try:
        dv.validate(manifest)
    except dv.DerivationError as e:
        raise CaseLoadError(f"manifest failed derivation.validate(): {e}") from e

    try:
        inputs = dv.active_inputs(manifest)
    except dv.DerivationError as e:
        raise CaseLoadError(str(e)) from e

    mask_entry = inputs.get("mask")
    mask_rel = mask_entry.get("path") if isinstance(mask_entry, dict) else None
    if not isinstance(mask_rel, str) or not mask_rel:
        raise CaseLoadError("active lineage inputs.mask.path missing or invalid")
    try:
        mask_path = Path(resolve_case_path(str(case_root_real), mask_rel, what="mask"))
    except ValueError as e:
        raise CaseLoadError(str(e)) from e
    if not mask_path.is_file():
        raise CaseLoadError(f"inputs.mask not on disk under case root: {mask_rel}")

    is_first, active_id = _is_first_active(manifest)
    if is_first:
        dwi_rel = Path("work") / "ss3t" / "dwi_preproc.mif"
    else:
        dwi_rel = Path("derivations") / active_id / "work" / "ss3t" / "dwi_preproc.mif"
    # Output layout comes from the shared ADR-0007 rule (first lineage:
    # work/profiles; later: derivations/<id>/profiles), never a local copy.
    try:
        out_dir = dv.artifact_dir(manifest, "profiles")
    except dv.DerivationError as e:
        raise CaseLoadError(str(e)) from e

    return {
        "case_id": case_id,
        "case_root": str(case_root_real),
        "active_derivation": active_id or "",
        "mask_path": str(mask_path),
        "dwi_path": str(case_root_real / dwi_rel),
        "out_dir": str(out_dir),
    }


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: _case_scalar_maps_inputs.py <case-dir>", file=sys.stderr)
        return 2
    try:
        info = resolve(argv[0])
    except CaseLoadError as e:
        print(f"REFUSE: {e}", file=sys.stderr)
        return 2
    for key in ("case_id", "case_root", "active_derivation", "mask_path", "dwi_path", "out_dir"):
        print(f"{key}={info[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
