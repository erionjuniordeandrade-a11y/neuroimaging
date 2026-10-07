#!/usr/bin/env python3
"""Build the C1b assigned-edge connectome (tck2connectome) for the demo case.

Runs on the 100k ACT subsample by default (pilot; ~119 MB). The 10M corpus
(10.5 GB) only runs behind --full — never implicitly, per the case brief.

Prints wall time, unassigned fraction, matrix shape, and exports three named
edges as a smoke test (two of the highest-count cells + one zero cell),
reporting the independent count check (matrix cell / assignment-row count /
extracted-tck count) for each.

Usage:
  ~/fsl/bin/python scripts/build_connectome.py
  ~/fsl/bin/python scripts/build_connectome.py --full
  ~/fsl/bin/python scripts/build_connectome.py --case-root /path/to/case
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tractlab import connectome as cx  # noqa: E402

# cases/<id>/{tracts,normative,nifti,...} are symlinks into the canonical
# checkout (see CLAUDE.md); resolve_case_path's containment check needs the
# manifest's own recorded case_root (the canonical, non-symlinked location),
# not this worktree's cases/ path — same convention TrackService uses in
# serve.py (`root = os.path.realpath(man["case_root"])`).
DEFAULT_CASE_DIR = REPO_ROOT / "cases" / "demo-leipzig-sub-010005"
PARC_KEY = "parc_schaefer200_yeo7"


class CaseLoadError(ValueError):
    """A ``--case-root`` directory failed the pre-flight checks below."""


def _load_case(case_root_arg: str | None) -> tuple[str, str, str]:
    """Return (case_id, case_root, manifest_path) read from
    ``<case_dir>/manifest.json``.

    ``case_dir`` is the directory passed via ``--case-root`` (default: the
    demo case). ``case_root`` is that manifest's own ``case_root`` field —
    the canonical, non-symlinked location cx.build reads from directly (same
    convention as the historical hard-coded demo-case constant) — but it
    must actually identify ``case_dir`` itself (or a location nested inside
    it): a manifest is untrusted input, and letting its own case_root field
    silently redirect execution to an unrelated directory the caller never
    named is a path-confusion hole, not a feature. Raises ``CaseLoadError``
    rather than exiting so callers (main, tests) control the exit code and
    message.

    ``manifest_path`` is the exact file this function read — the house
    indirection convention (manifest in the repo, data root elsewhere) means
    it is NOT always ``<case_root>/manifest.json``; this function passes it
    on explicitly to ``cx.build`` rather than letting that function guess
    wrong and refuse "manifest.json not found".
    """
    case_dir = Path(case_root_arg).resolve() if case_root_arg else DEFAULT_CASE_DIR
    manifest_path = case_dir / "manifest.json"
    if not manifest_path.is_file():
        raise CaseLoadError(f"no manifest.json under case root {case_dir}")
    manifest = json.loads(manifest_path.read_text())
    case_root = manifest.get("case_root")
    if not isinstance(case_root, str) or not case_root:
        raise CaseLoadError(f"manifest at {manifest_path} has no case_root")
    case_id = manifest.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise CaseLoadError(f"manifest at {manifest_path} has no case_id")

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
    return case_id, case_root, str(manifest_path)


def _top_off_diagonal_cells(matrix: list[list[float]], k: int = 2) -> list[tuple[int, int, int]]:
    n = len(matrix)
    cells = [
        (int(round(matrix[i][j])), i + 1, j + 1)
        for i in range(n) for j in range(i + 1, n)
    ]
    cells.sort(reverse=True)
    return cells[:k]


def _one_zero_cell(matrix: list[list[float]]) -> tuple[int, int, int] | None:
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            if int(round(matrix[i][j])) == 0:
                return (0, i + 1, j + 1)
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case-root", type=str, default=None,
                     help="directory containing the case's manifest.json (default: the demo case)")
    ap.add_argument("--full", action="store_true",
                     help="use the 10M corpus (10.5 GB; can take hours) instead of the 100k pilot")
    ap.add_argument("--yes", action="store_true",
                     help="required together with --full — confirms you mean to run the 10M corpus")
    ap.add_argument("--radius-mm", type=float, default=cx.DEFAULT_RADIUS_MM)
    args = ap.parse_args(argv)

    try:
        case_id, case_root, manifest_path = _load_case(args.case_root)
    except CaseLoadError as e:
        print(f"REFUSE: {e}", file=sys.stderr)
        return 2

    if args.full and not args.yes:
        print("Refusing to run the FULL 10M corpus (10.5 GB, hours of walltime) without --yes.")
        print("Re-run with: scripts/build_connectome.py --full --yes")
        return 2

    corpus_key = cx.WHOLEBRAIN_10M if args.full else cx.WHOLEBRAIN_100K
    out_dir = cx.connectome_out_dir(case_root)

    if args.full:
        print("CONFIRMED (--full --yes): running the FULL 10M corpus — this can take HOURS.")

    print(f"case_id:    {case_id}")
    print(f"case_root:  {case_root}")
    print(f"manifest:   {manifest_path}")
    print(f"corpus:     {corpus_key}")
    print(f"            ({'FULL 10M corpus — this can take HOURS' if args.full else '100k pilot subsample'})")
    print(f"out_dir:    {out_dir}")
    print(f"radius_mm:  {args.radius_mm}")
    print()

    result = cx.build(
        case_root, corpus_key, PARC_KEY, out_dir, radius_mm=args.radius_mm, manifest_path=manifest_path,
    )
    print(f"outcome:    {result.outcome.value}")
    print(f"wall_s:     {result.wall_s:.2f}")
    if result.outcome is not cx.Outcome.OK:
        print(f"error:      {result.error}")
        return 1

    prov = result.provenance
    matrix = cx.read_matrix(result.matrix_path)
    n = len(matrix)
    print(f"matrix shape:        {n}x{n}")
    print(f"n_streamlines:       {prov['n_streamlines']}")
    print(f"n_assigned:          {prov['n_assigned']}")
    print(f"unassigned_fraction: {prov['unassigned_fraction']:.4f}")
    print(f"mrtrix_version:      {prov['mrtrix_version']}")
    print(f"weighting:           {prov['weighting']}  (sift2: {prov['sift2']})")
    print(f"corpus sha256:       {prov['corpus']['sha256']}")
    print(f"parcellation sha256: {prov['parcellation']['sha256']}")

    top2 = _top_off_diagonal_cells(matrix, k=2)
    zero = _one_zero_cell(matrix)
    smoke = list(top2) + ([zero] if zero else [])

    print("\nsmoke test — three named edges (matrix cell / assignment-row count / extracted-tck count):")
    all_ok = True
    for _matrix_count, a, b in smoke:
        try:
            edge = cx.edge_streamlines(out_dir, a, b)
        except (cx.ConnectomeInvalid, cx.ConnectomeEngineError, cx.ConnectomeCountMismatch) as e:
            print(f"  edge {a}-{b}: FAILED — {e}")
            all_ok = False
            continue
        print(f"  edge {a}-{b}: matrix={edge['matrix_count']} "
              f"assignment_rows={edge['assignment_row_count']} "
              f"extracted={edge['extracted_count']}  OK -> {edge['tck_path_rel']}")

    print(f"\nsmoke test: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
