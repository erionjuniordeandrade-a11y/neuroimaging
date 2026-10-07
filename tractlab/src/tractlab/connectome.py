"""C1b — traceable assigned edges (tck2connectome / connectome2tck).

An **assigned edge** is a patient streamline tagged with a parcel pair by
``tck2connectome`` (ADR-0003: the tag is an assignment label, never cortex
identity). This module builds one auditable connectome layer per case:

  * a matrix from a declared corpus (whole-brain ACT .tck) + a declared
    parcellation (Schaefer-200/Yeo-7, fail-closed on unsigned QC — reuses
    parcellation.py's discovery rules, keyed by the exact input the caller
    asked for)
  * every streamline's node assignment saved (``-out_assignments``), so any
    matrix cell is traceable back to its exact streamlines
  * lesion/cavity intersection of an edge's streamlines, reusing
    connectotomy.streamline_hits_cavity — the SAME predicate C1 uses, so an
    edge's cut count can never silently diverge from the bank layer's.

WHITE-BOX like track.py: argv lists only, no shell, timed subprocess calls,
typed outcomes. Never invoked from a GET route — building is a script-owned
job; routes only read what a prior build already published.

No SIFT2 weights exist for the whole-brain ACT corpus this ships against:
every matrix here is raw streamline counts. ``weighting`` in the provenance
record says so explicitly — never silently implied.

Publication is atomic: each build() writes into a fresh temp directory under
``out_dir/generations/``, writes provenance.json LAST, renames the temp dir
into its permanent generation slot, then atomically swaps the ``current``
symlink onto it (a single inode rename). A reader always sees either the
prior generation or the new one in full, never a mix; a crash or a refused
build leaves ``current`` untouched.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .atlas_fetch import load_schaefer_lut, sha256_file
from .bank import load_tracks_cached
from .casepath import resolve_case_path
from .connectotomy import CavityInvalid, streamline_hits_cavity
from .grid import load_grid
from .parcellation import discover_parcellation, load_label_volume
from .recovery import load_lesion_bool
from . import derivation as dv

TCK2CONNECTOME = os.path.expanduser("~/mrtrix3/bin/tck2connectome")
CONNECTOME2TCK = os.path.expanduser("~/mrtrix3/bin/connectome2tck")
TCKINFO = os.path.expanduser("~/mrtrix3/bin/tckinfo")

DEFAULT_RADIUS_MM = 4.0
CONNECTOME_SUBDIR = os.path.join("work", "connectome")
GENERATIONS_SUBDIR = "generations"
CURRENT_LINK = "current"

# The 10M full corpus must never run implicitly — 10.5 GB, hours of walltime.
WHOLEBRAIN_100K = "tracts/connectome/pre/freesurfer/path-to-wm-v2/wholebrain_act_ifod2_100000.tck"
WHOLEBRAIN_10M = "tracts/connectome/pre/freesurfer/path-to-wm-v2/wholebrain_act_ifod2_10000000.tck"

# Per-(generation, a, b) locks so concurrent requests for the SAME edge
# serialize their connectome2tck run instead of racing each other — each
# extraction also writes to a unique temp file and os.replace()s it into
# place, so even without this lock a reader can never see a half-written
# file, but the lock avoids duplicate wasted subprocess runs. A process-wide
# registry (never persisted, never shared across worker processes) is
# correct here: the server this guards is single-process.
#
# Bounded: each entry counts its current users (holders + waiters) under
# _EDGE_LOCKS_GUARD and is evicted when the last user leaves, so the map
# only ever holds edges with a request in flight. A user registers BEFORE
# waiting on the lock, so an entry with a waiter is never evicted and two
# requests for the same key can never hold two different locks.
_EDGE_LOCKS_GUARD = threading.Lock()


class _EdgeLockEntry:
    __slots__ = ("lock", "users")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.users = 0


_EDGE_LOCKS: dict[tuple[str, int, int], _EdgeLockEntry] = {}


@contextmanager
def _edge_lock(generation_dir: str, a: int, b: int):
    key = (generation_dir, a, b)
    with _EDGE_LOCKS_GUARD:
        entry = _EDGE_LOCKS.get(key)
        if entry is None:
            entry = _EDGE_LOCKS[key] = _EdgeLockEntry()
        entry.users += 1
    try:
        with entry.lock:
            yield
    finally:
        with _EDGE_LOCKS_GUARD:
            entry.users -= 1
            if entry.users == 0 and _EDGE_LOCKS.get(key) is entry:
                del _EDGE_LOCKS[key]


class Outcome(str, Enum):
    OK = "ok"
    INVALID = "invalid"        # refused before/without running the engine
    ENGINE_ERROR = "engine_error"
    TIMEOUT = "timeout"


class ConnectomeInvalid(ValueError):
    """Invalid evidence: grid mismatch, unsigned parcellation QC, missing
    manifest input, malformed matrix/assignment contents, or a build whose
    recorded hashes / active_derivation / QC signature no longer match the
    current manifest (stale build). Never conflated with "not built".
    """


class ConnectomeEngineError(RuntimeError):
    """tck2connectome/connectome2tck exited non-zero, produced no output, or
    an independent count check (tckinfo) was unavailable — a missing
    cross-check is refused, never silently skipped.
    """


class ConnectomeCountMismatch(RuntimeError):
    """Two independently-computed counts for the same edge disagree — a
    server bug, must propagate loudly, never silently reconciled.
    """


@dataclass
class BuildResult:
    outcome: Outcome
    out_dir: str
    matrix_path: str | None
    assignments_path: str | None
    provenance_path: str | None
    provenance: dict[str, Any] | None
    wall_s: float
    argv: list[str] = field(default_factory=list)
    error: str | None = None


def connectome_out_dir(case_root: str) -> str:
    return os.path.join(os.path.realpath(case_root), CONNECTOME_SUBDIR)


# ---------------------------------------------------------------------------
# Content identity: sha256 cached by (dev, ino, size, mtime_ns) stamp, on
# disk, so a re-run (or a route re-validating a 10.5 GB corpus) does not
# reread bytes unless the file actually changed.
# ---------------------------------------------------------------------------

def _stat_stamp(path: Path) -> list[int]:
    st = path.stat()
    return [st.st_dev, st.st_ino, st.st_size, int(st.st_mtime_ns)]


def sha256_cached(path: str | Path, cache_file: str | Path) -> str:
    path = Path(path)
    cache_file = Path(cache_file)
    stamp = _stat_stamp(path)
    cache: dict[str, Any] = {}
    if cache_file.is_file():
        try:
            cache = json.loads(cache_file.read_text())
        except (OSError, ValueError):
            cache = {}
    key = str(path.resolve())
    entry = cache.get(key)
    if isinstance(entry, dict) and entry.get("stamp") == stamp and isinstance(entry.get("sha256"), str):
        return entry["sha256"]
    value = sha256_file(path)
    cache[key] = {"stamp": stamp, "sha256": value}
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cache))
    except OSError:
        pass  # cache is a speed optimization only; never fatal
    return value


def _record_path(root: str, path: str) -> str:
    """``path`` relative to ``root`` when nested under it; absolute otherwise.

    Every other provenance path (corpus, parcellation, matrix, assignments)
    is safely case-root-relative because ``resolve_case_path`` / parcellation
    discovery enforce containment. The manifest is the one path that is NOT
    guaranteed to be nested under case_root (a case may keep its manifest in
    the repo while its data root lives elsewhere) — falling back to an
    absolute path here, instead of a ``../../..``-laden relative one or a
    silently wrong case-root-relative guess, keeps the recorded location
    unambiguous regardless of where the manifest actually lives.
    """
    real_path = os.path.realpath(path)
    real_root = os.path.realpath(root)
    try:
        rel = os.path.relpath(real_path, real_root)
    except ValueError:
        return real_path
    return real_path if rel.startswith("..") else rel


# The repository's case directory (``<repo>/cases``). A case whose data lives
# outside the repo keeps its manifest here; recording it relative to this dir
# keeps provenance valid across checkouts/worktrees of the same repo.
REPO_CASES_DIR = str(Path(__file__).resolve().parents[2] / "cases")
_MANIFEST_BASE_REPO_CASES = "repo_cases"


def _record_manifest(root: str, mpath: str) -> dict[str, str]:
    """Provenance record for the manifest location (sha added by caller).

    Nested under case_root -> case-root-relative (historical form). Under the
    repo's ``cases/`` dir -> relative to it, tagged ``relative_to:
    "repo_cases"`` so it resolves against whichever checkout reads it. Anywhere
    else -> absolute (unchanged fallback).
    """
    recorded = _record_path(root, mpath)
    if not os.path.isabs(recorded):
        return {"path": recorded}
    rel = _record_path(REPO_CASES_DIR, mpath)
    if not os.path.isabs(rel):
        return {"path": rel, "relative_to": _MANIFEST_BASE_REPO_CASES}
    return {"path": recorded}


def _resolve_manifest_path(root: str, record: dict[str, Any]) -> str:
    """Reverse of ``_record_manifest``: rebuild the manifest's real path.

    Legacy records (no ``relative_to``) keep their historical meaning:
    absolute stays absolute (a missing file then fails closed in
    ``check_fresh``); relative resolves against case_root; no ``path`` at all
    (provenance written before the field existed) is
    ``<case_root>/manifest.json``. Records are never rewritten here.
    """
    recorded = record.get("path")
    base = record.get("relative_to")
    if base is None:
        if not recorded:
            return os.path.join(root, "manifest.json")
        return recorded if os.path.isabs(recorded) else os.path.join(root, recorded)
    if base != _MANIFEST_BASE_REPO_CASES:
        raise ConnectomeInvalid(
            f"connectome provenance: unknown manifest relative_to {base!r}"
        )
    if not isinstance(recorded, str) or not recorded or os.path.isabs(recorded):
        raise ConnectomeInvalid(
            "connectome provenance: repo_cases manifest path must be a relative path"
        )
    try:
        return resolve_case_path(REPO_CASES_DIR, recorded, what="manifest")
    except ValueError as e:
        raise ConnectomeInvalid(f"connectome provenance: {e}") from e


def _mrtrix_version(binary: str = TCK2CONNECTOME) -> str | None:
    try:
        out = subprocess.run([binary, "-version"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    first = (out.stdout or "").splitlines()
    return first[0].strip() if first else None


# ---------------------------------------------------------------------------
# Evidence contract: assignments.txt rows and matrix.csv cells are parsed
# through exactly these functions everywhere in the module — a malformed
# row/cell is a typed ConnectomeInvalid, never silently skipped or counted.
# ---------------------------------------------------------------------------

def _parse_assignment_line(line: str, lineno: int) -> tuple[int, int]:
    parts = line.split()
    if len(parts) != 2:
        raise ConnectomeInvalid(
            f"malformed assignments.txt line {lineno}: expected 2 columns, "
            f"got {len(parts)} ({line!r})"
        )
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as e:
        raise ConnectomeInvalid(
            f"malformed assignments.txt line {lineno}: non-integer node id ({line!r})"
        ) from e


def _iter_assignment_rows(assignments_path: str):
    with open(assignments_path) as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            yield lineno, _parse_assignment_line(line, lineno)


def _assignment_stats(assignments_path: str) -> tuple[int, int, float]:
    """(n_streamlines, n_assigned, unassigned_fraction).

    A row is "unassigned" if either endpoint is node 0 (radial search found
    no parcel within range for that endpoint) — matches connectome2tck's own
    "node 0 = unassigned" convention. Raises ConnectomeInvalid on any
    malformed row rather than skipping it.
    """
    n = 0
    n_unassigned = 0
    for _lineno, (x, y) in _iter_assignment_rows(assignments_path):
        n += 1
        if x == 0 or y == 0:
            n_unassigned += 1
    n_assigned = n - n_unassigned
    frac = (n_unassigned / n) if n else 0.0
    return n, n_assigned, frac


def _assignment_row_count(assignments_path: str, a: int, b: int) -> int:
    target = {a, b}
    count = 0
    for _lineno, (x, y) in _iter_assignment_rows(assignments_path):
        if {x, y} == target:
            count += 1
    return count


def read_matrix(matrix_path: str) -> list[list[float]]:
    """Parse matrix.csv, refusing (ConnectomeInvalid) anything that isn't a
    square matrix of finite, integer-valued (raw streamline count) cells.
    """
    rows: list[list[float]] = []
    with open(matrix_path) as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                values = [float(x) for x in line.split(",")]
            except ValueError as e:
                raise ConnectomeInvalid(
                    f"malformed matrix.csv line {lineno}: non-numeric value"
                ) from e
            rows.append(values)
    n = len(rows)
    for i, row in enumerate(rows, start=1):
        if len(row) != n:
            raise ConnectomeInvalid(
                f"matrix.csv is not square: {n} rows but row {i} has {len(row)} columns"
            )
        for v in row:
            if not np.isfinite(v):
                raise ConnectomeInvalid(f"matrix.csv row {i} has a non-finite value")
            if abs(v - round(v)) > 1e-6:
                raise ConnectomeInvalid(
                    f"matrix.csv row {i} has a non-integer value {v} "
                    "(raw counts must be integers)"
                )
    return rows


def _matrix_cell(matrix_path: str, a: int, b: int) -> int:
    matrix = read_matrix(matrix_path)
    n = len(matrix)
    if not (1 <= a <= n) or not (1 <= b <= n):
        raise ConnectomeInvalid(f"node id out of range for a {n}x{n} matrix: {a},{b}")
    return int(round(matrix[a - 1][b - 1]))


def _tck_content_sha256(tck_path: str) -> str:
    """sha256 of a .tck file's binary DATA ONLY — never the ASCII header.

    ``connectome2tck`` writes a fresh MRtrix invocation line into the header
    on every extraction (confirmed empirically: two back-to-back extractions
    of the identical edge produce different whole-file sha256 but identical
    data-segment sha256), so the whole-file hash is not usable as a stable
    identity for "did this edge's evidence change" — only the data segment
    is. The header's own ``file: . <offset>`` field names the exact byte
    offset the binary track data starts at (mrtrix pads between the ``END``
    marker and that offset, so searching for ``END\\n`` and adding 4 is
    wrong); this reads that field directly rather than guessing.
    """
    with open(tck_path, "rb") as f:
        data = f.read()
    marker = b"\nfile: . "
    idx = data.find(marker)
    if idx >= 0:
        start = idx + len(marker)
    elif data.startswith(b"file: . "):
        start = len(b"file: . ")
    else:
        raise ConnectomeEngineError(f"{tck_path}: no 'file: .' header offset field found")
    end = data.find(b"\n", start)
    if end < 0:
        raise ConnectomeEngineError(f"{tck_path}: malformed 'file: .' header line")
    try:
        offset = int(data[start:end].strip())
    except ValueError as e:
        raise ConnectomeEngineError(f"{tck_path}: non-integer data offset in 'file: .' header") from e
    if not (0 < offset <= len(data)):
        raise ConnectomeEngineError(f"{tck_path}: data offset {offset} out of range for a {len(data)}-byte file")
    return hashlib.sha256(data[offset:]).hexdigest()


def _tckinfo_count(tck_path: str) -> int | None:
    try:
        out = subprocess.run([TCKINFO, tck_path], capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0:
        return None
    for line in out.stdout.splitlines():
        s = line.strip()
        if s.startswith("count:"):
            try:
                return int(s.split(":", 1)[1])
            except ValueError:
                return None
    return None


def _rm(path: str) -> None:
    """Best-effort remove — cleanup only, never fatal to the caller's own error."""
    try:
        os.remove(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Publication: build into a fresh temp dir, provenance written last, then
# rename-into-slot + atomic symlink swap. A reader through _resolve_current
# always sees a complete generation or the previous one — never a mix.
# ---------------------------------------------------------------------------

def _fail(
    outcome: Outcome,
    out_dir: str,
    t0: float,
    error: str,
    *,
    argv: list[str] | None = None,
    tmp_dir: str | None = None,
) -> BuildResult:
    if tmp_dir is not None:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return BuildResult(outcome, out_dir, None, None, None, None,
                        time.time() - t0, argv=argv or [], error=error)


def _publish(tmp_dir: str, final_gen_dir: str, out_dir: str) -> None:
    """Atomically make ``tmp_dir`` the new ``current`` generation.

    Two OS-atomic operations: rename the finished temp build into its
    permanent generation slot, then swap the ``current`` symlink onto it via
    ``os.replace`` (a single inode rename). A crash/exception at any point
    leaves the previously-published generation exactly as it was — nothing
    here ever edits an existing generation in place. A separate function so
    tests can simulate an interrupted publish by monkeypatching this seam.
    """
    os.rename(tmp_dir, final_gen_dir)
    link = os.path.join(out_dir, CURRENT_LINK)
    tmp_link = os.path.join(out_dir, f".current-{uuid.uuid4().hex[:8]}")
    os.symlink(os.path.relpath(final_gen_dir, out_dir), tmp_link)
    os.replace(tmp_link, link)


def _resolve_current(out_dir: str) -> str:
    """Resolve the currently-published generation directory.

    Raises ConnectomeInvalid("connectome not built...") if nothing has been
    published yet — distinguishable from a "stale" build by message text;
    routes use is_built() for the 404 pre-check and this (via load_provenance
    / check_fresh) for everything after.
    """
    current = os.path.join(out_dir, CURRENT_LINK)
    provenance_path = os.path.join(current, "provenance.json")
    if not os.path.isfile(provenance_path):
        raise ConnectomeInvalid(f"connectome not built: {provenance_path} missing")
    return current


def is_built(out_dir: str) -> bool:
    try:
        _resolve_current(out_dir)
        return True
    except ConnectomeInvalid:
        return False


def pin_generation(out_dir: str) -> str:
    """Resolve the `current` symlink's REALPATH once — the concrete,
    immutable generation directory a single request pins itself to.

    `current` is a mutable symlink: `load_provenance`, `check_fresh`, and
    `artifact_paths` each resolving it independently (as they used to) means
    two reads within the same logical request can silently land on two
    different generations if a rebuild republishes `current` in between.
    Callers that serve one response for one edge (`edge_streamlines`) must
    call this ONCE and pass the result as `generation_dir` to every other
    function below, then re-call this at the end and refuse if it no longer
    matches — see `edge_streamlines`.
    """
    current = _resolve_current(out_dir)  # validates provenance.json exists
    return os.path.realpath(current)


def ensure_still_pinned(out_dir: str, generation_dir: str, *, what: str) -> None:
    """Refuse (typed ConnectomeInvalid) if `current` no longer resolves to
    the generation a request pinned. Call it as the LAST step before a
    response is handed off — after every byte of the response is assembled —
    so a rebuild published mid-request can never be served as "current"."""
    if pin_generation(out_dir) != generation_dir:
        raise ConnectomeInvalid(
            f"connectome generation changed while serving {what} "
            f"(pinned to {os.path.basename(generation_dir)}); refusing a response that could "
            "mix bytes from two generations"
        )


def artifact_paths(out_dir: str, generation_dir: str | None = None) -> dict[str, str]:
    """Resolve one generation's artefact paths — the pinned `generation_dir`
    when given (see `pin_generation`), else `current` resolved fresh."""
    current = generation_dir or _resolve_current(out_dir)
    return {
        "dir": current,
        "matrix": os.path.join(current, "matrix.csv"),
        "assignments": os.path.join(current, "assignments.txt"),
        "provenance": os.path.join(current, "provenance.json"),
        "edges": os.path.join(current, "edges"),
    }


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(
    case_root: str,
    corpus_key: str,
    parc_key: str,
    out_dir: str,
    radius_mm: float = DEFAULT_RADIUS_MM,
    timeout_s: float = 1800.0,
    manifest_path: str | None = None,
) -> BuildResult:
    """Run tck2connectome once; publish matrix.csv + assignments.txt +
    provenance.json as a new generation under ``out_dir`` (atomic — see
    module docstring).

    ``corpus_key`` is a case-root-relative path to the .tck (e.g.
    ``WHOLEBRAIN_100K``); ``parc_key`` is the manifest ``inputs`` key for the
    parcellation (e.g. ``"parc_schaefer200_yeo7"``) — discovery uses exactly
    this key (never a hardcoded default), and refuses if it is not a
    declared input. Refuses (typed ``Outcome.INVALID``, never a crash) if the
    parcellation is unavailable / QC-unsigned, or if its grid disagrees with
    the corpus's reference grid (the case's canonical DWI grid,
    ``inputs.mask``).

    ``manifest_path`` defaults to ``<case_root>/manifest.json`` only when
    nothing is passed. The house convention for a case whose data lives
    outside the repo (the private case keeps its manifest under
    ``cases/<id>/manifest.json`` in the repo while ``manifest["case_root"]``
    points at the data directory elsewhere) means the manifest is NOT always
    inside ``case_root`` — callers that resolved the manifest from a
    ``--case-root`` selection pass that exact path here rather than letting
    this function guess wrong and refuse "manifest.json not found".
    """
    t0 = time.time()
    root = os.path.realpath(case_root)
    mpath = os.path.realpath(manifest_path) if manifest_path else os.path.join(root, "manifest.json")
    if not os.path.isfile(mpath):
        return _fail(Outcome.INVALID, out_dir, t0, f"manifest.json not found at {mpath}")
    with open(mpath) as f:
        man = json.load(f)
    inputs = dv.active_inputs(man)

    try:
        corpus_path = resolve_case_path(root, corpus_key, what="corpus")
    except ValueError as e:
        return _fail(Outcome.INVALID, out_dir, t0, str(e))
    if not os.path.isfile(corpus_path):
        return _fail(Outcome.INVALID, out_dir, t0, f"corpus not found: {corpus_path}")

    if parc_key not in inputs:
        return _fail(Outcome.INVALID, out_dir, t0,
                      f"manifest.inputs.{parc_key} is not a declared parcellation input")
    parc = discover_parcellation(root, inputs, man, key=parc_key)
    if parc is None:
        return _fail(Outcome.INVALID, out_dir, t0,
                      f"parcellation input {parc_key!r} unavailable or QC unsigned "
                      "(parcellation_qc.approved_by/date/sheet_sha)")

    mask_meta = inputs.get("mask")
    if not isinstance(mask_meta, dict) or "path" not in mask_meta:
        return _fail(Outcome.INVALID, out_dir, t0,
                      "manifest.inputs.mask missing (corpus reference grid)")
    try:
        mask_path = resolve_case_path(root, mask_meta["path"], what="mask")
        ref_grid = load_grid(mask_path)
        load_label_volume(parc.path, ref_grid)  # raises ValueError on shape/affine mismatch
    except ValueError as e:
        return _fail(Outcome.INVALID, out_dir, t0,
                      f"parcellation grid mismatch vs corpus reference grid (inputs.mask): {e}")

    os.makedirs(os.path.join(out_dir, GENERATIONS_SUBDIR), exist_ok=True)
    gen_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    tmp_dir = os.path.join(out_dir, GENERATIONS_SUBDIR, f".tmp-{gen_id}")
    os.makedirs(tmp_dir)

    matrix_path = os.path.join(tmp_dir, "matrix.csv")
    assignments_path = os.path.join(tmp_dir, "assignments.txt")
    argv = [
        TCK2CONNECTOME, corpus_path, parc.path, matrix_path,
        "-assignment_radial_search", str(radius_mm),
        "-out_assignments", assignments_path,
        "-symmetric", "-zero_diagonal",
        "-force", "-quiet",
    ]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return _fail(Outcome.TIMEOUT, out_dir, t0,
                      f"tck2connectome exceeded {timeout_s}s and was killed",
                      argv=argv, tmp_dir=tmp_dir)
    wall = time.time() - t0
    if proc.returncode != 0:
        return _fail(Outcome.ENGINE_ERROR, out_dir, t0, (proc.stderr or "")[:800],
                      argv=argv, tmp_dir=tmp_dir)
    if not (os.path.isfile(matrix_path) and os.path.isfile(assignments_path)):
        return _fail(Outcome.ENGINE_ERROR, out_dir, t0,
                      "tck2connectome exited 0 but outputs are missing",
                      argv=argv, tmp_dir=tmp_dir)

    try:
        n_streamlines, n_assigned, unassigned_fraction = _assignment_stats(assignments_path)
        read_matrix(matrix_path)  # validates square/finite/integer; raises on corruption
    except ConnectomeInvalid as e:
        return _fail(Outcome.ENGINE_ERROR, out_dir, t0,
                      f"tck2connectome produced malformed output: {e}",
                      argv=argv, tmp_dir=tmp_dir)

    cache_file = os.path.join(out_dir, ".sha256_cache.json")
    corpus_sha = sha256_cached(corpus_path, cache_file)
    parc_sha = sha256_cached(parc.path, cache_file)
    lut_sha = sha256_cached(parc.lut_path, cache_file)
    matrix_sha = sha256_cached(matrix_path, cache_file)
    assignments_sha = sha256_cached(assignments_path, cache_file)
    manifest_sha = sha256_cached(mpath, cache_file)
    lut = load_schaefer_lut(Path(parc.lut_path))
    node_order = [{"id": i, "name": lut[i]["name"]} for i in sorted(lut)]

    provenance = {
        "case_root": root,
        "case_id": man.get("case_id"),
        "active_derivation": man.get("active_derivation"),
        # The manifest is not always nested under case_root (the repo-manifest
        # / elsewhere-data-root indirection) — recorded relative to case_root
        # when nested there, relative to the repo's cases/ dir when kept in
        # the repo (never a worktree-specific absolute path), absolute
        # otherwise. check_fresh() resolves this same record to re-read the
        # manifest live, never guessing "<case_root>/manifest.json".
        "manifest": {**_record_manifest(root, mpath), "sha256": manifest_sha},
        "corpus": {
            "key": corpus_key,
            "path": os.path.relpath(corpus_path, root),
            "sha256": corpus_sha,
        },
        "parcellation": {
            "key": parc_key,
            "path": os.path.relpath(parc.path, root),
            "sha256": parc_sha,
            "lut_path": os.path.relpath(parc.lut_path, root),
            "lut_sha256": lut_sha,
            "label": parc.label,
            "n_parcels": parc.n_parcels,
            "n_networks": parc.n_networks,
        },
        "matrix": {"path": "matrix.csv", "sha256": matrix_sha},
        "assignments": {"path": "assignments.txt", "sha256": assignments_sha},
        "radius_mm": float(radius_mm),
        "mrtrix_version": _mrtrix_version(),
        "n_streamlines": n_streamlines,
        "n_assigned": n_assigned,
        "unassigned_fraction": unassigned_fraction,
        "node_order": node_order,
        "weighting": "none (raw counts)",
        "assignment_method": "radial_search",
        "sift2": "none — no SIFT2 weights exist for this corpus; raw streamline counts only",
        "built_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "wall_s": round(wall, 3),
        "gen_id": gen_id,
        "argv": argv,
        "label_note": "ASSIGNED — parcel tag on a patient streamline, not cortex identity (ADR-0003)",
    }
    # Provenance last: publication (the rename + symlink swap below) is the
    # only thing that makes this generation visible to any reader.
    provenance_path = os.path.join(tmp_dir, "provenance.json")
    with open(provenance_path, "w") as f:
        json.dump(provenance, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    final_gen_dir = os.path.join(out_dir, GENERATIONS_SUBDIR, gen_id)
    try:
        _publish(tmp_dir, final_gen_dir, out_dir)
    except OSError as e:
        # tmp_dir may or may not have been renamed to final_gen_dir already;
        # clean up whichever one still exists as an orphan. Either way the
        # PREVIOUS "current" generation was never touched by this branch.
        shutil.rmtree(tmp_dir, ignore_errors=True)
        shutil.rmtree(final_gen_dir, ignore_errors=True)
        return _fail(Outcome.ENGINE_ERROR, out_dir, t0, f"failed to publish build: {e}", argv=argv)

    return BuildResult(
        Outcome.OK, out_dir,
        os.path.join(final_gen_dir, "matrix.csv"),
        os.path.join(final_gen_dir, "assignments.txt"),
        os.path.join(final_gen_dir, "provenance.json"),
        provenance, wall, argv=argv, error=None,
    )


# ---------------------------------------------------------------------------
# Read back / invalidate
# ---------------------------------------------------------------------------

def load_provenance(out_dir: str, generation_dir: str | None = None) -> dict[str, Any]:
    current = generation_dir or _resolve_current(out_dir)
    with open(os.path.join(current, "provenance.json")) as f:
        return json.load(f)


def _canonical_manifest_hash(man: dict[str, Any]) -> str:
    """Content hash of a manifest dict, independent of key order or on-disk
    formatting — so an in-memory dict and a freshly-read file can be compared
    for CONTENT equality without requiring byte-identical serialization."""
    canonical = json.dumps(man, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def check_fresh(
    provenance: dict[str, Any],
    out_dir: str,
    *,
    service_manifest: dict[str, Any] | None = None,
    generation_dir: str | None = None,
) -> None:
    """Fail closed if ANYTHING the build depended on has drifted since:
    corpus or parcellation bytes, the parcellation's LUT, the case's active
    derivation, the parcellation QC signature (re-run the same
    parcellation.py discovery rule at serve time, not just re-hashed), the
    published matrix.csv / assignments.txt themselves, or (when
    `service_manifest` is given) the manifest a server is ACTUALLY serving
    in memory no longer matching the one on disk this check just read — a
    caller (serve.py) holding a stale `service.manifest` after a derivation
    rebuild must never pair fresh tubes with a stale lesion/grid resolved
    from that stale copy. Names the stale artefact in the message.

    `generation_dir`, when given (see `pin_generation`), pins the published
    generation this check reads matrix.csv/assignments.txt from — never a
    fresh `current` resolution that could have moved since the caller
    pinned it.
    """
    case_root = provenance.get("case_root")
    if not case_root or not os.path.isdir(case_root):
        raise ConnectomeInvalid("connectome provenance: case_root missing or gone")
    # The manifest is not always nested under case_root (repo-manifest /
    # elsewhere-data-root indirection) — resolve the SAME path build()
    # recorded, never assume "<case_root>/manifest.json".
    manifest_prov = provenance.get("manifest") or {}
    manifest_path = _resolve_manifest_path(case_root, manifest_prov)
    if not os.path.isfile(manifest_path):
        raise ConnectomeInvalid(f"connectome provenance: manifest.json missing ({manifest_path})")
    cache_file = os.path.join(out_dir, ".sha256_cache.json")
    with open(manifest_path) as f:
        man = json.load(f)
    inputs = dv.active_inputs(man)

    corpus_prov = provenance.get("corpus") or {}
    corpus_path = os.path.join(case_root, corpus_prov.get("path", ""))
    if not os.path.isfile(corpus_path):
        raise ConnectomeInvalid(f"stale build: corpus missing ({corpus_prov.get('path')})")
    if sha256_cached(corpus_path, cache_file) != corpus_prov.get("sha256"):
        raise ConnectomeInvalid(f"stale build: corpus changed since build ({corpus_prov.get('path')})")

    if provenance.get("active_derivation") != man.get("active_derivation"):
        raise ConnectomeInvalid(
            "stale build: active_derivation changed since build "
            f"(built against {provenance.get('active_derivation')!r}, "
            f"manifest now {man.get('active_derivation')!r})"
        )

    parc_prov = provenance.get("parcellation") or {}
    parc_key = parc_prov.get("key", "parc_schaefer200_yeo7")
    parc = discover_parcellation(case_root, inputs, man, key=parc_key)
    if parc is None:
        raise ConnectomeInvalid(
            f"stale build: parcellation QC no longer signed or inputs.{parc_key} "
            "no longer discoverable"
        )

    # Manifest snapshot enforcement: after the specific, diagnostic checks
    # above (active_derivation, QC re-discovery) have had their chance to
    # name exactly what drifted, the manifest FILE ITSELF must still match
    # what build() recorded byte for byte — a catch-all for any OTHER field
    # drift those checks don't independently re-validate (e.g.
    # parcellation_qc.sheet_sha changed to a different, still-non-empty
    # value passes "QC still signed" untouched, but is a real edit to the
    # file this generation was built against).
    if sha256_cached(manifest_path, cache_file) != manifest_prov.get("sha256"):
        raise ConnectomeInvalid(f"stale build: manifest.json changed since build ({manifest_prov.get('path')})")
    # Second check (kept immediately after the disk-hash check): the
    # server's in-memory copy vs. the disk manifest just verified above —
    # catches a stale service.manifest surviving a derivation rebuild, never
    # pairing fresh tubes with a stale lesion/grid resolved from that stale
    # in-memory copy.
    if service_manifest is not None and _canonical_manifest_hash(man) != _canonical_manifest_hash(service_manifest):
        raise ConnectomeInvalid(
            "stale build: the server's in-memory manifest no longer matches the manifest "
            "on disk — refusing to pair this edge's tubes with a manifest that may name a "
            "different lesion or grid (re-open the case)"
        )

    if sha256_cached(parc.path, cache_file) != parc_prov.get("sha256"):
        raise ConnectomeInvalid(f"stale build: parcellation changed since build ({parc_prov.get('path')})")
    if sha256_cached(parc.lut_path, cache_file) != parc_prov.get("lut_sha256"):
        raise ConnectomeInvalid(
            f"stale build: parcellation LUT changed since build ({parc_prov.get('lut_path')})"
        )

    current = generation_dir or _resolve_current(out_dir)
    matrix_prov = provenance.get("matrix") or {}
    matrix_path = os.path.join(current, "matrix.csv")
    if not os.path.isfile(matrix_path):
        raise ConnectomeInvalid("stale build: matrix.csv missing from the published generation")
    if sha256_cached(matrix_path, cache_file) != matrix_prov.get("sha256"):
        raise ConnectomeInvalid("stale build: matrix.csv changed since build")

    assignments_prov = provenance.get("assignments") or {}
    assignments_path = os.path.join(current, "assignments.txt")
    if not os.path.isfile(assignments_path):
        raise ConnectomeInvalid("stale build: assignments.txt missing from the published generation")
    if sha256_cached(assignments_path, cache_file) != assignments_prov.get("sha256"):
        raise ConnectomeInvalid("stale build: assignments.txt changed since build")


# ---------------------------------------------------------------------------
# Per-edge traceability
# ---------------------------------------------------------------------------

def edge_streamlines(
    out_dir: str,
    a: int,
    b: int,
    timeout_s: float = 600.0,
    service_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract the exact streamlines assigned to edge (a, b); cross-check the
    count three independent ways (matrix cell, assignment-row count, tckinfo
    on the extracted file). Any disagreement raises ConnectomeCountMismatch;
    an UNAVAILABLE independent count (tckinfo missing/failed/timed out) is
    itself a typed ConnectomeEngineError — never a silent success with a
    null extracted count.

    Generation pinning: `current` is resolved to a concrete, immutable
    generation directory ONCE (`pin_generation`) at the top of this call and
    that same path is threaded through `load_provenance`, `check_fresh`, and
    `artifact_paths` — never re-resolved independently by each. After
    extraction, `current` is re-resolved and compared against the pin; if a
    rebuild republished a new generation while this call was running, the
    response is refused (typed `ConnectomeInvalid`) rather than served as if
    it were still "current".

    `service_manifest`, when given, is compared inside `check_fresh` against
    the manifest this call reads live from disk — see `check_fresh`.
    """
    gen_dir = pin_generation(out_dir)
    provenance = load_provenance(out_dir, generation_dir=gen_dir)
    check_fresh(provenance, out_dir, service_manifest=service_manifest, generation_dir=gen_dir)
    paths = artifact_paths(out_dir, generation_dir=gen_dir)
    matrix_path = paths["matrix"]
    assignments_path = paths["assignments"]
    corpus_path = os.path.join(provenance["case_root"], provenance["corpus"]["path"])
    cache_file = os.path.join(out_dir, ".sha256_cache.json")

    matrix_count = _matrix_cell(matrix_path, a, b)
    row_count = _assignment_row_count(assignments_path, a, b)

    edges_dir = paths["edges"]
    os.makedirs(edges_dir, exist_ok=True)
    prefix = os.path.join(edges_dir, f"{a}_{b}")
    tck_path = f"{prefix}.tck"

    # Serialize concurrent extractions of the SAME (generation, a, b) — and,
    # regardless of the lock, always write into a unique temp file first and
    # os.replace() it into the canonical tck_path: a concurrent reader can
    # never observe a half-written file, lock or no lock.
    with _edge_lock(gen_dir, a, b):
        tmp_prefix = os.path.join(edges_dir, f".{a}_{b}.{uuid.uuid4().hex[:8]}.tmp")
        tmp_tck = f"{tmp_prefix}.tck"
        argv = [
            CONNECTOME2TCK, corpus_path, assignments_path, tmp_prefix,
            "-nodes", f"{a},{b}", "-exclusive", "-files", "single",
            "-force", "-quiet",
        ]
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _rm(tmp_tck)
            raise ConnectomeEngineError(f"connectome2tck exceeded {timeout_s}s for edge {a}-{b}")
        if proc.returncode != 0 or not os.path.isfile(tmp_tck):
            _rm(tmp_tck)
            raise ConnectomeEngineError(
                f"connectome2tck failed for edge {a}-{b}: {(proc.stderr or '')[:400]}"
            )
        os.replace(tmp_tck, tck_path)  # atomic publish — same filesystem, single inode rename

    extracted_count = _tckinfo_count(tck_path)
    if extracted_count is None:
        raise ConnectomeEngineError(
            f"edge {a}-{b}: tckinfo on the extracted tck failed, timed out, or is "
            "unavailable — the independent count check cannot be skipped"
        )

    if matrix_count != row_count:
        raise ConnectomeCountMismatch(
            f"edge {a}-{b}: matrix cell={matrix_count} != assignment-row count={row_count}"
        )
    if extracted_count != matrix_count:
        raise ConnectomeCountMismatch(
            f"edge {a}-{b}: extracted tck count={extracted_count} != matrix cell={matrix_count}"
        )

    # Whole-file hash (kept for audit / the plain JSON route's tckSha256) is
    # NOT a stable identity — connectome2tck stamps a fresh invocation line
    # into the header on every extraction. content_sha256 (data segment only)
    # is what stays identical across repeated extractions of the same edge;
    # identity checks must key on it, never on tck_sha256.
    tck_sha256 = sha256_cached(tck_path, cache_file)
    content_sha256 = _tck_content_sha256(tck_path)

    # A single provenance snapshot — read once at the top of this call and
    # reused for every field below — so the reported gen_id/corpus/
    # parcellation identity can never mix with a generation published after
    # this snapshot was taken (a route calling load_provenance a second time
    # later would be racing a concurrent rebuild).
    corpus_prov = provenance["corpus"]
    parc_prov = provenance["parcellation"]
    matrix_prov = provenance["matrix"]
    assignments_prov = provenance["assignments"]

    # Assemble-time pin check: refuse if `current` moved since gen_dir was
    # pinned, rather than silently reporting an extraction as "current" when
    # a rebuild has already superseded it.
    ensure_still_pinned(out_dir, gen_dir, what=f"edge {a}-{b}")

    return {
        "a": a,
        "b": b,
        "matrix_count": matrix_count,
        "assignment_row_count": row_count,
        "extracted_count": extracted_count,
        "tck_path": tck_path,
        "tck_path_rel": os.path.relpath(tck_path, provenance["case_root"]),
        "tck_sha256": tck_sha256,
        "content_sha256": content_sha256,
        "gen_id": provenance["gen_id"],
        "radius_mm": provenance["radius_mm"],
        "corpus_path_rel": corpus_prov["path"],
        "corpus_sha256": corpus_prov["sha256"],
        "parcellation_path_rel": parc_prov["path"],
        "parcellation_sha256": parc_prov["sha256"],
        "matrix_path_rel": matrix_prov["path"],
        "matrix_sha256": matrix_prov["sha256"],
        "assignments_path_rel": assignments_prov["path"],
        "assignments_sha256": assignments_prov["sha256"],
        "case_root": provenance["case_root"],
        # Server-internal: the pinned generation directory. Callers that do
        # more work after this returns (lesion hits, tube packing) re-check
        # it with ensure_still_pinned() immediately before responding.
        "generation_dir": gen_dir,
        "argv": argv,
    }


def edge_cavity_hits(
    edge_tck: str,
    cavity_mask_path: str,
    case_root: str,
    manifest: dict[str, Any] | None = None,
    manifest_path: str | None = None,
) -> dict[str, Any]:
    """Cavity ∩ this edge's streamlines, reusing the C1 predicate exactly.

    Validates the cavity NIfTI against the CORPUS's own reference grid
    (manifest ``inputs.mask`` — the same rule build() uses), never against
    itself: checking a mask against a grid derived from that same mask is
    tautological and can never fail.

    ``manifest``, when given, is used directly — no disk read at all (the
    caller, e.g. a serve.py route, already has ``service.manifest`` loaded
    in memory). Otherwise ``manifest_path`` is read (defaulting to
    ``<case_root>/manifest.json`` only when nothing is passed) — the same
    "manifest is not always nested under case_root" allowance as build().
    """
    root = os.path.realpath(case_root)
    if manifest is not None:
        man = manifest
    else:
        mpath = os.path.realpath(manifest_path) if manifest_path else os.path.join(root, "manifest.json")
        if not os.path.isfile(mpath):
            raise ConnectomeInvalid(f"manifest.json not found at {mpath}")
        with open(mpath) as f:
            man = json.load(f)
    inputs = dv.active_inputs(man)
    mask_meta = inputs.get("mask")
    if not isinstance(mask_meta, dict) or "path" not in mask_meta:
        raise ConnectomeInvalid("manifest.inputs.mask missing (corpus reference grid)")
    mask_path = resolve_case_path(root, mask_meta["path"], what="mask")
    ref_grid = load_grid(mask_path)
    try:
        cavity = load_lesion_bool(cavity_mask_path, ref_grid)
    except ValueError as e:
        raise CavityInvalid(str(e)) from e
    lines, _ = load_tracks_cached(edge_tck)
    hits = 0
    total = 0
    for line in lines:
        total += 1
        if streamline_hits_cavity(np.asarray(line, dtype=np.float32), cavity, ref_grid):
            hits += 1
    return {
        "hits": hits,
        "total": total,
        "lesion_sha256": sha256_file(Path(cavity_mask_path)),
    }
