"""iFOD2 tracking wrapper — WHITE-BOX (owner reads every line).

This module shells out to ``tckgen`` on a painted seed. Security + honesty rules:

  * argv list only — never a shell string, never a client-supplied path or flag.
    The caller passes numbers (validated/bounded upstream) and file paths that the
    server resolved from the case manifest, never from the request body.
  * Every run gets a unique temp dir; nothing is shared between requests.
  * A hard timeout kills the whole process group (tckgen spawns threads).
  * Typed outcomes — a zero result is NOT the same as a failure:
      OK            rc==0, one valid tckinfo count, decoded count agrees
                    -> counts real
      ENGINE_ERROR  rc!=0 / missing / unparseable tckinfo  -> counts None
      TIMEOUT       killed by the deadline                  -> counts None
    Callers must never render TIMEOUT/ENGINE_ERROR as "tract absent".
  * run.json records the full argv + RNG seed + nthreads for reproducibility.

Determinism note: interactive tracking runs multithreaded and is therefore a
RNG-seeded *stochastic* sample, not bit-reproducible (MRtrix core/math/rng.h).
Deterministic acceptance fixtures must pass nthreads=0. See DESIGN v2 #3.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field, asdict
from enum import Enum

import numpy as np

from .connectotomy import reject_cavity_role

TCKGEN = os.path.expanduser("~/mrtrix3/bin/tckgen")
TCKINFO = os.path.expanduser("~/mrtrix3/bin/tckinfo")


class Outcome(str, Enum):
    OK = "ok"
    ENGINE_ERROR = "engine_error"
    TIMEOUT = "timeout"


# Live Commit knobs (commercial FA-threshold analogue is FOD amplitude -cutoff).
# Client may set cutoff/angle/minlength within these ranges; seeds/select only via density.
CUTOFF_DEFAULT = 0.05
CUTOFF_MIN = 0.02
CUTOFF_MAX = 0.20
ANGLE_DEFAULT = 45.0
MINLENGTH_DEFAULT = 20.0

# Named density presets — server maps label → seeds/select (client never sets raw seeds).
DENSITY_PRESETS: dict[str, tuple[int, int]] = {
    "sparse": (8_000, 800),
    "normal": (20_000, 1_500),
    "dense": (50_000, 1_500),
}


@dataclass(frozen=True)
class TrackParams:
    cutoff: float = CUTOFF_DEFAULT  # FOD amplitude (NOT tensor FA)
    angle: float = ANGLE_DEFAULT    # degrees, true iFOD2 knob
    minlength: float = MINLENGTH_DEFAULT  # mm
    maxlength: float = 250.0    # mm
    seeds: int = 20_000         # measured ~1.6s at this budget
    select: int = 1_500         # == display cap; no thinning needed
    nthreads: int = 4           # 0 => deterministic (slow); >0 => stochastic
    rng_seed: int = 42
    density: str = "normal"     # provenance label for seeds/select

    def validated(self) -> "TrackParams":
        if not (CUTOFF_MIN <= self.cutoff <= CUTOFF_MAX):
            raise ValueError(
                f"cutoff out of range [{CUTOFF_MIN}, {CUTOFF_MAX}] "
                f"(FOD amplitude, not FA)"
            )
        if not (0.0 < self.angle <= 90.0):
            raise ValueError("angle out of range")
        if not (0.0 < self.minlength < self.maxlength <= 500.0):
            raise ValueError("length bounds invalid")
        if not (0 < self.seeds <= 5_000_000):
            raise ValueError("seeds out of range")
        if not (0 < self.select <= 100_000):
            raise ValueError("select out of range")
        if not (0 <= self.nthreads <= 16):
            raise ValueError("nthreads out of range")
        if self.density not in DENSITY_PRESETS:
            raise ValueError(f"unknown density: {self.density!r}")
        return self

    def live_hash(self, *, seed_preset: str | None = None) -> str:
        """Provenance fingerprint of live Commit knobs (not bank recipeHash)."""
        payload = {
            "cutoff": round(self.cutoff, 4),
            "angle": round(self.angle, 2),
            "minlength": round(self.minlength, 2),
            "maxlength": round(self.maxlength, 2),
            "density": self.density,
            "seeds": self.seeds,
            "select": self.select,
            "seed_preset": seed_preset or "",
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class TrackResult:
    outcome: Outcome
    tck_path: str | None
    n_accepted: int | None
    wall_s: float
    argv: list[str]
    warning: str | None = None
    run_json: dict = field(default_factory=dict)


def _tckinfo_count(tck_path: str) -> int | None:
    """Return one nonnegative tckinfo count, or None if it is unverified."""
    try:
        out = subprocess.run(
            [TCKINFO, tck_path], capture_output=True, text=True, timeout=30
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0:
        return None
    stdout = out.stdout
    if isinstance(stdout, bytes):
        stdout = stdout.decode(errors="replace")
    if not isinstance(stdout, str):
        return None
    counts: list[int] = []
    for line in stdout.splitlines():
        s = line.strip()
        if s.startswith("count:"):
            token = s.split(":", 1)[1].strip()
            # ``int`` accepts signs and whitespace forms that are not a
            # trustworthy MRtrix count.  Keep the receipt grammar exact.
            if not token or not token.isdecimal():
                return None
            try:
                count = int(token)
            except (TypeError, ValueError, OverflowError):
                return None
            if count < 0:
                return None
            counts.append(count)
    if len(counts) != 1:
        return None
    return counts[0]


def _decoded_tck_count(tck_path: str) -> int | None:
    """Decode a completed TCK and return its verified population count.

    ``tckinfo`` is a useful receipt, but it is not the population itself.  A
    successful tracking result must agree with the rows that the server can
    actually decode and later publish/export.
    """
    try:
        import nibabel as nib

        streamlines = nib.streamlines.load(tck_path).streamlines
        count = 0
        for track in streamlines:
            arr = np.asarray(track, dtype=np.float32)
            if (
                arr.ndim != 2
                or arr.shape[1] != 3
                or arr.shape[0] < 2
                or not np.isfinite(arr).all()
            ):
                return None
            count += 1
        return count
    except (OSError, ValueError, TypeError, IndexError, ImportError, RuntimeError):
        return None
    except Exception:
        # Nibabel may raise a format-specific header/EOF exception. Any
        # undecodable output is an engine refusal, never an empty population.
        return None


def run_tckgen(
    fod_path: str,
    seed_path: str,
    mask_path: str,
    out_dir: str,
    params: TrackParams,
    timeout_s: float = 30.0,
    include_paths: list[str] | None = None,
    exclude_path: str | None = None,
    on_proc: Callable | None = None,
    seed_role: str | None = None,
    include_roles: list[str] | None = None,
    exclude_role: str | None = None,
) -> TrackResult:
    """Run one iFOD2 track job. Caller owns out_dir (unique per request).

    Optional Boolean ROI flags (server-owned paths only):
      include_paths → repeated ``-include`` (AND regions then optional OR union)
      exclude_path  → one ``-exclude`` (NOT union)
    on_proc(proc) is called immediately after Popen so the server can register
    the process for /api/cancel (kill process group).

    ``seed_role`` / ``include_roles`` / ``exclude_role`` are the typed-mask
    wall: a cavity-typed mask is never a tracking input (raises CavityRoleError
    before tckgen is spawned).
    """
    reject_cavity_role(seed_role, exclude_role, *(include_roles or []))
    params = params.validated()
    os.makedirs(out_dir, exist_ok=True)
    tck = os.path.join(out_dir, "track.tck")

    argv = [
        TCKGEN, fod_path, tck,
        "-algorithm", "iFOD2",
        "-seed_image", seed_path,
        "-mask", mask_path,
        "-select", str(params.select),
        "-seeds", str(params.seeds),
        "-minlength", str(params.minlength),
        "-maxlength", str(params.maxlength),
        "-cutoff", str(params.cutoff),
        "-angle", str(params.angle),
        "-nthreads", str(params.nthreads),
        "-force", "-quiet",
    ]
    for p in include_paths or []:
        argv.extend(["-include", p])
    if exclude_path:
        argv.extend(["-exclude", exclude_path])
    env = dict(os.environ, MRTRIX_RNG_SEED=str(params.rng_seed))

    t0 = time.time()
    proc = subprocess.Popen(argv, env=env, start_new_session=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if on_proc is not None:
        try:
            on_proc(proc)
        except Exception:
            pass
    try:
        _, err = proc.communicate(timeout=timeout_s)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            proc.communicate()
        wall = time.time() - t0
        return TrackResult(Outcome.TIMEOUT, None, None, wall, argv,
                           warning=f"tracking exceeded {timeout_s}s and was killed")
    wall = time.time() - t0
    # cancelled externally: non-zero rc after SIGTERM, no tck
    if rc is not None and rc != 0 and not os.path.exists(tck):
        msg = (err or b"").decode(errors="replace")[:400]
        if "killed" in msg.lower() or rc in (-15, -9, 143, 137):
            return TrackResult(Outcome.TIMEOUT, None, None, wall, argv,
                               warning="tracking cancelled",
                               run_json={"cancelled": True, "returncode": rc})

    run_json = {
        "argv": argv, "rng_seed": params.rng_seed, "nthreads": params.nthreads,
        "wall_s": round(wall, 3), "returncode": rc,
        "params": asdict(params),
        "deterministic": params.nthreads == 0,
    }

    if rc != 0:
        return TrackResult(Outcome.ENGINE_ERROR, None, None, wall, argv,
                           warning=(err or b"").decode(errors="replace")[:400],
                           run_json=run_json)

    count = _tckinfo_count(tck)
    if count is None:
        return TrackResult(Outcome.ENGINE_ERROR, tck, None, wall, argv,
                           warning="tckinfo failed or unparseable", run_json=run_json)

    decoded_count = _decoded_tck_count(tck)
    run_json.update({
        "tckinfoCount": count,
        "decodedCount": decoded_count,
        "countVerified": decoded_count is not None and decoded_count == count,
    })
    if decoded_count is None:
        return TrackResult(
            Outcome.ENGINE_ERROR,
            tck,
            None,
            wall,
            argv,
            warning="decoded TCK population is malformed or unavailable",
            run_json=run_json,
        )
    if decoded_count != count:
        return TrackResult(
            Outcome.ENGINE_ERROR,
            tck,
            None,
            wall,
            argv,
            warning=(
                f"tckinfo count {count} does not match decoded TCK count "
                f"{decoded_count}"
            ),
            run_json=run_json,
        )

    with open(os.path.join(out_dir, "run.json"), "w") as f:
        json.dump(run_json, f, indent=2)

    warning = None
    if decoded_count == 0:
        warning = "no streamlines accepted for this seed/budget (not tract absence)"
    return TrackResult(Outcome.OK, tck, decoded_count, wall, argv,
                       warning=warning, run_json=run_json)
