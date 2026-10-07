"""SIFT2 weights for bank tracts (optional quantitative colour / weighted p5)."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import numpy as np

from .evidence_identity import file_signature, sha256_file

TCKSIFT2 = os.path.expanduser("~/mrtrix3/bin/tcksift2")


def sift2_path_for_tck(tck_path: str) -> str:
    return str(Path(tck_path).with_suffix("")) + ".sift2.txt"


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def is_sha256(value: object) -> bool:
    """Return whether *value* is a syntactically valid SHA-256 digest."""
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def load_sift2_weights(
    path: str,
    n_expected: int | None = None,
    *,
    require_binding: bool = False,
    expected_bank_sha256: str | None = None,
    expected_weight_sha256: str | None = None,
) -> np.ndarray | None:
    """Load finite nonnegative weights with an optional identity receipt.

    The legacy unbound mode remains available to offline pruning callers. A
    publication path must pass ``require_binding=True`` and both hashes. The
    bank digest is checked by that caller against the bytes it loaded; this
    function checks its format, the weight-file digest, and that the file is
    stable across hashing and numeric parsing.
    """
    p = Path(path)
    if not p.is_file():
        return None
    if require_binding and (
        not is_sha256(expected_bank_sha256)
        or not is_sha256(expected_weight_sha256)
    ):
        return None
    if (expected_bank_sha256 is None) != (expected_weight_sha256 is None):
        # A one-sided receipt cannot establish which population the weights
        # belong to, even for callers that have not requested strict mode.
        return None
    if expected_bank_sha256 is not None and not is_sha256(expected_bank_sha256):
        return None
    if expected_weight_sha256 is not None and not is_sha256(expected_weight_sha256):
        return None
    try:
        before = file_signature(p)
        actual_weight_sha = sha256_file(p)
        if (
            expected_weight_sha256 is not None
            and actual_weight_sha.lower() != expected_weight_sha256.lower()
        ):
            return None
        w = np.loadtxt(str(p), dtype=np.float64)
        after = file_signature(p)
    except (OSError, ValueError, TypeError, EOFError):
        return None
    if before != after:
        return None
    if w.ndim != 1:
        w = w.ravel()
    if n_expected is not None and len(w) != n_expected:
        return None
    if not np.isfinite(w).all() or (w < 0).any():
        return None
    return np.asarray(w, dtype=np.float64)


def ensure_sift2(
    tck_path: str,
    fod_path: str,
    *,
    force: bool = False,
    timeout_s: int = 600,
) -> str | None:
    """Run tcksift2 if weights missing. Returns weights path or None on failure."""
    out = sift2_path_for_tck(tck_path)
    if not force and Path(out).is_file() and Path(out).stat().st_size > 0:
        return out
    if not Path(tck_path).is_file() or not Path(fod_path).is_file():
        return None
    if not Path(TCKSIFT2).is_file():
        return None
    try:
        subprocess.run(
            [TCKSIFT2, tck_path, fod_path, out, "-force", "-nthreads", "4"],
            check=True,
            timeout=timeout_s,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None
    return out if Path(out).is_file() else None


def weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """q in [0,100]."""
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    w = np.clip(w, 0, None)
    if w.sum() <= 0:
        return float(np.percentile(values, q))
    cw = np.cumsum(w)
    cw /= cw[-1]
    return float(np.interp(q / 100.0, cw, v))
