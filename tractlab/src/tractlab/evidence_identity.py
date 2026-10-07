"""Actual-bytes identity helpers shared by preflight and serve-time fidelity gates.

The recurring audit finding (S-01/S-05/N-03) is code that compares two
*declared* hash strings (e.g. a manifest field against a sidecar field)
without ever re-hashing the file that is actually being read right now. A
declaration can drift from reality; only the live bytes are ground truth.

``sha256_file`` is memoized by (path, size, mtime_ns) so a hot request path
(fidelity gate on every bank load) does not re-hash a large unchanged volume
on every call, while any real on-disk change — including a same-size
overwrite, since mtime_ns still advances — invalidates the cached digest.
"""
from __future__ import annotations

import hashlib
import os
import threading

_lock = threading.Lock()
_cache: dict[str, tuple[tuple, str]] = {}


def file_signature(path) -> tuple:
    st = os.stat(path)
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def sha256_file(path) -> str:
    """sha256 of the file's current bytes — never a stale declared value."""
    real = os.path.realpath(str(path))
    key = file_signature(real)
    with _lock:
        hit = _cache.get(real)
        if hit is not None and hit[0] == key:
            return hit[1]
    h = hashlib.sha256()
    with open(real, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if file_signature(real) != key:
        raise ValueError("evidence file changed while hashing; retry with stable inputs")
    with _lock:
        if len(_cache) >= 128:
            _cache.clear()
        _cache[real] = (key, digest)
    return digest


def clear_cache() -> None:
    """Test-only: drop memoized digests so a rewritten fixture re-hashes."""
    with _lock:
        _cache.clear()
