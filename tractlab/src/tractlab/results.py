"""Immutable result registry (S-02/03/04/07 — RESULT-API-CONTRACT.md).

A ``Result`` is a read-only snapshot of one successful producer's full
analytic population (bank load / filter / live track / recovery /
connectotomy cut). It is published only after every refusal gate — packing,
geometry checks, fidelity checks — has already passed; a rejected request
never calls ``publish`` and so can never alter or replace a prior result.
Margin and export read a result by id; they never reopen a mutable bank file
or fall back to "whatever was loaded last".

Result IDs identify populations. They are not secrets and not an
authorization mechanism — do not add a per-client authorization layer on
top of this registry (RESULT-API-CONTRACT.md).
"""
from __future__ import annotations

import hashlib
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

RESULT_SCHEMA = 1
DEFAULT_MAX_COUNT = 16
DEFAULT_MAX_TOTAL_BYTES = 256 * 1024 * 1024
DEFAULT_IDLE_EXPIRY_S = 30 * 60.0


class ResultUnavailable(LookupError):
    """Missing/expired/malformed resultId — maps to a typed 404 at the API."""


class ResultTooLarge(ValueError):
    """A single result alone exceeds the registry's byte retention budget."""


def digest_lines(lines) -> str:
    """Deterministic content digest: order-sensitive, shape-sensitive.

    Any change to geometry, streamline count, or streamline order changes
    this digest — it is the population's fingerprint, not a length count.
    """
    h = hashlib.sha256()
    for arr in lines:
        a = np.ascontiguousarray(arr, dtype=np.float32)
        h.update(int(a.shape[0]).to_bytes(8, "little", signed=False))
        h.update(a.tobytes(order="C"))
    return h.hexdigest()


def _lines_nbytes(lines) -> int:
    return sum(int(np.ascontiguousarray(a, dtype=np.float32).nbytes) for a in lines)


@dataclass(frozen=True)
class Result:
    """Copied, read-only geometry + weights + identity metadata."""

    id: str
    created_at: float
    source_population: str
    source_digest: str
    derivation_id: str | None
    grid_id: str
    volume_id: str
    lines: tuple[np.ndarray, ...]
    weights: np.ndarray | None
    source_ordinals: np.ndarray
    evidence: Mapping
    n_analytic_full: int
    nbytes: int
    schema: int = RESULT_SCHEMA

    @staticmethod
    def build(
        *,
        source_population: str,
        derivation_id: str | None,
        grid_id: str,
        volume_id: str,
        lines,
        weights=None,
        evidence=None,
        result_id: str | None = None,
    ) -> "Result":
        frozen: list[np.ndarray] = []
        for arr in lines:
            a = np.ascontiguousarray(arr, dtype=np.float32).copy()
            if a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all():
                raise ValueError("result geometry must be finite (N,3) arrays")
            if a.shape[0] < 2:
                # ``export_tck`` quite correctly omits zero/one-point rows.
                # Refuse those rows before publication so a result id can
                # never name a population different from the exported bytes.
                raise ValueError(
                    "result geometry streamlines must have at least 2 points"
                )
            a.setflags(write=False)
            frozen.append(a)
        frozen_lines = tuple(frozen)
        w = None
        if weights is not None:
            w = np.ascontiguousarray(weights, dtype=np.float64).copy()
            if w.shape != (len(frozen_lines),) or not np.isfinite(w).all() or (w < 0).any():
                raise ValueError("result weights must be finite nonnegative values, one per row")
            w.setflags(write=False)
        ordinals = np.arange(len(frozen_lines), dtype=np.uint32)
        ordinals.setflags(write=False)
        # These are rows within source_population (e.g. the filtered result),
        # not invented ordinals in an upstream corpus.
        metadata = dict(evidence or {})
        if any(not isinstance(v, (str, int, float, bool, type(None))) for v in metadata.values()):
            raise ValueError("result evidence must contain immutable scalar values")
        nbytes = _lines_nbytes(frozen_lines) + ordinals.nbytes + (int(w.nbytes) if w is not None else 0)
        return Result(
            id=result_id or uuid.uuid4().hex,
            created_at=time.time(),
            source_population=source_population,
            source_digest=digest_lines(frozen_lines),
            derivation_id=derivation_id,
            grid_id=grid_id,
            volume_id=volume_id,
            lines=frozen_lines,
            weights=w,
            source_ordinals=ordinals,
            evidence=MappingProxyType(metadata),
            n_analytic_full=len(frozen_lines),
            nbytes=nbytes,
        )


class ResultRegistry:
    """Bounded, thread-safe store of immutable ``Result`` objects.

    Publication is the only mutation. A caller that already holds a
    ``Result`` object (e.g. mid-export) keeps a valid, usable snapshot even
    if the registry subsequently evicts that id for capacity/age — Python
    keeps the referenced object alive; only a fresh ``get`` by id can fail.
    """

    def __init__(
        self,
        *,
        max_count: int = DEFAULT_MAX_COUNT,
        max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
        idle_expiry_s: float = DEFAULT_IDLE_EXPIRY_S,
    ):
        self._lock = threading.Lock()
        self._order: "OrderedDict[str, Result]" = OrderedDict()
        self._last_access: dict[str, float] = {}
        self.max_count = int(max_count)
        self.max_total_bytes = int(max_total_bytes)
        self.idle_expiry_s = float(idle_expiry_s)
        if self.max_count < 1 or self.max_total_bytes < 1 or not np.isfinite(self.idle_expiry_s) or self.idle_expiry_s <= 0:
            raise ValueError("result retention limits must be positive and finite")
        self._closed = False

    def _prune_locked(self, now: float) -> None:
        stale = [rid for rid, t in self._last_access.items()
                 if now - t > self.idle_expiry_s]
        for rid in stale:
            self._order.pop(rid, None)
            self._last_access.pop(rid, None)

    def _total_bytes_locked(self) -> int:
        return sum(r.nbytes for r in self._order.values())

    def publish(self, result: Result) -> Result:
        """Insert an already-built ``Result``, evicting oldest entries for
        capacity. Refuses only when the result alone exceeds the byte budget
        — that refusal happens before publication, so no other result is
        touched by a request that will not be honoured."""
        if result.nbytes > self.max_total_bytes:
            raise ResultTooLarge(
                f"result is {result.nbytes} bytes, over the "
                f"{self.max_total_bytes} byte retention budget"
            )
        with self._lock:
            if self._closed:
                raise ResultUnavailable("result registry is shut down")
            now = time.time()
            self._prune_locked(now)
            while self._order and (
                len(self._order) >= self.max_count
                or self._total_bytes_locked() + result.nbytes > self.max_total_bytes
            ):
                evicted_id, _ = self._order.popitem(last=False)
                self._last_access.pop(evicted_id, None)
            self._order[result.id] = result
            self._last_access[result.id] = now
            return result

    def get(self, result_id) -> Result:
        if not isinstance(result_id, str) or not result_id:
            raise ResultUnavailable("resultId must be a non-empty string")
        with self._lock:
            now = time.time()
            self._prune_locked(now)
            result = self._order.get(result_id)
            if result is None:
                raise ResultUnavailable(f"unknown or expired resultId {result_id!r}")
            self._last_access[result_id] = now
            self._order.move_to_end(result_id)
            return result

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._order.clear()
            self._last_access.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._order)

    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes_locked()
