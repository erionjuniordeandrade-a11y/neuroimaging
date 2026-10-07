"""S-02/03/04 — the immutable result registry (tractlab.results).

Synthetic only: no case fixtures, no network, no private data.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from tractlab.results import (
    Result,
    ResultRegistry,
    ResultTooLarge,
    ResultUnavailable,
    digest_lines,
)


def _lines(n, k=3):
    rng = np.random.default_rng(0)
    return [rng.random((k, 3)).astype(np.float32) for _ in range(n)]


def _build(lines=None, **kw):
    kw.setdefault("source_population", "bank:test")
    kw.setdefault("derivation_id", "d1")
    kw.setdefault("grid_id", "grid-1")
    kw.setdefault("volume_id", "vol-1")
    return Result.build(lines=lines if lines is not None else _lines(3), **kw)


def test_digest_is_order_and_content_sensitive():
    a = _lines(3)
    b = [x.copy() for x in a]
    assert digest_lines(a) == digest_lines(b)
    b[1] = b[1] + 1.0
    assert digest_lines(a) != digest_lines(b)
    # same content, different order -> different digest
    c = [a[1], a[0], a[2]]
    assert digest_lines(a) != digest_lines(c)


def test_digest_empty_population_is_stable_constant():
    assert digest_lines([]) == digest_lines([])


def test_result_geometry_is_copied_and_read_only():
    src = _lines(2)
    r = _build(lines=src)
    src[0][0, 0] = 999.0  # mutate the caller's array after building
    assert r.lines[0][0, 0] != 999.0, "Result must own a copy, not an alias"
    with pytest.raises(ValueError):
        r.lines[0][0, 0] = 1.0  # read-only


def test_result_n_analytic_full_matches_input_count():
    r = _build(lines=_lines(7))
    assert r.n_analytic_full == 7


def test_publish_assigns_unique_ids():
    reg = ResultRegistry()
    r1 = reg.publish(_build())
    r2 = reg.publish(_build())
    assert r1.id != r2.id
    assert reg.get(r1.id) is r1
    assert reg.get(r2.id) is r2


def test_get_unknown_id_is_typed_unavailable():
    reg = ResultRegistry()
    with pytest.raises(ResultUnavailable):
        reg.get("nope")
    with pytest.raises(ResultUnavailable):
        reg.get("")
    with pytest.raises(ResultUnavailable):
        reg.get(None)  # type: ignore[arg-type]


def test_capacity_eviction_keeps_most_recent():
    reg = ResultRegistry(max_count=3, max_total_bytes=10**9)
    ids = []
    for _ in range(5):
        r = reg.publish(_build())
        ids.append(r.id)
    assert len(reg) == 3
    # the two oldest are gone; the three newest remain gettable
    with pytest.raises(ResultUnavailable):
        reg.get(ids[0])
    with pytest.raises(ResultUnavailable):
        reg.get(ids[1])
    for rid in ids[2:]:
        assert reg.get(rid).id == rid


def test_single_result_over_byte_budget_refuses_before_publication():
    reg = ResultRegistry(max_total_bytes=100)
    big = _build(lines=_lines(1000, k=50))  # far over 100 bytes
    with pytest.raises(ResultTooLarge):
        reg.publish(big)
    assert len(reg) == 0


def test_oversized_publish_does_not_evict_existing_results():
    """A rejected publish must never alter a prior result (RESULT-API-CONTRACT.md)."""
    reg = ResultRegistry(max_total_bytes=10_000)
    kept = reg.publish(_build(lines=_lines(2)))
    huge = _build(lines=_lines(5000, k=50))
    with pytest.raises(ResultTooLarge):
        reg.publish(huge)
    assert reg.get(kept.id).id == kept.id


def test_idle_expiry_evicts_after_the_configured_window():
    reg = ResultRegistry(idle_expiry_s=0.05)
    r = reg.publish(_build())
    assert reg.get(r.id).id == r.id
    time.sleep(0.12)
    with pytest.raises(ResultUnavailable):
        reg.get(r.id)


def test_get_touches_recency_so_it_survives_capacity_pressure():
    reg = ResultRegistry(max_count=2, max_total_bytes=10**9)
    r1 = reg.publish(_build())
    r2 = reg.publish(_build())
    reg.get(r1.id)  # r1 is now more recently used than r2
    r3 = reg.publish(_build())
    # r2 (least recently touched) is evicted, not r1
    with pytest.raises(ResultUnavailable):
        reg.get(r2.id)
    assert reg.get(r1.id).id == r1.id
    assert reg.get(r3.id).id == r3.id


def test_close_makes_registry_unavailable_and_frees_entries():
    reg = ResultRegistry()
    r = reg.publish(_build())
    reg.close()
    assert len(reg) == 0
    with pytest.raises(ResultUnavailable):
        reg.get(r.id)
    with pytest.raises(ResultUnavailable):
        reg.publish(_build())


def test_holding_a_snapshot_survives_eviction_of_its_id():
    """Export-in-progress semantics: a caller that already holds the Result
    object keeps a valid, usable snapshot even after the registry evicts
    that id for capacity (RESULT-API-CONTRACT.md)."""
    reg = ResultRegistry(max_count=1, max_total_bytes=10**9)
    held = reg.publish(_build(lines=_lines(4)))
    reg.publish(_build())  # evicts `held`'s id
    with pytest.raises(ResultUnavailable):
        reg.get(held.id)
    # the already-held object is still a normal, complete Result
    assert held.n_analytic_full == 4
    assert len(held.lines) == 4


def test_concurrent_publish_and_get_are_consistent():
    """Controlled concurrency: many threads publishing/reading in parallel
    never corrupt the registry or hand back a torn/partial Result."""
    reg = ResultRegistry(max_count=1000, max_total_bytes=10**9)
    ids: list[str] = []
    ids_lock = threading.Lock()
    errors: list[Exception] = []

    def publisher():
        try:
            for _ in range(20):
                r = reg.publish(_build(lines=_lines(3)))
                with ids_lock:
                    ids.append(r.id)
        except Exception as e:  # pragma: no cover - failure path
            errors.append(e)

    def reader():
        try:
            for _ in range(50):
                with ids_lock:
                    snapshot = list(ids)
                for rid in snapshot[-5:]:
                    try:
                        r = reg.get(rid)
                        assert r.id == rid
                        assert r.n_analytic_full == 3
                    except ResultUnavailable:
                        pass  # legitimately evicted/expired mid-run
        except Exception as e:  # pragma: no cover - failure path
            errors.append(e)

    threads = [threading.Thread(target=publisher) for _ in range(4)]
    threads += [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not errors
    assert len(ids) == 80
    assert len(set(ids)) == 80  # every published id is unique


def test_rejected_producer_never_publishes():
    """A caller that never calls ``publish`` — modelling a refusal gate
    (fidelity/geometry) firing before publication — leaves a prior result
    fully intact and the registry size unchanged."""
    reg = ResultRegistry()
    kept = reg.publish(_build())
    before = len(reg)
    # Simulate a rejected request: build the "would-be" Result but do NOT
    # publish it, exactly like a bank load that fails fidelity after packing.
    _build(lines=_lines(9))
    assert len(reg) == before
    assert reg.get(kept.id).id == kept.id
