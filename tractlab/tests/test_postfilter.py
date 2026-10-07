"""Post-hoc inferior-reach filter — ranks descending fascicles after dense seed."""

from __future__ import annotations

import numpy as np

from tractlab.postfilter import InferiorReachFilter, filter_inferior_reach


def _line(z0: float, z1: float, n: int = 20) -> np.ndarray:
    z = np.linspace(z0, z1, n)
    return np.column_stack([np.full(n, 40.0), np.full(n, -40.0), z])


def test_keeps_only_inferior_reach():
    lines = [
        _line(40, 55),   # cortex only — drop
        _line(5, 50),    # reaches z=5 — keep
        _line(-15, 50),  # deeper — keep, longer span
    ]
    kept, st = filter_inferior_reach(lines, InferiorReachFilter(z_max=10.0, cap=10))
    assert st["n_pass_z"] == 2
    assert len(kept) == 2
    # deeper long span first
    assert float(kept[0][:, 2].min()) < float(kept[1][:, 2].min())


def test_cap_prefers_longer_zspan():
    lines = [_line(0, 20 + i) for i in range(5)]  # spans 20..24
    kept, st = filter_inferior_reach(lines, InferiorReachFilter(z_max=10.0, cap=2))
    assert st["n_pass_z"] == 5
    assert len(kept) == 2
    spans = [float(s[:, 2].max() - s[:, 2].min()) for s in kept]
    assert spans[0] >= spans[1]


def test_empty_input_is_honest_null():
    kept, st = filter_inferior_reach([], InferiorReachFilter(z_max=10.0))
    assert kept == []
    assert st["n_out"] == 0
