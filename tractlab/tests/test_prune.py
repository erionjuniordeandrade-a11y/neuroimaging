"""Outlier prune gates: length band, spatial MAD, SIFT2 floor."""

from __future__ import annotations

import numpy as np
import pytest

from tractlab.prune import (
    PruneParams,
    midpoint_mm,
    polyline_length_mm,
    prune_streamlines,
    prune_tck_file,
)


def _line(x0, x1, n=12, y=0.0, z=0.0):
    xs = np.linspace(x0, x1, n, dtype=np.float32)
    return np.column_stack(
        [xs, np.full(n, y, np.float32), np.full(n, z, np.float32)]
    )


def test_polyline_length_and_midpoint():
    s = _line(0, 100)
    assert polyline_length_mm(s) == pytest.approx(100.0, abs=0.01)
    m = midpoint_mm(s)
    assert m[0] == pytest.approx(50.0, abs=0.5)


def test_length_band_drops_stubs_and_giants():
    # 80 normal (~100mm), 10 stubs (~20mm), 10 giants (~250mm)
    core = [_line(0, 100) for _ in range(80)]
    stubs = [_line(0, 20) for _ in range(10)]
    giants = [_line(0, 250) for _ in range(10)]
    lines = stubs + core + giants
    kept, _, rep = prune_streamlines(
        lines,
        params=PruneParams(
            length_lo_pct=10, length_hi_pct=90,
            spatial_mad_k=100,  # effectively off
            sift2_drop_pct=0,
            min_keep=10,
        ),
    )
    assert rep.n_out < rep.n_in
    lengths = [polyline_length_mm(s) for s in kept]
    assert min(lengths) > 30
    assert max(lengths) < 200


def test_spatial_drops_far_midpoints():
    # tight cluster around y=0 + far outliers at y=80
    core = [_line(0, 80, y=float(np.random.default_rng(i).normal(0, 1))) for i in range(60)]
    far = [_line(0, 80, y=80.0) for _ in range(15)]
    lines = core + far
    kept, _, rep = prune_streamlines(
        lines,
        params=PruneParams(
            length_lo_pct=0, length_hi_pct=100,
            spatial_mad_k=2.5,
            spatial_max_pct=90,
            sift2_drop_pct=0,
            min_keep=10,
        ),
    )
    mids_y = [midpoint_mm(s)[1] for s in kept]
    assert all(abs(y) < 40 for y in mids_y)
    assert rep.n_out < rep.n_in


def test_sift2_floor_drops_near_zero_weight():
    lines = [_line(0, 80) for _ in range(50)]
    w = np.ones(50)
    w[:10] = 0.001  # junk ≪ 5% of median
    w[10:] = 1.0
    kept, kept_w, rep = prune_streamlines(
        lines,
        weights=w,
        params=PruneParams(
            length_lo_pct=0, length_hi_pct=100,
            spatial_mad_k=100,
            sift2_rel_floor=0.05,
            sift2_drop_pct=0,
            min_keep=5,
        ),
    )
    assert rep.sift2_floor is not None
    assert kept_w is not None
    assert float(kept_w.min()) >= rep.sift2_floor - 1e-9
    assert rep.n_out == 40  # only the 10 junk weights


def test_refuse_when_would_go_below_min_keep():
    lines = [_line(0, 50 + i) for i in range(20)]
    kept, _, rep = prune_streamlines(
        lines,
        params=PruneParams(
            length_lo_pct=40, length_hi_pct=60,
            spatial_mad_k=0.01,  # ultra tight → almost all die
            sift2_drop_pct=0,
            min_keep=15,
        ),
    )
    # either refused or kept enough
    if rep.refused:
        assert len(kept) == 20
    else:
        assert len(kept) >= 15


def test_prune_tck_file_roundtrip(tmp_path):
    from tractlab.export_tck import write_tck

    lines = [_line(0, 100, y=0) for _ in range(40)] + [_line(0, 100, y=50) for _ in range(10)]
    src = tmp_path / "in.tck"
    dst = tmp_path / "out.tck"
    write_tck(lines, src)
    rep = prune_tck_file(
        str(src), str(dst),
        params=PruneParams(
            length_lo_pct=5, length_hi_pct=95,
            spatial_mad_k=2.0, spatial_max_pct=90,
            sift2_drop_pct=0, min_keep=5,
        ),
    )
    assert dst.is_file()
    assert rep.n_out <= rep.n_in
    assert rep.n_out >= 5
