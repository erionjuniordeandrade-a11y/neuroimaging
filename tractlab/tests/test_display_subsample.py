"""Display subsample prefers core / SIFT2 — not longest-only noise."""

from __future__ import annotations

import numpy as np

from tractlab.serve import _display_subsample, _min_dist_to_shell_mm, DISPLAY_CAP


def _line(length_mm: float, npts: int = 10) -> np.ndarray:
    # straight line along x of given arc length
    xs = np.linspace(0, length_mm, npts, dtype=np.float32)
    return np.column_stack([xs, np.zeros(npts, np.float32), np.zeros(npts, np.float32)])


def test_display_subsample_under_cap_returns_all():
    lines = [_line(50 + i) for i in range(10)]
    out, idx = _display_subsample(lines, cap=20)
    assert len(out) == 10
    assert list(idx) == list(range(10))
    assert all(out[i] is lines[int(idx[i])] for i in range(10))


def test_display_subsample_avoids_longest_only():
    """Longest-rank would pick only the long outliers; core sample should not."""
    # 100 short-core + 200 medium + 50 very long
    short = [_line(40) for _ in range(100)]
    mid = [_line(80) for _ in range(200)]
    long = [_line(200) for _ in range(50)]
    lines = short + mid + long
    out, idx = _display_subsample(lines, cap=100, seed=1)
    assert len(out) == 100
    assert len(idx) == 100
    assert all(out[i] is lines[int(idx[i])] for i in range(len(out)))
    lengths = [
        float(np.sum(np.linalg.norm(np.diff(s, axis=0), axis=1))) for s in out
    ]
    # median of display set should sit in mid band, not ~200
    assert np.median(lengths) < 150
    # and not almost all longest
    n_long = sum(1 for L in lengths if L > 180)
    assert n_long < 40


def test_display_subsample_sift2_prefers_high_weight():
    lines = [_line(60) for _ in range(200)]
    w = np.ones(200)
    w[0] = 100.0  # first streamline is the "core"
    w[1] = 50.0
    out, idx = _display_subsample(lines, weights=w, cap=10, seed=0)
    assert len(out) == 10
    # top-weight lines should appear first in output order
    assert out[0] is lines[0]
    assert out[1] is lines[1]
    assert list(idx[:2]) == [0, 1]


def test_display_subsample_near_lesion_bias_reserves_quota():
    """With near-lesion bias, streamlines close to shell appear in the sample."""
    # Far lines (x far from origin) + near lines (x near 0)
    far = [_line(80) + np.array([100.0, 0, 0], dtype=np.float32) for _ in range(300)]
    near = []
    for i in range(50):
        # short segment around origin (lesion shell at origin)
        xs = np.linspace(0, 30, 8, dtype=np.float32)
        near.append(np.column_stack([xs, np.zeros(8, np.float32), np.zeros(8, np.float32)]))
    lines = far + near
    shell = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    out, idx = _display_subsample(
        lines,
        cap=40,
        seed=2,
        lesion_shell=shell,
        near_lesion_frac=0.5,
        near_radius_mm=15.0,
    )
    assert len(out) == 40
    assert len(idx) == 40
    assert all(out[i] is lines[int(idx[i])] for i in range(len(out)))
    # count how many sampled lines stay near origin (mean |x| small)
    n_nearish = 0
    for s in out:
        mx = float(np.mean(np.abs(s[:, 0])))
        if mx < 40:
            n_nearish += 1
    assert n_nearish >= 15  # at least ~half of near quota survived


def test_near_shell_distances_keep_mm_threshold_and_empty_identity():
    shell = np.array([[0., 0., 0.], [10., 0., 0.]])
    lines = [
        np.array([[0., 12., 0.]] * 9),
        np.array([[10., 12.001, 0.]] * 9),
        np.array([[1., 0., 4.]] * 9),
        np.empty((0, 3)),
    ]
    distance = _min_dist_to_shell_mm(lines, shell)
    np.testing.assert_allclose(distance, [12., 12.001, np.sqrt(17), np.inf], rtol=0, atol=1e-12)
    assert list(np.flatnonzero(distance <= 12)) == [0, 2]
