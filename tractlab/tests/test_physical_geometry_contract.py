"""Generated physical-geometry regressions for the N05/N06/N07 contract."""

from __future__ import annotations

import numpy as np
import pytest

from tractlab.bank import _hits_mask, filter_bank_memory
from tractlab.clearance import (
    ClearanceReport,
    clearance_report,
    clearance_report_weighted,
)
from tractlab.grid import Grid, voxel_to_world
from tractlab.margin import build_margin_hull
from tractlab.recovery import filter_near_lesion_memory
from tractlab.traversal import segment_voxel_hits


def identity_grid(shape: tuple[int, int, int]) -> Grid:
    return Grid(shape=shape, affine=np.eye(4), axcodes=("R", "A", "S"))


def lesion_cube(shape: tuple[int, int, int] = (24, 24, 24)) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[8:13, 8:13, 8:13] = True
    return mask


def sparse_crossing() -> np.ndarray:
    return np.array([[2.0, 10.0, 10.0], [18.0, 10.0, 10.0]])


def dense_crossing() -> np.ndarray:
    return np.linspace(sparse_crossing()[0], sparse_crossing()[1], 65)


def point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    direction = end - start
    denom = float(np.dot(direction, direction))
    if denom == 0.0:
        return float(np.linalg.norm(point - start))
    t = np.clip(float(np.dot(point - start, direction) / denom), 0.0, 1.0)
    return float(np.linalg.norm(point - (start + t * direction)))


def test_below_floor_display_keeps_raw_measurement_and_caution():
    report = ClearanceReport(
        p05_mm=1.0,
        p50_mm=4.0,
        n_streamlines=4,
        n_eligible=4,
        n_seed_only=0,
        geom_floor_mm=3.0,
        seed_exclude_mm=3.0,
        method="synthetic",
    )

    assert report.p05_mm == 1.0
    assert report.display_p05_mm == 1.0
    assert report.format_p05() == "<3"
    assert "below" in report.summary_line()
    assert "not navigation" in report.summary_line()
    assert ">=3" not in report.summary_line()


def test_mask_hits_are_invariant_to_collinear_vertex_insertion():
    grid = identity_grid((24, 24, 24))
    mask = lesion_cube()
    inv_affine = np.linalg.inv(grid.affine)

    for line in (sparse_crossing(), dense_crossing()):
        assert segment_voxel_hits(line, mask, grid) is True
        assert _hits_mask(line, mask, inv_affine) is True


def test_bank_filter_uses_continuous_path_for_include_and_exclude(monkeypatch):
    grid = identity_grid((24, 24, 24))
    zone = lesion_cube()
    miss = np.array([[2.0, 2.0, 2.0], [4.0, 2.0, 2.0]])
    tracks = [sparse_crossing(), miss]
    lengths = np.array([16.0, 2.0])

    monkeypatch.setattr(
        "tractlab.bank.load_tracks_cached",
        lambda _path: (tracks, lengths),
    )
    kept, meta = filter_bank_memory(
        bank_path="synthetic.tck",
        grid=grid,
        seed=None,
        and_masks=[zone],
        or_mask=None,
        not_mask=None,
        minlength=0.0,
        maxlength=100.0,
    )
    assert len(kept) == 1
    assert np.array_equal(kept[0], sparse_crossing())
    assert meta["rejected"]["and"] == 1

    kept_excluded, meta_excluded = filter_bank_memory(
        bank_path="synthetic.tck",
        grid=grid,
        seed=None,
        and_masks=[],
        or_mask=None,
        not_mask=zone,
        minlength=0.0,
        maxlength=100.0,
    )
    assert len(kept_excluded) == 1
    assert np.array_equal(kept_excluded[0], miss)
    assert meta_excluded["rejected"]["not"] == 1


def test_recovery_filter_uses_continuous_path(monkeypatch):
    grid = identity_grid((24, 24, 24))
    zone = lesion_cube()
    tracks = [sparse_crossing(), np.array([[2.0, 2.0, 2.0], [4.0, 2.0, 2.0]])]
    lengths = np.array([16.0, 2.0])

    monkeypatch.setattr(
        "tractlab.recovery.load_tracks_cached",
        lambda _path: (tracks, lengths),
    )
    kept, meta = filter_near_lesion_memory(
        bank_path="synthetic.tck",
        grid=grid,
        lesion_zone=zone,
        minlength=0.0,
        maxlength=100.0,
    )
    assert len(kept) == 1
    assert np.array_equal(kept[0], sparse_crossing())
    assert meta["rejected"]["miss_zone"] == 1


def test_clearance_measures_continuous_segments_not_only_vertices():
    lesion = np.array([[10.0, 10.0, 10.0]])
    sparse = clearance_report([sparse_crossing()], lesion, geom_floor_mm=3.0)
    dense = clearance_report([dense_crossing()], lesion, geom_floor_mm=3.0)

    assert sparse is not None and dense is not None
    assert sparse.p05_mm == 0.0
    assert dense.p05_mm == 0.0
    assert sparse.p05_mm == dense.p05_mm
    assert sparse.distance_error_bound_mm == 0.0


def test_clearance_sampling_fallback_reports_its_half_spacing_bound(monkeypatch):
    import tractlab.clearance as clearance

    # Force the bounded path for a tiny generated polyline so the numerical
    # contract is exercised without a large fixture.
    monkeypatch.setattr(clearance, "CLEARANCE_EXACT_PAIR_CAP", 1)
    line = np.array([[0.0, 0.0, 0.0], [0.0, 0.6, 0.0], [0.6, 0.6, 0.0]])
    lesion = np.array([[0.0, 0.3, 0.0]])
    report = clearance.clearance_report([line], lesion, geom_floor_mm=3.0)

    assert report is not None and report.p05_mm is not None
    assert report.distance_error_bound_mm == 0.25
    # The sampled nearest distance can only overestimate the exact path
    # minimum, by no more than the declared half-spacing bound.
    exact = point_segment_distance(lesion[0], line[0], line[1])
    assert exact <= report.p05_mm <= exact + report.distance_error_bound_mm + 1e-9


def test_seed_exclusion_uses_the_continuous_path_population():
    lesion = np.array([[10.0, 10.0, 10.0]])
    seed = np.array([[2.0, 10.0, 10.0]])
    report = clearance_report(
        [sparse_crossing()],
        lesion,
        seed_pts=seed,
        seed_exclude_mm=3.0,
        geom_floor_mm=3.0,
    )

    assert report is not None
    assert report.n_eligible == 1
    assert report.p05_mm == 0.0


def test_seed_exclusion_does_not_miss_a_short_eligible_arc_between_samples():
    """A sub-half-mm gap outside the seed balls is still part of the path."""
    lesion = np.array([[0.4, 0.0, 0.0]])
    line = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    # The three radius-0.295 balls leave only [0.395, 0.405] eligible.
    # The old 0.5 mm samples land at 0, 0.5, and 1, all inside the seed.
    seed = np.array([
        [0.1, 0.0, 0.0],
        [0.7, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])

    reports = [
        clearance_report(
            [path],
            lesion,
            seed_pts=seed,
            seed_exclude_mm=0.295,
            geom_floor_mm=3.0,
        )
        for path in (line, np.linspace(line[0], line[1], 101))
    ]

    assert all(report is not None for report in reports)
    assert all(report.n_eligible == 1 for report in reports if report is not None)
    assert all(report.p05_mm == 0.0 for report in reports if report is not None)
    assert all(report.distance_error_bound_mm == 0.0 for report in reports if report is not None)


def test_seed_exclusion_refuses_unbounded_exact_clipping(monkeypatch):
    import tractlab.clearance as clearance

    monkeypatch.setattr(clearance, "CLEARANCE_SEED_CLIP_PAIR_CAP", 1)
    line = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    seed = np.array([[0.25, 0.0, 0.0], [0.75, 0.0, 0.0]])
    with pytest.raises(ValueError, match="seed exclusion clipping"):
        clearance_report(
            [line],
            np.array([[0.5, 0.0, 0.0]]),
            seed_pts=seed,
            seed_exclude_mm=0.1,
            geom_floor_mm=3.0,
        )


def test_weighted_clearance_shares_the_same_path_population():
    lesion = np.array([[10.0, 10.0, 10.0]])
    lines = [sparse_crossing()]
    unweighted = clearance_report(lines, lesion, geom_floor_mm=3.0)
    weighted = clearance_report_weighted(
        lines,
        np.ones(2),
        lesion,
        geom_floor_mm=3.0,
    )

    assert unweighted is not None and weighted is not None
    assert weighted.p05_mm == unweighted.p05_mm
    assert weighted.n_eligible == unweighted.n_eligible


def test_margin_is_world_distance_envelope_and_sparse_dense_equivalent():
    affine = np.array(
        [
            [1.0, 0.25, 0.0, 10.0],
            [0.0, 1.0, 0.0, -5.0],
            [0.0, 0.0, 3.0, 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    grid = Grid(shape=(40, 40, 20), affine=affine, axcodes=("R", "A", "S"))
    line_vox = np.array([[5.0, 12.0, 6.0], [30.0, 12.0, 6.0]])
    sparse = voxel_to_world(grid, line_vox)
    dense = np.linspace(sparse[0], sparse[1], 101)

    sparse_verts, sparse_faces = build_margin_hull([sparse], grid, margin_mm=3.0, step=1)
    dense_verts, dense_faces = build_margin_hull([dense], grid, margin_mm=3.0, step=1)

    assert sparse_verts.shape[0] > 0 and sparse_faces.shape[0] > 0
    assert np.array_equal(sparse_faces, dense_faces)
    assert np.allclose(sparse_verts, dense_verts, atol=1e-6)

    start, end = sparse
    distances = np.array([
        point_segment_distance(vertex, start, end) for vertex in sparse_verts
    ])
    # The implementation must publish a bound at or below this synthetic-case
    # budget; the exact helper is asserted after the implementation exists.
    from tractlab.margin import margin_error_bound_mm

    error = margin_error_bound_mm()
    assert float(distances.min()) >= 3.0 - error - 1e-6
    assert float(distances.max()) <= 3.0 + error + 1e-6


def test_margin_rejects_extreme_finite_path_before_sampling():
    import tractlab.margin as margin

    # This assertion is deliberately before the call: the pre-fix code has no
    # path-sample cap and would attempt an astronomical np.linspace allocation.
    assert hasattr(margin, "MARGIN_MAX_PATH_SAMPLES_PER_STREAMLINE")
    line = np.array([[0.0, 0.0, 0.0], [1.0e12, 0.0, 0.0]])
    with pytest.raises(ValueError, match="path samples"):
        build_margin_hull([line], identity_grid((8, 8, 8)), margin_mm=3.0)


def test_margin_rejects_total_path_sample_budget_before_concatenation(monkeypatch):
    import tractlab.margin as margin

    assert hasattr(margin, "MARGIN_MAX_PATH_SAMPLES_TOTAL")
    monkeypatch.setattr(margin, "MARGIN_MAX_PATH_SAMPLES_TOTAL", 4)
    line = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="total path samples"):
        build_margin_hull([line, line], identity_grid((16, 16, 16)), margin_mm=3.0)


def test_margin_rejects_extreme_finite_raster_dimension_before_allocation():
    import tractlab.margin as margin

    # The path itself is already sampled here; this isolates the world-raster
    # shape/product guard from the path sampling guard.
    assert hasattr(margin, "MARGIN_MAX_RASTER_DIM")
    samples = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0e308]])
    with pytest.raises(ValueError, match="raster"):
        margin._world_raster(samples, 3.0, spacing_mm=0.5)
