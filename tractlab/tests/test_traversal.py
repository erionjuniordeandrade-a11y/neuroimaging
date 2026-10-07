"""S6/F2 — shared segment-voxel traversal: segments hit what vertices miss."""
import numpy as np

from tractlab.grid import Grid
from tractlab.traversal import hit_count, segment_voxel_hits


def identity_grid(shape) -> Grid:
    return Grid(shape=tuple(shape), affine=np.eye(4), axcodes=("R", "A", "S"))


def test_thin_mask_between_vertices_is_hit():
    # 1-voxel-thick wall at x-index 5; streamline vertices at x=2mm and x=9mm
    # (identity affine, 1mm voxels) — no VERTEX is inside, the SEGMENT crosses.
    mask = np.zeros((12, 12, 12), bool)
    mask[5, :, :] = True
    line = np.array([[2.0, 6.0, 6.0], [9.0, 6.0, 6.0]])
    assert segment_voxel_hits(line, mask, identity_grid((12, 12, 12))) is True


def test_miss_is_false_not_none():
    mask = np.zeros((12, 12, 12), bool)
    line = np.array([[2.0, 6.0, 6.0], [3.0, 6.0, 6.0]])
    assert segment_voxel_hits(line, mask, identity_grid((12, 12, 12))) is False


def test_vertex_hit_still_hits():
    mask = np.zeros((12, 12, 12), bool)
    mask[3, 6, 6] = True
    line = np.array([[3.0, 6.0, 6.0], [4.0, 6.0, 6.0]])
    assert segment_voxel_hits(line, mask, identity_grid((12, 12, 12))) is True


def test_out_of_bounds_discarded_never_clamped():
    # Line entirely outside the grid pointing at an edge voxel: clamping/face-snap
    # would fabricate a hit on the boundary voxel.
    mask = np.ones((12, 12, 12), bool)
    line = np.array([[-30.0, 6.0, 6.0], [-20.0, 6.0, 6.0]])
    assert segment_voxel_hits(line, mask, identity_grid((12, 12, 12))) is False


def test_step_mm_override_controls_sampling():
    # Sampling semantics, asserted both ways: the default (half-voxel) step
    # cannot jump the 1-voxel wall; a 4 mm override CAN and returns a miss.
    mask = np.zeros((12, 12, 12), bool)
    mask[5, :, :] = True
    line = np.array([[2.0, 6.0, 6.0], [9.0, 6.0, 6.0]])
    grid = identity_grid((12, 12, 12))
    assert segment_voxel_hits(line, mask, grid) is True
    assert segment_voxel_hits(line, mask, grid, step_mm=0.25) is True
    assert segment_voxel_hits(line, mask, grid, step_mm=4.0) is False


def test_bad_step_and_bad_mask_fail_loud():
    import pytest

    grid = identity_grid((12, 12, 12))
    mask = np.zeros((12, 12, 12), bool)
    line = np.array([[2.0, 6.0, 6.0], [9.0, 6.0, 6.0]])
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="step"):
            segment_voxel_hits(line, mask, grid, step_mm=bad)
    with pytest.raises(ValueError, match="shape"):
        segment_voxel_hits(line, np.zeros((5, 5, 5), bool), grid)


def test_non_finite_vertex_fails_loud_never_fabricates():
    import pytest

    grid = identity_grid((12, 12, 12))
    mask = np.ones((12, 12, 12), bool)
    line = np.array([[2.0, 6.0, 6.0], [np.nan, 6.0, 6.0]])
    with pytest.raises(ValueError, match="non-finite"):
        segment_voxel_hits(line, mask, grid)


def test_segment_hits_are_superset_of_vertex_hits():
    # Plan Task 3 invariant: every vertex-hit line is also a segment-hit line
    # (new counts >= old vertex-based counts, per-line).
    rng = np.random.default_rng(7)
    grid = identity_grid((12, 12, 12))
    mask = rng.random((12, 12, 12)) < 0.1
    from tractlab.grid import in_bounds, world_to_voxel

    def vertex_hits(line):
        vox = np.rint(world_to_voxel(grid, line)).astype(np.int64)
        good = in_bounds(grid, vox)
        vox = vox[good]
        return bool(len(vox)) and bool(mask[vox[:, 0], vox[:, 1], vox[:, 2]].any())

    for _ in range(50):
        line = rng.uniform(0, 11, size=(rng.integers(2, 6), 3))
        if vertex_hits(line):
            assert segment_voxel_hits(line, mask, grid) is True


def test_hit_count_counts_streamlines_not_vertices():
    mask = np.zeros((12, 12, 12), bool)
    mask[5, :, :] = True
    crossing = np.array([[2.0, 6.0, 6.0], [9.0, 6.0, 6.0]])
    missing = np.array([[2.0, 6.0, 6.0], [3.0, 6.0, 6.0]])
    assert hit_count([crossing, missing, crossing], mask, identity_grid((12, 12, 12))) == 2


def test_degenerate_lines_are_false():
    mask = np.ones((12, 12, 12), bool)
    grid = identity_grid((12, 12, 12))
    assert segment_voxel_hits(np.empty((0, 3)), mask, grid) is False
    assert segment_voxel_hits(np.array([[600.0, 6.0, 6.0]]), mask, grid) is False
