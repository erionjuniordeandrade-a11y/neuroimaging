"""C1 connectotomy — cavity ∩ analytic banks on a synthetic grid (no PHI)."""

from __future__ import annotations
import re

import numpy as np
import pytest

from tractlab.connectotomy import (
    CAVITY_ROLE,
    connectotomy_note,
    CavityRoleError,
    count_bank_cuts,
    report_from_banks,
    reject_cavity_in_request,
    reject_cavity_role,
    streamline_hits_cavity,
)
from tractlab.grid import Grid, voxel_to_world
from tractlab.roi_boolean import layers_from_request
from tractlab.track import TrackParams, run_tckgen


def _grid(shape=(10, 10, 10), spacing=2.0) -> Grid:
    aff = np.eye(4)
    aff[0, 0] = aff[1, 1] = aff[2, 2] = spacing
    return Grid(shape=shape, affine=aff, axcodes=("R", "A", "S"))


def _line_through(grid: Grid, voxels) -> np.ndarray:
    ijk = np.asarray(voxels, dtype=np.float64)
    return voxel_to_world(grid, ijk).astype(np.float32)


def test_segment_crossing_thin_cavity_counts_as_cut():
    # S6/F2: the cavity wall sits BETWEEN two vertices — vertex rounding said
    # "not cut"; the shared segment traversal must say "cut". A cut count that
    # depends on vertex spacing understates disconnection.
    grid = _grid()
    cavity = np.zeros(grid.shape, dtype=bool)
    cavity[5, :, :] = True
    jump = _line_through(grid, [[2, 5, 5], [8, 5, 5]])  # vertices 6 voxels apart
    n_cut, n_bank, idx = count_bank_cuts([jump], cavity, grid)
    assert (n_cut, n_bank) == (1, 1) and list(idx) == [0]


def test_known_intersection_exact_n_cut():
    grid = _grid()
    cavity = np.zeros(grid.shape, dtype=bool)
    cavity[5, 5, 5] = True
    hit = _line_through(grid, [[5, 5, 5], [6, 5, 5]])
    miss = _line_through(grid, [[1, 1, 1], [2, 1, 1]])
    n_cut, n_bank, idx = count_bank_cuts([hit, miss, hit], cavity, grid)
    assert n_bank == 3
    assert n_cut == 2
    assert list(idx) == [0, 2]


def test_empty_cavity_is_honest_null():
    grid = _grid()
    cavity = np.zeros(grid.shape, dtype=bool)
    banks = {"bank_cst_r": {"label": "CST-R", "lines": [_line_through(grid, [[5, 5, 5]])]}}
    assert report_from_banks(banks, cavity, grid, floor_mm=3.0) is None
    assert report_from_banks(banks, None, grid, floor_mm=3.0) is None


def test_out_of_bounds_vertices_never_clamp_into_hits():
    grid = _grid()
    cavity = np.zeros(grid.shape, dtype=bool)
    cavity[0, 0, 0] = True  # face voxel — clamp would fabricate hits here
    # World point far outside the volume (negative voxel index)
    oob = np.array([[-40.0, -40.0, -40.0], [-38.0, -40.0, -40.0]], dtype=np.float32)
    assert streamline_hits_cavity(oob, cavity, grid) is False
    n_cut, n_bank, idx = count_bank_cuts([oob], cavity, grid)
    assert n_bank == 1
    assert n_cut == 0
    assert list(idx) == []


def test_counts_differ_on_display_subsample():
    """Caller must pass the analytic file; a length-rank subsample changes n_cut."""
    grid = _grid()
    cavity = np.zeros(grid.shape, dtype=bool)
    cavity[5, 5, 5] = True
    hit = _line_through(grid, [[5, 5, 5], [6, 5, 5]])
    miss = _line_through(grid, [[1, 1, 1], [2, 1, 1]])
    full = [hit] * 5 + [miss] * 5
    subsample = full[:3]  # length-rank display would keep a different set
    n_full, _, _ = count_bank_cuts(full, cavity, grid)
    n_sub, _, _ = count_bank_cuts(subsample, cavity, grid)
    assert n_full == 5
    assert n_sub == 3
    assert n_full != n_sub


def test_geom_floor_is_copy_only_not_a_dilation():
    """A streamline in the neighbouring voxel is not a cut. 3 mm is copy only."""
    grid = _grid(spacing=2.0)
    cavity = np.zeros(grid.shape, dtype=bool)
    cavity[5, 5, 5] = True
    neighbour = _line_through(grid, [[6, 5, 5], [7, 5, 5]])
    assert streamline_hits_cavity(neighbour, cavity, grid) is False
    # (floor is per-case manifest data now; 3.0 is the 3T entry)


def test_c1_payload_has_no_assignment_or_network_fields():
    grid = _grid()
    cavity = np.zeros(grid.shape, dtype=bool)
    cavity[5, 5, 5] = True
    banks = {
        "bank_cst_r": {
            "label": "CST-R",
            "lines": [_line_through(grid, [[5, 5, 5]]), _line_through(grid, [[1, 1, 1]])],
        }
    }
    report = report_from_banks(banks, cavity, grid, floor_mm=3.0)
    assert report is not None
    assert report["cavity"] == "lesion"
    assert report["floor_mm"] == 3.0
    assert "margin" not in report["note"].lower()
    assert "at risk" not in report["note"].lower()
    assert "navigation" not in report["note"].lower()
    blob = str(report).lower()
    for forbidden in ("assigned", "assignment", "schaefer", "yeo", "network"):
        assert forbidden not in blob
    row = report["banks"][0]
    assert set(row) == {"id", "n_cut", "n_bank", "fraction"}
    assert row["n_cut"] == 1
    assert row["n_bank"] == 2
    assert row["fraction"] == 0.5


def test_note_is_the_shared_copy():
    note3 = connectotomy_note(3.0)
    assert "lesion cavity" in note3.lower()
    assert "3 mm" in note3
    # per-case: another case's floor prints ITS value, never 3 mm
    assert "1.5 mm" in connectotomy_note(1.5)
    for word in ("margin", "navigation", "at risk", "ASSIGNED"):
        assert word.lower() not in note3.lower()


def test_reject_cavity_role_is_typed():
    with pytest.raises(CavityRoleError, match="cavity"):
        reject_cavity_role(CAVITY_ROLE)
    reject_cavity_role(None, "seed", "include")  # other roles ok


def test_layers_from_request_rejects_cavity_typed_seed():
    with pytest.raises(CavityRoleError, match="cavity"):
        layers_from_request({
            "seed": {"points_mm": [[0.0, 0.0, 0.0]], "radius_mm": 4.0, "role": "cavity"},
        })
    with pytest.raises(CavityRoleError, match="cavity"):
        layers_from_request({
            "seed": {"points_mm": [[0.0, 0.0, 0.0]], "radius_mm": 4.0},
            "cavity": {"points_mm": [[1.0, 1.0, 1.0]], "radius_mm": 4.0},
        })


def test_run_tckgen_rejects_cavity_seed_role_before_engine(tmp_path):
    """Tracking entry point refuses a cavity-typed mask without calling tckgen."""
    with pytest.raises(CavityRoleError, match="cavity"):
        run_tckgen(
            "unused.mif",
            "unused_seed.nii.gz",
            "unused_mask.nii.gz",
            str(tmp_path),
            TrackParams(),
            seed_role="cavity",
        )


def test_reject_cavity_in_request_scans_include_exclude():
    with pytest.raises(CavityRoleError):
        reject_cavity_in_request({
            "seed": {"points_mm": [[0, 0, 0]], "radius_mm": 3, "role": "seed"},
            "and": [{"points_mm": [[1, 1, 1]], "radius_mm": 3, "role": "cavity"}],
        })


def test_connectotomy_note_prints_the_signed_corrected_floor_not_three():
    """S-14: on a reverse-PE corrected case with a signed 0.6 mm median the
    honesty copy must carry that figure; '3 mm' would be another case's floor."""
    note = connectotomy_note(0.6033)
    assert "~0.6 mm" in note
    assert re.search(r"(?<![\d.])~?3 mm", note) is None  # a bare 3 mm, not the 3 in 0.6033
    assert "no reverse-PE" not in note
