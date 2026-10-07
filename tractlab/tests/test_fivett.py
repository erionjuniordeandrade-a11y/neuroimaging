"""5TT / endpoint accounting, and the ".tck on disk" != ".tck finished" trap.

The lesion is modelled as normal WM (pv 0.92) in the unedited 5TT, but 6.3% of
lesion voxels carry a cortical-GM label, which makes them valid ACT termination
targets. These tests count endpoints landing inside the lesion and PRINT the
number, so the mechanism is named rather than assumed.

Two rules are pinned here because both were previously enforced in code and
tested nowhere:

* out-of-bounds endpoints are DISCARDED, never clamped onto a volume face;
* a ``.tck`` is read only when it is COMPLETE. The earlier gate was a 50 kB size
  heuristic, which a tckgen killed after 50 kB passes. Size is not a completion
  signal, so these tests build files that are large AND unfinished.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from tractlab.fivett import (
    count_endpoints_in_mask,
    endpoints_from_streamlines,
    endpoints_in_bool_mask,
    endpoints_in_mask,
    read_streamline_endpoints,
    read_streamlines,
    tck_completeness,
)
from tractlab.grid import Grid

CASE = os.path.expanduser("~/tractlab-data/cases/local-case")
LESION = f"{CASE}/tracts/roi/lesion1_dwi.nii.gz"
TCK = os.path.expanduser("~/tractlab-data/work0806/crossed_fat_raw.tck")


# --------------------------------------------------------------------------
# fixtures: synthetic .tck files, large enough to defeat a size heuristic
# --------------------------------------------------------------------------

def _lines(n: int) -> list[np.ndarray]:
    return [
        np.array([[float(i), 0.0, 0.0], [float(i), 1.0, 0.0], [float(i), 2.0, 0.0]],
                 dtype=np.float32)
        for i in range(n)
    ]


def _write_tck(path, lines, header=None):
    from nibabel.streamlines import TckFile, Tractogram

    TckFile(Tractogram(lines, affine_to_rasmm=np.eye(4)),
            header=dict(header or {})).save(str(path))
    return str(path)


def _unit_grid(shape=(4, 4, 4)) -> Grid:
    """Grid where voxel (i,j,k) sits at world (i,j,k) mm — 1 mm, no rotation."""
    return Grid(shape=shape, affine=np.eye(4), axcodes=("R", "A", "S"))


# --------------------------------------------------------------------------
# endpoint extraction
# --------------------------------------------------------------------------

def test_endpoints_shape_and_dtype_on_synthetic_input(tmp_path):
    """Endpoint extraction is testable without patient data."""
    lines = [
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0]], dtype=np.float32),
    ]
    p = _write_tck(tmp_path / "s.tck", lines)

    ep = read_streamline_endpoints(p)
    assert ep.shape == (2, 2, 3)
    np.testing.assert_allclose(ep[0, 0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(ep[0, 1], [2.0, 0.0, 0.0])
    np.testing.assert_allclose(ep[1, 1], [6.0, 6.0, 6.0])


def test_read_streamlines_keeps_every_interior_vertex(tmp_path):
    """The path gate needs interior vertices; endpoints alone cannot express it."""
    lines = [
        np.array([[0.0, 0.0, 0.0], [1.0, 7.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0]], dtype=np.float32),
    ]
    p = _write_tck(tmp_path / "s.tck", lines)

    got = read_streamlines(p)
    assert [g.shape for g in got] == [(3, 3), (2, 3)]
    np.testing.assert_allclose(got[0][1], [1.0, 7.0, 0.0])


def test_endpoints_from_streamlines_matches_reading_the_file_twice(tmp_path):
    """One read must serve both the endpoint gate and the path gate."""
    p = _write_tck(tmp_path / "s.tck", _lines(5))
    np.testing.assert_allclose(
        endpoints_from_streamlines(read_streamlines(p)),
        read_streamline_endpoints(p),
    )


# --------------------------------------------------------------------------
# the never-clamp rule — one implementation, both entry points
# --------------------------------------------------------------------------

def test_out_of_bounds_endpoint_is_discarded_not_clamped_onto_a_face():
    """A far-outside point must be False even when the face voxel is inside the mask.

    The mask is True at voxel (0,0,0) only. A clamping implementation maps
    (-50,-50,-50) onto (0,0,0) and returns True — a fabricated hit.
    """
    grid = _unit_grid()
    mask = np.zeros(grid.shape, dtype=bool)
    mask[0, 0, 0] = True

    endpoints = np.array([[[-50.0, -50.0, -50.0], [0.0, 0.0, 0.0]]])
    hit = endpoints_in_bool_mask(endpoints, grid, mask)

    assert hit.shape == (1, 2)
    assert not hit[0, 0], "out-of-bounds endpoint was clamped onto the (0,0,0) face"
    assert hit[0, 1], "an in-mask endpoint was missed"


def test_high_side_out_of_bounds_endpoint_is_also_discarded():
    """The upper face needs its own control; only the lower one used to be probed."""
    grid = _unit_grid()
    mask = np.zeros(grid.shape, dtype=bool)
    mask[3, 3, 3] = True

    hit = endpoints_in_bool_mask(
        np.array([[[3.0, 3.0, 3.0], [9999.0, 9999.0, 9999.0]]]), grid, mask
    )
    assert hit[0, 0], "the corner voxel itself must be a hit"
    assert not hit[0, 1], "out-of-bounds endpoint was clamped onto the (3,3,3) face"


def test_in_bounds_endpoint_outside_the_mask_is_false():
    grid = _unit_grid()
    mask = np.zeros(grid.shape, dtype=bool)
    mask[0, 0, 0] = True
    hit = endpoints_in_bool_mask(np.array([[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]]), grid, mask)
    assert not hit.any()


def test_path_and_array_entry_points_share_one_implementation(tmp_path):
    """Conformance test instead of two hand-kept copies of the never-clamp rule.

    ``endpoints_in_mask`` (loads a NIfTI) and ``endpoints_in_bool_mask`` (takes an
    in-memory mask) must agree exactly on the same data, including on the
    out-of-bounds points. Two independent copies of the rule is two chances to
    diverge silently.
    """
    import nibabel as nib

    rng = np.random.default_rng(0)
    mask = rng.random((6, 6, 6)) > 0.5
    affine = np.eye(4)
    path = tmp_path / "m.nii.gz"
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), str(path))

    grid = Grid(shape=(6, 6, 6), affine=affine, axcodes=("R", "A", "S"))
    pts = rng.uniform(-20.0, 25.0, size=(40, 2, 3))

    np.testing.assert_array_equal(
        endpoints_in_bool_mask(pts, grid, mask), endpoints_in_mask(pts, str(path))
    )


def test_empty_endpoint_array_gives_an_empty_result():
    grid = _unit_grid()
    mask = np.zeros(grid.shape, dtype=bool)
    assert endpoints_in_bool_mask(np.empty((0, 2, 3)), grid, mask).shape == (0, 2)


# --------------------------------------------------------------------------
# .tck completeness — size is not a completion signal
# --------------------------------------------------------------------------

def test_finished_tck_is_reported_complete(tmp_path):
    p = _write_tck(tmp_path / "full.tck", _lines(2000),
                   {"command_history": "tckgen in out -select 2000", "total_count": "2000"})
    c = tck_completeness(p)
    assert c.is_complete, c.reason
    assert c.n_read == 2000
    assert c.declared_count == 2000
    assert c.select_target == 2000


def test_truncated_tck_is_incomplete_even_though_it_beats_the_size_heuristic(tmp_path):
    """The exact failure the 50 kB heuristic passed: big file, unfinished write."""
    p = _write_tck(tmp_path / "full.tck", _lines(4000))
    blob = open(p, "rb").read()
    trunc = tmp_path / "trunc.tck"
    trunc.write_bytes(blob[: len(blob) // 2])

    assert os.path.getsize(trunc) >= 50_000, "fixture must defeat the old size guard"
    c = tck_completeness(str(trunc))
    assert not c.is_complete
    assert c.n_read is None
    assert "unreadable" in c.reason or "truncated" in c.reason


def test_header_count_disagreeing_with_the_data_is_incomplete(tmp_path):
    """A stale ``count:`` must not be believed just because the file parses.

    The count field is zero-padded to a fixed width, so it can be rewritten in
    place without moving the data offset.
    """
    p = _write_tck(tmp_path / "full.tck", _lines(2000))
    blob = open(p, "rb").read()
    assert b"count: 0000002000" in blob
    bad = tmp_path / "stale.tck"
    bad.write_bytes(blob.replace(b"count: 0000002000", b"count: 0000001000", 1))

    c = tck_completeness(str(bad))
    assert not c.is_complete
    assert c.declared_count == 1000 and c.n_read == 2000
    assert "declared" in c.reason


def test_tckgen_killed_before_reaching_select_is_incomplete(tmp_path):
    """A consistent but partial file: count == data, but far short of -select.

    MRtrix commits the header periodically, so a killed tckgen leaves a file that
    parses cleanly and whose declared count matches its data. Only the -select
    target in command_history, plus the seed budget, reveal that it never
    finished. This is the case a count-vs-data check alone cannot see.
    """
    p = _write_tck(
        tmp_path / "partial.tck",
        _lines(2000),
        {"command_history": "tckgen fod.mif out.tck -select 20000 -nthreads 0",
         "total_count": "5000",
         "max_num_seeds": "20000000"},
    )
    c = tck_completeness(p)
    assert not c.is_complete
    assert c.select_target == 20000
    assert "select" in c.reason


def test_seed_exhaustion_short_of_select_is_still_complete(tmp_path):
    """tckgen legitimately stops early when the seed budget runs out.

    Calling that "incomplete" would be a false alarm, and a check that cries wolf
    gets switched off. Severity must track what actually happened.
    """
    p = _write_tck(
        tmp_path / "exhausted.tck",
        _lines(2000),
        {"command_history": "tckgen fod.mif out.tck -select 20000",
         "total_count": "20000000",
         "max_num_seeds": "20000000"},
    )
    c = tck_completeness(p)
    assert c.is_complete, c.reason
    assert "seed" in c.reason


def test_missing_declared_count_fails_closed(tmp_path):
    """Unknown is not "complete". A gate that cannot verify must refuse."""
    p = _write_tck(tmp_path / "full.tck", _lines(100))
    blob = open(p, "rb").read()
    bad = tmp_path / "nocount.tck"
    # break the count key name, keeping the byte length so the offset still holds
    bad.write_bytes(blob.replace(b"count: ", b"xount: ", 1))

    c = tck_completeness(str(bad))
    assert not c.is_complete
    assert c.declared_count is None


def test_absent_file_is_incomplete_not_crashing():
    c = tck_completeness("/nonexistent/nope.tck")
    assert not c.is_complete
    assert c.n_read is None


def test_select_target_with_a_k_suffix_is_parsed():
    """MRtrix accepts -select 20k; failing to parse it must read as unknown, not 20."""
    from tractlab.fivett import parse_select_target

    assert parse_select_target("tckgen a b -select 20000") == 20000
    assert parse_select_target("tckgen a b -select 20k") == 20000
    assert parse_select_target("tckgen a b -seed_image s.nii") is None
    assert parse_select_target(None) is None


# --------------------------------------------------------------------------
# real data
# --------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(LESION), reason="lesion mask not present")
def test_out_of_bounds_endpoints_are_not_counted_as_hits():
    """A point far outside the volume must be False, never clamped onto a face."""
    far = np.array([[[1e4, 1e4, 1e4], [-1e4, -1e4, -1e4]]])
    inside = endpoints_in_mask(far, LESION)
    assert inside.shape == (1, 2)
    assert not inside.any(), "out-of-bounds endpoint was counted — clamping bug"


@pytest.mark.skipif(
    not (os.path.exists(TCK) and os.path.exists(LESION)),
    reason="raw crossed-FAT tractogram not present (run runbook step 6)",
)
def test_lesion_terminating_endpoint_count_is_reported():
    complete = tck_completeness(TCK)
    if not complete.is_complete:
        pytest.skip(f"tractogram is not finished: {complete.reason}")

    n = count_endpoints_in_mask(TCK, LESION)
    total = read_streamline_endpoints(TCK).shape[0] * 2
    assert total > 0, "tck parsed as complete but holds no streamlines"
    print(f"\nendpoints inside lesion: {n} / {total} ({100.0 * n / total:.2f}%)")
    assert 0 <= n <= total
    # `isinstance(n, int)` alone would assert nothing about the measurement. A
    # bundle seeded from bilateral SMA cannot terminate mostly inside a focal
    # frontal lesion: above a quarter, the lesion mask or the 5TT edit is
    # misaligned, which is a real defect and not a property of the anatomy.
    assert n < total * 0.25, (
        f"{n}/{total} endpoints terminate inside the lesion — check that "
        "5tt_path.mif and lesion1_dwi.nii.gz share the FOD grid"
    )
