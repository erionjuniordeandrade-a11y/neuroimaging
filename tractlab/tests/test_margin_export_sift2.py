"""Margin hull, .tck export, and SIFT2 util unit tests."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tractlab.export_tck import write_tck
from tractlab.grid import load_grid
from tractlab.margin import build_margin_hull
from tractlab.sift2_util import (
    load_sift2_weights,
    sift2_path_for_tck,
    weighted_percentile,
)

CASE = os.path.expanduser("~/tractlab-data/cases/local-case")
MASK = os.path.join(CASE, "nifti/mask_up.nii.gz")
CST = os.path.join(CASE, "tracts/bank/cst_r_motor_pons.tck")
needs_case = pytest.mark.skipif(not os.path.isfile(MASK), reason="case data absent")


def test_sift2_path_for_tck():
    assert sift2_path_for_tck("/a/b/c.tck") == "/a/b/c.sift2.txt"
    assert sift2_path_for_tck("foo.tck") == "foo.sift2.txt"


def test_weighted_percentile_uniform_and_skewed():
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    w = np.ones(5)
    # linear interpolation on cumulative weight mass → mid-range
    p50 = weighted_percentile(v, w, 50.0)
    assert 2.0 <= p50 <= 4.0
    # heavy weight on large values pulls p50 up
    w2 = np.array([1.0, 1.0, 1.0, 1.0, 100.0])
    assert weighted_percentile(v, w2, 50.0) > 4.0


def test_load_sift2_missing_returns_none(tmp_path):
    assert load_sift2_weights(str(tmp_path / "nope.txt")) is None


def test_load_sift2_length_mismatch_returns_none(tmp_path):
    p = tmp_path / "w.sift2.txt"
    p.write_text("1.0\n2.0\n3.0\n")
    assert load_sift2_weights(str(p), n_expected=5) is None
    w = load_sift2_weights(str(p), n_expected=3)
    assert w is not None and len(w) == 3


def test_write_tck_roundtrip(tmp_path):
    lines = [
        np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32),
        np.array([[0, 1, 0], [0, 2, 0]], dtype=np.float32),
    ]
    out = tmp_path / "out.tck"
    n = write_tck(lines, out)
    assert n == 2
    assert out.is_file() and out.stat().st_size > 0
    import nibabel as nib

    loaded = nib.streamlines.load(str(out))
    assert len(loaded.streamlines) == 2


def test_write_tck_empty_raises():
    with pytest.raises(ValueError, match="no streamlines"):
        write_tck([], "/tmp/should_not_exist.tck")


@needs_case
def test_margin_hull_nonempty_on_short_line():
    grid = load_grid(MASK)
    # short polyline through mid-brain-ish grid centre
    mid = np.array(grid.shape) / 2.0
    from tractlab.grid import voxel_to_world

    pts_vox = np.array(
        [mid + np.array([i * 2.0, 0, 0]) for i in range(8)], dtype=np.float64
    )
    line = voxel_to_world(grid, pts_vox)
    verts, faces = build_margin_hull([line], grid, margin_mm=5.0, step=2)
    assert verts.shape[0] > 10
    assert faces.shape[0] > 10
    assert verts.shape[1] == 3 and faces.shape[1] == 3


@needs_case
def test_margin_rejects_bad_mm():
    grid = load_grid(MASK)
    line = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="margin_mm"):
        build_margin_hull([line], grid, margin_mm=0.1)


@needs_case
def test_export_from_real_cst_subset(tmp_path):
    import nibabel as nib

    if not os.path.isfile(CST):
        pytest.skip("CST bank missing")
    tck = nib.streamlines.load(CST)
    lines = [np.asarray(s, dtype=np.float32) for s in list(tck.streamlines)[:20]]
    out = tmp_path / "cst_snip.tck"
    n = write_tck(lines, out)
    assert n == 20
    re = nib.streamlines.load(str(out))
    assert len(re.streamlines) == 20
