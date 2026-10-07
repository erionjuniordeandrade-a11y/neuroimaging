"""M4 track.py — real tckgen, typed outcomes, seed-membership, determinism."""

from __future__ import annotations

import os
import subprocess
import numpy as np
import pytest

from tractlab.grid import load_grid, voxel_to_world
from tractlab.seed import rasterize_points, write_seed_nifti
from tractlab.track import run_tckgen, TrackParams, Outcome, TCKGEN

CASE = os.path.expanduser("~/tractlab-data/cases/local-case")
FOD = f"{CASE}/nifti/wmfod_norm.mif"
MASK = f"{CASE}/nifti/mask_up.nii.gz"
FA_R_SEED = f"{CASE}/tracts/roi/fa_r_seed.nii.gz"

needs = pytest.mark.skipif(
    not (os.path.exists(FOD) and os.path.exists(TCKGEN)),
    reason="FOD or tckgen absent",
)


@needs
def test_ok_outcome_positive_control(tmp_path):
    """A known WM seed returns a plausible non-empty bundle (positive control)."""
    r = run_tckgen(FOD, FA_R_SEED, MASK, str(tmp_path),
                   TrackParams(seeds=20_000, select=1500), timeout_s=30)
    assert r.outcome is Outcome.OK
    assert r.n_accepted is not None and r.n_accepted > 0
    assert os.path.exists(r.tck_path)
    assert os.path.exists(os.path.join(str(tmp_path), "run.json"))


@needs
def test_every_returned_streamline_intersects_the_seed(tmp_path):
    """Independent check: tckedit -include seed count == accepted count."""
    r = run_tckgen(FOD, FA_R_SEED, MASK, str(tmp_path),
                   TrackParams(seeds=20_000, select=1500), timeout_s=30)
    assert r.outcome is Outcome.OK
    checked = str(tmp_path / "checked.tck")
    tckedit = TCKGEN.replace("tckgen", "tckedit")
    subprocess.run([tckedit, r.tck_path, checked, "-include", FA_R_SEED, "-force", "-quiet"],
                   check=True)
    tckinfo = TCKGEN.replace("tckgen", "tckinfo")
    out = subprocess.run([tckinfo, checked], capture_output=True, text=True)
    n_checked = next(int(l.split(":")[1]) for l in out.stdout.splitlines()
                     if l.strip().startswith("count:"))
    assert n_checked == r.n_accepted


@needs
def test_timeout_is_typed_not_zero(tmp_path):
    """An impossibly short deadline yields TIMEOUT with null counts, not a zero."""
    r = run_tckgen(FOD, FA_R_SEED, MASK, str(tmp_path),
                   TrackParams(seeds=2_000_000, select=100_000), timeout_s=0.5)
    assert r.outcome is Outcome.TIMEOUT
    assert r.n_accepted is None
    assert "killed" in (r.warning or "")


@needs
def test_serial_nthreads0_is_geometry_reproducible(tmp_path):
    """nthreads=0 -> identical STREAMLINE GEOMETRY across runs.

    NB: whole-file SHA is NOT stable — the .tck header embeds a `timestamp` and
    the (temp) output path in `command_history`, so file bytes differ even when
    the geometry is identical. Determinism is a property of the streamlines, not
    the provenance header; assert on geometry.
    """
    import hashlib
    import nibabel as nib

    def geohash(tck):
        h = hashlib.sha256()
        for arr in nib.streamlines.load(tck).streamlines:
            h.update(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
        return h.hexdigest()

    a = run_tckgen(FOD, FA_R_SEED, MASK, str(tmp_path / "a"),
                   TrackParams(seeds=8000, select=300, nthreads=0), timeout_s=60)
    b = run_tckgen(FOD, FA_R_SEED, MASK, str(tmp_path / "b"),
                   TrackParams(seeds=8000, select=300, nthreads=0), timeout_s=60)
    assert a.outcome is Outcome.OK and b.outcome is Outcome.OK
    assert geohash(a.tck_path) == geohash(b.tck_path)


@needs
def test_param_validation_rejects_out_of_range():
    with pytest.raises(ValueError):
        TrackParams(cutoff=2.0).validated()
    with pytest.raises(ValueError):
        TrackParams(cutoff=0.01).validated()  # below clinical floor 0.02
    with pytest.raises(ValueError):
        TrackParams(angle=200).validated()
    with pytest.raises(ValueError):
        TrackParams(minlength=300, maxlength=250).validated()
    with pytest.raises(ValueError):
        TrackParams(density="turbo").validated()


def test_live_hash_changes_with_cutoff():
    """Commercial threshold sweep must mint a new provenance hash."""
    a = TrackParams(cutoff=0.05, density="normal").validated().live_hash()
    b = TrackParams(cutoff=0.08, density="normal").validated().live_hash()
    c = TrackParams(cutoff=0.05, density="dense").validated().live_hash()
    assert a != b
    assert a != c
    assert len(a) == 16
