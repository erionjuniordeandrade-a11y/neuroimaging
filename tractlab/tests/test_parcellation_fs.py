"""Parcellation QC — laterality AND anteroposterior ordering.

The parcellation and the diffusion come from two different scan sessions
(~27-28 mm offset, ~6 deg rotation), so a header-only transfer covers only 56.8%
of the brain mask and puts rh-precentral 48% outside the brain. These tests fail
loudly on that, and the sabotage test proves they are not vacuous.
"""

from __future__ import annotations

import os
import numpy as np
import pytest

from tractlab.parcellation import (
    LABEL_IDS,
    load_parcellation,
    label_centroids_world,
    check_laterality_and_ap,
)

APARC = os.path.expanduser("~/tractlab-data/work0806/aparc_dwi.nii.gz")

pytestmark = pytest.mark.skipif(
    not os.path.exists(APARC),
    reason="aparc_dwi.nii.gz not present (run runbook step 3)",
)


def test_parcellation_shares_the_diffusion_grid():
    grid, labels = load_parcellation(APARC)
    assert grid.shape == (185, 185, 109)
    assert grid.axcodes == ("L", "P", "S")
    assert labels.shape == grid.shape


def test_required_labels_are_present_and_nonempty():
    grid, labels = load_parcellation(APARC)
    cents = label_centroids_world(grid, labels, LABEL_IDS)
    for name in ("lh_precentral", "rh_precentral",
                 "lh_superiorfrontal", "rh_superiorfrontal"):
        assert cents[name] is not None, f"{name} has zero voxels after resampling"


def test_laterality_right_is_positive_x():
    """World is RAS+: right-hemisphere centroids must have x > 0."""
    grid, labels = load_parcellation(APARC)
    cents = label_centroids_world(grid, labels, LABEL_IDS)
    assert cents["rh_precentral"][0] > 0.0
    assert cents["lh_precentral"][0] < 0.0
    assert cents["rh_superiorfrontal"][0] > 0.0
    assert cents["lh_superiorfrontal"][0] < 0.0


def test_anteroposterior_precentral_is_posterior_to_superiorfrontal():
    """The check that would have caught the .WRONG_ANTERIOR misplacement."""
    grid, labels = load_parcellation(APARC)
    cents = label_centroids_world(grid, labels, LABEL_IDS)
    for hemi in ("lh", "rh"):
        assert cents[f"{hemi}_precentral"][1] < cents[f"{hemi}_superiorfrontal"][1], (
            f"{hemi}-precentral is not posterior to {hemi}-superiorfrontal"
        )


def test_qc_reports_ok_on_real_data():
    grid, labels = load_parcellation(APARC)
    qc = check_laterality_and_ap(label_centroids_world(grid, labels, LABEL_IDS))
    assert qc.ok, f"QC failures: {qc.failures}"


def test_x_flip_sabotage_inverts_laterality_and_is_caught():
    """NON-VACUITY: an x-flipped affine must produce a WELL-FORMED WRONG answer
    that QC rejects — not an exception. Mirrors test_grid.py's sabotage gate.
    """
    grid, labels = load_parcellation(APARC)
    good = label_centroids_world(grid, labels, LABEL_IDS)
    assert check_laterality_and_ap(good).ok

    from dataclasses import replace
    flipped = grid.affine.copy()
    flipped[0, :] *= -1
    bad_grid = replace(grid, affine=flipped)
    bad = label_centroids_world(bad_grid, labels, LABEL_IDS)

    qc = check_laterality_and_ap(bad)
    assert not qc.ok, "x-flip was NOT caught — the laterality gate is vacuous"
    assert any("laterality" in f for f in qc.failures)
    # every lateral centroid must have moved to the opposite side
    for name in ("rh_precentral", "lh_precentral"):
        assert np.sign(bad[name][0]) == -np.sign(good[name][0])
