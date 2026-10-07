"""Crossed-FAT gate logic.

The gate is INVERTED relative to every other bundle in this project: midline
crossing is required, not vetoed. That inversion removed the contralateral veto
and left NO path condition at all, so the gate constrained only the two
endpoints' parcel labels and admitted 10-19 mm hairpins that never touch the
midline. Every test below that mentions "impossible" exists because of a
streamline the endpoint-only gate validated on real data.

Correctness is tested on synthetic geometry first, so the logic is provable
without patient data, then measured on the real tractogram.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from tractlab.crossedfat import (
    CC_LABEL_IDS,
    DEFAULT_CC_TOL_MM,
    DEFAULT_STANDOFF_MM,
    REASON_CROSSING_COUNT,
    REASON_FOOTPRINT,
    REASON_LESION,
    REASON_NOT_ANCHORED,
    REASON_STANDOFF,
    CrossedFatQC,
    callosal_voxels_world,
    classify_streamlines,
    midline_crossings,
    midsagittal_x_mm,
    sma_territory_mask,
    validate_crossed_fat,
)
from tractlab.fivett import tck_completeness
from tractlab.grid import Grid

OUT = os.path.expanduser("~/tractlab-data/work0806")
APARC = f"{OUT}/aparc_dwi.nii.gz"
TCK = f"{OUT}/crossed_fat_raw.tck"
LESION = os.path.expanduser(
    "~/tractlab-data/cases/local-case/tracts/roi/lesion1_dwi.nii.gz"
)

LPS = ("L", "P", "S")

# A synthetic callosal footprint: a rod of voxel centres on the midsagittal
# plane, running anteroposteriorly at z = 0.
def _cc_cloud(x_mid: float = 0.0) -> np.ndarray:
    return np.array([[x_mid, y, 0.0] for y in np.arange(-10.0, 10.5, 1.0)])


def _poly(*points) -> np.ndarray:
    return np.array(points, dtype=np.float64)


def _one_each(n: int = 1, swap: bool = False):
    """(in_lh, in_rh) where endpoint 0 is in lh and endpoint 1 is in rh."""
    lh = np.tile(np.array([[True, False]]), (n, 1))
    rh = np.tile(np.array([[False, True]]), (n, 1))
    return (rh, lh) if swap else (lh, rh)


def _no_lesion(n: int = 1) -> np.ndarray:
    return np.zeros((n, 2), dtype=bool)


def _classify(streamlines, in_lh, in_rh, in_lesion, *, x_mid=0.0, cc=None, **kw):
    return classify_streamlines(
        in_lh, in_rh, in_lesion, streamlines,
        x_mid_mm=x_mid,
        cc_world_xyz=_cc_cloud(x_mid) if cc is None else cc,
        **kw,
    )


# --------------------------------------------------------------------------
# midsagittal plane: measured, never assumed to be x = 0
# --------------------------------------------------------------------------

def _labels_with_cc(cc_i_slices=(40, 41, 42)) -> tuple[Grid, np.ndarray]:
    """Volume with lh/rh superiorfrontal plus a CC slab at known voxel columns."""
    shape = (100, 30, 10)
    labels = np.zeros(shape, dtype=np.int32)
    labels[20:25, :, :] = 1028
    labels[70:75, :, :] = 2028
    for n, i in enumerate(cc_i_slices):
        labels[i, 10:20, 4] = CC_LABEL_IDS[n % len(CC_LABEL_IDS)]
    # RAS+ affine with 1 mm voxels, origin shifted so the CC does not sit at x=0
    affine = np.eye(4)
    affine[0, 3] = -39.0
    return Grid(shape=shape, affine=affine, axcodes=("R", "A", "S")), labels


def test_midsagittal_plane_is_the_median_of_the_callosal_labels():
    """⛔ This subject's callosal midline is NOT x = 0.

    Labels 251-255 span world x [-5.13, +0.93] with median -1.93 mm in the real
    volume, because the parcellation was resampled across a ~27-28 mm
    inter-session registration. Assuming x = 0 carries ~2 mm of bias into every
    side and crossing decision.
    """
    grid, labels = _labels_with_cc(cc_i_slices=(40, 41, 42))
    cc = callosal_voxels_world(grid, labels)
    x_mid = midsagittal_x_mm(cc)

    assert cc.shape[1] == 3
    assert x_mid == pytest.approx(2.0)   # voxel i=41 -> world x = 41 - 39
    assert x_mid != 0.0


def test_absent_callosal_labels_raise_instead_of_falling_back_to_zero():
    """Unknown must be loud. A silent fallback to x = 0 is an honest-null breach."""
    grid, labels = _labels_with_cc()
    labels[np.isin(labels, CC_LABEL_IDS)] = 0
    with pytest.raises(ValueError, match="corpus callosum|251"):
        callosal_voxels_world(grid, labels)


# --------------------------------------------------------------------------
# crossing counting
# --------------------------------------------------------------------------

def test_a_straight_crossing_streamline_crosses_once():
    line = _poly([-20.0, 0.0, 0.0], [0.0, 0.0, 0.0], [20.0, 0.0, 0.0])
    crossings = midline_crossings(line, 0.0)
    assert crossings.shape == (1, 3)
    np.testing.assert_allclose(crossings[0], [0.0, 0.0, 0.0])


def test_a_streamline_staying_on_one_side_crosses_zero_times():
    line = _poly([-20.0, 0.0, 0.0], [-10.0, 5.0, 0.0], [-4.0, 0.0, 0.0])
    assert midline_crossings(line, 0.0).shape == (0, 3)


def test_a_zigzag_crosses_three_times():
    line = _poly([-20.0, 0.0, 0.0], [10.0, 1.0, 0.0], [-10.0, 2.0, 0.0],
                 [20.0, 3.0, 0.0])
    assert midline_crossings(line, 0.0).shape[0] == 3


def test_the_crossing_point_is_interpolated_not_snapped_to_a_vertex():
    line = _poly([-1.0, 0.0, 0.0], [3.0, 4.0, 8.0])
    crossings = midline_crossings(line, 0.0)
    # t = 1/4 along the segment
    np.testing.assert_allclose(crossings[0], [0.0, 1.0, 2.0])


def test_a_vertex_landing_exactly_on_the_midline_counts_as_one_crossing():
    line = _poly([-5.0, 0.0, 0.0], [0.0, 1.0, 0.0], [5.0, 2.0, 0.0])
    crossings = midline_crossings(line, 0.0)
    assert crossings.shape[0] == 1


def test_a_vertex_touching_the_midline_and_returning_is_not_a_crossing():
    """Grazing the plane and coming back is one side, not two."""
    line = _poly([-5.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-5.0, 2.0, 0.0])
    assert midline_crossings(line, 0.0).shape[0] == 0


def test_crossings_are_measured_against_the_supplied_midline():
    line = _poly([2.0, 0.0, 0.0], [20.0, 0.0, 0.0])
    assert midline_crossings(line, 0.0).shape[0] == 0
    assert midline_crossings(line, 10.0).shape[0] == 1


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------

def test_a_transcallosal_streamline_is_validated():
    """One endpoint left, one right, one crossing inside the callosal footprint."""
    line = _poly([-20.0, 0.0, 0.0], [0.0, 0.0, 0.0], [20.0, 0.0, 0.0])
    lh, rh = _one_each()
    qc = _classify([line], lh, rh, _no_lesion())

    assert qc.n_total == 1
    assert qc.n_validated == 1
    assert qc.n_impossible == 0
    assert qc.n_complete == 0
    assert qc.validated_indices == (0,)


def test_a_hairpin_whose_rh_endpoint_is_left_of_the_midline_is_not_validated():
    """⛔ The exact defect: 8 of 202 "validated" streamlines were 10-19 mm hairpins.

    The rh SMA territory leaks 73 voxels across the midline (to x = -3.58 mm)
    because the parcellation crossed a ~27-28 mm inter-session registration and
    the two territories are medial and adjacent. Both endpoints are physically in
    the LEFT hemisphere, so the streamline is not transcallosal — but its parcel
    LABELS say one endpoint per hemisphere, and the endpoint-only gate believed
    the labels.
    """
    line = _poly([-3.9, 0.0, 0.0], [-8.0, 6.0, 0.0], [-0.4, 12.0, 0.0])
    lh, rh = _one_each()
    qc = _classify([line], lh, rh, _no_lesion())

    assert qc.n_validated == 0, "a same-hemisphere hairpin was validated"
    assert qc.n_impossible == 1
    assert qc.rejections[REASON_STANDOFF] == 1


def test_an_endpoint_inside_the_standoff_band_is_rejected_even_across_the_midline():
    """A 1 mm-from-midline "hemispheric" endpoint is registration error, not anatomy."""
    line = _poly([-20.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 2.0, 0.0])
    lh, rh = _one_each()
    qc = _classify([line], lh, rh, _no_lesion())

    assert qc.n_validated == 0
    assert qc.rejections[REASON_STANDOFF] == 1


def test_a_zigzagging_streamline_is_impossible_not_validated():
    line = _poly([-20.0, 0.0, 0.0], [10.0, 1.0, 0.0], [-10.0, 2.0, 0.0],
                 [20.0, 3.0, 0.0])
    lh, rh = _one_each()
    qc = _classify([line], lh, rh, _no_lesion())

    assert qc.n_validated == 0
    assert qc.n_impossible == 1
    assert qc.rejections[REASON_CROSSING_COUNT] == 1


def test_a_crossing_outside_the_callosal_footprint_is_impossible():
    """⛔ ACT does not model the falx.

    An FOD peak can bridge the thin interhemispheric CSF gap between the two
    abutting medial gyri, so a "crossing" at the medial cortical surface is an
    artifact rather than a callosal fibre. The crossing must lie in the callosal
    footprint.
    """
    line = _poly([-20.0, 60.0, 60.0], [0.0, 60.0, 60.0], [20.0, 60.0, 60.0])
    lh, rh = _one_each()
    qc = _classify([line], lh, rh, _no_lesion())

    assert qc.n_validated == 0
    assert qc.n_impossible == 1
    assert qc.rejections[REASON_FOOTPRINT] == 1


def test_a_streamline_outside_both_territories_is_complete_not_impossible():
    """Never reaching the bundle's endpoints is ordinary, not a geometric fault."""
    line = _poly([-20.0, 0.0, 0.0], [-10.0, 1.0, 0.0])
    in_lh = np.array([[True, False]])
    in_rh = np.array([[False, False]])
    qc = _classify([line], in_lh, in_rh, _no_lesion())

    assert qc.n_validated == 0
    assert qc.n_impossible == 0
    assert qc.n_complete == 1
    assert qc.rejections[REASON_NOT_ANCHORED] == 1


def test_same_hemisphere_endpoints_are_rejected():
    """Both ends in the same SMA territory is an intrahemispheric U-fibre."""
    line = _poly([-20.0, 0.0, 0.0], [-10.0, 1.0, 0.0])
    in_lh = np.array([[True, True]])
    in_rh = np.array([[False, False]])
    qc = _classify([line], in_lh, in_rh, _no_lesion())

    assert qc.n_bihemispheric == 0
    assert qc.n_validated == 0


def test_lesion_terminating_streamline_is_rejected_even_if_transcallosal():
    """The 6.3% cortical-GM mislabel inside the lesion must not buy an endpoint."""
    line = _poly([-20.0, 0.0, 0.0], [0.0, 0.0, 0.0], [20.0, 0.0, 0.0])
    lh, rh = _one_each()
    qc = _classify([line], lh, rh, np.array([[False, True]]))

    assert qc.n_bihemispheric == 1
    assert qc.n_lesion_terminating == 1
    assert qc.n_validated == 0, "a lesion-terminating streamline was validated"
    assert qc.rejections[REASON_LESION] == 1


def test_swapping_the_lh_and_rh_territories_collapses_the_validated_count():
    """⛔ A left/right symmetric gate cannot notice a swapped pair of masks.

    "One endpoint in each" is symmetric, so swapping the territories was
    invisible. Tying the lh-labelled endpoint to the LEFT of the measured midline
    makes the swap fail loudly.
    """
    line = _poly([-20.0, 0.0, 0.0], [0.0, 0.0, 0.0], [20.0, 0.0, 0.0])
    ok_lh, ok_rh = _one_each()
    assert _classify([line], ok_lh, ok_rh, _no_lesion()).n_validated == 1

    sw_lh, sw_rh = _one_each(swap=True)
    assert _classify([line], sw_lh, sw_rh, _no_lesion()).n_validated == 0


def test_the_measured_midline_is_used_rather_than_x_equals_zero():
    """Endpoints on the same side of x=0 but opposite sides of the real midline."""
    line = _poly([2.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0])
    lh, rh = _one_each()
    cc = _cc_cloud(10.0)

    at_zero = _classify([line], lh, rh, _no_lesion(), x_mid=0.0, cc=cc)
    at_real = _classify([line], lh, rh, _no_lesion(), x_mid=10.0, cc=cc)

    assert at_zero.n_validated == 0
    assert at_real.n_validated == 1


# --------------------------------------------------------------------------
# the impossible : complete : validated partition
# --------------------------------------------------------------------------

def _mixed_population():
    """Six streamlines, one per bucket-reason."""
    lines = [
        _poly([-20.0, 0.0, 0.0], [0.0, 0.0, 0.0], [20.0, 0.0, 0.0]),   # validated
        _poly([-20.0, 1.0, 0.0], [0.0, 1.0, 0.0], [20.0, 1.0, 0.0]),   # lesion
        _poly([-3.9, 0.0, 0.0], [-8.0, 6.0, 0.0], [-0.4, 12.0, 0.0]),  # standoff
        _poly([-20.0, 0.0, 0.0], [10.0, 1.0, 0.0], [-10.0, 2.0, 0.0],
              [20.0, 3.0, 0.0]),                                        # zigzag
        _poly([-20.0, 60.0, 60.0], [0.0, 60.0, 60.0], [20.0, 60.0, 60.0]),  # falx
        _poly([-20.0, 0.0, 0.0], [-10.0, 1.0, 0.0]),                    # not anchored
    ]
    in_lh = np.array([[True, False]] * 5 + [[True, True]])
    in_rh = np.array([[False, True]] * 5 + [[False, False]])
    in_lesion = np.zeros((6, 2), dtype=bool)
    in_lesion[1, 1] = True
    return lines, in_lh, in_rh, in_lesion


def test_the_three_buckets_partition_the_total_exactly():
    """impossible + complete + validated == n_total, with no double counting."""
    lines, in_lh, in_rh, in_lesion = _mixed_population()
    qc = _classify(lines, in_lh, in_rh, in_lesion)

    assert qc.n_total == 6
    assert qc.n_validated == 1
    assert qc.n_impossible == 3
    assert qc.n_complete == 2
    assert qc.n_impossible + qc.n_complete + qc.n_validated == qc.n_total


def test_a_shadowed_reason_is_still_counted_independently():
    """⛔ A zero in the first-match tally must not read as "no such streamline".

    Reasons are assigned first-match, so a hairpin that also fails the standoff is
    filed under the standoff and the crossing-count row shows 0. On real data all
    7 zero-crossing hairpins are shadowed that way, and a reader would conclude
    the hairpins were gone. The independent per-conjunct counts say otherwise.
    """
    # one endpoint just inside the standoff band AND zero midline crossings
    line = _poly([-3.9, 0.0, 0.0], [-8.0, 6.0, 0.0], [-0.4, 12.0, 0.0])
    lh, rh = _one_each()
    qc = _classify([line], lh, rh, _no_lesion())

    assert qc.rejections[REASON_STANDOFF] == 1
    assert qc.rejections[REASON_CROSSING_COUNT] == 0, "expected the shadowing"
    assert qc.conjunct_failures[REASON_CROSSING_COUNT] == 1, (
        "the crossing failure vanished from the report entirely"
    )
    assert qc.conjunct_failures[REASON_STANDOFF] == 1
    assert "shadowed" in qc.report.lower()


def test_every_streamline_gets_exactly_one_auditable_reason():
    lines, in_lh, in_rh, in_lesion = _mixed_population()
    qc = _classify(lines, in_lh, in_rh, in_lesion)
    assert sum(qc.rejections.values()) + qc.n_validated == qc.n_total


def test_an_inconsistent_partition_cannot_be_constructed():
    """The dataclass enforces its own arithmetic rather than trusting the caller."""
    with pytest.raises(ValueError, match="partition"):
        CrossedFatQC(
            n_total=10, n_bihemispheric=5, n_lesion_terminating=1, n_validated=4,
            n_impossible=1, n_complete=1,   # 4 + 1 + 1 != 10
            n_candidate_lesion_terminating=1,
            x_mid_mm=-1.93, standoff_mm=5.0, cc_tol_mm=5.0,
        )


def test_mismatched_input_lengths_raise_instead_of_misaligning():
    """n_total came from in_lh alone; a shorter in_lesion silently mis-aligned."""
    line = _poly([-20.0, 0.0, 0.0], [20.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="shape|length"):
        _classify([line, line], np.array([[True, False]] * 2),
                  np.array([[False, True]] * 2), _no_lesion(1))
    with pytest.raises(ValueError, match="shape|length"):
        _classify([line], np.array([[True, False]] * 2),
                  np.array([[False, True]] * 2), _no_lesion(2))


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------

def _full_qc(**over) -> CrossedFatQC:
    kw = dict(
        n_total=100, n_bihemispheric=40, n_lesion_terminating=5, n_validated=35,
        n_impossible=4, n_complete=61, n_candidate_lesion_terminating=1,
        x_mid_mm=-1.93, standoff_mm=5.0, cc_tol_mm=5.0,
    )
    kw.update(over)
    return CrossedFatQC(**kw)


def test_report_names_the_denominator_and_the_triple():
    r = _full_qc().report
    assert "35" in r and "100" in r
    assert "raw" not in r.lower()
    for token in ("impossible", "complete", "validated"):
        assert token in r.lower()
    assert "4" in r and "61" in r


def test_report_states_the_thresholds_and_the_measured_midline():
    r = _full_qc().report
    assert "-1.93" in r
    assert "5.0" in r
    assert "standoff" in r.lower()


def test_report_names_the_lesion_overlap_so_subtraction_cannot_mislead():
    """⛔ 203 - 55 = 148 is wrong: only 1 of the 55 was also a candidate."""
    r = _full_qc().report
    assert "overlap" in r.lower()
    assert "1" in r


def test_report_states_the_territory_as_a_choice_and_carries_the_caveats():
    r = _full_qc().report
    assert "stated choice" in r
    assert "not an atlas label" in r
    assert "not navigation" in r
    assert "topup" in r
    assert "posterior third of the AP extent" in r


def test_report_does_not_fabricate_a_percentage_when_nothing_was_read():
    """⛔ 0.00% of nothing is a fabricated number, not a measurement."""
    r = _full_qc(n_total=0, n_bihemispheric=0, n_lesion_terminating=0,
                 n_validated=0, n_impossible=0, n_complete=0,
                 n_candidate_lesion_terminating=0).report
    assert "0.00%" not in r
    assert "unknown" in r.lower()


def test_report_marks_an_unknown_midline_as_unknown():
    r = _full_qc(x_mid_mm=None).report
    assert "unknown" in r.lower()


def test_report_carries_the_callosal_distance_sensitivity():
    """One cherry-picked tolerance hides the sensitivity; show the curve."""
    qc = _full_qc(
        callosal_distance_mm=(0.5, 1.5, 3.0, 7.0, 12.0),
        validated_by_cc_tol_mm={2.0: 2, 5.0: 3, 10.0: 4, 15.0: 5},
        validated_by_standoff_mm={0.0: 9, 2.0: 7, 5.0: 5, 10.0: 2},
    )
    r = qc.report
    assert "2 mm" in r or "2.0 mm" in r
    assert "15" in r


# --------------------------------------------------------------------------
# territory mask
# --------------------------------------------------------------------------

def _two_hemisphere_labels(shape=(10, 30, 10)) -> np.ndarray:
    """lh and rh superiorfrontal slabs, 30 voxels deep along voxel axis 1."""
    labels = np.zeros(shape, dtype=np.int32)
    labels[2:5, :, :] = 1028   # lh superiorfrontal
    labels[6:9, :, :] = 2028   # rh superiorfrontal
    return labels


def test_sma_territory_is_exactly_the_posterior_third_of_the_ap_extent():
    """30 AP slices -> exactly 10, at the HIGH-index (posterior) end for LPS.

    The previous implementation kept 11 of 30 (36.7%) because it took a third of
    the extent and then included the boundary slice as well. A "third" that is
    really 37% quietly inflates every downstream count.
    """
    labels = _two_hemisphere_labels()
    lh = sma_territory_mask(labels, "lh", LPS)

    j = np.argwhere(lh)[:, 1]
    assert (j.min(), j.max()) == (20, 29), "not exactly the posterior third"
    assert len(np.unique(j)) == 10
    assert lh.sum() == 3 * 10 * 10


def test_sma_territory_masks_are_hemisphere_specific_and_disjoint():
    labels = _two_hemisphere_labels()
    lh = sma_territory_mask(labels, "lh", LPS)
    rh = sma_territory_mask(labels, "rh", LPS)

    assert lh.sum() > 0 and rh.sum() > 0
    assert not (lh & rh).any(), "hemisphere masks overlap"
    assert lh.sum() == rh.sum()
    # each mask stays inside its own parent parcel
    assert (labels[lh] == 1028).all()
    assert (labels[rh] == 2028).all()


def test_an_anterior_posterior_flipped_volume_still_gets_the_POSTERIOR_third():
    """⛔ On ("L","A","S") voxel j increases ANTERIORLY, so posterior is LOW j.

    Hardcoding "posterior == high j" returns the ANTERIOR third here — still a
    third of the volume, still hemisphere-specific, so no downstream check
    notices. The direction is derived from the volume's own axcodes instead.
    """
    labels = _two_hemisphere_labels()
    lh = sma_territory_mask(labels, "lh", ("L", "A", "S"))

    j = np.argwhere(lh)[:, 1]
    assert (j.min(), j.max()) == (0, 9), "took the anterior third on an LAS volume"


def test_the_ap_axis_is_found_wherever_it_sits_not_assumed_to_be_axis_1():
    """A volume whose AP axis is voxel axis 2 must split along axis 2."""
    labels = np.zeros((10, 10, 30), dtype=np.int32)
    labels[2:5, :, :] = 1028
    lh = sma_territory_mask(labels, "lh", ("L", "S", "P"))

    k = np.argwhere(lh)[:, 2]
    assert (k.min(), k.max()) == (20, 29)


def test_a_volume_with_no_anterior_posterior_axis_raises():
    """Fail loud rather than split along an arbitrary axis."""
    labels = _two_hemisphere_labels()
    with pytest.raises(ValueError, match="anterior|posterior|axcodes"):
        sma_territory_mask(labels, "lh", ("L", "I", "R"))


def test_missing_superiorfrontal_raises_instead_of_returning_an_empty_mask():
    """⛔ An all-False mask reports "validated 0 / 20000 (0.00%)".

    That is indistinguishable from "checked, and genuinely zero", so a
    parcellation that failed to transfer looks like a negative result. Unknown
    must be loud.
    """
    labels = np.zeros((10, 30, 10), dtype=np.int32)
    labels[2:5, :, :] = 1028  # lh present, rh absent

    assert sma_territory_mask(labels, "lh", LPS).any()
    with pytest.raises(ValueError, match="rh_superiorfrontal|zero voxels"):
        sma_territory_mask(labels, "rh", LPS)


def test_bad_hemisphere_string_raises():
    with pytest.raises(ValueError, match="hemi"):
        sma_territory_mask(_two_hemisphere_labels(), "left", LPS)


# --------------------------------------------------------------------------
# the wiring: validate_crossed_fat on synthetic volumes
#
# The wiring had no coverage at all. Swapping the lh/rh masks, or passing the
# lesion mask where a territory belongs, would have passed every synthetic test
# above, because those tests call classify_streamlines directly and never
# exercise the function that assembles its arguments.
# --------------------------------------------------------------------------

def _synthetic_case(tmp_path, swap_hemispheres: bool = False):
    """A 1 mm RAS volume with lh/rh territories, a callosum, and a 3-streamline .tck.

    World x = voxel i - 20, so voxel i < 20 is the left hemisphere. axcodes are
    ("R","A","S"), where voxel j increases ANTERIORLY, so the posterior third of
    the AP extent is the LOW-j third.
    """
    import nibabel as nib
    from nibabel.streamlines import TckFile, Tractogram

    shape = (40, 30, 10)
    affine = np.eye(4)
    affine[0, 3] = -20.0

    lh_id, rh_id = (2028, 1028) if swap_hemispheres else (1028, 2028)
    labels = np.zeros(shape, dtype=np.int32)
    labels[5:10, :, :] = lh_id            # left slab,  world x in [-15, -11]
    labels[30:40, :, :] = rh_id           # right slab, world x in [+10, +19]
    labels[20, 10:21, 5] = 253            # CC_Central at world x = 0

    aparc = tmp_path / ("aparc_swapped.nii.gz" if swap_hemispheres else "aparc.nii.gz")
    nib.save(nib.Nifti1Image(labels, affine), str(aparc))

    def _lesion(name, voxel):
        mask = np.zeros(shape, dtype=np.uint8)
        mask[voxel] = 1
        path = tmp_path / name
        nib.save(nib.Nifti1Image(mask, affine), str(path))
        return str(path)

    lesion_far = _lesion("lesion_far.nii.gz", (1, 1, 1))
    lesion_on_endpoint = _lesion("lesion_hit.nii.gz", (32, 3, 5))

    lines = [
        # 0: transcallosal, both endpoints inside their posterior-third territory
        _poly([-13.0, 3.0, 5.0], [0.0, 15.0, 5.0], [12.0, 3.0, 5.0]),
        # 1: right endpoint far outside the volume — must be DISCARDED, and the
        #    clamped position (voxel 39, 0, 9) sits inside the rh territory, so a
        #    clamping implementation would count this as an anchored candidate.
        _poly([-13.0, 3.0, 5.0], [5000.0, -5000.0, 5000.0]),
        # 2: both endpoints in the ANTERIOR two thirds — outside both territories
        _poly([-13.0, 25.0, 5.0], [0.0, 15.0, 5.0], [12.0, 25.0, 5.0]),
    ]
    tck = tmp_path / ("tck_swapped.tck" if swap_hemispheres else "s.tck")
    TckFile(Tractogram(lines, affine_to_rasmm=np.eye(4))).save(str(tck))

    return str(tck), str(aparc), lesion_far, lesion_on_endpoint


def test_validate_crossed_fat_validates_only_the_transcallosal_streamline(tmp_path):
    tck, aparc, lesion_far, _ = _synthetic_case(tmp_path)
    qc = validate_crossed_fat(tck, aparc, lesion_far)

    assert qc.n_total == 3
    assert qc.n_validated == 1
    assert qc.validated_indices == (0,)
    assert qc.n_impossible + qc.n_complete + qc.n_validated == 3
    assert qc.x_mid_mm == pytest.approx(0.0)


def test_an_out_of_bounds_endpoint_is_not_clamped_into_a_territory(tmp_path):
    """⛔ The never-clamp rule, exercised through the real wiring.

    Streamline 1's far endpoint clamps onto voxel (39, 0, 9), which IS inside the
    rh territory, so a clamping lookup would report it as an endpoint-anchored
    candidate. Only the anchor count can show this; every later conjunct would
    reject the streamline for its own reasons and hide the bug.
    """
    tck, aparc, lesion_far, _ = _synthetic_case(tmp_path)
    qc = validate_crossed_fat(tck, aparc, lesion_far)

    assert qc.n_bihemispheric == 1, (
        "an out-of-bounds endpoint was clamped onto the volume face and counted "
        "as being inside the rh SMA territory"
    )


def test_swapping_the_hemisphere_labels_drops_the_validated_count_to_zero(tmp_path):
    """Sabotage control: the gate must bind each territory to a SIDE of x_mid.

    "One endpoint in each territory" is symmetric in lh/rh, so a swapped pair of
    masks was invisible to every earlier test. The swapped volume must validate
    nothing.
    """
    ok_tck, ok_aparc, lesion_far, _ = _synthetic_case(tmp_path)
    assert validate_crossed_fat(ok_tck, ok_aparc, lesion_far).n_validated == 1

    sw_tck, sw_aparc, sw_lesion, _ = _synthetic_case(tmp_path, swap_hemispheres=True)
    swapped = validate_crossed_fat(sw_tck, sw_aparc, sw_lesion)
    assert swapped.n_validated == 0, "swapped lh/rh territories still validated"
    assert swapped.rejections[REASON_STANDOFF] >= 1


def test_the_lesion_mask_argument_actually_excludes_an_endpoint(tmp_path):
    """Proves the lesion path is wired to the lesion slot rather than ignored."""
    tck, aparc, lesion_far, lesion_on_endpoint = _synthetic_case(tmp_path)

    assert validate_crossed_fat(tck, aparc, lesion_far).n_lesion_terminating == 0

    hit = validate_crossed_fat(tck, aparc, lesion_on_endpoint)
    assert hit.n_lesion_terminating == 1
    assert hit.n_candidate_lesion_terminating == 1
    assert hit.n_validated == 0
    assert hit.rejections[REASON_LESION] == 1


def test_validate_crossed_fat_honours_the_thresholds_it_is_given(tmp_path):
    """A 20 mm standoff cannot be met by an endpoint 13 mm from the midline."""
    tck, aparc, lesion_far, _ = _synthetic_case(tmp_path)
    assert validate_crossed_fat(tck, aparc, lesion_far, standoff_mm=20.0).n_validated == 0
    assert validate_crossed_fat(tck, aparc, lesion_far, cc_tol_mm=0.0).n_validated == 1


def test_validate_crossed_fat_refuses_an_unfinished_tractogram(tmp_path):
    """A partial .tck must raise, not quietly shrink the denominator."""
    from nibabel.streamlines import TckFile, Tractogram

    tck, aparc, lesion_far, _ = _synthetic_case(tmp_path)
    partial = tmp_path / "partial.tck"
    TckFile(
        Tractogram([_poly([-13.0, 3.0, 5.0], [0.0, 15.0, 5.0], [12.0, 3.0, 5.0])],
                   affine_to_rasmm=np.eye(4)),
        header={"command_history": "tckgen a b -select 20000",
                "total_count": "10", "max_num_seeds": "20000000"},
    ).save(str(partial))

    with pytest.raises(ValueError, match="not a finished tractogram"):
        validate_crossed_fat(str(partial), aparc, lesion_far)


def test_validate_crossed_fat_raises_when_the_callosum_is_absent(tmp_path):
    """No CC labels means no measurable midline; assuming x = 0 is forbidden."""
    import nibabel as nib

    tck, aparc, lesion_far, _ = _synthetic_case(tmp_path)
    img = nib.load(aparc)
    labels = np.asarray(img.dataobj).astype(np.int32)
    labels[np.isin(labels, CC_LABEL_IDS)] = 0
    no_cc = tmp_path / "aparc_no_cc.nii.gz"
    nib.save(nib.Nifti1Image(labels, img.affine), str(no_cc))

    with pytest.raises(ValueError, match="corpus callosum|251"):
        validate_crossed_fat(tck, str(no_cc), lesion_far)


# --------------------------------------------------------------------------
# real data
# --------------------------------------------------------------------------

def _real_inputs_ready() -> bool:
    return (
        os.path.exists(APARC)
        and os.path.exists(LESION)
        and tck_completeness(TCK).is_complete
    )


@pytest.mark.skipif(
    not _real_inputs_ready(),
    reason="crossed-FAT inputs not present or unfinished (runbook steps 3, 5, 6)",
)
def test_real_tractogram_reports_a_validated_count():
    qc = validate_crossed_fat(TCK, APARC, LESION)
    print("\n" + qc.report)

    assert qc.n_total > 0, "tck parsed as complete but holds no streamlines"
    assert qc.n_impossible + qc.n_complete + qc.n_validated == qc.n_total
    assert qc.x_mid_mm is not None


@pytest.mark.skipif(
    not _real_inputs_ready(),
    reason="crossed-FAT inputs not present or unfinished (runbook steps 3, 5, 6)",
)
def test_every_validated_streamline_independently_survives_the_path_conditions():
    """Re-derive the geometry from the .tck instead of trusting the gate's own count.

    `n_validated <= n_bihemispheric <= n_total` is an inequality the
    implementation cannot violate for any input — it measures itself. This
    recomputes each accepted streamline's crossings and callosal distance from the
    vertices and the CC labels, and would fail if the gate ever admitted a
    non-crossing hairpin again.
    """
    from scipy.spatial import cKDTree

    from tractlab.fivett import read_streamlines
    from tractlab.parcellation import load_parcellation

    qc = validate_crossed_fat(TCK, APARC, LESION)
    if qc.n_validated == 0:
        pytest.skip("nothing validated — no accepted streamline to re-check")

    grid, labels = load_parcellation(APARC)
    cc = callosal_voxels_world(grid, labels)
    x_mid = midsagittal_x_mm(cc)
    tree = cKDTree(cc)
    streamlines = read_streamlines(TCK)

    for i in qc.validated_indices:
        vertices = streamlines[i]
        crossings = midline_crossings(vertices, x_mid)
        assert crossings.shape[0] == 1, (
            f"streamline {i} was validated with {crossings.shape[0]} midline "
            "crossings — the path condition is not being applied"
        )
        assert tree.query(crossings)[0][0] <= qc.cc_tol_mm
        d0 = vertices[0, 0] - x_mid
        d1 = vertices[-1, 0] - x_mid
        assert min(d0, d1) <= -qc.standoff_mm and max(d0, d1) >= qc.standoff_mm

    print(
        f"\nre-derived {len(qc.validated_indices)} validated streamlines: "
        f"all cross the measured midline (x={x_mid:.2f} mm) exactly once, "
        f"within {qc.cc_tol_mm} mm of a callosal voxel"
    )


@pytest.mark.skipif(
    not _real_inputs_ready(),
    reason="crossed-FAT inputs not present or unfinished (runbook steps 3, 5, 6)",
)
def test_the_defaults_are_reported_not_tuned():
    """Guards against a later "improvement" that raises the yield by loosening."""
    qc = validate_crossed_fat(TCK, APARC, LESION)
    assert qc.standoff_mm == DEFAULT_STANDOFF_MM
    assert qc.cc_tol_mm == DEFAULT_CC_TOL_MM
    assert set(qc.validated_by_cc_tol_mm) >= {2.0, 5.0, 10.0, 15.0}
