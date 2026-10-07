"""Honest clearance: distribution + seed exclusion, not global min."""

from __future__ import annotations

import os
import numpy as np
import pytest

from tractlab.clearance import (
    lesion_surface_points,
    clearance_report,
    min_clearance_mm,
    per_vertex_distance_mm,
    SEED_EXCLUDE_MM,
)

LESION = os.path.expanduser(
    "~/tractlab-data/cases/local-case/tracts/roi/lesion1_dwi.nii.gz"
)
needs = pytest.mark.skipif(not os.path.exists(LESION), reason="lesion absent")


@needs
def test_lesion_shell_is_nonempty_world_points():
    pts = lesion_surface_points(LESION)
    assert pts.shape[0] > 20 and pts.shape[1] == 3


@needs
def test_clearance_report_p5_not_global_min():
    """A single near-lesion vertex must not dominate if most streamlines are far."""
    shell = lesion_surface_points(LESION)
    origin = shell.mean(axis=0)
    # 99 far streamlines + 1 that touches the lesion
    far = [origin + np.array([80.0, 0, 0]) + np.random.randn(20, 3) * 0.1
           for _ in range(99)]
    near = [np.vstack([shell[0], shell[0] + np.array([0.5, 0, 0])])]
    rep = clearance_report(far + near, shell, seed_pts=None, geom_floor_mm=3.0)
    assert rep is not None
    assert rep.p05_mm is not None and rep.p50_mm is not None
    # p5 should be large (most lines are far); global min would be ~0
    assert rep.p05_mm > 50.0
    assert min_clearance_mm(far + near, shell) < 1.0  # old metric still tiny


@needs
def test_seed_exclusion_prevents_roi_abutment_artifact():
    """If the only near points are on the seed, they are excluded."""
    shell = lesion_surface_points(LESION)
    # streamline sits on lesion AND we declare those points as seed
    on_lesion = [shell[:10].copy()]
    rep_no = clearance_report(on_lesion, shell, seed_pts=None, geom_floor_mm=3.0)
    assert rep_no is not None and rep_no.p05_mm is not None
    assert rep_no.p05_mm < 1.0
    # seed = same points → all verts excluded → null clearance, n_seed_only=1
    rep = clearance_report(on_lesion, shell, seed_pts=shell[:10], seed_exclude_mm=1.0, geom_floor_mm=3.0)
    assert rep is not None
    assert rep.n_eligible == 0
    assert rep.p05_mm is None
    assert rep.n_seed_only == 1


def test_format_never_claims_sub_floor_precision():
    from tractlab.clearance import ClearanceReport
    r = ClearanceReport(
        p05_mm=0.05, p50_mm=1.0, n_streamlines=10, n_eligible=10, n_seed_only=0,
        geom_floor_mm=3.0, seed_exclude_mm=3.0, method="test",
    )
    assert r.format_p05() == "<3"
    assert r.display_p05_mm == 0.05
    assert "below recorded floor" in r.summary_line()
    r2 = ClearanceReport(
        p05_mm=12.3, p50_mm=20.0, n_streamlines=10, n_eligible=10, n_seed_only=0,
        geom_floor_mm=3.0, seed_exclude_mm=3.0, method="test",
    )
    assert r2.format_p05() == "12"


def test_clearance_honest_null_when_no_lesion():
    assert clearance_report([np.zeros((3, 3))], np.empty((0, 3)), geom_floor_mm=3.0) is None


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("seed", [None, np.empty((0, 3))])
def test_bank_method_never_claims_exclusion_without_a_seed(weighted, empty, seed):
    from tractlab.clearance import clearance_report_weighted
    lesion = np.array([[0., 0., 0.]])
    lines = [] if empty else [np.array([[1., 0., 0.], [10., 0., 0.]])]
    if weighted:
        report = clearance_report_weighted(lines, np.ones(len(lines)), lesion, seed_pts=seed, geom_floor_mm=3.)
    else:
        report = clearance_report(lines, lesion, seed_pts=seed, geom_floor_mm=3.)
    assert report is not None
    assert "all vertices; no seed exclusion" in report.method
    assert "seed-excluded" not in report.method
    assert report.p05_mm == (None if empty else 1.)
    assert report.n_seed_only == 0
    assert report.n_eligible == len(lines)


def test_weighted_clearance_uses_one_distance_pass_and_keeps_seed_exclusion(monkeypatch):
    """SIFT2 p5 keeps its weighted population without a discarded base pass."""
    import tractlab.clearance as clearance

    def unexpected_unweighted_pass(*args, **kwargs):
        raise AssertionError("weighted clearance repeated the unweighted pass")

    monkeypatch.setattr(clearance, "clearance_report", unexpected_unweighted_pass)
    lesion = np.array([[0.0, 0.0, 0.0]])
    seed = np.array([[100.0, 0.0, 0.0]])
    lines = [
        np.array([[1.0, 0.0, 0.0]]),
        np.array([[10.0, 0.0, 0.0]]),
        np.array([[20.0, 0.0, 0.0]]),
        np.array([[100.0, 0.0, 0.0]]),  # seed-only: its large weight is excluded
    ]
    report = clearance.clearance_report_weighted(
        lines,
        np.array([1.0, 100.0, 1.0, 1_000_000.0]),
        lesion,
        seed_pts=seed,
        seed_exclude_mm=1.0,
        geom_floor_mm=3.0,
    )

    assert report is not None
    assert report.p05_mm == pytest.approx(1.369)
    assert report.p50_mm == pytest.approx(5.5)
    assert report.n_streamlines == 4
    assert report.n_eligible == 3
    assert report.n_seed_only == 1
    assert report.geom_floor_mm == 3.0
    assert report.method == "SIFT2-weighted p5 of per-streamline min; seed-excluded >=1 mm"

    empty_report = clearance.clearance_report_weighted(
        [np.array([[100.0, 0.0, 0.0]])],
        np.array([1.0]),
        lesion,
        seed_pts=seed,
        seed_exclude_mm=1.0,
        geom_floor_mm=3.0,
    )
    assert empty_report is not None
    assert empty_report.p05_mm is None
    assert empty_report.p50_mm is None
    assert empty_report.n_eligible == 0
    assert empty_report.n_seed_only == 1
    assert empty_report.method == "SIFT2-weighted p5 of per-streamline min; seed-excluded >=1 mm"


def test_analytic_subset_full_when_under_cap():
    from tractlab.clearance import analytic_subset, ANALYTIC_CAP

    lines = [np.zeros((4, 3)) for _ in range(10)]
    sub, meta, idx = analytic_subset(lines, cap=ANALYTIC_CAP)
    assert meta["clearancePopulation"] == "full"
    assert meta["nAnalytic"] == 10
    assert meta["nAnalyticFull"] == 10
    assert len(sub) == 10
    assert len(idx) == 10


def test_analytic_subset_samples_not_length_rank():
    """p5 population must not prefer long streamlines when over cap."""
    from tractlab.clearance import analytic_subset

    # 30 short near-origin + 5 long far — length-rank would pick the 5 long
    short = [np.array([[0, 0, 0], [1, 0, 0]], dtype=float) for _ in range(30)]
    long = [
        np.linspace([0, 0, 0], [100, 0, 0], 20) for _ in range(5)
    ]
    lines = short + long
    sub, meta, idx = analytic_subset(lines, cap=10, rng_seed=0)
    assert meta["clearancePopulation"] == "random_sample"
    assert meta["nAnalytic"] == 10
    assert meta["nAnalyticFull"] == 35
    assert len(sub) == 10
    assert len(idx) == 10


def test_clearance_on_full_not_display_rank_is_less_optimistic():
    """Length-ranked display can inflate p5; analytic full must use all lines."""
    from tractlab.clearance import clearance_report

    # Lesion at origin shell
    shell = np.array([[0.0, 0.0, 0.0], [0.5, 0, 0], [0, 0.5, 0]])
    # Many short streamlines grazing lesion (low clearance)
    near = [
        np.array([[2.0, 0, 0], [3.0, 0, 0]], dtype=float) for _ in range(40)
    ]
    # Few long streamlines far away (high clearance) — length-rank prefers these
    far = [
        np.linspace([50, 0, 0], [150, 0, 0], 30) for _ in range(5)
    ]
    all_lines = near + far
    # Display-style: keep 5 longest → all far
    lengths = [
        float(np.sum(np.linalg.norm(np.diff(s, axis=0), axis=1))) for s in all_lines
    ]
    order = np.argsort(-np.asarray(lengths))
    display = [all_lines[i] for i in order[:5]]
    rep_disp = clearance_report(display, shell, geom_floor_mm=3.0)
    rep_full = clearance_report(all_lines, shell, geom_floor_mm=3.0)
    assert rep_disp is not None and rep_full is not None
    assert rep_disp.p05_mm is not None and rep_full.p05_mm is not None
    # Full population p5 must be strictly closer (more conservative) than display-only
    assert rep_full.p05_mm < rep_disp.p05_mm


def test_per_vertex_distance_shape_and_values():
    shell = np.array([[0.0, 0.0, 0.0]])
    verts = np.array([
        [[0, 0, 0], [3, 0, 0], [4, 0, 0]],
        [[0, 0, 10], [0, 0, 20], [0, 0, 5]],
    ], dtype=float)
    d = per_vertex_distance_mm(verts, shell)
    assert d.shape == (2, 3)
    assert np.allclose(d[0], [0, 3, 4], atol=1e-5)


def test_geom_floor_constant_is_gone():
    import tractlab.clearance as C
    assert not hasattr(C, "GEOM_FLOOR_MM")  # quarantined: floors are per-case data
    assert SEED_EXCLUDE_MM >= 2.0


# --- Task 16: per-case geometric floor, fail-closed ------------------------

def test_geom_floor_mm_reads_manifest():
    from tractlab.clearance import geom_floor_mm

    assert geom_floor_mm({"acquisition": {"geom_floor_mm": 3.0}}) == 3.0
    assert geom_floor_mm({"acquisition": {"geom_floor_mm": 1.5}}) == 1.5


def test_geom_floor_mm_missing_refuses_never_defaults():
    import pytest

    from tractlab.clearance import geom_floor_mm

    for manifest in ({}, {"acquisition": {}}, {"acquisition": None}):
        with pytest.raises(ValueError, match="geom_floor_mm"):
            geom_floor_mm(manifest)


def test_geom_floor_mm_invalid_values_refuse():
    import pytest

    from tractlab.clearance import geom_floor_mm

    for bad in (0, -1.0, True, "3.0", float("nan"), float("inf")):
        with pytest.raises(ValueError, match="geom_floor_mm"):
            geom_floor_mm({"acquisition": {"geom_floor_mm": bad}})


# --- 2026-09-07 audit S-14: derivation-aware floor ------------------------

def _rpe(manifest, *, signed):
    out = dict(manifest, derivations={"d1": {"kind": "rpe_pair"}}, active_derivation="d1")
    if signed:
        out["delta_qc"] = {"auto": {"median_mm": 0.6}, "approved_by": "erion",
                           "date": "2026-08-18", "sheet_sha": "x", "derivation": "d1"}
    return out


def test_geom_floor_mm_rpe_pair_signed_uses_delta_qc_median_not_inherited_acquisition():
    from tractlab.clearance import geom_floor_mm

    man = _rpe({"acquisition": {"geom_floor_mm": 3.0}}, signed=True)
    assert geom_floor_mm(man) == 0.6  # ADR-0004: acquisition.* survived migration; ignored
    assert geom_floor_mm(_rpe({}, signed=True)) == 0.6  # no acquisition block needed


def test_geom_floor_mm_rpe_pair_unsigned_refuses_even_with_acquisition_floor():
    import pytest

    from tractlab.clearance import geom_floor_mm

    with pytest.raises(ValueError, match="unsigned"):
        geom_floor_mm(_rpe({"acquisition": {"geom_floor_mm": 3.0}}, signed=False))


def test_geom_floor_mm_uncorrected_and_unmigrated_keep_reading_acquisition():
    from tractlab.clearance import geom_floor_mm

    unc = dict({"acquisition": {"geom_floor_mm": 3.0}},
               derivations={"d1": {"kind": "uncorrected"}}, active_derivation="d1")
    assert geom_floor_mm(unc) == 3.0
    assert geom_floor_mm({"acquisition": {"geom_floor_mm": 2.5}}) == 2.5


def test_hud_clearance_strings_print_a_sub_millimetre_floor_honestly():
    """S-14 follow-up: a signed 0.6 mm delta-QC floor must read 0.6, not be
    rounded up to 1 by :.0f — same formatter as the connectotomy note."""
    from tractlab.clearance import ClearanceReport, format_floor_mm
    from tractlab.connectotomy import format_floor_mm as via_connectotomy

    assert via_connectotomy is format_floor_mm
    assert [format_floor_mm(v) for v in (3.0, 0.6033, 1.5, 0.04)] == ["3", "0.6", "1.5", "0"]
    r = ClearanceReport(
        p05_mm=0.4, p50_mm=5.0, n_streamlines=10, n_eligible=10, n_seed_only=0,
        geom_floor_mm=0.6033, seed_exclude_mm=3.0, method="test",
    )
    assert r.format_p05() == "<0.6"
    assert "below recorded floor 0.6 mm" in r.summary_line()
    assert "floor 1 mm" not in r.summary_line()
