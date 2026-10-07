"""Along-tract profiles — synthetic grids, no PHI."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from tractlab.export_tck import write_tck
from tractlab.profile import (
    CLAIM,
    HashMismatch,
    InvalidBankId,
    LesionInvalid,
    UnknownBank,
    ScalarMissing,
    ScalarUnsupported,
    UnknownFamily,
    ZERO_OR_MISSING_THRESHOLD,
    aggregate_nodes,
    compute_bank_profile,
    lesion_distance_track,
    orient_streamlines,
    parse_bank_id,
    resample,
    resolve_orientation,
    resolve_scalar_entry,
    sample_scalar,
    sha256_file,
    zero_or_missing_warning,
)
from tractlab.runtime_identity import BankSourceChanged
from tractlab.scalar_maps import PRODUCTS, StaleScalarMaps, decide, write_sidecar
from tractlab.serve import make_handler

TCKRESAMPLE = os.path.expanduser("~/mrtrix3/bin/tckresample")


def _affine(spacing: float = 1.0) -> np.ndarray:
    aff = np.eye(4)
    aff[0, 0] = aff[1, 1] = aff[2, 2] = spacing
    return aff


def _save_nifti(path, data, spacing: float = 1.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = nib.Nifti1Image(np.asarray(data), _affine(spacing))
    nib.save(img, path)
    return path


def _line(xyz0, xyz1, n=11) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    a = np.asarray(xyz0, dtype=np.float64)
    b = np.asarray(xyz1, dtype=np.float64)
    return a[None, :] + t[:, None] * (b - a)[None, :]


def test_parse_bank_id_families():
    assert parse_bank_id("bank_cst_r") == ("cst", "r", False)
    assert parse_bank_id("bank_fat_l_soft") == ("fat", "l", True)
    assert parse_bank_id("bank_slf3_r") == ("slf3", "r", False)
    assert parse_bank_id("bank_or_r_meyer") == ("or", "r", False)
    with pytest.raises(UnknownFamily, match="no orientation rule"):
        parse_bank_id("bank_cc_forceps_major")


def test_parse_bank_id_rejects_noncanonical_side_tokens():
    with pytest.raises(InvalidBankId, match="side token"):
        parse_bank_id("bank_fat_right")
    with pytest.raises(InvalidBankId, match="side token"):
        parse_bank_id("bank_cst_left")
    with pytest.raises(InvalidBankId, match="side token"):
        parse_bank_id("bank_slf3_R")


def test_straight_line_matches_linear_scalar_gradient(tmp_path):
    # Identity 1 mm affine. data[i,j,k] = i / 19 so world-x samples as x/19.
    data = np.zeros((20, 8, 8), dtype=np.float32)
    for i in range(20):
        data[i, :, :] = i / 19.0
    fa = _save_nifti(str(tmp_path / "fa.nii.gz"), data)
    line = _line((0, 4, 4), (19, 4, 4), n=20)
    samples = sample_scalar(resample([line], 100), fa)
    assert samples.shape == (1, 100)
    expected = np.linspace(0.0, 1.0, 100)
    np.testing.assert_allclose(samples[0], expected, atol=1e-6)
    nodes = aggregate_nodes(samples)
    for k in (0, 25, 50, 75, 99):
        assert nodes["median"][k] == pytest.approx(expected[k], abs=1e-6)
        assert nodes["n"][k] == 1
        assert nodes["n_valid"][k] == 1
    # Node 0 sits on FA==0 (the ramp origin). Display threshold flags exact zeros.
    assert nodes["n_zero"][0] == 1 and nodes["zero_or_missing_flag"][0] is True
    assert nodes["zero_or_missing_flag"][25] is False
    warning = zero_or_missing_warning(nodes)
    assert warning is not None and "nodes 0" in warning
    assert "masked" not in warning.lower()


def test_curved_line_constant_scalar_is_flat(tmp_path):
    data = np.full((12, 12, 12), 0.42, dtype=np.float32)
    fa = _save_nifti(str(tmp_path / "fa.nii.gz"), data)
    t = np.linspace(0, np.pi, 40)
    curve = np.column_stack([6 + 4 * np.cos(t), 6 + 4 * np.sin(t), np.full_like(t, 6.0)])
    samples = sample_scalar(resample([curve], 50), fa)
    np.testing.assert_allclose(samples[0], 0.42, atol=1e-6)
    nodes = aggregate_nodes(samples)
    assert nodes["mean"][0] == pytest.approx(0.42, abs=1e-6)
    assert nodes["p25"][10] == pytest.approx(0.42, abs=1e-6)


def test_cst_orientation_flips_reversed_streamlines(tmp_path):
    up = _line((0, 0, 10), (0, 0, 90), n=9)
    down = up[::-1].copy()
    result = resolve_orientation("bank_cst_r", [up, down], str(tmp_path))
    rec = result["record"]
    assert rec["rule"] == "inferior"
    assert rec["n_flipped"] == 1
    for line in result["streamlines"]:
        assert line[0, 2] < line[-1, 2]
    assert rec["anchor_mm"][2] == pytest.approx(10.0)


def test_fat_uses_first_roi_centroid_then_anterior_fallback(tmp_path):
    roi = np.zeros((10, 10, 10), dtype=np.uint8)
    roi[5, 9, 5] = 1  # high-y voxel → world (5, 9, 5)
    _save_nifti(str(tmp_path / "tracts" / "roi" / "fat_r_sfg_dil1.nii.gz"), roi)
    posterior = _line((5, 1, 5), (5, 9, 5), n=7)
    reversed_line = posterior[::-1].copy()
    result = resolve_orientation("bank_fat_r", [posterior, reversed_line], str(tmp_path))
    rec = result["record"]
    assert rec["rule"] == "roi_centroid"
    assert rec["roi_path"].endswith("fat_r_sfg_dil1.nii.gz")
    roi_file = tmp_path / "tracts" / "roi" / "fat_r_sfg_dil1.nii.gz"
    assert rec["roi_sha256"] == sha256_file(str(roi_file))
    np.testing.assert_allclose(rec["anchor_mm"], [5.0, 9.0, 5.0], atol=1e-6)
    for line in result["streamlines"]:
        assert line[0, 1] > line[-1, 1]

    empty = tmp_path / "empty_case"
    empty.mkdir()
    fallback = resolve_orientation("bank_fat_r", [reversed_line], str(empty))
    assert fallback["record"]["rule"] == "anterior"
    assert fallback["streamlines"][0][0, 1] > fallback["streamlines"][0][-1, 1]


def test_tie_keeps_original_order():
    line = _line((0, 0, 0), (10, 0, 0), n=5)
    # Anchor equidistant from both ends.
    oriented, n_flipped = orient_streamlines([line], np.array([5.0, 0.0, 0.0]))
    assert n_flipped == 0
    np.testing.assert_array_equal(oriented[0], line)


def test_lesion_track_matches_hand_computed_distances(tmp_path):
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[5, 5, 5] = 1
    lesion = _save_nifti(str(tmp_path / "lesion.nii.gz"), mask)
    s1 = np.array([[5.0, 5.0, 0.0], [5.0, 5.0, 5.0], [5.0, 5.0, 10.0]])
    s2 = np.array([[5.0, 5.0, 2.0], [5.0, 5.0, 4.0], [5.0, 5.0, 8.0]])
    track = lesion_distance_track([s1, s2], lesion)
    assert track["reason"] is None
    np.testing.assert_allclose(track["track_mm"], [3.0, 0.0, 3.0], atol=1e-5)


def test_lesion_track_null_without_mask():
    line = _line((0, 0, 0), (1, 0, 0), n=4)
    track = lesion_distance_track([line], None)
    assert track == {"track_mm": None, "reason": "no lesion mask"}


def _mini_case(tmp_path, *, with_fa=True, with_lesion=False):
    case = tmp_path / "case"
    (case / "tracts" / "bank").mkdir(parents=True)
    (case / "work" / "profiles").mkdir(parents=True)
    line = _line((2, 4, 4), (17, 4, 4), n=16)
    tck = case / "tracts" / "bank" / "cst_r.tck"
    write_tck([line, line[::-1].copy()], tck)
    data = np.zeros((20, 8, 8), dtype=np.float32)
    for i in range(20):
        data[i, :, :] = i / 19.0
    inputs = {
        "bank_cst_r": {
            "path": "tracts/bank/cst_r.tck",
            "sha256": sha256_file(str(tck)),
        },
    }
    if with_fa:
        fa = _save_nifti(str(case / "work" / "profiles" / "fa.nii.gz"), data)
        inputs["profile_fa"] = {
            "path": "work/profiles/fa.nii.gz",
            "sha256": sha256_file(fa),
        }
    lesion_path = None
    if with_lesion:
        mask = np.zeros((20, 8, 8), dtype=np.uint8)
        mask[10, 4, 4] = 1
        lesion_path = _save_nifti(str(case / "work" / "profiles" / "lesion.nii.gz"), mask)
        inputs["lesion"] = {"path": "work/profiles/lesion.nii.gz"}
    manifest = {"case_id": "synth", "inputs": inputs}
    banks = {"bank_cst_r": SimpleNamespace(path=str(tck))}
    return case, manifest, banks, lesion_path


def test_compute_profile_provenance_and_cache(tmp_path):
    case, manifest, banks, _ = _mini_case(tmp_path)
    payload = compute_bank_profile(
        case_root=str(case),
        manifest=manifest,
        banks=banks,
        bank_id="bank_cst_r",
        scalar="fa",
    )
    for key in (
        "bank_id", "bank_path", "bank_sha256", "scalar", "scalar_path",
        "scalar_sha256", "n_points", "n_streamlines", "orientation",
        "mrtrix_version", "claim", "nodes", "zero_or_missing_warning",
        "zero_or_missing_threshold", "active_derivation",
        "streamline_mean_histogram", "lesion_distance",
    ):
        assert key in payload
    assert "n_valid" in payload["nodes"] and "n_zero" in payload["nodes"]
    assert "zero_or_missing_flag" in payload["nodes"]
    assert payload["zero_or_missing_threshold"] == ZERO_OR_MISSING_THRESHOLD
    assert payload["claim"] == CLAIM
    assert payload["bank_id"] == "bank_cst_r"
    assert payload["n_streamlines"] == 2
    assert payload["n_points"] == 100
    assert payload["orientation"]["rule"] == "inferior"
    assert payload["lesion_distance"]["reason"] == "no lesion mask"
    assert payload["lesion_distance"]["track_mm"] is None
    hist = payload["streamline_mean_histogram"]
    assert hist["n"] == 2
    assert sum(hist["counts"]) == 2
    cache_file = case / "work" / "profiles" / "bank_cst_r.fa.json"
    assert cache_file.is_file()
    again = compute_bank_profile(
        case_root=str(case),
        manifest=manifest,
        banks=banks,
        bank_id="bank_cst_r",
        scalar="fa",
    )
    assert again["elapsed_s"] == payload["elapsed_s"]  # served from cache

    # File-hash change with matching recorded sha256 forces recompute.
    fa_path = case / "work" / "profiles" / "fa.nii.gz"
    img = nib.load(str(fa_path))
    nib.save(nib.Nifti1Image(np.asarray(img.dataobj) * 0.5, img.affine), str(fa_path))
    manifest["inputs"]["profile_fa"]["sha256"] = sha256_file(str(fa_path))
    fresh = compute_bank_profile(
        case_root=str(case),
        manifest=manifest,
        banks=banks,
        bank_id="bank_cst_r",
        scalar="fa",
    )
    assert fresh["scalar_sha256"] != payload["scalar_sha256"]
    assert fresh["cache"]["scalar_sha256"] == fresh["scalar_sha256"]


def test_unknown_bank_and_missing_scalar(tmp_path):
    case, manifest, banks, _ = _mini_case(tmp_path, with_fa=False)
    with pytest.raises(UnknownBank, match="unknown bank"):
        compute_bank_profile(
            case_root=str(case), manifest=manifest, banks=banks,
            bank_id="bank_nope", scalar="fa",
        )
    with pytest.raises(ScalarMissing, match="no fa map"):
        compute_bank_profile(
            case_root=str(case), manifest=manifest, banks=banks,
            bank_id="bank_cst_r", scalar="fa",
        )
    with pytest.raises(ScalarUnsupported):
        resolve_scalar_entry(manifest, "rd")
    # Manifest-only bank that was never loaded into `banks` is still unknown.
    manifest["inputs"]["bank_fat_r"] = {"path": "tracts/bank/missing.tck"}
    with pytest.raises(UnknownBank):
        compute_bank_profile(
            case_root=str(case), manifest=manifest, banks=banks,
            bank_id="bank_fat_r", scalar="fa",
        )


def _stub_route(tmp_path, service):
    route = object.__new__(make_handler(service, str(tmp_path)))
    captured = {}

    def _json(code, body):
        captured["code"] = code
        captured["body"] = body
        return code, body

    def _send(code, body=b"", ctype="application/octet-stream", extra=None):
        captured["code"] = code
        captured["body"] = body
        captured["ctype"] = ctype
        return code, body

    route._json = _json
    route._send = _send
    route.server = SimpleNamespace(server_address=("127.0.0.1", 18995))
    return route, captured


def test_profile_route_404_and_409(tmp_path):
    case, manifest, banks, _ = _mini_case(tmp_path, with_fa=False)
    service = SimpleNamespace(
        case_root=str(case),
        manifest=manifest,
        banks=banks,
        _lesion_path=None,
    )
    route, captured = _stub_route(tmp_path, service)
    route.path = "/api/profile/bank_nope?scalar=fa"
    route._handle_profile()
    assert captured["code"] == 404
    assert captured["body"]["code"] == "unknown_bank"

    route.path = "/api/profile/bank_cst_r?scalar=fa"
    route._handle_profile()
    assert captured["code"] == 409
    assert captured["body"]["code"] == "scalar_missing"

    route.path = "/api/profile/bank_cst_r?scalar=rd"
    route._handle_profile()
    assert captured["code"] == 409
    assert captured["body"]["code"] == "scalar_unsupported"


def test_profile_route_200_has_provenance(tmp_path):
    case, manifest, banks, _ = _mini_case(tmp_path)
    service = SimpleNamespace(
        case_root=str(case),
        manifest=manifest,
        banks=banks,
        _lesion_path=None,
    )
    route, captured = _stub_route(tmp_path, service)
    route.path = "/api/profile/bank_cst_r"
    route._handle_profile()
    assert captured["code"] == 200
    body = captured["body"]
    assert body["claim"] == CLAIM
    assert body["n_streamlines"] == 2
    assert len(body["nodes"]["median"]) == 100
    assert "zero_or_missing_flag" in body["nodes"]
    assert "zero_or_missing_warning" in body
    assert body["zero_or_missing_threshold"] == ZERO_OR_MISSING_THRESHOLD
    assert "rule" in body["orientation"] and "anchor_mm" in body["orientation"]


def test_zeroed_slab_zero_or_missing_flags_only_those_nodes(tmp_path):
    """A zeroed slab under part of a straight bundle flags those nodes only."""
    data = np.full((20, 8, 8), 0.5, dtype=np.float32)
    data[:5, :, :] = 0.0  # world x in [0, 4]
    fa = _save_nifti(str(tmp_path / "fa.nii.gz"), data)
    lines = [_line((0, y, 4), (19, y, 4), n=20) for y in (3.0, 3.5, 4.0, 4.5, 5.0)]
    samples = sample_scalar(resample(lines, 20), fa)
    nodes = aggregate_nodes(samples)
    assert samples.shape == (5, 20)
    for k in range(5):
        assert nodes["n_zero"][k] == 5
        assert nodes["n_nan"][k] == 0
        assert nodes["n_valid"][k] == 5
        assert nodes["zero_or_missing_flag"][k] is True
        assert nodes["median"][k] == pytest.approx(0.0)
    for k in range(5, 20):
        assert nodes["n_zero"][k] == 0
        assert nodes["zero_or_missing_flag"][k] is False
        assert nodes["median"][k] == pytest.approx(0.5)
    warning = zero_or_missing_warning(nodes)
    assert warning is not None
    assert "nodes 0,1,2,3,4" in warning
    assert "5," not in warning.split(":", 1)[0]
    assert "masked" not in warning.lower()


@pytest.mark.skipif(not os.path.isfile(TCKRESAMPLE), reason="tckresample absent")
def test_numpy_resample_matches_tckresample(tmp_path):
    # Dense polyline so both linear arc-length interpolators see the same chords.
    t = np.linspace(0, np.pi / 2, 200)
    line = np.column_stack([20.0 * np.cos(t), 20.0 * np.sin(t), np.linspace(0, 10, 200)])
    src = tmp_path / "in.tck"
    dst = tmp_path / "out.tck"
    write_tck([line], src)
    subprocess.run(
        [TCKRESAMPLE, str(src), str(dst), "-num_points", "25", "-quiet"],
        check=True,
        timeout=30,
    )
    oracle = np.asarray(nib.streamlines.load(str(dst)).streamlines[0], dtype=np.float64)
    got = resample([line], 25)[0]
    np.testing.assert_allclose(got[0], oracle[0], atol=1e-5)
    np.testing.assert_allclose(got[-1], oracle[-1], atol=1e-5)
    np.testing.assert_allclose(got, oracle, atol=5e-2, rtol=0)


def test_profile_route_refuses_changed_bank_source(tmp_path):
    case, manifest, banks, _ = _mini_case(tmp_path)

    def boom(_bank_id):
        raise BankSourceChanged("named bank source changed after server boot")

    service = SimpleNamespace(
        case_root=str(case),
        manifest=manifest,
        banks=banks,
        _lesion_path=None,
        _lock=threading.Lock(),
        bank_source_hash=boom,
    )
    route, captured = _stub_route(tmp_path, service)
    route.path = "/api/profile/bank_cst_r?scalar=fa"
    route._handle_profile()
    assert captured["code"] == 409
    assert captured["body"]["code"] == "bank_source_changed"
    assert captured["body"]["restartRequired"] is True
    assert captured["body"]["bankId"] == "bank_cst_r"


def test_profile_refuses_manifest_hash_mismatch(tmp_path):
    case, manifest, banks, _ = _mini_case(tmp_path)
    manifest["inputs"]["profile_fa"]["sha256"] = "0" * 64
    with pytest.raises(HashMismatch, match="sha256 mismatch"):
        compute_bank_profile(
            case_root=str(case), manifest=manifest, banks=banks,
            bank_id="bank_cst_r", scalar="fa",
        )
    manifest["inputs"]["profile_fa"]["sha256"] = sha256_file(
        str(case / "work" / "profiles" / "fa.nii.gz")
    )
    manifest["inputs"]["bank_cst_r"]["sha256"] = "1" * 64
    with pytest.raises(HashMismatch, match="bank sha256 mismatch"):
        compute_bank_profile(
            case_root=str(case), manifest=manifest, banks=banks,
            bank_id="bank_cst_r", scalar="fa",
        )


def test_profile_cache_key_includes_active_derivation(tmp_path):
    case, manifest, banks, _ = _mini_case(tmp_path)
    manifest["active_derivation"] = "d0"
    first = compute_bank_profile(
        case_root=str(case), manifest=manifest, banks=banks,
        bank_id="bank_cst_r", scalar="fa",
    )
    assert first["active_derivation"] == "d0"
    assert first["cache"]["active_derivation"] == "d0"
    manifest["active_derivation"] = "d1"
    second = compute_bank_profile(
        case_root=str(case), manifest=manifest, banks=banks,
        bank_id="bank_cst_r", scalar="fa",
    )
    assert second["active_derivation"] == "d1"
    assert second["cache"]["active_derivation"] == "d1"


def test_profile_refuses_declared_missing_empty_or_escaping_lesion(tmp_path, monkeypatch):
    case, manifest, banks, _ = _mini_case(tmp_path)
    kwargs = dict(
        case_root=str(case), manifest=manifest, banks=banks,
        bank_id="bank_cst_r", scalar="fa",
    )
    manifest["inputs"]["lesion"] = {"path": "work/profiles/nope.nii.gz"}
    with pytest.raises(LesionInvalid, match="declared lesion missing"):
        compute_bank_profile(**kwargs)

    manifest["inputs"]["lesion"] = {"path": "../outside.nii.gz"}
    with pytest.raises(LesionInvalid, match="declared lesion escapes case_root"):
        compute_bank_profile(**kwargs)

    empty = np.zeros((8, 8, 8), dtype=np.uint8)
    _save_nifti(str(case / "work" / "profiles" / "empty_lesion.nii.gz"), empty)
    manifest["inputs"]["lesion"] = {"path": "work/profiles/empty_lesion.nii.gz"}
    with pytest.raises(LesionInvalid, match="declared lesion empty"):
        compute_bank_profile(**kwargs)

    mask = np.zeros((20, 8, 8), dtype=np.uint8)
    mask[10, 4, 4] = 1
    les = _save_nifti(str(case / "work" / "profiles" / "lesion.nii.gz"), mask)
    manifest["inputs"]["lesion"] = {"path": "work/profiles/lesion.nii.gz"}
    monkeypatch.setattr(
        "tractlab.profile.load_prebuilt_bundle", lambda *a, **k: ([], {}),
    )
    with pytest.raises(LesionInvalid, match="declared lesion has no streamline samples"):
        compute_bank_profile(**kwargs)
    monkeypatch.undo()
    ok = compute_bank_profile(**kwargs)
    assert ok["lesion_distance"]["track_mm"] is not None
    assert ok["lesion_path"] == "work/profiles/lesion.nii.gz"
    assert ok["lesion_sha256"] == sha256_file(les)


def _fat_case(tmp_path):
    case, manifest, banks, _ = _mini_case(tmp_path)
    fat = case / "tracts" / "bank" / "fat_r.tck"
    write_tck([_line((5, 1, 5), (5, 9, 5), n=8)], fat)
    roi = np.zeros((10, 10, 10), dtype=np.uint8)
    roi[5, 9, 5] = 1
    roi_path = _save_nifti(str(case / "tracts" / "roi" / "fat_r_sfg_dil1.nii.gz"), roi)
    manifest["inputs"]["bank_fat_r"] = {
        "path": "tracts/bank/fat_r.tck",
        "sha256": sha256_file(str(fat)),
    }
    banks["bank_fat_r"] = SimpleNamespace(path=str(fat))
    return case, manifest, banks, roi_path


def test_roi_hash_invalidates_cache_and_is_served(tmp_path):
    case, manifest, banks, roi_path = _fat_case(tmp_path)
    first = compute_bank_profile(
        case_root=str(case), manifest=manifest, banks=banks,
        bank_id="bank_fat_r", scalar="fa",
    )
    assert first["orientation"]["rule"] == "roi_centroid"
    assert first["orientation"]["roi_sha256"] == sha256_file(roi_path)
    assert first["cache"]["roi_sha256"] == first["orientation"]["roi_sha256"]
    cached = compute_bank_profile(
        case_root=str(case), manifest=manifest, banks=banks,
        bank_id="bank_fat_r", scalar="fa",
    )
    assert cached["elapsed_s"] == first["elapsed_s"]
    img = nib.load(roi_path)
    data = np.asarray(img.dataobj).copy()
    data[5, 8, 5] = 1
    nib.save(nib.Nifti1Image(data, img.affine), roi_path)
    fresh = compute_bank_profile(
        case_root=str(case), manifest=manifest, banks=banks,
        bank_id="bank_fat_r", scalar="fa",
    )
    assert fresh["orientation"]["roi_sha256"] != first["orientation"]["roi_sha256"]
    assert fresh["cache"]["roi_sha256"] == fresh["orientation"]["roi_sha256"]


def test_concurrent_profile_requests_are_isolated(tmp_path):
    case, manifest, banks, _roi = _fat_case(tmp_path)
    service = SimpleNamespace(
        case_root=str(case),
        manifest=manifest,
        banks=banks,
        _lesion_path=None,
        _lock=threading.Lock(),
    )
    results = {}

    def hit(bid):
        route, captured = _stub_route(tmp_path, service)
        route.path = f"/api/profile/{bid}?scalar=fa"
        route._handle_profile()
        results[bid] = captured

    threads = [
        threading.Thread(target=hit, args=("bank_cst_r",)),
        threading.Thread(target=hit, args=("bank_fat_r",)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results["bank_cst_r"]["code"] == 200
    assert results["bank_fat_r"]["code"] == 200
    assert results["bank_cst_r"]["body"]["bank_id"] == "bank_cst_r"
    assert results["bank_fat_r"]["body"]["bank_id"] == "bank_fat_r"
    cst = json.loads((case / "work" / "profiles" / "bank_cst_r.fa.json").read_text())
    fat = json.loads((case / "work" / "profiles" / "bank_fat_r.fa.json").read_text())
    assert cst["bank_id"] == "bank_cst_r" and fat["bank_id"] == "bank_fat_r"
    assert cst["n_streamlines"] != fat["n_streamlines"] or cst["orientation"]["rule"] != fat["orientation"]["rule"]


def test_bank_load_started_during_profile_fetch_remains_responsive(tmp_path, monkeypatch):
    """A profile compute must not hold the single-flight lock.

    ``_handle_bank_load`` (serve.py) takes ``service._lock`` with
    ``blocking=False`` and answers 409 "a track job is already running" when it
    cannot. Holding the lock across the profile compute therefore made a bank
    load clicked during a profile fetch fail. This asserts the exact predicate
    that handler uses: while a profile is computing, the lock is free.
    """
    import tractlab.serve as serve_module

    case, manifest, banks, _ = _mini_case(tmp_path)
    service = SimpleNamespace(
        case_root=str(case),
        manifest=manifest,
        banks=banks,
        _lesion_path=None,
        _lock=threading.Lock(),
    )

    started = threading.Event()
    release = threading.Event()
    real_compute = serve_module.compute_bank_profile

    def slow_compute(**kwargs):
        started.set()
        # Stand in for a real resample/sample pass; bounded so the test cannot
        # hang if the lock is (wrongly) held across it.
        assert release.wait(timeout=10), "profile compute was never released"
        return real_compute(**kwargs)

    monkeypatch.setattr(serve_module, "compute_bank_profile", slow_compute)

    captured_profile = {}

    def run_profile():
        route, captured = _stub_route(tmp_path, service)
        route.path = "/api/profile/bank_cst_r?scalar=fa"
        route._handle_profile()
        captured_profile.update(captured)

    worker = threading.Thread(target=run_profile)
    worker.start()
    try:
        assert started.wait(timeout=10), "profile compute never started"
        # The bank-load predicate, unchanged: non-blocking acquire must win
        # while the profile is mid-compute.
        acquired = service._lock.acquire(blocking=False)
        assert acquired, "a bank load during a profile compute would answer 409 busy"
        service._lock.release()
    finally:
        release.set()
        worker.join(timeout=30)

    assert not worker.is_alive()
    assert captured_profile["code"] == 200
    assert captured_profile["body"]["bank_id"] == "bank_cst_r"
    # The lock is not leaked once the request finishes.
    assert service._lock.acquire(blocking=False)
    service._lock.release()


def test_profile_route_still_waits_for_a_running_track_job(tmp_path):
    """The brief catalog hold keeps the existing busy contract."""
    case, manifest, banks, _ = _mini_case(tmp_path)
    service = SimpleNamespace(
        case_root=str(case),
        manifest=manifest,
        banks=banks,
        _lesion_path=None,
        _lock=threading.Lock(),
    )
    route, captured = _stub_route(tmp_path, service)
    route.path = "/api/profile/bank_cst_r?scalar=fa"
    service._lock.acquire()
    try:
        holder = threading.Thread(target=route._handle_profile)
        holder.start()
        holder.join(timeout=2)
        assert holder.is_alive(), "the profile must wait for a running track job"
    finally:
        service._lock.release()
        holder.join(timeout=30)
    assert captured["code"] == 200


def test_profile_route_rejects_blank_and_duplicate_scalar(tmp_path):
    case, manifest, banks, _ = _mini_case(tmp_path)
    service = SimpleNamespace(
        case_root=str(case),
        manifest=manifest,
        banks=banks,
        _lesion_path=None,
    )
    route, captured = _stub_route(tmp_path, service)
    route.path = "/api/profile/bank_cst_r?scalar="
    route._handle_profile()
    assert captured["code"] == 409
    assert captured["body"]["code"] == "scalar_unsupported"
    route.path = "/api/profile/bank_cst_r?scalar=fa&scalar=md"
    route._handle_profile()
    assert captured["code"] == 409
    assert captured["body"]["code"] == "scalar_unsupported"


def test_make_scalar_maps_refuses_stale_products(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    dwi = tmp_path / "dwi.bin"
    mask = tmp_path / "mask.bin"
    dwi.write_bytes(b"dwi-v1")
    mask.write_bytes(b"mask-v1")
    for name in PRODUCTS:
        (out / name).write_bytes(b"product")
    write_sidecar(str(out), str(dwi), str(mask))
    assert decide(str(out), str(dwi), str(mask)) == "reuse"
    dwi.write_bytes(b"dwi-v2")
    with pytest.raises(StaleScalarMaps, match="dwi_sha256"):
        decide(str(out), str(dwi), str(mask))
    assert decide(str(out), str(dwi), str(mask), force=True) == "build"
    dwi.write_bytes(b"dwi-v1")
    mask.write_bytes(b"mask-v1")
    write_sidecar(str(out), str(dwi), str(mask))
    mask.write_bytes(b"mask-v2")
    with pytest.raises(StaleScalarMaps, match="mask_sha256"):
        decide(str(out), str(dwi), str(mask))
