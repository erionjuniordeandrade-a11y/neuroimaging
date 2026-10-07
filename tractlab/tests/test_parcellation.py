"""A2 parcellation prior: LUT, auto-QC broken fixtures, fail-closed discovery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import nibabel as nib
import pytest

from tractlab.atlas_fetch import (
    YEO7_NETWORKS,
    load_schaefer_lut,
    label_to_network_map,
    _network_id_from_name,
)
from tractlab.atlas_prep import auto_qc_parcellation
from tractlab.parcellation import discover_parcellation, parcellation_qc_ok

CASE = Path.home() / "tractlab-data/cases/local-case"
MASK = CASE / "nifti/mask_up.nii.gz"
needs = pytest.mark.skipif(not MASK.is_file(), reason="case absent")


def test_yeo7_network_table_complete():
    assert len(YEO7_NETWORKS) == 7
    assert YEO7_NETWORKS[7] == "Default"


def test_network_id_from_schaefer_name_tokens():
    assert _network_id_from_name("7Networks_LH_Vis_1") == 1
    assert _network_id_from_name("7Networks_RH_Default_1") == 7
    assert _network_id_from_name("7Networks_LH_Cont_PFCl_1") == 6


def test_discover_parcellation_fail_closed():
    man = {"parcellation_qc": {"approved_by": None}, "inputs": {}}
    assert discover_parcellation("/tmp", {}, man) is None
    assert not parcellation_qc_ok(man)


def test_parcellation_signed_false_when_qc_scoped_to_other_derivation():
    man = {
        "active_derivation": "d1",
        "derivations": {"d1": {"kind": "rpe_pair"}},
        "parcellation_qc": {
            "approved_by": "tester",
            "date": "2026-08-09",
            "sheet_sha": "abc",
            "derivation": "d0-old",
        },
    }
    assert parcellation_qc_ok(man) is False


@needs
def test_auto_qc_label_completeness_broken(tmp_path):
    """Missing half the labels must fail completeness."""
    g = nib.load(str(MASK))
    aff = g.affine
    sh = g.shape
    vol = np.zeros(sh, dtype=np.int32)
    # only labels 1..50
    vol[10:20, 10:20, 10:20] = 1
    for i in range(1, 51):
        vol.flat[i] = i
    p = tmp_path / "parc.nii.gz"
    nib.save(nib.Nifti1Image(vol, aff), str(p))
    # synthetic incomplete LUT
    lut = {
        i: {"name": f"L_{i}", "network_id": (i % 7) + 1, "hemi": "L" if i <= 100 else "R"}
        for i in range(1, 201)
    }
    qc = auto_qc_parcellation(
        labels_path=p, mask_path=str(MASK), lut=lut, hull_min=0.0, lr_ratio_lo=0.0, lr_ratio_hi=100.0,
    )
    assert not qc.label_complete
    assert not qc.ok


@needs
def test_nn_tripwire_rejects_non_integer_like_labels(tmp_path):
    """Unique labels must be LUT keys — fractional junk would appear as unknown ints after NN;
    here we inject label 999 outside LUT."""
    g = nib.load(str(MASK))
    aff = g.affine
    vol = np.zeros(g.shape, dtype=np.int32)
    vol[20:30, 20:30, 20:30] = 999
    p = tmp_path / "parc.nii.gz"
    nib.save(nib.Nifti1Image(vol, aff), str(p))
    lut = {i: {"name": f"x{i}", "network_id": 1, "hemi": "L"} for i in range(1, 201)}
    present = set(int(x) for x in np.unique(vol) if int(x) > 0)
    assert not present.issubset(set(lut.keys()))


# ── Wave 3: /api/parcellation/lut reports the LUT's case-relative path with its sha ──


def test_parcellation_lut_route_reports_relative_path_with_sha(tmp_path):
    import hashlib
    import importlib
    import os
    from types import SimpleNamespace

    case = tmp_path / "case"
    lut = case / "normative" / "Schaefer2018_200Parcels_7Networks_order.txt"
    lut.parent.mkdir(parents=True)
    rows = []
    for i in range(1, 201):
        net = ((i - 1) % 7) + 1
        hemi = "LH" if i <= 100 else "RH"
        rows.append(f"{i}\t7Networks_{hemi}_Net{net}_{i}\t{100 + net}\t{10 + net}\t{200 - net}\t0")
    lut.write_text("\n".join(rows) + "\n")

    serve = importlib.import_module("tractlab.serve")
    service = SimpleNamespace(
        case_root=str(case),
        parcellation=SimpleNamespace(lut_path=os.path.realpath(lut)),
    )
    route = object.__new__(serve.make_handler(service, str(tmp_path)))
    route._runtime_current = lambda: True
    route._enforce_browser_origin_policy = lambda: None
    route._json = lambda code, obj: (code, obj)
    route.path = "/api/parcellation/lut"
    status, body = route.do_GET()
    assert status == 200
    assert body["lutSha256"] == hashlib.sha256(lut.read_bytes()).hexdigest()
    assert body["lutPath"] == "normative/Schaefer2018_200Parcels_7Networks_order.txt"
    assert not os.path.isabs(body["lutPath"])


def test_parcellation_lut_route_refuses_lut_outside_case_root(tmp_path):
    import importlib
    from types import SimpleNamespace

    (tmp_path / "case").mkdir()
    outside = tmp_path / "elsewhere.txt"
    outside.write_text("x\n")
    serve = importlib.import_module("tractlab.serve")
    service = SimpleNamespace(case_root=str(tmp_path / "case"),
                              parcellation=SimpleNamespace(lut_path=str(outside)))
    route = object.__new__(serve.make_handler(service, str(tmp_path)))
    route._runtime_current = lambda: True
    route._enforce_browser_origin_policy = lambda: None
    route._json = lambda code, obj: (code, obj)
    route.path = "/api/parcellation/lut"
    status, body = route.do_GET()
    assert status == 500 and "outside case_root" in body["error"]
