"""C1b traceable assigned edges — synthetic corpus + parcellation, no PHI.

Real tck2connectome/connectome2tck runs (skipped if ~/mrtrix3/bin is absent,
matching the repo's other real-engine tests, e.g. tests/test_track.py).
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from tractlab import connectome as cx

needs = pytest.mark.skipif(
    not (os.path.exists(cx.TCK2CONNECTOME) and os.path.exists(cx.CONNECTOME2TCK)),
    reason="mrtrix3 tck2connectome/connectome2tck absent",
)


def _write_lut(path: Path) -> None:
    """A minimal-but-valid 200-row Schaefer order file (load_schaefer_lut requires exactly 200)."""
    lines = []
    for i in range(1, 201):
        net = ((i - 1) % 7) + 1
        hemi = "LH" if i <= 100 else "RH"
        lines.append(f"{i}\t7Networks_{hemi}_Net{net}_{i}\t{100 + net}\t{10 + net}\t{200 - net}\t0")
    path.write_text("\n".join(lines) + "\n")


def _grid_affine(spacing: float = 2.0) -> np.ndarray:
    aff = np.eye(4)
    aff[0, 0] = aff[1, 1] = aff[2, 2] = spacing
    return aff


def _vox2world(aff: np.ndarray, ijk) -> np.ndarray:
    ijk = np.asarray(ijk, dtype=np.float64)
    h = np.concatenate([ijk, np.ones((ijk.shape[0], 1))], axis=1)
    return (aff @ h.T).T[:, :3].astype(np.float32)


def _make_case(
    tmp_path, *, with_lesion: bool = False, lesion_wrong_grid: bool = False,
    manifest_dir=None,
) -> dict:
    """A tiny synthetic case: 12^3 grid, parcels 1<->2 connected, 3 isolated.

    with_lesion=True adds inputs.lesion sitting on the 1<->2 streamlines'
    shared midpoint voxel (5,5,5) — on the CANONICAL grid unless
    lesion_wrong_grid=True, which puts it on a different shape/spacing grid
    to exercise the cavity-vs-corpus-reference-grid check.

    manifest_dir=<path>, when given, writes manifest.json THERE instead of
    under the data root — the house indirection convention (repo manifest,
    data root elsewhere). Default (None) keeps the manifest nested under
    case_root, matching every case built before this convention existed.
    """
    root = tmp_path / "case"
    (root / "normative").mkdir(parents=True)
    (root / "nifti").mkdir(parents=True)
    (root / "tracts" / "connectome").mkdir(parents=True)

    shape = (12, 12, 12)
    aff = _grid_affine()

    mask = np.ones(shape, dtype=np.uint8)
    nib.save(nib.Nifti1Image(mask, aff), str(root / "nifti" / "mask_up.nii.gz"))

    labels = np.zeros(shape, dtype=np.int16)
    labels[1, 1, 1] = 1  # parcel 1
    labels[9, 9, 9] = 2  # parcel 2
    labels[1, 9, 1] = 3  # parcel 3 — isolated, nothing assigns to it
    nib.save(nib.Nifti1Image(labels, aff), str(root / "normative" / "parc_schaefer200_yeo7.nii.gz"))

    _write_lut(root / "normative" / "Schaefer2018_200Parcels_7Networks_order.txt")

    # 3 streamlines connecting parcel 1 <-> parcel 2 (through voxel 5,5,5);
    # 1 unassigned (far from any parcel)
    lines = [_vox2world(aff, [[1, 1, 1], [5, 5, 5], [9, 9, 9]]) for _ in range(3)]
    lines.append(_vox2world(aff, [[5, 0, 0], [5, 1, 0], [5, 2, 0]]))
    tgram = nib.streamlines.Tractogram(lines, affine_to_rasmm=np.eye(4))
    corpus_path = root / "tracts" / "connectome" / "wholebrain_synthetic.tck"
    nib.streamlines.save(tgram, str(corpus_path))

    manifest = {
        "case_id": "synthetic-connectome",
        "case_root": str(root),
        "active_derivation": None,
        "inputs": {
            "mask": {"path": "nifti/mask_up.nii.gz"},
            "parc_schaefer200_yeo7": {
                "path": "normative/parc_schaefer200_yeo7.nii.gz",
                "lut_path": "normative/Schaefer2018_200Parcels_7Networks_order.txt",
                "label": "Synthetic Schaefer-200/Yeo-7",
                "n_parcels": 200,
                "n_networks": 7,
            },
        },
        "parcellation_qc": {
            "approved_by": "test",
            "date": "2026-09-14",
            "sheet_sha": "test-sheet-sha",
        },
    }

    if with_lesion:
        if lesion_wrong_grid:
            wrong_aff = _grid_affine(spacing=3.0)
            lesion = np.zeros((8, 8, 8), dtype=np.uint8)
            lesion[2, 2, 2] = 1
            nib.save(nib.Nifti1Image(lesion, wrong_aff), str(root / "nifti" / "lesion.nii.gz"))
        else:
            lesion = np.zeros(shape, dtype=np.uint8)
            lesion[5, 5, 5] = 1
            nib.save(nib.Nifti1Image(lesion, aff), str(root / "nifti" / "lesion.nii.gz"))
        manifest["inputs"]["lesion"] = {"path": "nifti/lesion.nii.gz"}

    manifest_path = (Path(manifest_dir) / "manifest.json") if manifest_dir else (root / "manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest))

    return {
        "root": str(root),
        "corpus_key": str(corpus_path.relative_to(root)),
        "parc_key": "parc_schaefer200_yeo7",
        "manifest_path": str(manifest_path),
    }


def _route(service):
    """Build a route object with _send/_json patched to return plain values,
    mirroring tests/test_analytic_routes.py's pattern (no real socket).
    """
    module = importlib.import_module("tractlab.serve")
    route = object.__new__(module.make_handler(service, "."))
    route._send = lambda code, body=b"", ctype="application/octet-stream", extra=None: (code, body, extra)
    route._json = lambda code, obj: (code, obj, None)
    return route


@pytest.fixture
def demo_case(tmp_path):
    return _make_case(tmp_path)


# ---------------------------------------------------------------------------
# Build + provenance basics
# ---------------------------------------------------------------------------

@needs
def test_build_ok_with_typed_provenance(demo_case, tmp_path):
    out_dir = str(tmp_path / "work" / "connectome")
    result = cx.build(demo_case["root"], demo_case["corpus_key"], demo_case["parc_key"], out_dir)
    assert result.outcome is cx.Outcome.OK, result.error
    assert result.provenance["n_streamlines"] == 4
    assert result.provenance["n_assigned"] == 3
    assert result.provenance["unassigned_fraction"] == pytest.approx(0.25)
    assert result.provenance["weighting"] == "none (raw counts)"
    assert result.provenance["assignment_method"] == "radial_search"
    assert result.provenance["corpus"]["sha256"]
    assert result.provenance["parcellation"]["sha256"]

    matrix = cx.read_matrix(result.matrix_path)
    assert int(round(matrix[0][1])) == 3  # node1 x node2 == the 3 assigned streamlines


@needs
def test_discover_parcellation_refuses_unsigned_qc(demo_case, tmp_path):
    man_path = Path(demo_case["root"]) / "manifest.json"
    man = json.loads(man_path.read_text())
    man["parcellation_qc"]["approved_by"] = None
    man_path.write_text(json.dumps(man))
    out_dir = str(tmp_path / "work" / "connectome")
    result = cx.build(demo_case["root"], demo_case["corpus_key"], demo_case["parc_key"], out_dir)
    assert result.outcome is cx.Outcome.INVALID
    assert "QC unsigned" in result.error or "unavailable" in result.error


@needs
def test_build_refuses_parcellation_grid_mismatch(tmp_path):
    root = tmp_path / "case"
    (root / "normative").mkdir(parents=True)
    (root / "nifti").mkdir(parents=True)
    (root / "tracts" / "connectome").mkdir(parents=True)

    aff = _grid_affine()
    mask = np.ones((12, 12, 12), dtype=np.uint8)
    nib.save(nib.Nifti1Image(mask, aff), str(root / "nifti" / "mask_up.nii.gz"))

    wrong_aff = _grid_affine(spacing=3.0)
    labels = np.zeros((8, 8, 8), dtype=np.int16)
    labels[1, 1, 1] = 1
    nib.save(nib.Nifti1Image(labels, wrong_aff), str(root / "normative" / "parc_schaefer200_yeo7.nii.gz"))
    _write_lut(root / "normative" / "Schaefer2018_200Parcels_7Networks_order.txt")

    lines = [_vox2world(aff, [[1, 1, 1], [5, 5, 5]])]
    tgram = nib.streamlines.Tractogram(lines, affine_to_rasmm=np.eye(4))
    corpus_path = root / "tracts" / "connectome" / "wholebrain_synthetic.tck"
    nib.streamlines.save(tgram, str(corpus_path))

    manifest = {
        "case_id": "grid-mismatch",
        "case_root": str(root),
        "active_derivation": None,
        "inputs": {
            "mask": {"path": "nifti/mask_up.nii.gz"},
            "parc_schaefer200_yeo7": {
                "path": "normative/parc_schaefer200_yeo7.nii.gz",
                "lut_path": "normative/Schaefer2018_200Parcels_7Networks_order.txt",
            },
        },
        "parcellation_qc": {"approved_by": "test", "date": "2026-09-14", "sheet_sha": "x"},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))

    out_dir = str(tmp_path / "work" / "connectome")
    result = cx.build(str(root), str(corpus_path.relative_to(root)), "parc_schaefer200_yeo7", out_dir)
    assert result.outcome is cx.Outcome.INVALID
    assert "grid mismatch" in result.error


# ---------------------------------------------------------------------------
# Item 7 — parc_key identity: used for discovery, refused if undeclared,
# recorded in provenance.
# ---------------------------------------------------------------------------

@needs
def test_parc_key_must_be_declared_and_recorded(demo_case, tmp_path):
    out_dir = str(tmp_path / "work" / "connectome")
    undeclared = cx.build(demo_case["root"], demo_case["corpus_key"], "parc_does_not_exist", out_dir)
    assert undeclared.outcome is cx.Outcome.INVALID
    assert "not a declared parcellation input" in undeclared.error

    ok = cx.build(demo_case["root"], demo_case["corpus_key"], demo_case["parc_key"], out_dir)
    assert ok.outcome is cx.Outcome.OK, ok.error
    assert ok.provenance["parcellation"]["key"] == demo_case["parc_key"]


@needs
def test_build_uses_the_passed_parc_key_not_a_hardcoded_default(demo_case, tmp_path):
    man_path = Path(demo_case["root"]) / "manifest.json"
    man = json.loads(man_path.read_text())
    man["inputs"]["parc_alt"] = dict(man["inputs"]["parc_schaefer200_yeo7"])
    man_path.write_text(json.dumps(man))

    out_dir = str(tmp_path / "work" / "connectome")
    result = cx.build(demo_case["root"], demo_case["corpus_key"], "parc_alt", out_dir)
    assert result.outcome is cx.Outcome.OK, result.error
    assert result.provenance["parcellation"]["key"] == "parc_alt"


# ---------------------------------------------------------------------------
# Per-edge traceability + independent-count discipline (items 3, 5)
# ---------------------------------------------------------------------------

@needs
def test_edge_streamlines_matches_matrix_and_extraction_exactly(demo_case, tmp_path):
    out_dir = str(tmp_path / "work" / "connectome")
    built = cx.build(demo_case["root"], demo_case["corpus_key"], demo_case["parc_key"], out_dir)
    assert built.outcome is cx.Outcome.OK, built.error

    edge = cx.edge_streamlines(out_dir, 1, 2)
    assert edge["matrix_count"] == 3
    assert edge["assignment_row_count"] == 3
    assert edge["extracted_count"] == 3
    assert os.path.isfile(edge["tck_path"])
    assert len(edge["tck_sha256"]) == 64

    zero_edge = cx.edge_streamlines(out_dir, 1, 3)  # parcel 3 is isolated
    assert zero_edge["matrix_count"] == 0
    assert zero_edge["assignment_row_count"] == 0
    assert zero_edge["extracted_count"] == 0


@needs
def test_edge_refuses_unavailable_extracted_count(demo_case, tmp_path, monkeypatch):
    out_dir = str(tmp_path / "work" / "connectome")
    built = cx.build(demo_case["root"], demo_case["corpus_key"], demo_case["parc_key"], out_dir)
    assert built.outcome is cx.Outcome.OK, built.error

    monkeypatch.setattr(cx, "_tckinfo_count", lambda tck_path: None)
    with pytest.raises(cx.ConnectomeEngineError, match="independent count check"):
        cx.edge_streamlines(out_dir, 1, 2)


@needs
def test_served_artifacts_have_path_and_sha256(tmp_path):
    case = _make_case(tmp_path, with_lesion=True)
    out_dir = str(tmp_path / "work" / "connectome")
    built = cx.build(case["root"], case["corpus_key"], case["parc_key"], out_dir)
    assert built.outcome is cx.Outcome.OK, built.error

    prov = built.provenance
    assert prov["matrix"]["path"] == "matrix.csv"
    assert len(prov["matrix"]["sha256"]) == 64
    assert prov["assignments"]["path"] == "assignments.txt"
    assert len(prov["assignments"]["sha256"]) == 64

    edge = cx.edge_streamlines(out_dir, 1, 2)
    assert len(edge["tck_sha256"]) == 64

    lesion_path = os.path.join(case["root"], "nifti", "lesion.nii.gz")
    hits = cx.edge_cavity_hits(edge["tck_path"], lesion_path, case["root"])
    assert len(hits["lesion_sha256"]) == 64
    assert hits["total"] == 3
    assert hits["hits"] == 3  # all 3 assigned streamlines pass through voxel (5,5,5)


# ---------------------------------------------------------------------------
# Item 2 — cavity grid check must validate against the CORPUS reference
# grid, never against itself.
# ---------------------------------------------------------------------------

@needs
def test_edge_cavity_hits_refuses_noncanonical_grid(tmp_path):
    case = _make_case(tmp_path, with_lesion=True, lesion_wrong_grid=True)
    out_dir = str(tmp_path / "work" / "connectome")
    built = cx.build(case["root"], case["corpus_key"], case["parc_key"], out_dir)
    assert built.outcome is cx.Outcome.OK, built.error

    edge = cx.edge_streamlines(out_dir, 1, 2)
    lesion_path = os.path.join(case["root"], "nifti", "lesion.nii.gz")
    with pytest.raises(cx.CavityInvalid, match="grid mismatch"):
        cx.edge_cavity_hits(edge["tck_path"], lesion_path, case["root"])


# ---------------------------------------------------------------------------
# Item 6 — malformed evidence refuses (no mrtrix needed — pure parsing)
# ---------------------------------------------------------------------------

def test_malformed_assignments_refuse(tmp_path):
    bad = tmp_path / "assignments.txt"
    bad.write_text("1 2\n1 2 3\n")  # second row has 3 columns
    with pytest.raises(cx.ConnectomeInvalid, match="malformed assignments.txt"):
        cx._assignment_stats(str(bad))
    with pytest.raises(cx.ConnectomeInvalid, match="malformed assignments.txt"):
        cx._assignment_row_count(str(bad), 1, 2)

    bad2 = tmp_path / "assignments2.txt"
    bad2.write_text("1 2\nfoo bar\n")  # non-integer node id
    with pytest.raises(cx.ConnectomeInvalid, match="non-integer node id"):
        cx._assignment_stats(str(bad2))


def test_matrix_contract_refuses_corruption(tmp_path):
    non_square = tmp_path / "m1.csv"
    non_square.write_text("0,1\n1,0,2\n")
    with pytest.raises(cx.ConnectomeInvalid, match="not square"):
        cx.read_matrix(str(non_square))

    non_finite = tmp_path / "m2.csv"
    non_finite.write_text("0,nan\nnan,0\n")
    with pytest.raises(cx.ConnectomeInvalid, match="non-finite"):
        cx.read_matrix(str(non_finite))

    non_integer = tmp_path / "m3.csv"
    non_integer.write_text("0,2.5\n2.5,0\n")
    with pytest.raises(cx.ConnectomeInvalid, match="non-integer"):
        cx.read_matrix(str(non_integer))


# ---------------------------------------------------------------------------
# Item 1 — serve-time freshness: hashes, active_derivation, QC signature,
# published-output hashes.
# ---------------------------------------------------------------------------

@needs
def test_hash_mismatch_refuses_stale_build(demo_case, tmp_path):
    out_dir = str(tmp_path / "work" / "connectome")
    built = cx.build(demo_case["root"], demo_case["corpus_key"], demo_case["parc_key"], out_dir)
    assert built.outcome is cx.Outcome.OK, built.error

    corpus_path = os.path.join(demo_case["root"], demo_case["corpus_key"])
    with open(corpus_path, "ab") as f:
        f.write(b"\x00" * 16)  # mutate after build: different size + mtime

    provenance = cx.load_provenance(out_dir)
    with pytest.raises(cx.ConnectomeInvalid, match="stale build"):
        cx.check_fresh(provenance, out_dir)
    with pytest.raises(cx.ConnectomeInvalid, match="stale build"):
        cx.edge_streamlines(out_dir, 1, 2)


@needs
def test_route_refuses_qc_revocation_or_derivation_drift(demo_case, tmp_path):
    out_dir = cx.connectome_out_dir(demo_case["root"])
    built = cx.build(demo_case["root"], demo_case["corpus_key"], demo_case["parc_key"], out_dir)
    assert built.outcome is cx.Outcome.OK, built.error

    man_path = Path(demo_case["root"]) / "manifest.json"
    man = json.loads(man_path.read_text())

    # QC revoked after build
    revoked = json.loads(json.dumps(man))
    revoked["parcellation_qc"]["approved_by"] = None
    man_path.write_text(json.dumps(revoked))
    with pytest.raises(cx.ConnectomeInvalid, match="QC no longer signed"):
        cx.check_fresh(cx.load_provenance(out_dir), out_dir)

    # restore, then drift active_derivation
    man_path.write_text(json.dumps(man))
    drifted = json.loads(json.dumps(man))
    drifted["active_derivation"] = "d2"
    man_path.write_text(json.dumps(drifted))
    with pytest.raises(cx.ConnectomeInvalid, match="active_derivation changed"):
        cx.check_fresh(cx.load_provenance(out_dir), out_dir)

    # Same derivation drift surfaced through the route as 409
    service = SimpleNamespace(case_root=demo_case["root"], manifest=drifted)
    route = _route(service)
    status, body, _ = route._handle_connectome()
    assert status == 409
    assert body["error"] == "connectome-stale"


# ---------------------------------------------------------------------------
# Item 4 — atomic publication: an interrupted rebuild must never mix
# generations; the previous published generation stays intact.
# ---------------------------------------------------------------------------

@needs
def test_interrupted_rebuild_never_publishes_mixed_generation(demo_case, tmp_path, monkeypatch):
    out_dir = str(tmp_path / "work" / "connectome")
    gen1 = cx.build(demo_case["root"], demo_case["corpus_key"], demo_case["parc_key"], out_dir)
    assert gen1.outcome is cx.Outcome.OK, gen1.error
    gen1_provenance = cx.load_provenance(out_dir)
    gen1_cell = cx._matrix_cell(cx.artifact_paths(out_dir)["matrix"], 1, 2)
    assert gen1_cell == 3

    def boom(tmp_dir, final_gen_dir, out_dir):
        raise OSError("simulated interruption during publish")

    monkeypatch.setattr(cx, "_publish", boom)
    gen2 = cx.build(demo_case["root"], demo_case["corpus_key"], demo_case["parc_key"], out_dir)
    assert gen2.outcome is cx.Outcome.ENGINE_ERROR
    assert "failed to publish" in gen2.error

    # gen1 is untouched: same provenance, same matrix cell, still resolvable.
    assert cx.is_built(out_dir)
    assert cx.load_provenance(out_dir) == gen1_provenance
    assert cx._matrix_cell(cx.artifact_paths(out_dir)["matrix"], 1, 2) == gen1_cell


# ---------------------------------------------------------------------------
# Item 8 — route tests: success shape, stale 409, count-mismatch 500,
# lesion-declared-but-invalid (missing file / wrong grid), 404.
# ---------------------------------------------------------------------------

@needs
def test_route_connectome_success_shape(demo_case, tmp_path):
    out_dir = cx.connectome_out_dir(demo_case["root"])
    built = cx.build(demo_case["root"], demo_case["corpus_key"], demo_case["parc_key"], out_dir)
    assert built.outcome is cx.Outcome.OK, built.error

    man = json.loads((Path(demo_case["root"]) / "manifest.json").read_text())
    service = SimpleNamespace(case_root=demo_case["root"], manifest=man)
    route = _route(service)

    status, body, _ = route._handle_connectome()
    assert status == 200
    assert len(body["matrix"]) == 3 and len(body["matrix"][0]) == 3
    assert body["nodeLabels"][:2] == ["7Networks_LH_Net1_1", "7Networks_LH_Net2_2"]
    assert "case_root" not in body["provenance"] and "argv" not in body["provenance"]
    assert body["provenance"]["matrix"]["sha256"]

    route.path = "/api/connectome/edge/1/2"
    status, body, _ = route._handle_connectome_edge()
    assert status == 200
    assert body["matrixCount"] == body["assignmentRowCount"] == body["extractedCount"] == 3
    assert len(body["tckSha256"]) == 64
    assert body["lesion"] is None
    assert body["lesionReason"] == "manifest has no inputs.lesion"


@needs
def test_route_connectome_stale_hash_409(demo_case, tmp_path):
    out_dir = cx.connectome_out_dir(demo_case["root"])
    built = cx.build(demo_case["root"], demo_case["corpus_key"], demo_case["parc_key"], out_dir)
    assert built.outcome is cx.Outcome.OK, built.error

    corpus_path = os.path.join(demo_case["root"], demo_case["corpus_key"])
    with open(corpus_path, "ab") as f:
        f.write(b"\x00" * 16)

    man = json.loads((Path(demo_case["root"]) / "manifest.json").read_text())
    service = SimpleNamespace(case_root=demo_case["root"], manifest=man)
    route = _route(service)
    status, body, _ = route._handle_connectome()
    assert status == 409
    assert body["error"] == "connectome-stale"


@needs
def test_route_connectome_edge_count_mismatch_500(demo_case, tmp_path, monkeypatch):
    out_dir = cx.connectome_out_dir(demo_case["root"])
    built = cx.build(demo_case["root"], demo_case["corpus_key"], demo_case["parc_key"], out_dir)
    assert built.outcome is cx.Outcome.OK, built.error

    monkeypatch.setattr(cx, "_assignment_row_count", lambda *a, **k: 999)

    man = json.loads((Path(demo_case["root"]) / "manifest.json").read_text())
    service = SimpleNamespace(case_root=demo_case["root"], manifest=man)
    route = _route(service)
    route.path = "/api/connectome/edge/1/2"
    status, body, _ = route._handle_connectome_edge()
    assert status == 500
    assert body["error"] == "connectome-count-mismatch"


@needs
def test_route_edge_lesion_declared_but_missing_file(demo_case, tmp_path):
    out_dir = cx.connectome_out_dir(demo_case["root"])
    built = cx.build(demo_case["root"], demo_case["corpus_key"], demo_case["parc_key"], out_dir)
    assert built.outcome is cx.Outcome.OK, built.error

    man_path = Path(demo_case["root"]) / "manifest.json"
    man = json.loads(man_path.read_text())
    man["inputs"]["lesion"] = {"path": "nifti/does_not_exist.nii.gz"}
    # Written to disk AND rebuilt: service.manifest must match what's on disk
    # (W8 fix round 2), and the connectome's own provenance.manifest.sha256
    # must match the disk bytes too (W8 final round, item 2) — this test
    # exercises "lesion file missing", not "stale manifest" of either kind.
    man_path.write_text(json.dumps(man))
    built2 = cx.build(demo_case["root"], demo_case["corpus_key"], demo_case["parc_key"], out_dir)
    assert built2.outcome is cx.Outcome.OK, built2.error
    service = SimpleNamespace(case_root=demo_case["root"], manifest=man)
    route = _route(service)
    route.path = "/api/connectome/edge/1/2"
    status, body, _ = route._handle_connectome_edge()
    assert status == 200
    assert body["lesion"] is None
    assert "missing" in body["lesionReason"]


@needs
def test_route_edge_lesion_wrong_grid_reports_reason(tmp_path):
    case = _make_case(tmp_path, with_lesion=True, lesion_wrong_grid=True)
    out_dir = cx.connectome_out_dir(case["root"])
    built = cx.build(case["root"], case["corpus_key"], case["parc_key"], out_dir)
    assert built.outcome is cx.Outcome.OK, built.error

    man = json.loads((Path(case["root"]) / "manifest.json").read_text())
    service = SimpleNamespace(case_root=case["root"], manifest=man)
    route = _route(service)
    route.path = "/api/connectome/edge/1/2"
    status, body, _ = route._handle_connectome_edge()
    assert status == 200
    assert body["lesion"] is None
    assert "grid mismatch" in body["lesionReason"]


def test_route_404_when_not_built(tmp_path):
    """Pure route logic — no mrtrix needed, mirrors tests/test_analytic_routes.py's pattern."""
    service = SimpleNamespace(case_root=str(tmp_path), manifest={"inputs": {}})
    route = _route(service)

    status, body, _ = route._handle_connectome()
    assert status == 404

    route.path = "/api/connectome/edge/1/2"
    status, body, _ = route._handle_connectome_edge()
    assert status == 404


# ---------------------------------------------------------------------------
# W8 — GET /api/connectome/edge/<a>/<b>/tubes: same binary tube response
# builder as /api/connectotomy/<id>/cut (self._pack_lines), same freshness
# check as the other two connectome routes. Not bank-backed: X-edgeSourceHash
# / X-corpusSourceHash replace X-bankSourceHash.
# ---------------------------------------------------------------------------

def _tubes_service(case_root, manifest):
    """A service double with the extra fields _pack_lines needs beyond
    case_root/manifest — geom_floor_mm=None and an empty lesion_shell skip
    the clearance/per-vertex-distance branches entirely (no acquisition
    metadata exists on this synthetic case), matching how a case with no
    signed geometric floor is served today.
    """
    return SimpleNamespace(
        case_root=case_root,
        manifest=manifest,
        geom_floor_mm=None,
        lesion_shell=np.zeros((0, 3), dtype=np.float32),
        space_id="synthetic-connectome",
        _geom_floor_error="case manifest has no acquisition.geom_floor_mm",
    )


@needs
def test_route_edge_tubes_success_shape_and_headers(demo_case, tmp_path):
    out_dir = cx.connectome_out_dir(demo_case["root"])
    built = cx.build(demo_case["root"], demo_case["corpus_key"], demo_case["parc_key"], out_dir)
    assert built.outcome is cx.Outcome.OK, built.error

    man = json.loads((Path(demo_case["root"]) / "manifest.json").read_text())
    service = _tubes_service(demo_case["root"], man)
    route = _route(service)
    route.path = "/api/connectome/edge/1/2/tubes"
    status, buf, extra = route._handle_connectome_edge_tubes()

    assert status == 200
    assert isinstance(buf, (bytes, bytearray)) and len(buf) > 0
    assert extra["X-engine"] == "CONNECTOME | edge 1-2"
    assert extra["X-a"] == "1" and extra["X-b"] == "2"
    assert extra["X-matrixCount"] == extra["X-assignmentRowCount"] == extra["X-extractedCount"] == "3"
    assert extra["X-nReturned"] == 3
    assert len(extra["X-edgeSourceHash"]) == 64
    assert len(extra["X-corpusSourceHash"]) == 64
    assert len(extra["X-parcellationSourceHash"]) == 64
    assert extra["X-generationId"] == built.provenance["gen_id"]
    assert extra["X-corpusPath"] == built.provenance["corpus"]["path"]
    assert extra["X-parcellationPath"] == built.provenance["parcellation"]["path"]
    assert extra["X-edgeTckPath"].endswith("1_2.tck")
    assert "X-bankSourceHash" not in extra
    assert "ASSIGNED" in extra["X-labelNote"] and "ADR-0003" in extra["X-labelNote"]
    # No lesion on this synthetic case (no inputs.lesion) — honest reason, empty headers.
    assert extra["X-lesionHits"] == "" and extra["X-lesionTotal"] == ""
    assert extra["X-lesionReason"] == "manifest has no inputs.lesion"


@needs
def test_route_edge_tubes_reports_lesion_hits_in_one_request(tmp_path):
    """Show extracts once: lesion hits ride the SAME tubes response, no second request needed."""
    case = _make_case(tmp_path, with_lesion=True)
    out_dir = cx.connectome_out_dir(case["root"])
    built = cx.build(case["root"], case["corpus_key"], case["parc_key"], out_dir)
    assert built.outcome is cx.Outcome.OK, built.error

    man = json.loads((Path(case["root"]) / "manifest.json").read_text())
    service = _tubes_service(case["root"], man)
    route = _route(service)
    route.path = "/api/connectome/edge/1/2/tubes"
    status, buf, extra = route._handle_connectome_edge_tubes()
    assert status == 200
    assert extra["X-lesionHits"] == "3" and extra["X-lesionTotal"] == "3"
    assert extra["X-lesionReason"] == ""


def test_route_edge_tubes_404_when_not_built(tmp_path):
    service = SimpleNamespace(case_root=str(tmp_path), manifest={"inputs": {}})
    route = _route(service)
    route.path = "/api/connectome/edge/1/2/tubes"
    status, body, _ = route._handle_connectome_edge_tubes()
    assert status == 404


@needs
def test_route_edge_tubes_stale_409(demo_case, tmp_path):
    out_dir = cx.connectome_out_dir(demo_case["root"])
    built = cx.build(demo_case["root"], demo_case["corpus_key"], demo_case["parc_key"], out_dir)
    assert built.outcome is cx.Outcome.OK, built.error

    corpus_path = os.path.join(demo_case["root"], demo_case["corpus_key"])
    with open(corpus_path, "ab") as f:
        f.write(b"\x00" * 16)  # mutate after build: different size + mtime

    man = json.loads((Path(demo_case["root"]) / "manifest.json").read_text())
    service = _tubes_service(demo_case["root"], man)
    route = _route(service)
    route.path = "/api/connectome/edge/1/2/tubes"
    status, body, _ = route._handle_connectome_edge_tubes()
    assert status == 409
    assert body["error"] == "connectome-stale"


# ---------------------------------------------------------------------------
# W8 fix round item 0 (root cause): edge identity must key on the extracted
# tck's DATA-segment hash, never the whole-file hash — connectome2tck stamps
# a fresh invocation timestamp into the header on every extraction, so only
# the data segment is stable across repeated extractions of the same edge.
# ---------------------------------------------------------------------------

@needs
def test_edge_content_hash_stable_across_extractions_whole_file_is_not(demo_case, tmp_path):
    out_dir = str(tmp_path / "work" / "connectome")
    built = cx.build(demo_case["root"], demo_case["corpus_key"], demo_case["parc_key"], out_dir)
    assert built.outcome is cx.Outcome.OK, built.error

    edge1 = cx.edge_streamlines(out_dir, 1, 2)
    edge2 = cx.edge_streamlines(out_dir, 1, 2)
    assert len(edge1["content_sha256"]) == 64
    assert edge1["content_sha256"] == edge2["content_sha256"], (
        f"content hash must be identical across repeated extractions of the same edge: "
        f"{edge1['content_sha256']} != {edge2['content_sha256']}"
    )
    # Documents the root cause this hash exists to work around: the WHOLE-FILE
    # hash is not required to be stable (mrtrix stamps a fresh invocation
    # timestamp into the header every run) — it is fine if it happens to
    # match, but the test never depends on that; only content_sha256 does.


def test_tck_content_hash_reads_the_files_own_data_offset(tmp_path):
    """Pure parsing — no mrtrix needed. The offset comes from the file's own
    'file: . <offset>' header field, never a guessed 'END\\n + 4' position
    (mrtrix pads between END and the declared offset)."""
    header = b"mrtrix tracks\ndatatype: Float32LE\nfile: . 64\ncount: 1\nEND\n"
    assert len(header) < 64, "the header itself must fit before the declared data offset"
    padding = b"\x00" * (64 - len(header))
    payload = b"\x01\x02\x03\x04"
    tck = tmp_path / "t.tck"
    tck.write_bytes(header + padding + payload)
    import hashlib
    assert cx._tck_content_sha256(str(tck)) == hashlib.sha256(payload).hexdigest()


def test_tck_content_hash_refuses_missing_offset_field(tmp_path):
    tck = tmp_path / "bad.tck"
    tck.write_bytes(b"mrtrix tracks\nEND\nsome data")
    with pytest.raises(cx.ConnectomeEngineError, match="no 'file: .' header"):
        cx._tck_content_sha256(str(tck))


# ---------------------------------------------------------------------------
# Item 3: a single provenance snapshot — edge_streamlines() must read
# load_provenance exactly once and the route must reuse that same result for
# every identity/path header, never a second read that could race a
# concurrent rebuild and mix bytes from one generation with an id from
# another.
# ---------------------------------------------------------------------------

@needs
def test_edge_tubes_generation_swap_cannot_mix_provenance(demo_case, tmp_path, monkeypatch):
    out_dir = cx.connectome_out_dir(demo_case["root"])
    built = cx.build(demo_case["root"], demo_case["corpus_key"], demo_case["parc_key"], out_dir)
    assert built.outcome is cx.Outcome.OK, built.error

    calls = {"n": 0}
    real_load = cx.load_provenance

    def counting_load(out_dir_arg, generation_dir=None):
        calls["n"] += 1
        return real_load(out_dir_arg, generation_dir=generation_dir)

    monkeypatch.setattr(cx, "load_provenance", counting_load)

    man = json.loads((Path(demo_case["root"]) / "manifest.json").read_text())
    service = _tubes_service(demo_case["root"], man)
    route = _route(service)
    route.path = "/api/connectome/edge/1/2/tubes"
    status, buf, extra = route._handle_connectome_edge_tubes()

    assert status == 200
    assert calls["n"] == 1, (
        f"the route must use a single provenance snapshot from inside edge_streamlines(), "
        f"never a second connectome_mod.load_provenance() call (saw {calls['n']})"
    )
    assert extra["X-generationId"] == built.provenance["gen_id"]
    assert extra["X-corpusSourceHash"] == built.provenance["corpus"]["sha256"]
    assert extra["X-parcellationSourceHash"] == built.provenance["parcellation"]["sha256"]


@needs
def test_edge_tubes_refuses_missing_path_or_sha256(demo_case, tmp_path, monkeypatch):
    """Item 5: no empty-header fallback — a missing required field 500s by name."""
    out_dir = cx.connectome_out_dir(demo_case["root"])
    built = cx.build(demo_case["root"], demo_case["corpus_key"], demo_case["parc_key"], out_dir)
    assert built.outcome is cx.Outcome.OK, built.error

    real_edge_streamlines = cx.edge_streamlines

    def missing_hash(out_dir_arg, a, b, **kw):
        result = dict(real_edge_streamlines(out_dir_arg, a, b, **kw))
        result["parcellation_sha256"] = ""
        return result

    monkeypatch.setattr(cx, "edge_streamlines", missing_hash)

    man = json.loads((Path(demo_case["root"]) / "manifest.json").read_text())
    service = _tubes_service(demo_case["root"], man)
    route = _route(service)
    route.path = "/api/connectome/edge/1/2/tubes"
    status, body, _ = route._handle_connectome_edge_tubes()
    assert status == 500
    assert body["error"] == "connectome-engine-error"
    assert "parcellation_sha256" in body["reason"]


# ---------------------------------------------------------------------------
# Item 2: Show extracts once — the tubes route must never trigger a second
# connectome2tck run (e.g. for the lesion block); one GET must suffice.
# ---------------------------------------------------------------------------

@needs
def test_show_edge_extracts_once_and_preserves_source_hash(tmp_path, monkeypatch):
    case = _make_case(tmp_path, with_lesion=True)
    out_dir = cx.connectome_out_dir(case["root"])
    built = cx.build(case["root"], case["corpus_key"], case["parc_key"], out_dir)
    assert built.outcome is cx.Outcome.OK, built.error

    calls = {"n": 0}
    real_run = cx.subprocess.run

    def counting_run(argv, *a, **k):
        if argv and argv[0] == cx.CONNECTOME2TCK:
            calls["n"] += 1
        return real_run(argv, *a, **k)

    monkeypatch.setattr(cx.subprocess, "run", counting_run)

    man = json.loads((Path(case["root"]) / "manifest.json").read_text())
    service = _tubes_service(case["root"], man)
    route = _route(service)
    route.path = "/api/connectome/edge/1/2/tubes"
    status, buf, extra = route._handle_connectome_edge_tubes()

    assert status == 200
    assert calls["n"] == 1, f"one Show must run connectome2tck exactly once (ran {calls['n']} times)"
    assert extra["X-lesionHits"] == "3" and extra["X-lesionTotal"] == "3", (
        "lesion hits came from the SAME extraction the tubes were built from"
    )
    assert len(extra["X-edgeSourceHash"]) == 64


# ---------------------------------------------------------------------------
# W8 addendum: the manifest is not always nested under case_root — the house
# convention for a case whose data lives outside the repo keeps the manifest
# in the repo (cases/<id>/manifest.json) while manifest["case_root"] points
# elsewhere. build(), check_fresh(), and edge_cavity_hits() must all accept
# the manifest explicitly instead of assuming "<case_root>/manifest.json".
# ---------------------------------------------------------------------------

@needs
def test_build_with_manifest_outside_case_root(tmp_path):
    manifest_dir = tmp_path / "repo-manifest"
    case = _make_case(tmp_path, manifest_dir=manifest_dir)
    assert not (Path(case["root"]) / "manifest.json").exists(), \
        "the manifest must NOT be nested under case_root for this test to mean anything"
    assert case["manifest_path"] == str(manifest_dir / "manifest.json")

    out_dir = str(tmp_path / "work" / "connectome")
    result = cx.build(
        case["root"], case["corpus_key"], case["parc_key"], out_dir,
        manifest_path=case["manifest_path"],
    )
    assert result.outcome is cx.Outcome.OK, result.error
    # Not nested under case_root -> recorded absolute, never a guessed relative path.
    assert result.provenance["manifest"]["path"] == str(Path(case["manifest_path"]).resolve())
    assert len(result.provenance["manifest"]["sha256"]) == 64


def test_build_without_manifest_path_still_defaults_to_case_root(tmp_path):
    """The old, always-worked convention: no manifest_path -> <case_root>/manifest.json."""
    case = _make_case(tmp_path)  # manifest_dir=None -> nested under case_root
    out_dir = str(tmp_path / "work" / "connectome")
    result = cx.build(case["root"], case["corpus_key"], "parc_does_not_exist", out_dir)
    # Refused for an unrelated reason (bad parc_key), but it must have FOUND
    # the manifest at all to get that far — this is a pure "did it look in
    # the right place" check for the manifest-load, not a full build.
    assert result.outcome is cx.Outcome.INVALID
    assert "manifest.json not found" not in (result.error or "")
    assert "not a declared parcellation input" in result.error


@needs
def test_check_fresh_and_edge_streamlines_resolve_manifest_outside_case_root(tmp_path):
    manifest_dir = tmp_path / "repo-manifest"
    case = _make_case(tmp_path, manifest_dir=manifest_dir)
    out_dir = str(tmp_path / "work" / "connectome")
    built = cx.build(
        case["root"], case["corpus_key"], case["parc_key"], out_dir,
        manifest_path=case["manifest_path"],
    )
    assert built.outcome is cx.Outcome.OK, built.error

    # check_fresh must re-resolve the SAME (non-case-root-nested) manifest
    # location from provenance alone — never guess "<case_root>/manifest.json".
    cx.check_fresh(built.provenance, out_dir)  # must not raise

    # edge_streamlines() calls check_fresh() internally — the whole per-edge
    # path must work end to end with an externally-located manifest too.
    edge = cx.edge_streamlines(out_dir, 1, 2)
    assert edge["matrix_count"] == edge["assignment_row_count"] == edge["extracted_count"] == 3

    # Editing the case_root-nested guess location (which does not even hold
    # the manifest here) must not fool check_fresh; only the real manifest's
    # own drift matters. Prove check_fresh still catches a REAL edit: mutate
    # the manifest actually in use (active_derivation drift).
    man = json.loads(Path(case["manifest_path"]).read_text())
    man["active_derivation"] = "d2"
    Path(case["manifest_path"]).write_text(json.dumps(man))
    with pytest.raises(cx.ConnectomeInvalid, match="active_derivation changed"):
        cx.check_fresh(cx.load_provenance(out_dir), out_dir)


@needs
def test_edge_cavity_hits_accepts_manifest_dict_directly(tmp_path):
    """serve.py already has service.manifest loaded in memory — no disk re-read needed."""
    case = _make_case(tmp_path, with_lesion=True)
    out_dir = str(tmp_path / "work" / "connectome")
    built = cx.build(case["root"], case["corpus_key"], case["parc_key"], out_dir)
    assert built.outcome is cx.Outcome.OK, built.error

    edge = cx.edge_streamlines(out_dir, 1, 2)
    lesion_path = os.path.join(case["root"], "nifti", "lesion.nii.gz")
    man = json.loads(Path(case["manifest_path"]).read_text())

    # Remove the manifest.json a naive implementation would go looking for
    # under case_root — a passed-in dict must be used as-is, no disk read.
    os.remove(case["manifest_path"])
    hits = cx.edge_cavity_hits(edge["tck_path"], lesion_path, case["root"], manifest=man)
    assert hits["total"] == 3 and hits["hits"] == 3


# ---------------------------------------------------------------------------
# W8 fix round 2 (six HIGH items from the second-pass review)
# ---------------------------------------------------------------------------

@needs
def test_edge_tubes_generation_swap_pins_or_refuses(demo_case, tmp_path, monkeypatch):
    """Item 1: generation pinning. edge_streamlines() must pin `current`'s
    realpath ONCE and refuse (typed error) if it no longer matches when the
    response is assembled — never silently serve a superseded generation."""
    out_dir = cx.connectome_out_dir(demo_case["root"])
    gen1 = cx.build(demo_case["root"], demo_case["corpus_key"], demo_case["parc_key"], out_dir)
    assert gen1.outcome is cx.Outcome.OK, gen1.error
    gen1_dir = cx.pin_generation(out_dir)

    gen2 = cx.build(demo_case["root"], demo_case["corpus_key"], demo_case["parc_key"], out_dir, radius_mm=6.0)
    assert gen2.outcome is cx.Outcome.OK, gen2.error
    gen2_dir = cx.pin_generation(out_dir)
    assert gen1_dir != gen2_dir, "the rebuild must have published a different generation directory"

    real_pin = cx.pin_generation
    calls = {"n": 0}

    def flaky_pin(out_dir_arg):
        calls["n"] += 1
        # First call: what the request pinned "at entry" (simulates current
        # having been gen1 at that instant). Every later call: the REAL
        # current, which by then has moved on to gen2 — a rebuild mid-request.
        return gen1_dir if calls["n"] == 1 else real_pin(out_dir_arg)

    monkeypatch.setattr(cx, "pin_generation", flaky_pin)
    with pytest.raises(cx.ConnectomeInvalid, match="generation changed"):
        cx.edge_streamlines(out_dir, 1, 2)
    assert calls["n"] >= 2, "the assemble-time check must re-pin, not trust the entry pin blindly"


@needs
def test_edge_tubes_after_derivation_rebuild_refuses_stale_service_manifest(tmp_path):
    """Item 4: a server holding a STALE in-memory manifest (loaded before a
    derivation rebuild) must be refused, even though the connectome's own
    provenance-vs-disk checks all pass (both were rebuilt together)."""
    import copy

    case = _make_case(tmp_path, with_lesion=True)
    out_dir = cx.connectome_out_dir(case["root"])
    built_v1 = cx.build(case["root"], case["corpus_key"], case["parc_key"], out_dir)
    assert built_v1.outcome is cx.Outcome.OK, built_v1.error

    man_v1 = json.loads(Path(case["manifest_path"]).read_text())
    stale_service_manifest = copy.deepcopy(man_v1)

    # Simulate a derivation rebuild: the lesion moves to a new file, active_derivation
    # UNCHANGED (so the pre-existing active_derivation check alone would not catch this).
    man_v2 = copy.deepcopy(man_v1)
    man_v2["inputs"]["lesion"] = {"path": "nifti/lesion_v2.nii.gz"}
    Path(case["manifest_path"]).write_text(json.dumps(man_v2))
    assert man_v2.get("active_derivation") == man_v1.get("active_derivation")

    # The connectome itself gets rebuilt too (a real deployment would rebuild
    # both together) — its provenance now matches the DISK manifest exactly.
    built_v2 = cx.build(case["root"], case["corpus_key"], case["parc_key"], out_dir)
    assert built_v2.outcome is cx.Outcome.OK, built_v2.error

    # Without service_manifest: the pre-existing checks alone pass (disk == provenance).
    cx.check_fresh(built_v2.provenance, out_dir)  # must not raise

    # With the STALE in-memory manifest: must refuse.
    with pytest.raises(cx.ConnectomeInvalid, match="in-memory manifest no longer matches"):
        cx.check_fresh(built_v2.provenance, out_dir, service_manifest=stale_service_manifest)

    # And the route itself, wired the same way serve.py wires it.
    service = _tubes_service(case["root"], stale_service_manifest)
    route = _route(service)
    route.path = "/api/connectome/edge/1/2/tubes"
    status, body, _ = route._handle_connectome_edge_tubes()
    assert status == 409
    assert body["error"] == "connectome-stale"

    # A FRESH service.manifest (the rebuilt one) is accepted normally.
    fresh_service = _tubes_service(case["root"], man_v2)
    route2 = _route(fresh_service)
    route2.path = "/api/connectome/edge/1/2/tubes"
    status2, buf2, extra2 = route2._handle_connectome_edge_tubes()
    assert status2 == 200


@needs
def test_edge_tubes_carries_complete_count_source_provenance(tmp_path):
    """Item 5: matrix/assignments path+sha256 (and lesion path+sha256, when
    used) must be reported on the tubes response — never only the edge's own
    extraction identity."""
    case = _make_case(tmp_path, with_lesion=True)
    out_dir = cx.connectome_out_dir(case["root"])
    built = cx.build(case["root"], case["corpus_key"], case["parc_key"], out_dir)
    assert built.outcome is cx.Outcome.OK, built.error

    man = json.loads(Path(case["manifest_path"]).read_text())
    service = _tubes_service(case["root"], man)
    route = _route(service)
    route.path = "/api/connectome/edge/1/2/tubes"
    status, buf, extra = route._handle_connectome_edge_tubes()
    assert status == 200

    assert extra["X-matrixPath"] == built.provenance["matrix"]["path"]
    assert extra["X-matrixSha256"] == built.provenance["matrix"]["sha256"]
    assert len(extra["X-matrixSha256"]) == 64
    assert extra["X-assignmentsPath"] == built.provenance["assignments"]["path"]
    assert extra["X-assignmentsSha256"] == built.provenance["assignments"]["sha256"]
    assert len(extra["X-assignmentsSha256"]) == 64
    assert extra["X-assignmentRadiusMm"] == str(built.provenance["radius_mm"])

    assert extra["X-lesionHits"] == "3" and extra["X-lesionTotal"] == "3"
    assert extra["X-lesionPath"] == "nifti/lesion.nii.gz"
    assert len(extra["X-lesionSha256"]) == 64


@needs
def test_concurrent_edge_extractions_are_isolated_or_locked(demo_case, tmp_path):
    """Item 6: concurrent requests for the SAME edge must never read a file
    another is truncating — each extracts into a unique temp file and
    os.replace()s it into place, serialized per (generation, a, b) by a lock."""
    import threading

    out_dir = cx.connectome_out_dir(demo_case["root"])
    built = cx.build(demo_case["root"], demo_case["corpus_key"], demo_case["parc_key"], out_dir)
    assert built.outcome is cx.Outcome.OK, built.error

    results = []
    errors = []
    result_lock = threading.Lock()

    def worker():
        try:
            edge = cx.edge_streamlines(out_dir, 1, 2)
            with result_lock:
                results.append(edge)
        except Exception as e:  # noqa: BLE001 — captured for the assertion below
            with result_lock:
                errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=90)

    assert not errors, f"concurrent extraction raised: {errors}"
    assert len(results) == 6
    content_hashes = {r["content_sha256"] for r in results}
    assert len(content_hashes) == 1, f"all concurrent extractions must agree on content (got {content_hashes})"
    for r in results:
        assert r["matrix_count"] == r["assignment_row_count"] == r["extracted_count"] == 3

    # The file on disk afterward is complete and valid — no torn write survives.
    tck_path = results[0]["tck_path"]
    assert os.path.isfile(tck_path)
    assert cx._tckinfo_count(tck_path) == 3
    assert not any(name.endswith(".tck") and name.startswith(".") for name in os.listdir(os.path.dirname(tck_path))), \
        "no orphaned temp .tck file left behind"


# ---------------------------------------------------------------------------
# W8 final round, item 2: check_fresh must hash the manifest FILE on disk
# against provenance["manifest"]["sha256"] — a field that changes (e.g.
# parcellation_qc.sheet_sha to a different non-empty value) but isn't caught
# by any OTHER independent re-check must still be refused.
# ---------------------------------------------------------------------------

@needs
def test_check_fresh_refuses_manifest_hash_drift_when_service_matches_disk(demo_case, tmp_path):
    out_dir = cx.connectome_out_dir(demo_case["root"])
    built = cx.build(demo_case["root"], demo_case["corpus_key"], demo_case["parc_key"], out_dir)
    assert built.outcome is cx.Outcome.OK, built.error

    man_path = Path(demo_case["root"]) / "manifest.json"
    original_text = man_path.read_text()
    man = json.loads(original_text)
    # Change ONLY parcellation_qc.sheet_sha — still non-empty, so
    # discover_parcellation's "QC still signed" check (approved_by/date/
    # sheet_sha all truthy) passes untouched; active_derivation, corpus, and
    # parcellation bytes are all unchanged too. No other check below the
    # manifest-hash one would catch this.
    man["parcellation_qc"]["sheet_sha"] = "a-different-sheet-sha"
    man_path.write_text(json.dumps(man))

    # service_manifest matches disk exactly (both hold the edited manifest) —
    # so the SECOND check (service vs disk) would pass; only the FIRST check
    # (disk vs provenance's recorded build-time hash) can catch this.
    provenance = cx.load_provenance(out_dir)
    with pytest.raises(cx.ConnectomeInvalid, match="manifest.json changed since build"):
        cx.check_fresh(provenance, out_dir, service_manifest=man)

    # And without service_manifest at all — the disk-hash check alone still refuses.
    with pytest.raises(cx.ConnectomeInvalid, match="manifest.json changed since build"):
        cx.check_fresh(provenance, out_dir)

    # Restoring the exact original bytes must pass again (proves this isn't
    # a permanently-broken generation, only a genuinely-changed manifest).
    man_path.write_text(original_text)
    cx.check_fresh(provenance, out_dir)  # must not raise


# ---------------------------------------------------------------------------
# Wave 3: a manifest kept in the repo's cases/ dir is recorded RELATIVE to that
# dir, never as a worktree-specific absolute path. Legacy absolute records
# still load (and still fail closed when the recorded file is gone).
# ---------------------------------------------------------------------------

def _repo_case(tmp_path, monkeypatch, repo_name="repo-a"):
    cases_dir = tmp_path / repo_name / "cases"
    monkeypatch.setattr(cx, "REPO_CASES_DIR", str(cases_dir))
    case = _make_case(tmp_path, manifest_dir=cases_dir / "synthetic-connectome")
    return cases_dir, case


@needs
def test_build_records_repo_manifest_relative_to_repo_cases_dir(tmp_path, monkeypatch):
    cases_dir, case = _repo_case(tmp_path, monkeypatch)
    out_dir = str(tmp_path / "work" / "connectome")
    result = cx.build(
        case["root"], case["corpus_key"], case["parc_key"], out_dir,
        manifest_path=case["manifest_path"],
    )
    assert result.outcome is cx.Outcome.OK, result.error
    rec = result.provenance["manifest"]
    assert rec["path"] == "synthetic-connectome/manifest.json"
    assert rec["relative_to"] == "repo_cases"
    assert str(tmp_path) not in json.dumps(rec)  # no worktree-specific absolute path


@needs
def test_repo_relative_manifest_record_resolves_in_another_worktree(tmp_path, monkeypatch):
    cases_dir, case = _repo_case(tmp_path, monkeypatch)
    out_dir = str(tmp_path / "work" / "connectome")
    built = cx.build(
        case["root"], case["corpus_key"], case["parc_key"], out_dir,
        manifest_path=case["manifest_path"],
    )
    assert built.outcome is cx.Outcome.OK, built.error
    # A second checkout holds the same manifest; the first checkout is gone.
    other = tmp_path / "repo-b" / "cases" / "synthetic-connectome"
    other.mkdir(parents=True)
    (other / "manifest.json").write_bytes(Path(case["manifest_path"]).read_bytes())
    Path(case["manifest_path"]).unlink()
    monkeypatch.setattr(cx, "REPO_CASES_DIR", str(tmp_path / "repo-b" / "cases"))
    cx.check_fresh(cx.load_provenance(out_dir), out_dir)  # must not raise


def test_legacy_absolute_manifest_record_still_resolves(tmp_path):
    m = tmp_path / "elsewhere" / "manifest.json"
    assert cx._resolve_manifest_path(str(tmp_path / "root"), {"path": str(m)}) == str(m)
    # legacy relative (case-root nested) and missing (pre-field) records unchanged
    root = str(tmp_path / "root")
    assert cx._resolve_manifest_path(root, {"path": "manifest.json"}) == os.path.join(root, "manifest.json")
    assert cx._resolve_manifest_path(root, {}) == os.path.join(root, "manifest.json")


def test_unknown_manifest_record_base_fails_closed(tmp_path):
    with pytest.raises(cx.ConnectomeInvalid, match="relative_to"):
        cx._resolve_manifest_path(str(tmp_path), {"path": "x/manifest.json", "relative_to": "moon"})


def test_repo_relative_manifest_record_refuses_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(cx, "REPO_CASES_DIR", str(tmp_path / "repo" / "cases"))
    with pytest.raises(cx.ConnectomeInvalid):
        cx._resolve_manifest_path(str(tmp_path), {"path": "../../x/manifest.json", "relative_to": "repo_cases"})


@needs
def test_legacy_absolute_manifest_record_fails_closed_when_gone(tmp_path):
    manifest_dir = tmp_path / "outside-repo"
    case = _make_case(tmp_path, manifest_dir=manifest_dir)
    out_dir = str(tmp_path / "work" / "connectome")
    built = cx.build(
        case["root"], case["corpus_key"], case["parc_key"], out_dir,
        manifest_path=case["manifest_path"],
    )
    assert built.outcome is cx.Outcome.OK, built.error
    prov = cx.load_provenance(out_dir)
    assert os.path.isabs(prov["manifest"]["path"]) and "relative_to" not in prov["manifest"]
    cx.check_fresh(prov, out_dir)  # absolute legacy form still loads
    Path(case["manifest_path"]).unlink()
    with pytest.raises(cx.ConnectomeInvalid, match="manifest.json missing"):
        cx.check_fresh(prov, out_dir)


# ---------------------------------------------------------------------------
# Wave 3: _EDGE_LOCKS is bounded (unused locks are evicted) and a generation
# swap between the final pin check and route packing is refused.
# ---------------------------------------------------------------------------

def test_edge_lock_serializes_and_evicts_when_unused():
    import threading

    key = ("/gen/x", 1, 2)
    assert key not in cx._EDGE_LOCKS
    inside = threading.Event()
    release = threading.Event()
    order = []

    def holder():
        with cx._edge_lock(*key):
            order.append("holder-in")
            inside.set()
            release.wait(5)
            order.append("holder-out")

    def waiter():
        inside.wait(5)
        with cx._edge_lock(*key):
            order.append("waiter-in")

    t1 = threading.Thread(target=holder)
    t2 = threading.Thread(target=waiter)
    t1.start()
    t2.start()
    inside.wait(5)
    # The waiter is blocked on the SAME lock, so the entry must still exist.
    for _ in range(100):
        if cx._EDGE_LOCKS.get(key) is not None and _lock_users(key) == 2:
            break
        threading.Event().wait(0.01)
    assert key in cx._EDGE_LOCKS
    assert order == ["holder-in"]
    release.set()
    t1.join(5)
    t2.join(5)
    assert order == ["holder-in", "holder-out", "waiter-in"]
    assert key not in cx._EDGE_LOCKS, "an unused edge lock must be evicted"


def _lock_users(key):
    entry = cx._EDGE_LOCKS.get(key)
    return getattr(entry, "users", None) if entry is not None else 0


def test_edge_lock_map_stays_bounded_over_many_edges():
    for a in range(1, 200):
        with cx._edge_lock("/gen/y", a, a + 1):
            pass
    assert not any(k[0] == "/gen/y" for k in cx._EDGE_LOCKS)


@needs
def test_edge_streamlines_leaves_no_lock_behind(demo_case):
    out_dir = cx.connectome_out_dir(demo_case["root"])
    built = cx.build(demo_case["root"], demo_case["corpus_key"], demo_case["parc_key"], out_dir)
    assert built.outcome is cx.Outcome.OK, built.error
    edge = cx.edge_streamlines(out_dir, 1, 2)
    assert not any(k[0] == edge["generation_dir"] for k in cx._EDGE_LOCKS)


def _point_current_at(out_dir, gen_dir):
    import uuid as _uuid
    tmp_link = os.path.join(out_dir, f".current-{_uuid.uuid4().hex[:8]}")
    os.symlink(os.path.relpath(gen_dir, out_dir), tmp_link)
    os.replace(tmp_link, os.path.join(out_dir, cx.CURRENT_LINK))


@needs
def test_edge_tubes_refuses_generation_swap_during_packing(demo_case, monkeypatch):
    """Deterministic interleaving: `current` moves AFTER edge_streamlines'
    own final pin check but while the route is loading/packing the tubes.
    The route must refuse (409) rather than ship a superseded generation."""
    out_dir = cx.connectome_out_dir(demo_case["root"])
    gen1 = cx.build(demo_case["root"], demo_case["corpus_key"], demo_case["parc_key"], out_dir)
    assert gen1.outcome is cx.Outcome.OK, gen1.error
    gen1_dir = cx.pin_generation(out_dir)
    gen2 = cx.build(demo_case["root"], demo_case["corpus_key"], demo_case["parc_key"], out_dir, radius_mm=6.0)
    assert gen2.outcome is cx.Outcome.OK, gen2.error
    gen2_dir = cx.pin_generation(out_dir)
    assert gen1_dir != gen2_dir
    _point_current_at(out_dir, gen1_dir)

    serve = importlib.import_module("tractlab.serve")
    real_load = serve.load_prebuilt_bundle

    def load_then_republish(*args, **kwargs):
        out = real_load(*args, **kwargs)
        _point_current_at(out_dir, gen2_dir)  # a rebuild publishes mid-pack
        return out

    monkeypatch.setattr(serve, "load_prebuilt_bundle", load_then_republish)
    man = json.loads((Path(demo_case["root"]) / "manifest.json").read_text())
    route = _route(_tubes_service(demo_case["root"], man))
    route.path = "/api/connectome/edge/1/2/tubes"
    status, body, _ = route._handle_connectome_edge_tubes()
    assert status == 409, (status, body if isinstance(body, dict) else None)
    assert body["error"] == "connectome-stale"
    assert "generation changed" in body["reason"]


@needs
def test_edge_json_refuses_generation_swap_after_extraction(demo_case, monkeypatch):
    out_dir = cx.connectome_out_dir(demo_case["root"])
    gen1 = cx.build(demo_case["root"], demo_case["corpus_key"], demo_case["parc_key"], out_dir)
    assert gen1.outcome is cx.Outcome.OK, gen1.error
    gen1_dir = cx.pin_generation(out_dir)
    gen2 = cx.build(demo_case["root"], demo_case["corpus_key"], demo_case["parc_key"], out_dir, radius_mm=6.0)
    assert gen2.outcome is cx.Outcome.OK, gen2.error
    gen2_dir = cx.pin_generation(out_dir)
    _point_current_at(out_dir, gen1_dir)

    man = json.loads((Path(demo_case["root"]) / "manifest.json").read_text())
    route = _route(_tubes_service(demo_case["root"], man))
    real_info = route._edge_lesion_info

    def info_then_republish(edge):
        out = real_info(edge)
        _point_current_at(out_dir, gen2_dir)
        return out

    route._edge_lesion_info = info_then_republish
    route.path = "/api/connectome/edge/1/2"
    status, body, _ = route._handle_connectome_edge()
    assert status == 409, (status, body)
    assert "generation changed" in body["reason"]
