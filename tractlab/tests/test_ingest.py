from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest

from tractlab import ingest
from dicom_synth import expected_t1_affine, write_test_series


def _series(
    name: str,
    *,
    metadata: dict | None = None,
    bvals: list[float] | None = None,
    bvecs: list[list[float]] | None = None,
    nvolumes: int = 1,
) -> dict:
    return {
        "metadata": {"SeriesDescription": name, **(metadata or {})},
        "bvals": bvals,
        "bvecs": bvecs,
        "nvolumes": nvolumes,
        "nifti": f"raw/dcm2niix/{name}.nii.gz",
        "json": f"raw/dcm2niix/{name}.json",
        "bval": f"raw/dcm2niix/{name}.bval" if bvals is not None else None,
        "bvec": f"raw/dcm2niix/{name}.bvec" if bvecs is not None else None,
    }


def _dwi(name: str = "dwi", *, nvolumes: int = 7) -> dict:
    bvals = [0.0, *([1000.0] * (nvolumes - 1))]
    directions = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [2**-0.5, 2**-0.5, 0],
        [2**-0.5, 0, 2**-0.5],
        [0, 2**-0.5, 2**-0.5],
    ]
    bvecs = [
        [0, 0, 0],
        *(directions[idx % len(directions)] for idx in range(nvolumes - 1)),
    ]
    return _series(name, bvals=bvals, bvecs=bvecs, nvolumes=nvolumes)


def _t1(name: str = "mprage", **metadata) -> dict:
    return _series(
        name,
        metadata={
            "MRAcquisitionType": "3D",
            "ScanningSequence": "GR",
            "SequenceName": "tfl3d1_16",
            "SeriesDescription": "MPRAGE",
            "VoxelSize": [1.0, 1.0, 1.0],
            **metadata,
        },
    )


def test_classify_picks_largest_dwi_and_excludes_derived():
    small = _dwi("small")
    large = _dwi("large", nvolumes=10)
    derived = _dwi("derived", nvolumes=40)
    derived["metadata"]["ImageType"] = ["DERIVED", "ADC"]
    result = ingest.classify([small, large, derived, _t1()])
    assert result["dwi"]["json"] == "raw/dcm2niix/large.json"


def test_classify_pairs_opposite_epi_and_rejects_missing_or_unknown_polarity():
    dwi = _dwi()
    dwi["metadata"].update(
        {
            "PhaseEncodingDirection": "j",
            "TotalReadoutTime": 0.049,
            "MatrixSize": [8, 8, 6],
            "VoxelSize": [1.0, 1.0, 1.0],
        }
    )
    reverse = _series(
        "reverse",
        metadata={
            "SeriesDescription": "EPI b0",
            "ScanningSequence": "EP\\SE",
            "PhaseEncodingDirection": "j-",
            "MatrixSize": [8, 8, 6],
            "VoxelSize": [1.0, 1.0, 1.0],
        },
        bvals=[0.0],
        bvecs=[[0, 0, 0]],
    )
    result = ingest.classify([dwi, reverse, _t1()])
    assert result["rpe"]["json"] == "raw/dcm2niix/reverse.json"
    assert result["rpe_mode"] == "pair"

    without_reverse = ingest.classify([dwi, _t1()])
    assert without_reverse["rpe"] is None
    assert without_reverse["rpe_mode"] == "none"
    assert any("reverse" in warning.lower() for warning in without_reverse["warnings"])

    dwi["metadata"]["PhaseEncodingDirection"] = None
    unknown = ingest.classify([dwi, reverse, _t1()])
    assert unknown["rpe"] is None
    assert unknown["rpe_mode"] == "none"
    assert any("polarity" in warning.lower() for warning in unknown["warnings"])


def test_classify_selects_high_resolution_t1_and_warns_post_contrast():
    low_resolution = _t1("coarse", VoxelSize=[1.5, 1.0, 1.0])
    post_contrast = _t1(
        "post",
        SeriesDescription="MPRAGE post contrast",
        ContrastBolusAgent="gadolinium",
    )
    result = ingest.classify([_dwi(), low_resolution, post_contrast])
    assert result["t1"]["json"] == "raw/dcm2niix/post.json"
    assert result["t1_post_contrast"] is True
    assert any("contrast" in warning.lower() for warning in result["warnings"])


def test_classify_refuses_missing_dwi_or_t1():
    no_dwi = ingest.classify([_t1()])
    assert any("dwi" in reason.lower() for reason in no_dwi["refusals"])

    derived = _dwi("derived")
    derived["metadata"]["ImageType"] = ["DERIVED", "ADC"]
    derived_only = ingest.classify([derived, _t1()])
    assert any("dwi" in reason.lower() for reason in derived_only["refusals"])

    no_t1 = ingest.classify([_dwi()])
    assert any("t1" in reason.lower() for reason in no_t1["refusals"])


def test_end_to_end_synthetic_dicom_round_trip_and_phi_anonymization(tmp_path):
    dicom_dir = tmp_path / "input"
    write_test_series(dicom_dir)
    cases_root = tmp_path / "cases"

    result = ingest.run_ingest(
        dicom_dir,
        cases_root=cases_root,
        label="synthetic validation",
    )

    assert result["ok"] is True
    case_root = Path(result["case_root"])
    assert case_root.parent == cases_root
    case = json.loads((case_root / "case.json").read_text())
    assert case["schema"] == "tractlab.case/1"
    assert case["source"]["n_files"] == 48
    assert case["inputs"]["dwi"]["bval"]
    assert case["inputs"]["dwi"]["bvec"]
    bvals = np.loadtxt(case_root / case["inputs"]["dwi"]["bval"]).reshape(-1)
    assert np.count_nonzero(bvals) == 6
    assert sorted(bvals[bvals > 0].tolist()) == [1000.0] * 6

    t1_path = case_root / case["inputs"]["t1"]["nifti"]
    t1_image = nib.load(str(t1_path))
    np.testing.assert_allclose(t1_image.affine, expected_t1_affine(), atol=1e-4, rtol=0)

    manifest = json.loads((case_root / "manifest.json").read_text())
    assert manifest["case_id"] == case["case_id"]
    assert manifest["case_root"] == "."
    assert manifest["disclaimer"]
    assert manifest["status"] == "ingested"
    assert manifest["inputs"] == {
        "b0": {"path": "nifti/b0.nii.gz"},
        "fod": {"path": "nifti/wmfod_norm.mif"},
        "mask": {"path": "nifti/mask_up.nii.gz"},
        "t1": {"path": "nifti/t1_brain_dwi.nii.gz"},
    }
    for key in ("t1_qc", "atlas_prior_qc", "parcellation_qc"):
        assert key in manifest
        assert manifest[key]["approved_by"] is None
        assert manifest[key]["date"] is None
        assert manifest[key]["sheet_path"] is None
        assert manifest[key]["sheet_sha"] is None
        assert isinstance(manifest[key]["auto"], dict)
        assert manifest[key]["derivation"] == ""

    sidecars = list((case_root / "raw/dcm2niix").glob("*.json"))
    assert sidecars
    forbidden = {
        "PatientName",
        "PatientID",
        "PatientBirthDate",
        "AccessionNumber",
        "InstitutionName",
        "InstitutionAddress",
        "AcquisitionDateTime",
        "SeriesInstanceUID",
        "StudyInstanceUID",
    }
    for path in sidecars:
        data = json.loads(path.read_text())
        assert forbidden.isdisjoint(ingest._walk_keys(data))
        assert "SeriesDescription" in data
        assert "SYNTHETIC" not in path.read_text()
    assert str(dicom_dir) not in (case_root / "case.json").read_text()
    assert "SYNTHETIC" not in (case_root / "case.json").read_text()


def test_phi_key_injection_deletes_case_and_returns_exit_three(tmp_path, monkeypatch):
    source = tmp_path / "synthetic-dicom"
    source.mkdir()
    (source / "placeholder.dcm").write_bytes(b"synthetic")

    def injected_writer(_dicom_dir: Path, output_dir: Path, _binary: str) -> None:
        output_dir.mkdir(parents=True)
        (output_dir / "injected.json").write_text(
            json.dumps({"PatientName": "synthetic", "ImageType": ["ORIGINAL"]})
        )

    monkeypatch.setattr(ingest, "run_dcm2niix", injected_writer)
    result = ingest.run_ingest(
        source,
        cases_root=tmp_path / "cases",
        label="phi test",
    )
    assert result["exit_code"] == 3
    assert result["ok"] is False
    assert not list((tmp_path / "cases").glob("case-*"))


def test_cases_root_inside_repo_is_refused(tmp_path):
    result = ingest.run_ingest(
        tmp_path / "synthetic-dicom",
        cases_root=Path(ingest.__file__).resolve().parents[2] / "cases" / "must-not-exist",
    )
    assert result["exit_code"] == 2
    assert result["ok"] is False
    assert not (Path(ingest.__file__).resolve().parents[2] / "cases" / "must-not-exist").exists()


def test_make_demo_dicom_smoke_on_small_cropped_raw_volume(tmp_path):
    raw = tmp_path / "raw"
    (raw / "dwi").mkdir(parents=True)
    (raw / "anat").mkdir()
    affine = np.eye(4)
    dwi = np.stack(
        [np.full((8, 8, 6), index + 1, dtype=np.int16) for index in range(7)],
        axis=3,
    )
    dwi_path = raw / "dwi" / "sub-demo_ses-01_dwi.nii.gz"
    nib.save(nib.Nifti1Image(dwi, affine), dwi_path)
    np.savetxt(dwi_path.with_suffix("").with_suffix(".bval"), [0, *([1000] * 6)])
    np.savetxt(
        dwi_path.with_suffix("").with_suffix(".bvec"),
        np.asarray(
            [
                [0, 1, 0, 0, 2**-0.5, 2**-0.5, 0],
                [0, 0, 1, 0, 2**-0.5, 0, 2**-0.5],
                [0, 0, 0, 1, 0, 2**-0.5, 2**-0.5],
            ]
        ),
    )
    t1_data = np.arange(8 * 8 * 6, dtype=np.int16).reshape((8, 8, 6))
    nib.save(
        nib.Nifti1Image(t1_data, affine),
        raw / "anat" / "sub-demo_ses-01_T1w.nii.gz",
    )

    output = tmp_path / "synthetic-dicom"
    maker = Path(ingest.__file__).resolve().parents[2] / "scripts/make_demo_dicom.py"
    completed = subprocess.run(
        [sys.executable, str(maker), str(output), "--raw", str(raw)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "field maps were omitted" in completed.stdout
    result = ingest.run_ingest(output, cases_root=tmp_path / "cases", label="demo smoke")
    assert result["ok"] is True
    case_root = Path(result["case_root"])
    case = json.loads((case_root / "case.json").read_text())
    bvals = np.loadtxt(case_root / case["inputs"]["dwi"]["bval"]).reshape(-1)
    assert np.count_nonzero(bvals) == 6

    refused = subprocess.run(
        [
            sys.executable,
            str(maker),
            str(Path(ingest.__file__).resolve().parents[2] / "cases" / "must-not-exist"),
            "--raw",
            str(raw),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert refused.returncode == 2


# --- GE portal export (SIGNA Pioneer) regressions ---------------------------------


def _ge_dwi(name: str = "ORIG: Ax DWI CSD 60DIR B3000 10BO", *, nvolumes: int = 70) -> dict:
    series = _dwi(name, nvolumes=nvolumes)
    series["metadata"].update(
        {
            "Manufacturer": "GE",
            "ScanningSequence": "EP\\SE",
            "PhaseEncodingDirection": "j-",
            "AcquisitionMatrixPE": 130,
            "ReconMatrixPE": 256,
            "MatrixSize": [256, 256, 80],
            "VoxelSize": [0.9375, 0.9375, 2.0],
        }
    )
    return series


def _ge_reverse(name: str = "ORIG: POLARIDADE INVERTIDA", *, mini_dwi: bool = True) -> dict:
    bvals = [0.0, 0.0, *([3000.0] * 20)] if mini_dwi else [0.0, 0.0, 0.0]
    bvecs = [[0, 0, 0], [0, 0, 0], *([[1, 0, 0]] * (len(bvals) - 2))]
    return _series(
        name,
        metadata={
            "Manufacturer": "GE",
            "ScanningSequence": "EP\\RM",
            "PhaseEncodingDirection": "j",
            "MatrixSize": [256, 256, 80],
            "VoxelSize": [0.9375, 0.9375, 2.0],
        },
        bvals=bvals,
        bvecs=bvecs,
        nvolumes=len(bvals),
    )


def test_ge_reverse_pe_mini_dwi_is_accepted_and_pure_b0_preferred():
    dwi = _ge_dwi()
    mini = _ge_reverse()
    result = ingest.classify([dwi, mini, _t1()])
    assert result["rpe"] is mini
    assert result["rpe_mode"] == "pair"
    assert any("reverse-pe" in w.lower() and "b=0" in w.lower() for w in result["warnings"])

    pure = _ge_reverse("ORIG: SE-EPI b0 reverse", mini_dwi=False)
    preferred = ingest.classify([dwi, mini, pure, _t1()])
    assert preferred["rpe"] is pure


def test_ge_reverse_pe_without_any_b0_is_rejected():
    dwi = _ge_dwi()
    no_b0 = _ge_reverse()
    no_b0["bvals"] = [3000.0] * 22
    result = ingest.classify([dwi, no_b0, _t1()])
    assert result["rpe"] is None


def test_ge_orig_series_preferred_on_volume_tie():
    filtered = _ge_dwi("Ax DWI CSD 60DIR B3000 10BO")
    orig = _ge_dwi()
    for order in ([filtered, orig], [orig, filtered]):
        result = ingest.classify([*order, _t1()])
        assert result["dwi"] is orig
        assert any("ORIG:" in w for w in result["warnings"])

    rpe_filtered = _ge_reverse("POLARIDADE INVERTIDA")
    rpe_orig = _ge_reverse()
    result = ingest.classify([orig, rpe_filtered, rpe_orig, _t1()])
    assert result["rpe"] is rpe_orig


def test_post_contrast_detects_ge_and_portuguese_labels():
    for label in ("3D Ax T1 +C", "T1 C+ 3D", "3D T1 BRAVO GD", "T1 GADO", "3D T1 POS CONTRASTE",
                  "3D T1 PÓS-CONTRASTE", "MPRAGE post contrast"):
        assert ingest._post_contrast({"SeriesDescription": label}), label
    for label in ("T1 SPGR", "MPRAGE", "Ax DWI CSD 60DIR", "CORONAL", "3D T1 BRAVO", "SAG CUBE T2"):
        assert not ingest._post_contrast({"SeriesDescription": label}), label


def test_readout_time_sidecar_derived_or_nominal_with_warning():
    value, source, warning = ingest._readout_time({"TotalReadoutTime": 0.049})
    assert (value, source, warning) == (0.049, "sidecar", None)

    value, source, warning = ingest._readout_time({"EffectiveEchoSpacing": 0.0004, "ReconMatrixPE": 128})
    assert source == "derived" and warning is None
    assert value == pytest.approx(0.0004 * 127)

    value, source, warning = ingest._readout_time({"Manufacturer": "GE"})
    assert (value, source) == (0.05, "nominal")
    assert warning and "invariant" in warning.lower()


def test_regrid_voxel_only_for_zero_filled_reconstructions():
    meta = _ge_dwi()["metadata"]
    assert ingest._regrid_voxel(meta) == [1.846, 1.846, 2.0]
    native = dict(meta, ReconMatrixPE=130)
    assert ingest._regrid_voxel(native) is None
    assert ingest._regrid_voxel({"VoxelSize": [2.0, 2.0, 2.0]}) is None


def test_run_dcm2niix_tolerates_latin1_output(tmp_path):
    fake = tmp_path / "fake-dcm2niix"
    fake.write_text(
        "#!/bin/sh\nprintf 'Convert 1 DICOM as ADC mm\\262/s\\n'\nprintf 'warn \\262\\n' >&2\nexit 0\n"
    )
    fake.chmod(0o755)
    ingest.run_dcm2niix(tmp_path / "in", tmp_path / "out", str(fake))


def test_site_and_staff_fields_are_scrubbed_and_case_kept(tmp_path, monkeypatch):
    source = tmp_path / "synthetic-dicom"
    write_test_series(source)
    real = ingest.run_dcm2niix
    site_keys = sorted(ingest._SITE_KEYS)

    def writer_with_site_fields(dicom_dir: Path, output_dir: Path, binary: str) -> None:
        real(dicom_dir, output_dir, binary)
        for path in output_dir.glob("*.json"):
            data = json.loads(path.read_text())
            data.update({key: "synthetic-site" for key in site_keys})
            path.write_text(json.dumps(data))

    monkeypatch.setattr(ingest, "run_dcm2niix", writer_with_site_fields)
    result = ingest.run_ingest(source, cases_root=tmp_path / "cases", label="site scrub")
    assert result["ok"] is True, result
    case_root = Path(result["case_root"])
    for path in (case_root / "raw/dcm2niix").glob("*.json"):
        assert "synthetic-site" not in path.read_text()
        assert set(site_keys).isdisjoint(ingest._walk_keys(json.loads(path.read_text())))
    assert {"InstitutionName", "StationName", "OperatorsName", "DeviceSerialNumber"} <= ingest._SITE_KEYS
    assert ingest._SITE_KEYS <= ingest._PHI_KEYS


def test_ge_like_export_writes_runnable_case_json(tmp_path, monkeypatch):
    """A GE-like conversion (no TRT, mini-DWI reverse-PE, +C T1, zero-filled) yields a runnable case."""
    source = tmp_path / "synthetic-dicom"
    source.mkdir()
    (source / "placeholder.dcm").write_bytes(b"synthetic")
    shape = (8, 8, 4)
    affine = np.diag([0.9375, 0.9375, 2.0, 1.0])

    def ge_writer(_dicom_dir: Path, output_dir: Path, _binary: str) -> None:
        output_dir.mkdir(parents=True)
        dwi_bvals = [0, 0, 3000, 3000, 3000, 3000, 3000, 3000]
        dirs = [[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                [2**-0.5, 2**-0.5, 0], [2**-0.5, 0, 2**-0.5], [0, 2**-0.5, 2**-0.5]]
        rpe_bvals = [0, 3000, 0, 3000]
        common = {"Manufacturer": "GE", "ImageType": ["ORIGINAL", "PRIMARY"],
                  "InstitutionName": "synthetic-site", "StationName": "synthetic-station"}
        series = {
            "800_ORIG_DWI": (dwi_bvals, dirs, {"SeriesDescription": "ORIG: Ax DWI CSD 60DIR B3000 10BO",
                                               "ScanningSequence": "EP\\SE", "PhaseEncodingDirection": "j-",
                                               "AcquisitionMatrixPE": 130, "ReconMatrixPE": 256}),
            "810_ORIG_RPE": (rpe_bvals, [[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0]],
                             {"SeriesDescription": "ORIG: POLARIDADE INVERTIDA", "ScanningSequence": "EP\\RM",
                              "PhaseEncodingDirection": "j"}),
        }
        for stem, (bvals, vecs, meta) in series.items():
            data = np.stack([np.full(shape, 100 + i, dtype=np.int16) for i in range(len(bvals))], axis=3)
            nib.save(nib.Nifti1Image(data, affine), output_dir / f"{stem}.nii.gz")
            np.savetxt(output_dir / f"{stem}.bval", [bvals], fmt="%g")
            np.savetxt(output_dir / f"{stem}.bvec", np.asarray(vecs, dtype=float).T, fmt="%.6f")
            (output_dir / f"{stem}.json").write_text(json.dumps({**common, **meta}))
        t1 = np.zeros((8, 8, 8), dtype=np.int16)
        nib.save(nib.Nifti1Image(t1, np.eye(4)), output_dir / "500_T1.nii.gz")
        (output_dir / "500_T1.json").write_text(json.dumps({
            **common, "SeriesDescription": "3D Ax T1 +C", "MRAcquisitionType": "3D",
            "ScanningSequence": "GR"}))

    monkeypatch.setattr(ingest, "run_dcm2niix", ge_writer)
    result = ingest.run_ingest(source, cases_root=tmp_path / "cases", label="ge synthetic")
    assert result["ok"] is True, result
    case_root = Path(result["case_root"])
    case = json.loads((case_root / "case.json").read_text())
    dwi = case["inputs"]["dwi"]
    assert dwi["pe_dir"] == "j-"
    assert dwi["total_readout_time"] == 0.05
    assert dwi["total_readout_time_source"] == "nominal"
    assert dwi["regrid_voxel_mm"] == [1.846, 1.846, 2.0]
    assert case["inputs"]["t1"]["post_contrast"] is True
    rpe = case["inputs"]["rpe"]
    assert rpe["mode"] == "pair"
    assert rpe["nifti"].endswith("810_ORIG_RPE_rpe_b0.nii.gz")
    rpe_img = nib.load(str(case_root / rpe["nifti"]))
    assert rpe_img.shape == (*shape, 2)
    np.testing.assert_array_equal(rpe_img.get_fdata()[0, 0, 0, :], [100, 102])
    np.testing.assert_allclose(rpe_img.affine, affine)
    rpe_sidecar = json.loads((case_root / rpe["json"]).read_text())
    assert rpe_sidecar["PhaseEncodingDirection"] == "j"
    for path in (case_root / "raw/dcm2niix").glob("*.json"):
        assert "synthetic-site" not in path.read_text()
        assert "synthetic-station" not in path.read_text()

    helper = Path(ingest.__file__).resolve().parents[2] / "scripts/pipeline/runner_state.py"
    completed = subprocess.run([sys.executable, str(helper), "config", str(case_root)],
                               capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr.decode()
    fields = completed.stdout.split(b"\0")[:-1]
    assert len(fields) == 13
    assert fields[6] == b"0.05"
    assert fields[7] == b"pair"
    assert fields[12] == b"1.846,1.846,2"


def test_manifest_skeleton_uses_the_packaged_template_and_stays_unsigned(tmp_path, monkeypatch):
    """A clean clone has no cases/ folder, so the skeleton must not read one."""
    from tractlab import ingest as ingest_module

    template = Path(ingest_module.__file__).with_name("templates") / "manifest_skeleton.json"
    assert template.is_file()
    monkeypatch.setattr(ingest_module, "_repo_root", lambda: tmp_path / "no-repo-here")

    ingest_module.write_manifest_skeleton(tmp_path, "case-0000abcd")

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["case_id"] == "case-0000abcd"
    assert manifest["disclaimer"].startswith("research/preview only")
    for key in ("t1_qc", "atlas_prior_qc", "parcellation_qc"):
        block = manifest[key]
        assert block["approved_by"] is None
        assert block["sheet_sha"] is None
        assert block["auto"]["ok"] is False
