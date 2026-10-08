"""Patient archive on synthetic DICOM only: import, de-duplication, search, scenes, server routes."""

from __future__ import annotations

import http.client
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, MRImageStorage, generate_uid

from neuro_workbench.archive import Archive, study_case_id


def write_series(folder: Path, *, patient: str, pid: str, study_uid: str, modality: str, description: str,
                 slices: int, number: int, study_desc: str = "SYNTHETIC HEAD") -> str:
    folder.mkdir(parents=True, exist_ok=True)
    series_uid = generate_uid()
    for z in range(slices):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage if modality == "CT" else MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(str(folder / f"{series_uid[-6:]}-{z}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
        ds.PatientName, ds.PatientID, ds.PatientBirthDate, ds.PatientSex = patient, pid, "19700101", "O"
        ds.StudyInstanceUID, ds.SeriesInstanceUID = study_uid, series_uid
        ds.StudyDate, ds.StudyTime, ds.StudyDescription, ds.AccessionNumber = "20260101", "120000", study_desc, "SYN1"
        ds.Modality, ds.SeriesDescription, ds.SeriesNumber, ds.InstanceNumber = modality, description, number, z + 1
        ds.ImagePositionPatient, ds.ImageOrientationPatient = [0, 0, float(z) * 2], [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing, ds.SliceThickness = [1, 1], 2
        ds.Rows = ds.Columns = 16
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.PixelData = (np.arange(256, dtype=np.int16).reshape(16, 16) + z).tobytes()
        ds.save_as(ds.filename, enforce_file_format=True)
    return series_uid


@pytest.fixture()
def dicom_folder(tmp_path: Path) -> Path:
    src = tmp_path / "incoming"
    study = generate_uid()
    write_series(src / "a", patient="SYNTHETIC^ALPHA", pid="SYN-001", study_uid=study, modality="CT",
                 description="CTA HEAD", slices=5, number=2)
    write_series(src / "b", patient="SYNTHETIC^ALPHA", pid="SYN-001", study_uid=study, modality="MR",
                 description="T1 AX", slices=4, number=3)
    write_series(src / "c", patient="SYNTHETIC^ALPHA", pid="SYN-001", study_uid=study, modality="CT",
                 description="SCOUT", slices=1, number=1)
    write_series(src / "d", patient="SYNTHETIC^BETA", pid="SYN-002", study_uid=generate_uid(), modality="MR",
                 description="FLAIR", slices=3, number=1, study_desc="SYNTHETIC SPINE")
    (src / "notes.txt").write_text("not a DICOM file")
    (src / ".DS_Store").write_bytes(b"\0")
    return src


def test_import_copies_indexes_and_skips_duplicates(tmp_path, dicom_folder):
    archive = Archive(tmp_path / "archive")
    first = archive.import_path(dicom_folder)
    assert first == {"files_seen": 14, "added": 13, "already_in_archive": 0, "not_dicom": 1,
                     "patients": 2, "studies": 2, "series": 4}
    again = archive.import_path(dicom_folder)
    assert again["added"] == 0 and again["already_in_archive"] == 13
    assert stat.S_IMODE(os.stat(tmp_path / "archive").st_mode) == 0o700
    copied = list((tmp_path / "archive" / "dicom").rglob("*.dcm"))
    assert len(copied) == 13 and all(stat.S_IMODE(f.stat().st_mode) == 0o600 for f in copied)
    assert not any("SYN" in str(f) or "ALPHA" in str(f) for f in copied)  # file names carry no identifiers


def test_import_from_a_zip_leaves_no_staging_files(tmp_path, dicom_folder):
    import shutil
    zipped = shutil.make_archive(str(tmp_path / "cd"), "zip", dicom_folder)
    archive = Archive(tmp_path / "archive")
    report = archive.import_path(Path(zipped))
    assert report["added"] == 13 and report["studies"] == 2
    assert not list((tmp_path / "archive").glob(".import-*"))


def test_a_truncated_zip_is_refused_with_a_clear_message(tmp_path, dicom_folder):
    import shutil
    zipped = Path(shutil.make_archive(str(tmp_path / "cd"), "zip", dicom_folder))
    cut = tmp_path / "cut.zip"
    cut.write_bytes(zipped.read_bytes()[: zipped.stat().st_size // 2])
    with pytest.raises(ValueError, match="incomplete"):
        Archive(tmp_path / "archive").import_path(cut)


def test_search_by_name_id_and_modality(tmp_path, dicom_folder):
    archive = Archive(tmp_path / "archive")
    archive.import_path(dicom_folder)
    assert sorted(p["name"] for p in archive.patients()) == ["SYNTHETIC ALPHA", "SYNTHETIC BETA"]
    alpha = archive.patients("alpha")
    assert len(alpha) == 1 and alpha[0]["patient_id"] == "SYN-001"
    study = alpha[0]["studies"][0]
    assert study["modalities"] == ["CT", "MR"] and [s["images"] for s in study["series"]] == [1, 5, 4]
    assert [p["patient_id"] for p in archive.patients("flair")] == ["SYN-002"]
    assert archive.patients("syn-002 ct") == []


def test_study_scene_draws_ct_and_mr_and_skips_the_scout(tmp_path, dicom_folder):
    archive = Archive(tmp_path / "archive")
    archive.import_path(dicom_folder)
    study = archive.patients("alpha")[0]["studies"][0]
    scene, files = archive.study_scene(archive.study_uid(study["id"]))
    assert [(v["label"], v["kind"]) for v in scene["volumes"]] == [("CTA HEAD", "CTA"), ("T1 AX", "MR")]
    assert scene["volumes"][0]["cal_min"] == 100.0 and scene["skipped_series"] == 1
    import nibabel as nib
    assert nib.load(files[scene["volumes"][0]["id"]]).shape == (16, 16, 5)


def _req(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    conn.request(method, path, body=json.dumps(body) if body is not None else None,
                 headers={"Host": f"127.0.0.1:{port}", **(headers or {})})
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, json.loads(data)


def test_server_imports_lists_and_opens_a_study(tmp_path, dicom_folder):
    proc = subprocess.Popen([sys.executable, "-m", "neuro_workbench.server", "--port", "0",
                             "--cache-dir", str(tmp_path / "cache"), "--archive-dir", str(tmp_path / "archive"),
                             "--phantom", str(tmp_path / "none.html")], stdout=subprocess.PIPE, text=True)
    try:
        port = int(json.loads(proc.stdout.readline())["url"].rsplit(":", 1)[1].strip("/"))
        json_h = {"Content-Type": "application/json"}
        assert _req(port, "POST", "/api/archive/import", {"path": str(dicom_folder)},
                    {**json_h, "Origin": "http://evil.example"})[0] == 403
        assert _req(port, "POST", "/api/archive/import", {"path": str(dicom_folder)})[0] == 403  # not JSON
        status, started = _req(port, "POST", "/api/archive/import", {"path": str(dicom_folder)}, json_h)
        assert status == 200 and started["state"] in ("running", "done")
        for _ in range(100):
            state = _req(port, "GET", "/api/archive/import")[1]
            if state["state"] != "running":
                break
            time.sleep(0.1)
        assert state["state"] == "done" and state["report"]["added"] == 13
        patients = _req(port, "GET", "/api/archive?q=beta")[1]["patients"]
        assert [p["patient_id"] for p in patients] == ["SYN-002"]
        cases = _req(port, "GET", "/api/cases")[1]["cases"]
        study_ids = [c["id"] for c in cases if c.get("archive")]
        assert len(study_ids) == 2 and patients[0]["studies"][0]["id"] in study_ids
        case_id = study_case_id(pydicom.dcmread(next((dicom_folder / "a").iterdir())).StudyInstanceUID)
        for _ in range(100):
            scene_state = _req(port, "GET", f"/api/case/{case_id}/case")[1]
            if scene_state["state"] != "starting":
                break
            time.sleep(0.2)
        assert scene_state["state"] == "ready", scene_state
        scene = _req(port, "GET", f"/case/{case_id}/scene.json")[1]
        assert [v["kind"] for v in scene["volumes"]] == ["CTA", "MR"]
    finally:
        proc.terminate()
        proc.wait(timeout=10)
