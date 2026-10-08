"""Derived products of archive studies on synthetic DICOM only: DTI detection, capsule and tract builds."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from pydicom.uid import generate_uid

from neuro_workbench import derived as derived_mod
from neuro_workbench.archive import Archive, is_diffusion, study_case_id
from neuro_workbench.derived import Derived, capsule_series

from test_archive import write_series


@pytest.mark.parametrize("modality,description,images,expected", [
    ("MR", "ep2d_diff_mddw_30", 31, True),
    ("MR", "DTI 64 dir", 65, True),
    ("MR", "Ax DWI b1000", 8, True),
    ("MR", "DTI_ADC", 30, False),           # a scanner map, not the acquisition
    ("MR", "DTI_ColFA", 30, False),
    ("MR", "dwi_TRACEW", 30, False),
    ("MR", "DWI b1000", 3, False),          # too few images for a tensor
    ("CT", "DIFF", 40, False),
    ("MR", "T1 MPRAGE", 176, False),
])
def test_is_diffusion(modality, description, images, expected):
    assert is_diffusion(modality, description, images) is expected


@pytest.fixture()
def dti_archive(tmp_path: Path):
    src = tmp_path / "incoming"
    study = generate_uid()
    common = dict(patient="SYNTHETIC^GAMMA", pid="SYN-003", study_uid=study)
    write_series(src / "t1", modality="MR", description="T1 MPRAGE", slices=6, number=3, **common)
    write_series(src / "flair", modality="MR", description="FLAIR AX", slices=5, number=4, **common)
    write_series(src / "dti", modality="MR", description="DTI 30dir", slices=8, number=7, **common)
    write_series(src / "adc", modality="MR", description="DTI_ADC", slices=8, number=8, **common)
    write_series(src / "loc", modality="MR", description="Localizer", slices=3, number=1, **common)
    write_series(src / "cr", modality="CR", description="CHEST", slices=1, number=9, **common)
    archive = Archive(tmp_path / "archive")
    archive.import_path(src)
    return archive, study_case_id(study), study


def test_archive_flags_dti_series_and_counts_them(dti_archive):
    archive, case_id, uid = dti_archive
    study = archive.patients()[0]["studies"][0]
    assert study["dti"] == 1
    assert [s["number"] for s in study["series"] if s["dti"]] == [7]
    assert [s["number"] for s in archive.study_series(uid)] == [1, 3, 4, 7, 8, 9]


def test_capsule_uses_volumes_reference_first_and_skips_diffusion_and_localizers(dti_archive):
    archive, _, uid = dti_archive
    assert capsule_series(archive.study_series(uid)) == [3, 4]


def test_status_offers_capsule_and_reports_missing_pipeline_tools(dti_archive, monkeypatch):
    archive, case_id, _ = dti_archive
    monkeypatch.setattr(derived_mod, "missing_tools", lambda: ["MRtrix3"])
    st = Derived(Path("/nonexistent-repo"), archive).status(case_id)
    assert st["capsule"] == {"series": [3, 4], "state": "absent"}
    assert st["tracts"] == {"dti": [{"number": 7, "images": 8}], "state": "unavailable", "missing": ["MRtrix3"]}
    monkeypatch.setattr(derived_mod, "missing_tools", lambda: [])
    assert Derived(Path("/nonexistent-repo"), archive).status(case_id)["tracts"]["state"] == "absent"


def test_study_without_dti_says_so(tmp_path):
    src = tmp_path / "in"
    write_series(src, patient="SYNTHETIC^DELTA", pid="SYN-004", study_uid=(u := generate_uid()),
                 modality="CT", description="CTA", slices=4, number=2)
    archive = Archive(tmp_path / "archive")
    archive.import_path(src)
    st = Derived(tmp_path, archive).status(study_case_id(u))
    assert st["tracts"]["state"] == "nodti" and st["capsule"]["state"] == "absent"


def _wait(fn, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        out = fn()
        if out:
            return out
        time.sleep(0.05)
    raise AssertionError("timed out")


def test_capsule_build_runs_the_cli_and_is_deleted_with_the_study(dti_archive, monkeypatch):
    archive, case_id, uid = dti_archive
    calls = []

    def fake_run(self, kind, cmd, cwd, log, env=None):
        calls.append(cmd)
        Path(cmd[cmd.index("-o") + 1]).write_text("<html>capsule</html>")
        log.write_text("ok\n")
        return 0

    monkeypatch.setattr(Derived, "_run", fake_run)
    d = Derived(Path("/repo"), archive)
    assert d.build(case_id, "capsule")["state"] in ("running", "ready")
    _wait(lambda: d.status(case_id)["capsule"]["state"] == "ready")
    cmd = calls[0]
    assert cmd[1:4] == ["-m", "capsule.cli", "build"] and cmd[cmd.index("--series") + 1] == "3,4"
    assert cmd[4] == str(archive.study_dicom_dir(uid))
    assert cmd[cmd.index("--anatomy") + 1] == "auto" and cmd[cmd.index("--brain-mask") + 1] == "synthstrip"
    assert d.capsule_file(case_id).read_text() == "<html>capsule</html>"
    archive.delete_study(uid)
    assert not archive.derived_dir(case_id).exists()


def test_failed_capsule_build_reports_one_line_and_can_retry(dti_archive, monkeypatch):
    archive, case_id, _ = dti_archive
    monkeypatch.setattr(Derived, "_run", lambda self, kind, cmd, cwd, log, env=None: (log.write_text("x\nboom\n"), 2)[1])
    d = Derived(Path("/repo"), archive)
    d.build(case_id, "capsule")
    st = _wait(lambda: (s := d.status(case_id)["capsule"])["state"] == "error" and s)
    assert st["error"] == "capsule build failed (exit 2): boom"


def test_tract_build_ingests_runs_the_pipeline_and_becomes_ready(dti_archive, monkeypatch):
    archive, case_id, uid = dti_archive
    monkeypatch.setattr(derived_mod, "missing_tools", lambda: [])
    steps = []

    def fake_run(self, kind, cmd, cwd, log, env=None):
        steps.append(Path(cmd[1]).name if cmd[0] == "bash" else cmd[2])
        root = self._dir(case_id, "tracts")
        if cmd[0] != "bash":  # tractlab.ingest
            case = root / "case-0123abcd"
            case.mkdir()
            (case / "case.json").write_text("{}")
            log.write_text(json.dumps({"ok": True, "case_id": "case-0123abcd", "refusals": []}) + "\n")
            return 0
        case = root / "case-0123abcd"
        (case / "manifest.json").write_text("{}")
        (case / "status.json").write_text(json.dumps({"schema": "tractlab.status/1", "state": "done", "stage": "finish",
                                                      "stages": [{"name": n, "state": "done"} for n in "abcd"]}))
        return 0

    monkeypatch.setattr(Derived, "_run", fake_run)
    d = Derived(Path("/repo"), archive)
    d.build(case_id, "tracts")
    _wait(lambda: d.status(case_id)["tracts"]["state"] == "ready")
    assert steps == ["tractlab.ingest", "run_case.sh"]
    assert d.manifest(case_id).name == "manifest.json"


def test_tract_pipeline_failure_shows_the_stage_and_offers_resume(dti_archive, monkeypatch):
    archive, case_id, _ = dti_archive
    monkeypatch.setattr(derived_mod, "missing_tools", lambda: [])
    case = archive.derived_dir(case_id) / "tracts" / "case-0123abcd"
    case.mkdir(parents=True)
    (case / "case.json").write_text("{}")
    (case / "status.json").write_text(json.dumps({"state": "failed", "stage": "t1reg", "stages": [
        {"name": "ss3t", "state": "done"}, {"name": "t1reg", "state": "failed"},
        {"name": "banks", "state": "pending"}, {"name": "scalar_maps", "state": "pending"}]}))
    st = Derived(Path("/repo"), archive).status(case_id)["tracts"]
    assert st["state"] == "error" and st["stage"] == "t1reg" and (st["done"], st["total"]) == (1, 4)


def test_build_refuses_studies_without_input(tmp_path):
    src = tmp_path / "in"
    write_series(src, patient="SYNTHETIC^EPS", pid="SYN-005", study_uid=(u := generate_uid()),
                 modality="CR", description="CHEST", slices=1, number=1)
    archive = Archive(tmp_path / "archive")
    archive.import_path(src)
    d = Derived(tmp_path, archive)
    with pytest.raises(RuntimeError, match="no CT or MR"):
        d.build(study_case_id(u), "capsule")
    with pytest.raises(RuntimeError, match="no diffusion"):
        d.build(study_case_id(u), "tracts")
