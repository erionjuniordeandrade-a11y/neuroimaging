"""Local patient archive: DICOM import, a patient/study/series index, and studies as scenes.

The archive is one folder that only the owner account can open (mode 0700). It
holds copies of imported DICOM files, a SQLite index and a NIfTI cache. Patient
names and IDs are kept as they are: the archive is not de-identified, so it must
stay on the machine that imported it. Disk encryption comes from the operating
system (FileVault on macOS).

Nothing here logs a path, a name or an identifier. Import results are counts.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path

import pydicom
from pydicom.errors import InvalidDicomError

from neuro_workbench.scene import SCENE_SCHEMA, _window_for

DICOMDIR_CLASS = "1.2.840.10008.1.3.10"
# Series with fewer files are scouts or single reports; they are indexed but not drawn.
MIN_SLICES_FOR_VOLUME = 3
SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
  key TEXT PRIMARY KEY, patient_id TEXT, name TEXT, birth_date TEXT, sex TEXT);
CREATE TABLE IF NOT EXISTS studies (
  uid TEXT PRIMARY KEY, patient_key TEXT NOT NULL REFERENCES patients(key),
  date TEXT, time TEXT, description TEXT, accession TEXT, imported_at REAL);
CREATE TABLE IF NOT EXISTS series (
  uid TEXT PRIMARY KEY, study_uid TEXT NOT NULL REFERENCES studies(uid),
  modality TEXT, description TEXT, number INTEGER, folder TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS instances (
  sop_uid TEXT PRIMARY KEY, series_uid TEXT NOT NULL REFERENCES series(uid), file TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS instances_series ON instances(series_uid);
"""


def default_archive_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Eidos" / "archive"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "eidos" / "archive"


def _h(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:n]


def study_case_id(study_uid: str) -> str:
    return f"study-{_h(study_uid, 12)}"


def _text(ds, keyword: str) -> str:
    value = ds.get(keyword)
    return "" if value is None else str(value).strip()


def _volume_kind(modality: str, description: str) -> str:
    text = f"{modality} {description}".upper()
    if "CTA" in text or "ANGIO" in text:
        return "CTA"
    return modality.upper() or "OT"


class Archive:
    def __init__(self, root: Path):
        self.root = root
        self._lock = threading.Lock()
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)
        for sub in ("dicom", "cache"):
            (self.root / sub).mkdir(mode=0o700, exist_ok=True)
        with self._db() as db:
            db.executescript(SCHEMA)
        os.chmod(self.root / "index.sqlite", 0o600)

    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.root / "index.sqlite", timeout=30)
        db.row_factory = sqlite3.Row
        return db

    # Import ---------------------------------------------------------------
    def import_path(self, source: Path, progress=None) -> dict:
        """Copy every DICOM file under source (a folder, a CD or a .zip) into the archive. Returns counts only."""
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError("the import folder does not exist")
        if source.is_file() and zipfile.is_zipfile(source):
            # Unpack inside the private archive folder, never in a shared temp folder.
            staging = Path(tempfile.mkdtemp(prefix=".import-", dir=self.root))
            try:
                with zipfile.ZipFile(source) as zf:
                    zf.extractall(staging)  # extractall drops absolute paths and ".." parts
                return self.import_path(staging, progress)
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        if source.is_file() and source.suffix.lower() == ".zip":
            # A zip without a central directory is usually a download that stopped early.
            raise ValueError("this .zip is incomplete or damaged (no file index at its end); download it again")
        files = [source] if source.is_file() else _walk(source)
        report = {"files_seen": len(files), "added": 0, "already_in_archive": 0, "not_dicom": 0,
                  "patients": set(), "studies": set(), "series": set()}
        with self._lock, self._db() as db:
            for i, path in enumerate(files):
                if progress and i % 50 == 0:
                    progress(i, len(files))
                try:
                    # Large values (pixel data) are deferred, so this reads only the header.
                    ds = pydicom.dcmread(path, defer_size=1024)
                except (InvalidDicomError, OSError, ValueError, AttributeError):
                    report["not_dicom"] += 1
                    continue
                sop, series_uid, study_uid = (_text(ds, k) for k in ("SOPInstanceUID", "SeriesInstanceUID", "StudyInstanceUID"))
                is_dicomdir = _text(ds.get("file_meta", {}), "MediaStorageSOPClassUID") == DICOMDIR_CLASS
                if is_dicomdir or not (sop and series_uid and study_uid) or "PixelData" not in ds:
                    report["not_dicom"] += 1
                    continue
                pid, name, birth = _text(ds, "PatientID"), _text(ds, "PatientName"), _text(ds, "PatientBirthDate")
                pkey = _h(f"{pid}|{name}|{birth}")
                report["patients"].add(pkey)
                report["studies"].add(study_uid)
                report["series"].add(series_uid)
                if db.execute("SELECT 1 FROM instances WHERE sop_uid=?", (sop,)).fetchone():
                    report["already_in_archive"] += 1
                    continue
                folder = Path("dicom") / _h(study_uid) / _h(series_uid)
                (self.root / folder).mkdir(mode=0o700, parents=True, exist_ok=True)
                rel = folder / f"{_h(sop)}.dcm"
                tmp = self.root / folder / f".{_h(sop)}.{os.getpid()}.tmp"
                shutil.copyfile(path, tmp)
                os.chmod(tmp, 0o600)
                os.replace(tmp, self.root / rel)
                db.execute("INSERT OR IGNORE INTO patients VALUES (?,?,?,?,?)",
                           (pkey, pid, name, birth, _text(ds, "PatientSex")))
                db.execute("INSERT OR IGNORE INTO studies VALUES (?,?,?,?,?,?,?)",
                           (study_uid, pkey, _text(ds, "StudyDate"), _text(ds, "StudyTime"),
                            _text(ds, "StudyDescription"), _text(ds, "AccessionNumber"), time.time()))
                number = ds.get("SeriesNumber")
                db.execute("INSERT OR IGNORE INTO series VALUES (?,?,?,?,?,?)",
                           (series_uid, study_uid, _text(ds, "Modality"), _text(ds, "SeriesDescription"),
                            int(number) if number not in (None, "") else None, str(folder)))
                db.execute("INSERT INTO instances VALUES (?,?,?)", (sop, series_uid, str(rel)))
                report["added"] += 1
        if progress:
            progress(len(files), len(files))
        return {k: len(v) if isinstance(v, set) else v for k, v in report.items()}

    # Queries --------------------------------------------------------------
    def patients(self, query: str = "") -> list[dict]:
        """Patients with their studies and series, newest study first, filtered by a free-text query."""
        with self._db() as db:
            series = db.execute("""SELECT s.uid, s.study_uid, s.modality, s.description, s.number,
                                          COUNT(i.sop_uid) AS n FROM series s
                                   LEFT JOIN instances i ON i.series_uid = s.uid GROUP BY s.uid""").fetchall()
            studies = db.execute("SELECT * FROM studies ORDER BY date DESC, time DESC").fetchall()
            patients = db.execute("SELECT * FROM patients").fetchall()
        by_study: dict[str, list[dict]] = {}
        for s in series:
            by_study.setdefault(s["study_uid"], []).append(
                {"modality": s["modality"], "description": s["description"], "number": s["number"], "images": s["n"]})
        by_patient: dict[str, list[dict]] = {}
        for st in studies:
            items = sorted(by_study.get(st["uid"], []), key=lambda x: (x["number"] is None, x["number"] or 0))
            by_patient.setdefault(st["patient_key"], []).append({
                "id": study_case_id(st["uid"]), "date": st["date"], "description": st["description"],
                "accession": st["accession"], "modalities": sorted({x["modality"] for x in items if x["modality"]}),
                "series": items})
        out = []
        terms = [t for t in query.lower().split() if t]
        for p in patients:
            studies_p = by_patient.get(p["key"], [])
            hay = " ".join([p["name"], p["patient_id"], p["birth_date"]] + [
                f"{s['date']} {s['description']} {s['accession']} {' '.join(s['modalities'])} "
                + " ".join(x["description"] for x in s["series"]) for s in studies_p]).lower()
            if all(t in hay for t in terms):
                out.append({"name": p["name"].replace("^", " ").strip(), "patient_id": p["patient_id"],
                            "birth_date": p["birth_date"], "sex": p["sex"], "studies": studies_p})
        out.sort(key=lambda p: max((s["date"] for s in p["studies"]), default=""), reverse=True)
        return out

    def studies(self) -> list[dict]:
        """One entry per study, for the case list."""
        return [{"patient": p["name"], "patient_id": p["patient_id"], **s} for p in self.patients() for s in p["studies"]]

    def study_uid(self, case_id: str) -> str | None:
        with self._db() as db:
            for (uid,) in db.execute("SELECT uid FROM studies"):
                if study_case_id(uid) == case_id:
                    return uid
        return None

    # Scenes ---------------------------------------------------------------
    def study_scene(self, study_uid: str) -> tuple[dict, dict[str, Path]]:
        """Convert each image series of a study to NIfTI once and describe them as layers."""
        import SimpleITK as sitk

        with self._db() as db:
            study = db.execute("SELECT * FROM studies WHERE uid=?", (study_uid,)).fetchone()
            if study is None:
                raise KeyError("study not in archive")
            series = db.execute("""SELECT s.*, COUNT(i.sop_uid) AS n FROM series s JOIN instances i ON i.series_uid=s.uid
                                   WHERE s.study_uid=? GROUP BY s.uid ORDER BY s.number""", (study_uid,)).fetchall()
        scene = {"schema": SCENE_SCHEMA, "frame": "scanner RAS mm", "source": "archive",
                 "title": study["description"] or "Study", "date": study["date"] or "", "volumes": [], "labels": [], "tracts": []}
        files: dict[str, Path] = {}
        skipped = 0
        for s in series:
            if s["n"] < MIN_SLICES_FOR_VOLUME:
                skipped += 1
                continue
            key = f"vol-{_h(s['uid'], 10)}"
            target = self.root / "cache" / f"{_h(s['uid'])}.nii.gz"
            if not target.is_file():
                folder = str(self.root / s["folder"])
                names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(folder, s["uid"])
                if len(names) < MIN_SLICES_FOR_VOLUME:
                    skipped += 1
                    continue
                reader = sitk.ImageSeriesReader()
                reader.SetFileNames(names)
                try:
                    image = reader.Execute()
                except RuntimeError:
                    skipped += 1
                    continue
                tmp = target.with_name(f".{target.stem}.{os.getpid()}.tmp.nii.gz")
                sitk.WriteImage(image, str(tmp), useCompression=True)
                os.chmod(tmp, 0o600)
                os.replace(tmp, target)
            files[key] = target
            label = s["description"] or s["modality"] or "Series"
            kind = _volume_kind(s["modality"] or "", s["description"] or "")
            scene["volumes"].append({"id": key, "label": label, "kind": kind,
                                     **_window_for("CT" if kind in ("CT", "CTA") else kind, label)})
        if skipped:
            scene["skipped_series"] = skipped
        if not scene["volumes"]:
            raise ValueError("this study has no series with enough slices to draw")
        return scene, files



def _walk(source: Path) -> list[Path]:
    """Every non-hidden file under source, hidden folders skipped."""
    out = []
    for folder, dirs, names in os.walk(source):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        out.extend(Path(folder) / n for n in sorted(names) if not n.startswith("."))
    return out
