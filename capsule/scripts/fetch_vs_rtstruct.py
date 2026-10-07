"""Fetch the RTSTRUCT series that belong to the local Vestibular-Schwannoma-SEG demo MR subject.

The subject is identified in code from the local T1 header (PatientID + StudyInstanceUID) and
matched against the anonymous NBIA v1 API; no identifier is printed or written anywhere except
inside the downloaded DICOM files themselves. Each RTSTRUCT is saved as rtstruct_<T1|T2>.dcm
according to the image series it references. Existing files are never overwritten.

Run:
  uv run python scripts/fetch_vs_rtstruct.py \
      --mri data/public/Vestibular-Schwannoma-SEG/brain-mri \
      --out data/public/Vestibular-Schwannoma-SEG/rtstruct
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import urllib.parse
import urllib.request
import zipfile

import pydicom

API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
COLLECTION = "Vestibular-Schwannoma-SEG"


def _header(folder: Path) -> pydicom.Dataset:
    files = sorted(folder.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"no DICOM in {folder}")
    return pydicom.dcmread(files[0], stop_before_pixels=True)


def _referenced_series(ds: pydicom.Dataset) -> set[str]:
    refs = set()
    for frame in ds.get("ReferencedFrameOfReferenceSequence", []):
        for study in frame.get("RTReferencedStudySequence", []):
            for series in study.get("RTReferencedSeriesSequence", []):
                refs.add(str(series.SeriesInstanceUID))
    return refs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mri", default=f"data/public/{COLLECTION}/brain-mri")
    parser.add_argument("--out", default=f"data/public/{COLLECTION}/rtstruct")
    args = parser.parse_args(argv)
    mri = Path(args.mri)
    local = {"T1": _header(mri / "T1_fl3d"), "T2": _header(mri / "T2_tse3d")}
    subject, study = str(local["T1"].PatientID), str(local["T1"].StudyInstanceUID)
    query = urllib.parse.urlencode({"Collection": COLLECTION, "PatientID": subject})
    rows = json.load(urllib.request.urlopen(f"{API}/getSeries?{query}", timeout=60))
    rtstructs = [r for r in rows if r.get("Modality") == "RTSTRUCT" and r.get("StudyInstanceUID") == study]
    print(f"subject matched in code: {len(rows)} series, {len(rtstructs)} RTSTRUCT in the same study")
    if not rtstructs:
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for row in rtstructs:
        query = urllib.parse.urlencode({"SeriesInstanceUID": row["SeriesInstanceUID"]})
        data = urllib.request.urlopen(f"{API}/getImage?{query}", timeout=120).read()
        archive = zipfile.ZipFile(io.BytesIO(data))  # also holds a licence text file
        for name in [n for n in archive.namelist() if n.lower().endswith(".dcm")]:
            blob = archive.read(name)
            ds = pydicom.dcmread(io.BytesIO(blob))
            refs = _referenced_series(ds)
            on = next((k for k, h in local.items() if str(h.SeriesInstanceUID) in refs), None)
            if on is None:
                print("skipped an RTSTRUCT that references neither local series")
                continue
            target = out / f"rtstruct_{on}.dcm"
            if target.exists():
                print(f"{target} exists; not overwritten")
                continue
            target.write_bytes(blob)
            rois = [str(s.ROIName) for s in ds.StructureSetROISequence]
            print(f"{target}: references local {on}, ROIs {rois}, sha256 {hashlib.sha256(blob).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
