"""Header-only DICOM discovery and selected-series conversion."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess

import pydicom

from .deid import mr_weighting, safe_description, safe_sequence, study_year


@dataclass
class Series:
    uid: str
    number: int
    modality: str
    description: str
    scanning_sequence: str
    sequence_name: str
    contrast: bool
    rows: int
    columns: int
    pixel_spacing: tuple[float, float]
    slice_thickness: float | None
    transfer_syntax: str
    year: int | None
    files: list[Path]
    weighting: str | None = None


def scan_series(root: str | Path) -> tuple[list[Series], int]:
    root = Path(root)
    if not root.is_dir():
        raise ValueError("DICOM input must be a directory")
    groups: dict[str, list[tuple[Path, pydicom.Dataset]]] = defaultdict(list)
    skipped = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            if not getattr(ds, "SeriesInstanceUID", None) or not getattr(ds, "SOPClassUID", None):
                skipped += 1
                continue
            groups[str(ds.SeriesInstanceUID)].append((path, ds))
        except (pydicom.errors.InvalidDicomError, OSError, ValueError):
            skipped += 1
    result = []
    for uid, entries in groups.items():
        ds = entries[0][1]
        spacing = getattr(ds, "PixelSpacing", [0, 0])
        syntax = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
        result.append(Series(
            uid=uid,
            number=int(getattr(ds, "SeriesNumber", 0)),
            modality=str(getattr(ds, "Modality", "OT")),
            description=safe_description(getattr(ds, "SeriesDescription", "")),
            scanning_sequence=safe_sequence(getattr(ds, "ScanningSequence", "")),
            sequence_name=safe_sequence(getattr(ds, "SequenceName", "")),
            contrast=bool(getattr(ds, "ContrastBolusAgent", "")),
            rows=int(getattr(ds, "Rows", 0)),
            columns=int(getattr(ds, "Columns", 0)),
            pixel_spacing=(float(spacing[0]), float(spacing[1])) if len(spacing) == 2 else (0, 0),
            slice_thickness=float(ds.SliceThickness) if getattr(ds, "SliceThickness", None) else None,
            transfer_syntax=str(syntax) if syntax else "unknown",
            year=study_year(getattr(ds, "StudyDate", "")),
            files=[path for path, _ in entries],
            weighting=mr_weighting(getattr(ds, "SeriesDescription", "")),
        ))
    return sorted(result, key=lambda s: (s.number, s.modality, s.uid)), skipped


def dcm2niix_executable() -> str:
    env = os.environ.get("DCM2NIIX")
    if env:
        if not Path(env).is_file() and not shutil.which(env):
            raise FileNotFoundError("DCM2NIIX executable not found")
        return env
    return shutil.which("dcm2niix") or os.path.expanduser("~/fsl/bin/dcm2niix")


def convert_series(series: Series, work: Path) -> Path:
    """Give dcm2niix only selected files, never the full input directory."""
    source = work / f"input_{series.number}"
    output = work / f"output_{series.number}"
    source.mkdir()
    output.mkdir()
    for index, path in enumerate(series.files):
        shutil.copyfile(path, source / f"{index:05d}.dcm")
    cmd = [dcm2niix_executable(), "-z", "n", "-b", "y", "-ba", "y", "-f", "series_%s", "-o", str(output), str(source)]
    run = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if run.returncode:
        raise RuntimeError(f"dcm2niix failed for series {series.number} (exit {run.returncode})")
    nifti = sorted(output.glob("*.nii"))
    tilt_corrected = [path for path in nifti if path.stem.endswith("_Tilt_1")]
    if len(nifti) == 2 and len(tilt_corrected) == 1:
        # Gantry-tilted CT: dcm2niix writes the raw sheared stack plus a resampled
        # orthogonal "_Tilt_1" volume. Only the latter has a geometry SimpleITK can
        # represent (a direction matrix cannot hold the shear).
        nifti = tilt_corrected
    if len(nifti) != 1:
        raise RuntimeError(f"dcm2niix produced {len(nifti)} NIfTI files for series {series.number}")
    return nifti[0]
