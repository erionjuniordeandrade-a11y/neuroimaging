#!/usr/bin/env python3
"""Create synthetic classic-MR DICOM series from the CC0 demo NIfTI inputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.tag import Tag
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid


DEFAULT_RAW = Path(
    str(Path.home()) + "/tractlab/cases/demo-leipzig-sub-010005/"
    "raw/sub-010005/ses-01"
)


def _is_inside_repo(path: Path) -> bool:
    repo = Path(__file__).resolve().parents[1]
    try:
        path.resolve().relative_to(repo)
    except ValueError:
        return False
    return True


def _nifti_stem(path: Path) -> str:
    if path.name.endswith(".nii.gz"):
        return path.name[:-7]
    if path.name.endswith(".nii"):
        return path.name[:-4]
    raise ValueError("A NIfTI file is required.")


def _one_nifti(folder: Path, suffix: str) -> Path:
    candidates = sorted(
        path
        for path in folder.glob("*.nii*")
        if path.name.endswith(f"_{suffix}.nii.gz") or path.name.endswith(f"_{suffix}.nii")
    )
    if len(candidates) != 1:
        raise ValueError("The expected demo NIfTI input is missing or ambiguous.")
    return candidates[0]


def _load_dwi_tables(nifti_path: Path, nvolumes: int) -> tuple[np.ndarray, np.ndarray]:
    stem = _nifti_stem(nifti_path)
    try:
        bvals = np.asarray(
            np.loadtxt(nifti_path.with_name(stem + ".bval")), dtype=float
        ).reshape(-1)
        bvecs = np.asarray(
            np.loadtxt(nifti_path.with_name(stem + ".bvec")), dtype=float
        )
    except (OSError, ValueError) as exc:
        raise ValueError("The demo DWI gradient tables could not be read.") from exc
    if bvecs.shape == (3, nvolumes):
        bvecs = bvecs.T
    if bvals.shape != (nvolumes,) or bvecs.shape != (nvolumes, 3):
        raise ValueError("The demo DWI gradient tables do not match the volume count.")
    return bvals, bvecs


def _pixel_encoding(data: np.ndarray) -> tuple[np.dtype, int, float, float]:
    if np.issubdtype(data.dtype, np.integer):
        minimum = int(np.min(data))
        maximum = int(np.max(data))
        if minimum >= 0 and maximum <= 65535:
            return np.dtype(np.uint16), 0, 1.0, 0.0
        if minimum >= -32768 and maximum <= 32767:
            return np.dtype(np.int16), 1, 1.0, 0.0

    finite = np.asarray(data, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("A NIfTI input contains no finite image values.")
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    slope = (upper - lower) / 65535.0 if upper > lower else 1.0
    return np.dtype(np.uint16), 0, slope, lower


def _to_pixel_array(
    image_slice: np.ndarray,
    *,
    dtype: np.dtype,
    pixel_representation: int,
    slope: float,
    intercept: float,
) -> np.ndarray:
    values = np.asarray(image_slice)
    if slope == 1.0 and intercept == 0.0:
        return np.asarray(values, dtype=dtype)
    encoded = np.rint((values.astype(np.float64) - intercept) / slope)
    encoded = np.clip(encoded, 0, 65535)
    return encoded.astype(dtype)


def _lps_gradient(affine: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    basis_ras = affine[:3, :3] / spacing[np.newaxis, :]
    ras = basis_ras @ np.asarray(gradient, dtype=float)
    lps = np.asarray((-ras[0], -ras[1], ras[2]), dtype=float)
    norm = float(np.linalg.norm(lps))
    return lps / norm if norm > 0 else np.zeros(3, dtype=float)


def _write_classic_series(
    output_dir: Path,
    image: nib.spatialimages.SpatialImage,
    *,
    series_number: int,
    series_description: str,
    sequence_name: str,
    scanning_sequence: str,
    acquisition_type: str,
    bvals: np.ndarray | None = None,
    bvecs: np.ndarray | None = None,
) -> int:
    if len(image.shape) == 3:
        shape = (*image.shape, 1)
    elif len(image.shape) == 4:
        shape = image.shape
    else:
        raise ValueError("Only 3D or 4D NIfTI inputs can be converted.")
    nvolumes = int(shape[3])
    if bvals is not None and bvecs is not None:
        if len(bvals) != nvolumes or len(bvecs) != nvolumes:
            raise ValueError("Diffusion tables do not match the NIfTI volume count.")

    data = np.asanyarray(image.dataobj)
    if data.ndim == 3:
        data = data[..., np.newaxis]
    dtype, pixel_representation, slope, intercept = _pixel_encoding(data)
    affine = np.asarray(image.affine, dtype=float)
    lps_affine = np.diag((-1.0, -1.0, 1.0, 1.0)) @ affine
    spacing = np.linalg.norm(lps_affine[:3, :3], axis=0)
    if np.any(spacing <= 0) or not np.all(np.isfinite(spacing)):
        raise ValueError("A NIfTI affine has invalid voxel spacing.")
    orientation = np.concatenate(
        (
            lps_affine[:3, 0] / spacing[0],
            lps_affine[:3, 1] / spacing[1],
        )
    )

    study_uid = generate_uid()
    series_uid = generate_uid()
    output_dir.mkdir(parents=True, exist_ok=True)
    instance_number = 0
    rows, columns = int(shape[1]), int(shape[0])
    for volume_index in range(nvolumes):
        for slice_index in range(int(shape[2])):
            instance_number += 1
            file_meta = FileMetaDataset()
            file_meta.MediaStorageSOPClassUID = MRImageStorage
            file_meta.MediaStorageSOPInstanceUID = generate_uid()
            file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            destination = output_dir / f"image{instance_number:06d}.dcm"
            dataset = FileDataset(
                str(destination),
                {},
                file_meta=file_meta,
                preamble=b"\0" * 128,
            )
            dataset.SOPClassUID = MRImageStorage
            dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
            dataset.StudyInstanceUID = study_uid
            dataset.SeriesInstanceUID = series_uid
            dataset.Modality = "MR"
            dataset.PatientName = "DEMO^CC0"
            dataset.PatientID = "DEMO-CC0"
            dataset.AccessionNumber = "DEMO-CC0"
            dataset.SeriesNumber = series_number
            dataset.InstanceNumber = instance_number
            dataset.AcquisitionNumber = volume_index + 1
            dataset.TemporalPositionIdentifier = volume_index + 1
            dataset.ImageType = (
                ["ORIGINAL", "PRIMARY", "DIFFUSION"]
                if bvals is not None
                else ["ORIGINAL", "PRIMARY"]
            )
            dataset.SeriesDescription = series_description
            dataset.ProtocolName = series_description
            dataset.SequenceName = sequence_name
            dataset.ScanningSequence = scanning_sequence
            dataset.MRAcquisitionType = acquisition_type
            dataset.Rows = rows
            dataset.Columns = columns
            dataset.SamplesPerPixel = 1
            dataset.PhotometricInterpretation = "MONOCHROME2"
            dataset.BitsAllocated = 16
            dataset.BitsStored = 16
            dataset.HighBit = 15
            dataset.PixelRepresentation = pixel_representation
            dataset.PixelSpacing = [float(spacing[1]), float(spacing[0])]
            dataset.SliceThickness = float(spacing[2])
            dataset.SpacingBetweenSlices = float(spacing[2])
            dataset.ImageOrientationPatient = [float(value) for value in orientation]
            position = lps_affine @ np.asarray(
                (0.0, 0.0, float(slice_index), 1.0), dtype=float
            )
            dataset.ImagePositionPatient = [float(value) for value in position[:3]]
            dataset.RescaleSlope = float(slope)
            dataset.RescaleIntercept = float(intercept)
            dataset.RepetitionTime = 3500.0
            dataset.EchoTime = 80.0 if bvals is not None else 3.0
            dataset.FlipAngle = 90 if bvals is not None else 9
            if bvals is not None and bvecs is not None:
                dataset.add_new(Tag(0x0018, 0x9087), "FD", float(bvals[volume_index]))
                gradient = _lps_gradient(affine, bvecs[volume_index])
                dataset.add_new(Tag(0x0018, 0x9089), "FD", gradient.tolist())

            nifti_slice = data[:, :, slice_index, volume_index]
            pixel_slice = _to_pixel_array(
                nifti_slice,
                dtype=dtype,
                pixel_representation=pixel_representation,
                slope=slope,
                intercept=intercept,
            )
            dataset.PixelData = np.ascontiguousarray(pixel_slice.T).tobytes()
            dataset.save_as(destination, enforce_file_format=True)
    return instance_number


def make_demo_dicom(out_dir: Path, raw_dir: Path) -> int:
    if _is_inside_repo(out_dir):
        print("Output folder must be outside the TractLab repository.", file=sys.stderr)
        return 2
    if not raw_dir.is_dir():
        print("The selected demo raw folder is unavailable.", file=sys.stderr)
        return 2
    if out_dir.exists():
        if not out_dir.is_dir() or any(out_dir.iterdir()):
            print("Output folder must be empty.", file=sys.stderr)
            return 2

    try:
        dwi_path = _one_nifti(raw_dir / "dwi", "dwi")
        t1_path = _one_nifti(raw_dir / "anat", "T1w")
        dwi_image = nib.load(str(dwi_path))
        t1_image = nib.load(str(t1_path))
        if len(dwi_image.shape) != 4:
            raise ValueError("The demo DWI input must be 4D.")
        bvals, bvecs = _load_dwi_tables(dwi_path, int(dwi_image.shape[3]))
        if len(t1_image.shape) != 3:
            raise ValueError("The demo T1 input must be 3D.")
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        dwi_count = _write_classic_series(
            out_dir / "dwi",
            dwi_image,
            series_number=1,
            series_description="DWI",
            sequence_name="demo_epi",
            scanning_sequence="EP\\SE",
            acquisition_type="2D",
            bvals=bvals,
            bvecs=bvecs,
        )
        t1_count = _write_classic_series(
            out_dir / "t1",
            t1_image,
            series_number=2,
            series_description="MP2RAGE T1w",
            sequence_name="MP2RAGE_UNI",
            scanning_sequence="GR",
            acquisition_type="3D",
        )
    except Exception:
        print("Synthetic DICOM generation failed.", file=sys.stderr)
        return 1

    print(
        f"Created {dwi_count + t1_count} synthetic DICOM instances in two series. "
        "AP/PA field maps were omitted because this generic encoding does not "
        "carry a verified phase-encoding polarity for dcm2niix."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="make_demo_dicom.py")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    args = parser.parse_args(argv)
    return make_demo_dicom(args.out_dir.expanduser(), args.raw.expanduser())


if __name__ == "__main__":
    raise SystemExit(main())
