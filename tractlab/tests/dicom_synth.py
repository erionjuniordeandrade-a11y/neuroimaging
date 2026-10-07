"""Small, PHI-bearing synthetic MR series for ingest tests."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.tag import Tag
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid


SYNTHETIC_PATIENT_NAME = "SYNTHETIC^PHI"
SYNTHETIC_PATIENT_ID = "SYNTHETIC-ID-001"
_DIRECTIONS = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (2**-0.5, 2**-0.5, 0.0),
    (2**-0.5, 0.0, 2**-0.5),
    (0.0, 2**-0.5, 2**-0.5),
)


def expected_t1_affine() -> np.ndarray:
    """RAS affine implied by the synthetic DICOM's LPS origin and orientation."""
    return np.asarray(
        (
            (1.0, 0.0, 0.0, -3.5),
            (0.0, 1.0, 0.0, -3.5),
            (0.0, 0.0, 1.0, -2.5),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=float,
    )


def write_test_series(root: Path) -> None:
    """Write a 7-volume DWI and a 3D 1 mm MPRAGE series."""
    root.mkdir(parents=True, exist_ok=True)
    dwi_data = np.stack(
        [np.full((8, 8, 6), idx + 1, dtype=np.uint16) for idx in range(7)],
        axis=3,
    )
    write_classic_mr_series(
        root / "dwi",
        data=dwi_data,
        series_number=1,
        series_description="DWI",
        sequence_name="ep_b1000",
        scanning_sequence="EP\\SE",
        acquisition_type="2D",
        bvalues=(0.0, *(1000.0 for _ in _DIRECTIONS)),
        gradients=((0.0, 0.0, 0.0), *(_DIRECTIONS[idx % len(_DIRECTIONS)] for idx in range(6))),
        voxel_size=(1.0, 1.0, 1.0),
    )

    t1_data = np.arange(8 * 8 * 6, dtype=np.uint16).reshape((8, 8, 6)) + 100
    write_classic_mr_series(
        root / "t1",
        data=t1_data,
        series_number=2,
        series_description="MPRAGE",
        sequence_name="tfl3d1_16",
        scanning_sequence="GR",
        acquisition_type="3D",
        voxel_size=(1.0, 1.0, 1.0),
    )


def write_classic_mr_series(
    root: Path,
    *,
    data: np.ndarray,
    series_number: int,
    series_description: str,
    sequence_name: str,
    scanning_sequence: str,
    acquisition_type: str,
    bvalues: Sequence[float] | None = None,
    gradients: Sequence[Sequence[float]] | None = None,
    voxel_size: Sequence[float] = (1.0, 1.0, 1.0),
    image_type: Iterable[str] = ("ORIGINAL", "PRIMARY"),
) -> None:
    """Write an axial classic-MR series with optional per-volume diffusion tags."""
    root.mkdir(parents=True, exist_ok=True)
    if data.ndim == 3:
        volumes = data[..., np.newaxis]
    elif data.ndim == 4:
        volumes = data
    else:
        raise ValueError("data must be a 3D or 4D array")

    nvolumes = volumes.shape[3]
    if bvalues is not None and len(bvalues) != nvolumes:
        raise ValueError("one b-value is required per volume")
    if gradients is not None and len(gradients) != nvolumes:
        raise ValueError("one gradient is required per volume")

    study_uid = generate_uid()
    series_uid = generate_uid()
    spacing_x, spacing_y, spacing_z = (float(value) for value in voxel_size)
    lps_origin = (-3.5 * spacing_x, -3.5 * spacing_y, -2.5 * spacing_z)

    for volume_index in range(nvolumes):
        for slice_index in range(volumes.shape[2]):
            instance_number = volume_index * volumes.shape[2] + slice_index + 1
            file_meta = FileMetaDataset()
            file_meta.MediaStorageSOPClassUID = MRImageStorage
            file_meta.MediaStorageSOPInstanceUID = generate_uid()
            file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            dataset = FileDataset(
                str(root / f"image{instance_number:04d}.dcm"),
                {},
                file_meta=file_meta,
                preamble=b"\0" * 128,
            )
            dataset.SOPClassUID = MRImageStorage
            dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
            dataset.StudyInstanceUID = study_uid
            dataset.SeriesInstanceUID = series_uid
            dataset.Modality = "MR"
            dataset.PatientName = SYNTHETIC_PATIENT_NAME
            dataset.PatientID = SYNTHETIC_PATIENT_ID
            dataset.PatientBirthDate = "19700101"
            dataset.PatientSex = "O"
            dataset.AccessionNumber = "SYNTH-ACC-01"
            dataset.AcquisitionDateTime = "20260926120000"
            dataset.SeriesNumber = series_number
            dataset.InstanceNumber = instance_number
            dataset.AcquisitionNumber = volume_index + 1
            dataset.TemporalPositionIdentifier = volume_index + 1
            dataset.ImageType = list(image_type)
            dataset.SeriesDescription = series_description
            dataset.ProtocolName = series_description
            dataset.SequenceName = sequence_name
            dataset.ScanningSequence = scanning_sequence
            dataset.MRAcquisitionType = acquisition_type
            dataset.Rows = int(volumes.shape[0])
            dataset.Columns = int(volumes.shape[1])
            dataset.SamplesPerPixel = 1
            dataset.PhotometricInterpretation = "MONOCHROME2"
            dataset.BitsAllocated = 16
            dataset.BitsStored = 16
            dataset.HighBit = 15
            dataset.PixelRepresentation = 0
            dataset.PixelSpacing = [spacing_x, spacing_y]
            dataset.SliceThickness = spacing_z
            dataset.SpacingBetweenSlices = spacing_z
            dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            dataset.ImagePositionPatient = [
                lps_origin[0],
                lps_origin[1],
                lps_origin[2] + slice_index * spacing_z,
            ]
            dataset.RepetitionTime = 3500.0
            dataset.EchoTime = 80.0 if bvalues is not None else 3.0
            dataset.FlipAngle = 90 if bvalues is not None else 9
            if bvalues is not None and gradients is not None:
                dataset.add_new(Tag(0x0018, 0x9087), "FD", float(bvalues[volume_index]))
                dataset.add_new(
                    Tag(0x0018, 0x9089),
                    "FD",
                    [float(value) for value in gradients[volume_index]],
                )
            pixel_slice = np.ascontiguousarray(volumes[:, :, slice_index, volume_index])
            dataset.PixelData = pixel_slice.tobytes()
            dataset.save_as(root / f"image{instance_number:04d}.dcm", enforce_file_format=True)
