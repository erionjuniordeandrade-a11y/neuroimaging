"""Small, deterministic DICOM phantoms for pipeline tests and demos.

The CT and MR pixel arrays describe the same anatomy.  Their DICOM patient
coordinates differ by a known rigid transform, so the files exercise both
header-based geometry and image registration without requiring patient data.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian


_CT_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.2"
_MR_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.4"
_RAS_TO_LPS = np.diag([-1.0, -1.0, 1.0])


def _dicom_ds(value: float) -> str:
    """Format a DICOM Decimal String within its 16-character limit."""
    return format(float(value), ".10g")


def _stable_uid(seed: int, label: str) -> str:
    """Return a reproducible, valid 2.25 UID for this phantom field."""
    digest = hashlib.sha256(f"case-capsule:{seed}:{label}".encode("utf-8")).digest()
    return f"2.25.{int.from_bytes(digest[:16], byteorder='big')}"


def _canary_values(seed: int) -> tuple[dict[str, str], list[str]]:
    """Build unique synthetic strings for DICOM identity fields."""
    nonce = hashlib.sha256(f"phantom-canary:{seed}".encode("utf-8")).hexdigest()[:10].upper()
    abbreviations = {
        "patient_name": "PNM",
        "patient_id": "PID",
        "other_patient_ids": "OPI",
        "issuer_of_patient_id": "IPI",
        "patient_birth_name": "PBN",
        "patient_address": "PAD",
        "accession_number": "ACC",
        "study_id": "SID",
        "study_description": "SD",
        "patient_comments": "PC",
        "study_comments": "SC",
        "referring_physician": "RPH",
        "performing_physician": "PPH",
        "operators_name": "OPN",
        "institution_name": "IN",
        "institution_address": "IA",
        "department_name": "DEP",
        "station_name": "STN",
        "device_serial_number": "DSN",
        "manufacturer": "MFG",
        "manufacturer_model": "MDM",
        "requesting_physician": "RQP",
        "requested_procedure_id": "RPI",
        "requested_procedure_description": "RPD",
    }
    # Keep each value within 16 characters, including DICOM SH fields.
    values = {name: f"CANARY-{nonce[:4]}-{suffix}" for name, suffix in abbreviations.items()}
    # Date VRs must remain valid DICOM DA values; their exact synthetic values
    # join the canary set so the de-identification check also catches dates.
    return values, [*values.values(), "19700101", "20260101"]


def _make_dataset(
    *,
    path: Path,
    modality: str,
    pixel_data: np.ndarray,
    instance_number: int,
    rows: int,
    columns: int,
    ipp_lps: np.ndarray,
    iop_lps: np.ndarray,
    pixel_spacing_mm: tuple[float, float],
    slice_spacing_mm: float,
    study_uid: str,
    series_uid: str,
    frame_uid: str,
    sop_uid: str,
    series_number: int,
    canary_fields: dict[str, str] | None,
) -> None:
    sop_class = _CT_SOP_CLASS if modality == "CT" else _MR_SOP_CLASS
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = sop_class
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = "1.2.826.0.1.3680043.10.543.1"

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.SpecificCharacterSet = "ISO_IR 192"
    ds.SOPClassUID = sop_class
    ds.SOPInstanceUID = sop_uid
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.FrameOfReferenceUID = frame_uid
    ds.Modality = modality
    ds.SeriesNumber = series_number
    ds.InstanceNumber = instance_number
    ds.SeriesDescription = "Synthetic Head CT" if modality == "CT" else "Synthetic Head MR T1"
    ds.StudyDescription = "Synthetic imaging phantom"
    ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
    ds.PatientPosition = "HFS"
    ds.StudyDate = "20260101"
    ds.SeriesDate = "20260101"
    ds.AcquisitionDate = "20260101"
    ds.ContentDate = "20260101"
    ds.StudyTime = "120000"
    ds.SeriesTime = "120000"
    ds.AcquisitionTime = "120000"
    ds.ContentTime = "120000"
    ds.PatientName = "Synthetic^Phantom"
    ds.PatientID = "SYNTHETIC-PHANTOM"
    ds.PatientBirthDate = "19700101"
    ds.PatientSex = "O"
    ds.AccessionNumber = "SYNTHETIC-ACC"
    ds.StudyID = "SYNTHETIC-STUDY"
    ds.InstitutionName = "Synthetic Imaging Lab"
    ds.InstitutionalDepartmentName = "Synthetic Radiology"
    ds.ReferringPhysicianName = "Synthetic^Referrer"
    ds.Manufacturer = "Case Capsule Phantom"
    ds.ManufacturerModelName = "Synthetic Scanner"
    ds.DeviceSerialNumber = "SYNTHETIC-SERIAL"
    ds.StationName = "PHANTOM"
    ds.OperatorsName = "Synthetic^Operator"

    if canary_fields is not None:
        ds.PatientName = canary_fields["patient_name"]
        ds.PatientID = canary_fields["patient_id"]
        ds.OtherPatientIDs = canary_fields["other_patient_ids"]
        ds.IssuerOfPatientID = canary_fields["issuer_of_patient_id"]
        ds.PatientBirthName = canary_fields["patient_birth_name"]
        ds.PatientAddress = canary_fields["patient_address"]
        ds.AccessionNumber = canary_fields["accession_number"]
        ds.StudyID = canary_fields["study_id"]
        ds.StudyDescription = canary_fields["study_description"]
        ds.PatientComments = canary_fields["patient_comments"]
        ds.StudyComments = canary_fields["study_comments"]
        ds.ReferringPhysicianName = canary_fields["referring_physician"]
        ds.PerformingPhysicianName = canary_fields["performing_physician"]
        ds.OperatorsName = canary_fields["operators_name"]
        ds.InstitutionName = canary_fields["institution_name"]
        ds.InstitutionAddress = canary_fields["institution_address"]
        ds.InstitutionalDepartmentName = canary_fields["department_name"]
        ds.StationName = canary_fields["station_name"]
        ds.DeviceSerialNumber = canary_fields["device_serial_number"]
        ds.Manufacturer = canary_fields["manufacturer"]
        ds.ManufacturerModelName = canary_fields["manufacturer_model"]
        ds.RequestingPhysician = canary_fields["requesting_physician"]
        ds.RequestedProcedureID = canary_fields["requested_procedure_id"]
        ds.RequestedProcedureDescription = canary_fields["requested_procedure_description"]
        # These fields use DICOM DA (date) values.  Keep them syntactically
        # valid while making their exact values searchable in the canary gate.
        ds.PatientBirthDate = "19700101"
        ds.StudyDate = "20260101"

    ds.Rows = rows
    ds.Columns = columns
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1 if modality == "CT" else 0
    ds.PixelSpacing = [_dicom_ds(pixel_spacing_mm[0]), _dicom_ds(pixel_spacing_mm[1])]
    ds.SliceThickness = _dicom_ds(slice_spacing_mm)
    ds.SpacingBetweenSlices = _dicom_ds(slice_spacing_mm)
    ds.ImagePositionPatient = [_dicom_ds(v) for v in ipp_lps]
    ds.ImageOrientationPatient = [_dicom_ds(v) for v in iop_lps]
    ds.RescaleSlope = "1"
    ds.RescaleIntercept = "-1024" if modality == "CT" else "0"
    if modality == "CT":
        ds.RescaleType = "HU"
        ds.KVP = "120"
        ds.ConvolutionKernel = "BRAIN"
        ds.ScanOptions = "HELICAL"
    else:
        ds.RescaleType = "US"
        ds.ScanningSequence = "SE"
        ds.SequenceVariant = "SK"
        ds.SequenceName = "T1_SE"
        ds.MRAcquisitionType = "3D"
        ds.MagneticFieldStrength = "3"
        ds.RepetitionTime = "500"
        ds.EchoTime = "10"
        ds.FlipAngle = "90"
        ds.ContrastBolusAgent = ""
    ds.PixelData = np.ascontiguousarray(pixel_data).tobytes()
    ds.save_as(path, write_like_original=False)


def write_phantom(
    out_dir: str | Path,
    *,
    seed: int,
    tilt_deg: float = 0,
    canary: bool = True,
) -> dict[str, Any]:
    """Write compact CT and MR T1 DICOM series containing the same phantom.

    The MR image's voxel geometry is the CT geometry transformed by a known
    7-degree RAS-z rotation and 4 mm RAS-x translation.  ``mr_to_ct_ras`` is
    the inverse map from MR physical coordinates to CT physical coordinates.
    Pixel intensities remain in modality-native units (CT HU; MR arbitrary
    units), and all DICOM pixel values are stored as 16-bit integers.

    Returns paths, series numbers, anatomical truth coordinates in RAS mm,
    both directions of the known transform, and all synthetic canary strings.
    Existing output files are never overwritten.
    """
    if not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if not math.isfinite(float(tilt_deg)) or abs(float(tilt_deg)) > 20.0:
        raise ValueError("tilt_deg must be finite and between -20 and 20 degrees")

    root = Path(out_dir)
    ct_dir = root / "CT"
    mr_dir = root / "MR_T1"
    ct_dir.mkdir(parents=True, exist_ok=True)
    mr_dir.mkdir(parents=True, exist_ok=True)
    if any(ct_dir.iterdir()) or any(mr_dir.iterdir()):
        raise FileExistsError("phantom output series directories must be empty")

    # Odd dimensions make the central voxel exactly RAS (0, 0, 0). The head is
    # adult-sized and clearly anisotropic: a near-circular phantom hides
    # rotation about z from mutual-information registration.
    nx, ny, nz = 101, 123, 47
    sx = sy = 1.5
    sz = 3.0
    center_index = np.array([(nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2], dtype=float)
    tilt = math.radians(float(tilt_deg))
    ct_x_dir = np.array([1.0, 0.0, 0.0])
    ct_y_dir = np.array([0.0, 1.0, 0.0])
    ct_slice_step = np.array([math.sin(tilt) * sz, 0.0, math.cos(tilt) * sz])

    z_idx, y_idx, x_idx = np.indices((nz, ny, nx), dtype=np.float32)
    dx = (x_idx - center_index[0]) * sx
    dy = (y_idx - center_index[1]) * sy
    dz = (z_idx - center_index[2]) * sz
    # A gantry-tilted CT stack shifts each slice laterally in patient space.
    dx = dx + (z_idx - center_index[2]) * math.sin(tilt) * sz
    x_mm, y_mm, z_mm = dx, dy, dz
    # Internal structures are drawn in unit coordinates scaled up to mm.
    k = 2.2
    xu, yu, zu = x_mm / k, y_mm / k, z_mm / k

    rx, ry, rz = 31.0, 38.0, 29.0
    ellipsoid_r = np.sqrt((xu / rx) ** 2 + (yu / ry) ** 2 + (zu / rz) ** 2)
    nose = ((xu - 2.0) / 5.0) ** 2 + ((yu - 37.0) / 7.0) ** 2 + ((zu + 6.0) / 8.0) ** 2 <= 1
    ellipsoid_r = np.where(nose, np.minimum(ellipsoid_r, 0.85), ellipsoid_r)
    head = ellipsoid_r <= 1.0
    bone = head & (ellipsoid_r >= 0.91)
    core = head & ~bone
    lesion_center = np.array([14.0, -9.0, 4.0])
    marker_center = np.array([-56.0, 8.0, 12.0])
    lesion = (
        (x_mm - lesion_center[0]) ** 2
        + (y_mm - lesion_center[1]) ** 2
        + (z_mm - lesion_center[2]) ** 2
    ) <= 10.0**2
    marker = (
        (x_mm - marker_center[0]) ** 2
        + (y_mm - marker_center[1]) ** 2
        + (z_mm - marker_center[2]) ** 2
    ) <= 4.0**2

    # Asymmetric structures give mutual-information registration enough
    # internal detail even though CT and MR have very different contrasts.
    vent_left = ((xu + 3.4) / 2.0) ** 2 + ((yu - 1.5) / 7.0) ** 2 + (zu / 9.0) ** 2 <= 1
    vent_right = ((xu - 1.0) / 2.4) ** 2 + ((yu - 2.0) / 5.0) ** 2 + (zu / 8.0) ** 2 <= 1
    deep_structure = ((xu - 8.0) / 7.0) ** 2 + ((yu + 5.0) / 5.0) ** 2 + ((zu - 2.0) / 6.0) ** 2 <= 1
    cortical_ridge = (
        ((xu + 11.0) / 8.0) ** 2 + ((yu + 4.0) / 5.0) ** 2 + ((zu + 2.0) / 7.0) ** 2 <= 1
    )

    texture = (
        2.1 * np.sin(0.27 * xu + 0.11 * yu + 0.07 * zu)
        + 1.4 * np.cos(0.09 * xu - 0.23 * yu + 0.16 * zu)
        + 0.8 * np.sin(0.18 * xu + 0.21 * yu - 0.19 * zu)
    )
    ct = np.full((nz, ny, nx), -1000, dtype=np.int16)
    ct[core] = 40
    ct[core & cortical_ridge] = 48
    ct[core & deep_structure] = 35
    ct[core & (vent_left | vent_right)] = 10
    ct[lesion] = 60
    ct[bone] = 1000
    ct[marker] = 1800

    mr = np.zeros((nz, ny, nx), dtype=np.uint16)
    mr[core] = np.clip(100.0 + 5.0 * texture[core], 1, 65535).astype(np.uint16)
    white_matter = core & (ellipsoid_r < 0.59)
    mr[white_matter] = np.clip(145.0 + 4.0 * texture[white_matter], 1, 65535).astype(np.uint16)
    mr[core & cortical_ridge] = np.clip(80.0 + 4.0 * texture[core & cortical_ridge], 1, 65535).astype(np.uint16)
    mr[core & deep_structure] = np.clip(175.0 + 4.0 * texture[core & deep_structure], 1, 65535).astype(np.uint16)
    mr[core & (vent_left | vent_right)] = 18
    mr[bone] = 14
    mr[lesion] = 220
    mr[marker] = 245

    rotation_z = math.radians(7.0)
    c, s = math.cos(rotation_z), math.sin(rotation_z)
    ct_to_mr = np.eye(4, dtype=float)
    ct_to_mr[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    ct_to_mr[:3, 3] = np.array([4.0, 0.0, 0.0])
    mr_to_ct = np.linalg.inv(ct_to_mr)
    mr_rotation = ct_to_mr[:3, :3]
    mr_x_dir = mr_rotation @ ct_x_dir
    mr_y_dir = mr_rotation @ ct_y_dir
    mr_slice_step = mr_rotation @ ct_slice_step

    ct_origin_ras = -center_index[0] * sx * ct_x_dir - center_index[1] * sy * ct_y_dir - center_index[2] * ct_slice_step
    mr_origin_ras = mr_rotation @ ct_origin_ras + ct_to_mr[:3, 3]
    ct_iop_lps = np.concatenate((_RAS_TO_LPS @ ct_x_dir, _RAS_TO_LPS @ ct_y_dir))
    mr_iop_lps = np.concatenate((_RAS_TO_LPS @ mr_x_dir, _RAS_TO_LPS @ mr_y_dir))

    if canary:
        canary_fields, canary_strings = _canary_values(int(seed))
    else:
        canary_fields, canary_strings = None, []

    study_uid = _stable_uid(int(seed), "study")
    ct_series_uid = _stable_uid(int(seed), "ct-series")
    mr_series_uid = _stable_uid(int(seed), "mr-series")
    ct_frame_uid = _stable_uid(int(seed), "ct-frame")
    mr_frame_uid = _stable_uid(int(seed), "mr-frame")
    if canary:
        canary_strings.extend([study_uid, ct_series_uid, mr_series_uid, ct_frame_uid, mr_frame_uid])

    for k in range(nz):
        ct_pos_ras = ct_origin_ras + k * ct_slice_step
        mr_pos_ras = mr_origin_ras + k * mr_slice_step
        ct_sop_uid = _stable_uid(int(seed), f"ct-sop-{k + 1}")
        mr_sop_uid = _stable_uid(int(seed), f"mr-sop-{k + 1}")
        if canary:
            canary_strings.extend([ct_sop_uid, mr_sop_uid])
        _make_dataset(
            path=ct_dir / f"IM{k + 1:04d}.dcm",
            modality="CT",
            pixel_data=(ct[k].astype(np.int32) + 1024).astype(np.int16),
            instance_number=k + 1,
            rows=ny,
            columns=nx,
            ipp_lps=_RAS_TO_LPS @ ct_pos_ras,
            iop_lps=ct_iop_lps,
            pixel_spacing_mm=(sy, sx),
            slice_spacing_mm=sz,
            study_uid=study_uid,
            series_uid=ct_series_uid,
            frame_uid=ct_frame_uid,
            sop_uid=ct_sop_uid,
            series_number=1,
            canary_fields=canary_fields,
        )
        _make_dataset(
            path=mr_dir / f"IM{k + 1:04d}.dcm",
            modality="MR",
            pixel_data=mr[k],
            instance_number=k + 1,
            rows=ny,
            columns=nx,
            ipp_lps=_RAS_TO_LPS @ mr_pos_ras,
            iop_lps=mr_iop_lps,
            pixel_spacing_mm=(sy, sx),
            slice_spacing_mm=sz,
            study_uid=study_uid,
            series_uid=mr_series_uid,
            frame_uid=mr_frame_uid,
            sop_uid=mr_sop_uid,
            series_number=2,
            canary_fields=canary_fields,
        )

    return {
        "ct_dir": ct_dir,
        "mr_dir": mr_dir,
        "series": {"CT": 1, "MR": 2},
        "lesion_center_ras_mm": tuple(float(v) for v in lesion_center),
        "marker_center_ras_mm": tuple(float(v) for v in marker_center),
        "lesion_center_mr_ras_mm": tuple(float(v) for v in (ct_to_mr @ np.r_[lesion_center, 1.0])[:3]),
        "marker_center_mr_ras_mm": tuple(float(v) for v in (ct_to_mr @ np.r_[marker_center, 1.0])[:3]),
        "mr_to_ct_ras": mr_to_ct.tolist(),
        "ct_to_mr_ras": ct_to_mr.tolist(),
        "canary_strings": canary_strings,
        "spacing_mm": (sx, sy, sz),
        "dimensions_xyz": (nx, ny, nz),
        "ct_values_hu": {"air": -1000, "soft_tissue": 40, "lesion": 60, "bone": 1000, "marker": 1800},
        "mr_values_au": {"air": 0, "background_tissue": 100, "white_matter": 145, "lesion": 220, "marker": 245},
    }
