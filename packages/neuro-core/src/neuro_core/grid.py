"""Grid / affine contract — the single source of truth for voxel<->world.

⛔ For NIfTI, the authoritative IJK->world matrix is nibabel's ``img.affine``
(the NIfTI sform/qform selection), NOT MRtrix ``mrinfo -transform``. For this
case the two disagree by 42.3 mm at the volume center because
``mrinfo -transform`` prints a normalized +diagonal matrix with strides handled
separately, whereas the NIfTI array affine is L/P/S with a negative diagonal.
Using the wrong one silently moves a painted seed to a different part of the
brain while every API type still looks valid. Native MIF inputs use the
validated, stride-aware canonical reconstruction in ``load_mif_grid``; no MIF
transform is substituted for a NIfTI array affine. See docs/DESIGN-slice1.md v2
#2.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import nibabel as nib


@dataclass(frozen=True)
class Grid:
    """An imaging grid: shape + the authoritative IJK->world affine.

    ``affine`` is always ``nibabel.Nifti1Image.affine``. Serialization order for
    any u8 volume derived from this grid is i-fastest (Fortran / column-major on
    the (i,j,k) axes), recorded here so the browser and server agree.
    """

    shape: tuple[int, int, int]
    affine: np.ndarray  # 4x4, nibabel img.affine
    axcodes: tuple[str, str, str]
    order: str = "i-fastest"
    # NIfTI permits an unknown spatial unit. Preserve that header fact on the
    # generic Grid API; strict ingestion decides whether the documented legacy
    # mm assumption is authorized by the case manifest.
    spatial_unit: str = "unknown"
    unit_semantics: str | None = None

    def __post_init__(self) -> None:
        shape = tuple(int(s) for s in self.shape)
        if len(shape) != 3 or any(s <= 0 for s in shape):
            raise ValueError(f"grid shape must be three positive dimensions, got {self.shape!r}")
        object.__setattr__(self, "shape", shape)
        affine = validate_affine(self.affine).copy()
        affine.setflags(write=False)
        object.__setattr__(self, "affine", affine)
        unit = _normalise_spatial_unit(self.spatial_unit)
        if unit in {"meter", "micron"}:
            raise ValueError(
                f"grid spatial unit {unit!r} is unsupported: numeric world "
                "outputs are millimetres; normalize the image before loading"
            )
        object.__setattr__(self, "spatial_unit", unit)
        semantics = self.unit_semantics
        if semantics is None:
            semantics = "explicit_mm" if unit == "mm" else "legacy_assumed_mm"
        semantics = str(semantics).strip().lower()
        if semantics not in {"explicit_mm", "legacy_assumed_mm", "native_mif_mm"}:
            raise ValueError(f"unsupported grid unit semantics {semantics!r}")
        if unit == "unknown" and semantics != "legacy_assumed_mm":
            raise ValueError(
                "unknown NIfTI spatial units must be represented as "
                "legacy_assumed_mm"
            )
        if unit == "mm" and semantics == "legacy_assumed_mm":
            raise ValueError("explicit mm spatial units cannot use legacy_assumed_mm semantics")
        object.__setattr__(self, "unit_semantics", semantics)

    @property
    def inv_affine(self) -> np.ndarray:
        return np.linalg.inv(self.affine)

    @property
    def numeric_world_unit(self) -> str:
        """Unit of every numeric world coordinate emitted by TractLab."""
        return "mm"

    @property
    def unit_note(self) -> str:
        """Human-readable unit provenance suitable for viewer/API metadata."""
        if self.unit_semantics == "legacy_assumed_mm":
            return "world coordinates are mm by legacy assumption; NIfTI unit was unknown"
        if self.unit_semantics == "native_mif_mm":
            return "world coordinates are mm from the native MRtrix header"
        return "world coordinates are mm from an explicit NIfTI mm unit"

    @property
    def unit_provenance(self) -> dict[str, object]:
        return {
            "numeric_world_unit": self.numeric_world_unit,
            "declared_spatial_unit": self.spatial_unit,
            "unit_semantics": self.unit_semantics,
            "unit_status": self.unit_semantics,
            "assumed_unknown_spatial_units_mm": self.unit_semantics == "legacy_assumed_mm",
        }


_UNIT_ALIASES = {
    "": "unknown",
    "none": "unknown",
    "unknown": "unknown",
    "mm": "mm",
    "millimeter": "mm",
    "millimeters": "mm",
    "meter": "meter",
    "meters": "meter",
    "m": "meter",
    "micron": "micron",
    "microns": "micron",
    "micrometer": "micron",
    "micrometers": "micron",
    "um": "micron",
}


def _normalise_spatial_unit(unit: object) -> str:
    raw = str(unit or "unknown").strip().lower()
    normalised = _UNIT_ALIASES.get(raw)
    if normalised is None:
        raise ValueError(
            f"unsupported spatial unit {raw!r}; numeric world outputs require mm"
        )
    return normalised


def unknown_units_assumption_from_manifest(manifest: dict) -> bool:
    """Return the explicit case-level permission for unknown NIfTI units.

    A missing key is false. The value is deliberately strict: strings such as
    ``"true"`` must never silently authorize a physical-unit assumption.
    """
    acquisition = manifest.get("acquisition")
    if acquisition is None:
        return False
    if not isinstance(acquisition, dict):
        raise ValueError("manifest acquisition must be an object")
    value = acquisition.get("assume_unknown_spatial_units_mm", False)
    if type(value) is not bool:  # noqa: E721 - reject bool-like strings/ints
        raise ValueError(
            "acquisition.assume_unknown_spatial_units_mm must be a strict boolean"
        )
    return value


def validate_affine(affine: np.ndarray, *, label: str = "affine") -> np.ndarray:
    """Return a finite, invertible homogeneous world affine.

    The array affine is the source of truth for NIfTI data.  This guard is
    intentionally strict about the spatial 3x3 block, but does not rewrite an
    affine or substitute a tool-specific transform.
    """
    aff = np.asarray(affine, dtype=np.float64)
    if aff.shape != (4, 4):
        raise ValueError(f"{label}: expected a (4,4) affine, got {aff.shape}")
    if not np.isfinite(aff).all():
        raise ValueError(f"{label}: affine contains non-finite values")
    if not np.allclose(aff[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8, rtol=0.0):
        raise ValueError(f"{label}: affine has an invalid homogeneous row")
    det = float(np.linalg.det(aff[:3, :3]))
    if not np.isfinite(det) or abs(det) <= 1e-12:
        raise ValueError(f"{label}: affine spatial block is singular")
    return aff


def _spatial_unit(image) -> str:
    """Read the declared NIfTI spatial unit without changing coordinates."""
    try:
        unit, _time_unit = image.header.get_xyzt_units()
    except (AttributeError, TypeError, ValueError):
        unit = None
    return _normalise_spatial_unit(unit)


def _nifti_affine_codes(image) -> tuple[int, int]:
    header = getattr(image, "header", None)
    if header is None:
        return (0, 0)
    try:
        _sform, s_code = header.get_sform(coded=True)
        _qform, q_code = header.get_qform(coded=True)
        return int(s_code or 0), int(q_code or 0)
    except (AttributeError, TypeError, ValueError, KeyError):
        try:
            return int(header["sform_code"]), int(header["qform_code"])
        except (KeyError, TypeError, ValueError):
            return (0, 0)


def grid_from_image(
    image,
    *,
    source: str = "image",
    require_mm: bool = False,
    assume_unknown_spatial_units_mm: bool = False,
    require_authoritative_affine: bool = False,
) -> Grid:
    """Build a :class:`Grid` from an already-loaded NIfTI-like image.

    Generic loading preserves unknown-unit metadata and records the legacy mm
    interpretation. Strict ingestion sets ``require_mm`` and must pass the
    manifest's explicit ``assume_unknown_spatial_units_mm`` permission. An
    image with both NIfTI affine codes zero is rejected at strict ingestion;
    its apparent nibabel affine could be synthesized from pixdim and is not a
    trustworthy world-coordinate contract.
    """
    if type(assume_unknown_spatial_units_mm) is not bool:  # noqa: E721
        raise ValueError(
            f"{source}: assume_unknown_spatial_units_mm must be a strict boolean"
        )
    aff = getattr(image, "affine", None)
    if aff is None:
        raise ValueError(f"{source}: image has no affine")
    shape = tuple(int(s) for s in getattr(image, "shape", ())[:3])
    if len(shape) != 3:
        raise ValueError(f"{source}: expected at least three spatial dimensions")
    spatial_unit = _spatial_unit(image)
    if spatial_unit in {"meter", "micron"}:
        raise ValueError(
            f"{source}: explicit spatial unit {spatial_unit!r} is unsupported; "
            "numeric world outputs require millimetres and the image must be "
            "normalized before loading"
        )
    if require_authoritative_affine:
        s_code, q_code = _nifti_affine_codes(image)
        if s_code == 0 and q_code == 0:
            raise ValueError(
                f"{source}: both sform_code and qform_code are zero; refusing "
                "a world affine synthesized from pixdim"
            )
    if require_mm and spatial_unit == "unknown" and not assume_unknown_spatial_units_mm:
        raise ValueError(
            f"{source}: NIfTI spatial unit is unknown; numeric world outputs "
            "require explicit mm units or manifest "
            "acquisition.assume_unknown_spatial_units_mm=true"
        )
    semantics = "explicit_mm" if spatial_unit == "mm" else "legacy_assumed_mm"
    return Grid(
        shape=shape,
        affine=validate_affine(aff, label=f"{source} affine"),
        axcodes=tuple(nib.aff2axcodes(aff)),
        spatial_unit=spatial_unit,
        unit_semantics=semantics,
    )


def load_grid(
    nifti_path: str,
    *,
    require_mm: bool = False,
    assume_unknown_spatial_units_mm: bool = False,
    require_authoritative_affine: bool = False,
) -> Grid:
    """Load the authoritative grid from a NIfTI file via nibabel.

    Raises if the file is 4-D-with-nontrivial-4th-dim in a way that would make
    a 3-D grid ambiguous; the FOD (185x185x109x45) is handled by taking its
    spatial shape, which shares the same affine as the b0.
    """
    if str(nifti_path).lower().endswith(".mif"):
        raise ValueError(f"{nifti_path}: native MIF requires load_mif_grid()")
    return grid_from_image(
        nib.load(nifti_path), source=str(nifti_path), require_mm=require_mm,
        assume_unknown_spatial_units_mm=assume_unknown_spatial_units_mm,
        require_authoritative_affine=require_authoritative_affine,
    )


def _mrinfo_executable() -> str:
    configured = os.environ.get("TRACTLAB_MRINFO")
    candidates = [configured, shutil.which("mrinfo"), str(Path.home() / "mrtrix3/bin/mrinfo")]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
        if candidate and shutil.which(candidate):
            return candidate
    raise ValueError(
        "native MIF grid cannot be validated: mrinfo is unavailable; "
        "set TRACTLAB_MRINFO or install MRtrix3"
    )


def _mrinfo_field(executable: str, option: str, path: str) -> str:
    try:
        proc = subprocess.run(
            [executable, option, path],
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"{path}: mrinfo {option} timed out") from exc
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise ValueError(f"{path}: mrinfo {option} failed: {detail.strip()}") from exc
    return proc.stdout.strip()


def _parse_mrinfo_numbers(text: str, *, option: str, path: str) -> np.ndarray:
    tokens = str(text).split()
    number = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
    if not tokens or any(number.fullmatch(token) is None for token in tokens):
        raise ValueError(
            f"{path}: mrinfo {option} returned malformed numeric output"
        )
    try:
        values = np.asarray([float(token) for token in tokens], dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{path}: mrinfo {option} returned malformed numeric output") from exc
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"{path}: mrinfo {option} returned no finite numeric values")
    return values


def load_mif_grid(mif_path: str, *, mrinfo_path: str | None = None) -> Grid:
    """Read a native MIF grid in a canonical array-axis representation.

    ``mrinfo -transform`` is *not* an NIfTI array affine: MRtrix reports a
    normalized direction transform and stores voxel spacing and data strides as
    separate header fields. The reconstruction below applies the symbolic
    stride permutation and signs exactly once to produce the equivalent
    canonical array grid. It never feeds the normalized transform directly to
    NIfTI or replaces a nibabel affine.

    MRtrix image coordinates and spacing are millimetres.  A native MIF with a
    header whose orientation/stride semantics cannot be read is refused rather
    than guessed or silently resampled.
    """
    path = str(mif_path)
    if not path.lower().endswith(".mif"):
        raise ValueError(f"{path}: expected a native .mif image")
    executable = mrinfo_path or _mrinfo_executable()
    size = _parse_mrinfo_numbers(_mrinfo_field(executable, "-size", path), option="-size", path=path)
    spacing = _parse_mrinfo_numbers(
        _mrinfo_field(executable, "-spacing", path), option="-spacing", path=path,
    )
    strides = _parse_mrinfo_numbers(
        _mrinfo_field(executable, "-strides", path), option="-strides", path=path,
    )
    transform_values = _parse_mrinfo_numbers(
        _mrinfo_field(executable, "-transform", path), option="-transform", path=path,
    )
    if size.size < 3 or spacing.size < 3 or strides.size < 3:
        raise ValueError(f"{path}: native MIF header lacks three spatial dimensions")
    if transform_values.size != 16:
        raise ValueError(f"{path}: mrinfo -transform did not return a 4x4 matrix")
    mif_shape = tuple(int(x) for x in size[:3])
    if any(x <= 0 or not np.isclose(x, round(x)) for x in size[:3]):
        raise ValueError(f"{path}: invalid native MIF dimensions {size[:3].tolist()}")
    spacing3 = np.asarray(spacing[:3], dtype=np.float64)
    strides3 = np.asarray(strides[:3], dtype=np.float64)
    if np.any(spacing3 <= 0) or not np.isfinite(spacing3).all():
        raise ValueError(f"{path}: invalid native MIF voxel spacing")
    if np.any(strides3 == 0) or not np.isfinite(strides3).all():
        raise ValueError(f"{path}: invalid native MIF data strides")
    # MRtrix strides rank axes by memory order across all dimensions, so a
    # volume-contiguous 4D FOD reports spatial strides such as -2,3,4 (with 1
    # on the volume axis). Only distinct integer magnitudes are required.
    stride_symbols = {int(abs(x)) for x in strides3 if float(x).is_integer()}
    if len(stride_symbols) != 3:
        raise ValueError(
            f"{path}: native MIF spatial strides are not three distinct integers"
        )

    transform = validate_affine(
        transform_values.reshape(4, 4), label=f"{path} mrinfo transform",
    )
    direction = transform[:3, :3]
    column_norms = np.linalg.norm(direction, axis=0)
    if not np.allclose(column_norms, 1.0, atol=1e-3, rtol=0.0):
        raise ValueError(
            f"{path}: native MIF transform has non-unit direction columns; "
            "orientation semantics are unresolved"
        )

    order = np.argsort(np.abs(strides3)).astype(np.int64)
    signs = np.where(strides3[order] < 0.0, -1.0, 1.0)
    shape = tuple(mif_shape[int(axis)] for axis in order)
    affine = np.eye(4, dtype=np.float64)
    # MRtrix's transform is the logical MIF mapping. To compare it with an
    # array affine from NIfTI, express the same image axes in canonical storage
    # order. A negative stride reverses that axis and moves the corner origin;
    # a permutation reorders both shape and affine columns. This is the same
    # one-time representation conversion used by MRtrix's NIfTI exporter.
    affine[:3, :3] = direction[:, order] @ np.diag(spacing3[order] * signs)
    origin = transform[:3, 3].copy()
    for axis, sign in zip(order, signs):
        if sign < 0.0:
            origin += direction[:, axis] * spacing3[axis] * (mif_shape[axis] - 1)
    affine[:3, 3] = origin
    return Grid(
        shape=shape,
        affine=validate_affine(affine, label=f"{path} logical affine"),
        axcodes=tuple(nib.aff2axcodes(affine)),
        spatial_unit="mm",
        unit_semantics="native_mif_mm",
    )


def load_image_grid(
    path: str,
    *,
    require_mm: bool = False,
    assume_unknown_spatial_units_mm: bool = False,
    require_authoritative_affine: bool = False,
) -> Grid:
    """Load a NIfTI or native MIF grid with an explicit format branch."""
    if str(path).lower().endswith(".mif"):
        return load_mif_grid(path)
    return load_grid(
        path, require_mm=require_mm,
        assume_unknown_spatial_units_mm=assume_unknown_spatial_units_mm,
        require_authoritative_affine=require_authoritative_affine,
    )


def assert_grids_match(
    reference: Grid,
    candidate: Grid,
    *,
    atol: float = 1e-3,
    context: str = "grid",
    require_verified_units: bool = False,
) -> Grid:
    """Fail closed on shape, canonical-unit, or affine mismatch."""
    if reference.shape != candidate.shape:
        raise ValueError(f"{context}: shape {candidate.shape} != reference {reference.shape}")
    if require_verified_units and (
        reference.unit_semantics == "legacy_assumed_mm"
        or candidate.unit_semantics == "legacy_assumed_mm"
    ):
        raise ValueError(
            f"{context}: spatial units are not verified mm "
            f"(reference={reference.unit_note}; candidate={candidate.unit_note})"
        )
    if not np.allclose(reference.affine, candidate.affine, atol=atol, rtol=0.0):
        raise ValueError(f"{context}: affine mismatch vs reference grid")
    return candidate


def voxel_to_world(grid: Grid, ijk: np.ndarray) -> np.ndarray:
    """Map voxel indices (...,3) to world mm (...,3) via the nibabel affine."""
    ijk = np.asarray(ijk, dtype=np.float64)
    homog = np.concatenate([ijk, np.ones(ijk.shape[:-1] + (1,))], axis=-1)
    return (homog @ grid.affine.T)[..., :3]


def world_to_voxel(grid: Grid, xyz_mm: np.ndarray) -> np.ndarray:
    """Map world mm (...,3) to fractional voxel indices (...,3).

    Callers that need integer voxels round with ``np.round`` and MUST discard
    out-of-bounds results (never clamp — clamping snaps exiting points onto the
    volume face and fabricates ROI hits; see docs/DESIGN-slice1.md defect list).
    """
    xyz_mm = np.asarray(xyz_mm, dtype=np.float64)
    homog = np.concatenate([xyz_mm, np.ones(xyz_mm.shape[:-1] + (1,))], axis=-1)
    return (homog @ grid.inv_affine.T)[..., :3]


def in_bounds(grid: Grid, ijk_int: np.ndarray) -> np.ndarray:
    """Boolean mask of which integer voxels lie inside the grid. No clamping."""
    ijk_int = np.asarray(ijk_int)
    shape = np.asarray(grid.shape)
    return np.all((ijk_int >= 0) & (ijk_int < shape), axis=-1)


def grid_id(grid: Grid) -> str:
    """Fingerprint of geometry, storage order, and unit provenance.

    Binds a request to THIS geometry and the unit semantics used to establish it.

    Note: two volumes with the same geometry but different voxel *contents* share
    a grid_id — content identity is a separate ``volume_id`` (b0 SHA + window
    params + serialized-u8 SHA), by design (DESIGN-slice1.md v2 #2, finding 2.3).
    """
    h = hashlib.sha256()
    h.update(np.asarray(grid.shape, dtype=np.int64).tobytes())
    # round the affine to 6 decimals so float noise below 1e-6 mm doesn't churn the id
    h.update(np.round(grid.affine, 6).astype(np.float64).tobytes())
    h.update(grid.order.encode())
    # The numerical world unit is canonical mm, but the distinction between an
    # explicit mm header and a legacy unknown-unit assumption is part of grid
    # identity so a request cannot silently cross that provenance boundary.
    h.update(grid.numeric_world_unit.encode())
    h.update(grid.spatial_unit.encode())
    h.update(str(grid.unit_semantics).encode())
    return h.hexdigest()


LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])


def lps_to_ras_affine(direction, spacing, origin) -> np.ndarray:
    """RAS index->world affine from an LPS direction (9 values), spacing and origin.

    The index order is SimpleITK's (x, y, z); a [k, j, i] numpy array from
    ``sitk.GetArrayFromImage`` indexes the same voxels reversed.
    """
    lps = np.eye(4)
    lps[:3, :3] = np.asarray(direction, dtype=float).reshape(3, 3) @ np.diag(np.asarray(spacing, dtype=float))
    lps[:3, 3] = np.asarray(origin, dtype=float)
    return LPS_TO_RAS @ lps
