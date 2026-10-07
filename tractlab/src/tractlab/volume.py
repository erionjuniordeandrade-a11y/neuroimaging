"""u8 volume contract — quantize the b0 ONCE for the browser MPR slicer.

The whole b0 is windowed with a single global window (masked p2/p98) and
quantized to u8, then serialized i-fastest so the browser can slice any plane.
This replaces the old per-plane-windowed baked planes (which could not be
byte-compared and were sourced from T1). See DESIGN-slice1.md v2 #8.

``volume_id`` fingerprints the *contents* (source SHA + window params + u8 SHA),
distinct from ``grid_id`` which fingerprints only geometry. Both are bound into
every request so a voxel-swap that keeps the grid still fails loud (v2 #2, 2.3).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import nibabel as nib
from neuro_core.hashing import sha256_file

from .grid import (
    Grid,
    assert_grids_match,
    grid_id,
    load_grid,
    load_image_grid,
)


@dataclass(frozen=True)
class U8Volume:
    grid: Grid
    data_u8: np.ndarray          # shape == grid.shape, dtype uint8
    window: tuple[float, float]  # (lo, hi) intensities mapped to (0, 255)
    source_sha256: str
    order: str = "i-fastest"

    def to_bytes(self) -> bytes:
        """Serialize i-fastest (Fortran order on i,j,k). Browser reads i fastest."""
        return np.asfortranarray(self.data_u8).tobytes(order="F")

    @property
    def volume_id(self) -> str:
        h = hashlib.sha256()
        h.update(self.source_sha256.encode())
        h.update(np.asarray(self.window, dtype=np.float64).tobytes())
        h.update(hashlib.sha256(self.to_bytes()).digest())
        h.update(grid_id(self.grid).encode())
        return h.hexdigest()


_sha256_file = sha256_file


def build_binary_u8_volume(
    mask_path: str,
    reference_grid: Grid | None = None,
    *,
    require_mm: bool = False,
    assume_unknown_spatial_units_mm: bool = False,
    require_authoritative_affine: bool = False,
) -> U8Volume:
    """A 0/255 u8 volume from a binary mask (for MPR overlays like the lesion).

    Not percentile-windowed — a mask has no intensity to window.
    """
    img = nib.load(mask_path)
    grid = load_grid(
        mask_path,
        require_mm=require_mm,
        assume_unknown_spatial_units_mm=assume_unknown_spatial_units_mm,
        require_authoritative_affine=require_authoritative_affine,
    )
    if reference_grid is not None:
        assert_grids_match(
            reference_grid, grid, context=f"{mask_path}: mask grid",
        )
    m = (np.asarray(img.dataobj) > 0).astype(np.uint8) * 255
    if m.ndim != 3 or m.shape != grid.shape:
        raise ValueError(f"{mask_path}: expected 3-D mask shape {grid.shape}, got {m.shape}")
    return U8Volume(grid=grid, data_u8=m, window=(0.0, 1.0),
                    source_sha256=_sha256_file(mask_path))


def build_u8_volume(
    b0_path: str,
    mask_path: str | None = None,
    lo_pct: float = 2.0,
    hi_pct: float = 98.0,
    *,
    require_mm: bool = False,
    assume_unknown_spatial_units_mm: bool = False,
    require_authoritative_affine: bool = False,
) -> U8Volume:
    """Window the b0 globally at masked lo/hi percentiles and quantize to u8.

    The window is computed over in-mask voxels only (so background air does not
    drag the percentiles), then applied to the WHOLE volume. One window, whole
    volume — never per-plane. OOB/clip is by design here (intensity clamp), which
    is unrelated to the coordinate no-clamp rule.
    """
    img = nib.load(b0_path)
    b0_grid = load_grid(
        b0_path,
        require_mm=require_mm,
        assume_unknown_spatial_units_mm=assume_unknown_spatial_units_mm,
        require_authoritative_affine=require_authoritative_affine,
    )
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"{b0_path}: expected 3-D b0, got shape {data.shape}")

    if mask_path is not None:
        mask_img = nib.load(mask_path)
        mask_grid = load_grid(
            mask_path,
            require_mm=require_mm,
            assume_unknown_spatial_units_mm=assume_unknown_spatial_units_mm,
            require_authoritative_affine=require_authoritative_affine,
        )
        assert_grids_match(
            b0_grid, mask_grid, context=f"{mask_path}: mask grid",
        )
        m = np.asarray(mask_img.dataobj) > 0
        if m.shape != data.shape:
            raise ValueError("mask shape != b0 shape")
        sample = data[m]
    else:
        sample = data[data > 0]
    if sample.size == 0:
        raise ValueError("no in-mask/positive voxels to window")

    lo = float(np.percentile(sample, lo_pct))
    hi = float(np.percentile(sample, hi_pct))
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        raise ValueError(f"degenerate window lo={lo} hi={hi}")

    scaled = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    u8 = np.round(scaled * 255).astype(np.uint8)

    return U8Volume(
        grid=b0_grid,
        data_u8=u8,
        window=(lo, hi),
        source_sha256=_sha256_file(b0_path),
    )


@dataclass(frozen=True)
class RGBVolume:
    """3-channel u8 volume for direction-encoded colour (DEC) MPR underlay.

    Serialized as three full spatial planes concatenated (R then G then B), each
    i-fastest Fortran order — same spatial layout as ``U8Volume.to_bytes()``.
    Header should carry ``X-Channels: 3`` and ``X-Format: rgb-planes``.
    """

    grid: Grid
    data_u8: np.ndarray  # shape (nx, ny, nz, 3), dtype uint8
    source_sha256: str
    order: str = "i-fastest"

    def to_bytes(self) -> bytes:
        if self.data_u8.ndim != 4 or self.data_u8.shape[-1] != 3:
            raise ValueError("RGBVolume data must be (nx,ny,nz,3)")
        planes = [
            np.asfortranarray(self.data_u8[..., c]).tobytes(order="F")
            for c in range(3)
        ]
        return b"".join(planes)

    @property
    def volume_id(self) -> str:
        h = hashlib.sha256()
        h.update(self.source_sha256.encode())
        h.update(hashlib.sha256(self.to_bytes()).digest())
        h.update(grid_id(self.grid).encode())
        return h.hexdigest()


def build_rgb_u8_volume(
    rgb_path: str,
    reference_grid: Grid,
    *,
    affine_atol: float = 1e-3,
    require_mm: bool = False,
    assume_unknown_spatial_units_mm: bool = False,
    require_authoritative_affine: bool = False,
) -> RGBVolume:
    """Load a 3-channel DEC (or similar) and quantize each channel to u8 [0,255].

    Expects float RGB already roughly in [0, max]; scales by the global max
    across channels so relative colour is preserved. Grid shape + affine must
    match the tracking/reference grid (fail closed).
    """
    img = nib.load(rgb_path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim != 4 or data.shape[-1] != 3:
        raise ValueError(f"{rgb_path}: expected (X,Y,Z,3), got {data.shape}")
    g = load_grid(
        rgb_path,
        require_mm=require_mm,
        assume_unknown_spatial_units_mm=assume_unknown_spatial_units_mm,
        require_authoritative_affine=require_authoritative_affine,
    )
    assert_grids_match(
        reference_grid, g, atol=affine_atol, context=f"{rgb_path}: DEC grid",
    )
    peak = float(np.nanmax(data))
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError(f"{rgb_path}: non-positive peak {peak}")
    scaled = np.clip(data / peak, 0.0, 1.0)
    u8 = np.round(scaled * 255).astype(np.uint8)
    return RGBVolume(
        grid=g,
        data_u8=u8,
        source_sha256=_sha256_file(rgb_path),
    )


def assert_grid_match(
    path: str,
    reference_grid: Grid,
    *,
    atol: float = 1e-3,
    require_mm: bool = False,
    assume_unknown_spatial_units_mm: bool = False,
    require_authoritative_affine: bool = False,
) -> Grid:
    """Fail closed if an image is not on the tracking grid.

    NIfTI paths use nibabel's selected affine.  Native MIF paths use the
    explicit MRtrix header reconstruction in :mod:`tractlab.grid`.
    """
    g = load_image_grid(
        path,
        require_mm=require_mm,
        assume_unknown_spatial_units_mm=assume_unknown_spatial_units_mm,
        require_authoritative_affine=require_authoritative_affine,
    )
    return assert_grids_match(
        reference_grid, g, atol=atol, context=f"{path}: grid",
    )
