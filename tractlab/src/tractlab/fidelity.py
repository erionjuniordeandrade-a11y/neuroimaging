"""E1 — data-fidelity evaluation half: real-SH amplitudes along geometry.

MRtrix real-SH convention (pinned by tests/test_fidelity_sh.py against an
sh2amp golden fixture — the tolerance there is a contract):

  even l only; within each l, m runs -l..l;
  m < 0 → sqrt(2) * Im(Y_l^{|m|}),  m = 0 → Y_l^0,  m > 0 → sqrt(2) * Re(Y_l^m).

Amplitude at a point = trilinear-interpolated coefficient vector · basis row
for the sampling direction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from .grid import assert_grids_match, grid_from_image, validate_affine

try:  # scipy >= 1.15: sph_harm removed; sph_harm_y(l, m, polar, azimuth)
    from scipy.special import sph_harm_y

    def _ylm(m_abs: int, l: int, az: np.ndarray, el: np.ndarray) -> np.ndarray:
        return sph_harm_y(l, m_abs, el, az)
except ImportError:  # scipy < 1.15: sph_harm(m, l, azimuth, polar)
    from scipy.special import sph_harm

    def _ylm(m_abs: int, l: int, az: np.ndarray, el: np.ndarray) -> np.ndarray:
        return sph_harm(m_abs, l, az, el)


def lmax_from_ncoef(ncoef: int) -> int:
    """Inverse of ncoef = (lmax+1)(lmax+2)/2 for even lmax; loud otherwise."""
    lmax = 0
    while (lmax + 1) * (lmax + 2) // 2 < ncoef:
        lmax += 2
    if (lmax + 1) * (lmax + 2) // 2 != ncoef:
        raise ValueError(f"{ncoef} is not a real-SH coefficient count")
    return lmax


def unit_dirs(dirs: np.ndarray) -> np.ndarray:
    """Normalize an (N,3) cartesian direction set; reject anything else."""
    d = np.asarray(dirs, dtype=np.float64)
    if d.ndim != 2 or d.shape[1] != 3:
        raise ValueError(f"expected (N,3) cartesian directions, got {d.shape}")
    norm = np.linalg.norm(d, axis=1, keepdims=True)
    if not np.all(norm > 0):
        raise ValueError("zero-length direction")
    return d / norm


def real_sh_basis(dirs: np.ndarray, lmax: int) -> np.ndarray:
    """(N, ncoef) MRtrix real-SH basis for unit cartesian directions."""
    d = np.asarray(dirs, dtype=np.float64)
    az = np.arctan2(d[:, 1], d[:, 0])          # azimuth phi
    el = np.arccos(np.clip(d[:, 2], -1.0, 1.0))  # polar theta from +z
    cols = []
    for l in range(0, lmax + 1, 2):
        for m in range(-l, l + 1):
            y = _ylm(abs(m), l, az, el)
            if m < 0:
                cols.append(np.sqrt(2.0) * y.imag)
            elif m == 0:
                cols.append(y.real)
            else:
                cols.append(np.sqrt(2.0) * y.real)
    return np.stack(cols, axis=1)


def amplitude_along(
    coeffs_img: np.ndarray,
    affine: np.ndarray,
    points_mm: np.ndarray,
    dirs: np.ndarray,
) -> np.ndarray:
    """fODF amplitude at each (point, direction) pair.

    points_mm: (N,3) world coordinates; dirs: (N,3) unit directions, one per
    point. Returns (N,). Coefficients are trilinearly interpolated
    (map_coordinates order=1); points outside the volume are returned as
    ``NaN``. Edge padding is not evidence: an out-of-FOV sample must remain
    explicitly unmeasurable.
    """
    vol = np.asarray(coeffs_img, dtype=np.float64)
    pts = np.asarray(points_mm, dtype=np.float64)
    d = np.asarray(dirs, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or d.shape != pts.shape:
        raise ValueError("points_mm and dirs must both be (N,3)")
    if not np.isfinite(pts).all() or not np.isfinite(d).all():
        raise ValueError("points_mm and dirs must be finite")
    if vol.ndim != 4:
        raise ValueError(f"SH volume must be 4-D (X,Y,Z,C), got {vol.shape}")
    validate_affine(affine, label="SH affine")
    ncoef = vol.shape[-1]
    lmax = lmax_from_ncoef(ncoef)

    inv = np.linalg.inv(np.asarray(affine, dtype=np.float64))
    vox = (pts @ inv[:3, :3].T) + inv[:3, 3]
    valid = _fractional_in_bounds(vox, vol.shape[:3])
    out = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    if not valid.any():
        return out

    # Only absorb tiny inverse-affine roundoff at an exact boundary. Real
    # out-of-FOV points never reach map_coordinates and are not edge-clamped.
    valid_vox = np.clip(vox[valid], 0.0, np.asarray(vol.shape[:3], dtype=float) - 1.0)
    coords = valid_vox.T  # (3, N_valid)
    interp = np.empty((int(valid.sum()), ncoef), dtype=np.float64)
    for c in range(ncoef):
        interp[:, c] = map_coordinates(
            vol[..., c], coords, order=1, mode="nearest"
        )

    basis = real_sh_basis(d[valid], lmax)  # (N_valid, ncoef)
    out[valid] = np.einsum("nc,nc->n", interp, basis)
    return out


# ── E1 support ratios on SOURCE geometry ─────────────────────────────────────

# Relative floor: a peak below this fraction of the volume's max peak has no
# meaningful lobe to normalize by → ratio 0. Relative, so a globally rescaled
# FOD yields identical ratios (Sol batch-4: an absolute floor broke the
# scale-invariance claim). _PEAK_ABS_GUARD only stops literal /0 on empty maps.
PEAK_FLOOR_REL = 1e-6
_PEAK_ABS_GUARD = 1e-30


class FidelityRefusal(RuntimeError):
    """Missing/mismatched provenance: fidelity refuses, never guesses."""


def _verified_sh_image(source, supplied_conversion=None):
    """Read the pinned source, proving any supplied conversion's voxel data.

    A digest of a scratch file alone does not prove it came from the FOD.
    Native MIF is converted afresh by MRtrix in private temporary storage;
    both the affine and every coefficient must match a supplied conversion.
    The returned image owns its data before temporary storage is removed.
    This offline operation does not run tracking or modify the source.
    """
    import nibabel as nib
    import shutil
    import subprocess
    import tempfile

    def read_owned(path):
        image = nib.load(str(path))
        return nib.Nifti1Image(np.array(image.get_fdata(), copy=True), image.affine, image.header.copy())

    source = Path(source)
    if source.suffix.lower() == ".mif":
        executable = shutil.which("mrconvert") or str(Path.home() / "mrtrix3/bin/mrconvert")
        if not Path(executable).is_file():
            raise FidelityRefusal("MRtrix mrconvert is required to verify native MIF coefficients")
        with tempfile.TemporaryDirectory(prefix="tractlab-fod-proof-") as work:
            canonical = Path(work) / "source.nii"
            try:
                subprocess.run([executable, str(source), str(canonical), "-force", "-quiet"],
                               check=True, capture_output=True, timeout=120)
            except (OSError, subprocess.SubprocessError) as exc:
                raise FidelityRefusal("could not verify native MIF conversion") from exc
            image = read_owned(canonical)
    else:
        image = read_owned(source)
    if supplied_conversion is not None and Path(supplied_conversion).resolve() != source.resolve():
        converted = nib.load(str(supplied_conversion))
        if (converted.shape != image.shape
                or not np.allclose(converted.affine, image.affine, atol=1e-5, rtol=0)
                or not np.array_equal(converted.get_fdata(), image.get_fdata(), equal_nan=True)):
            raise FidelityRefusal("SH conversion does not match pinned FOD geometry and coefficients")
    return image


def _summarise_ratios(ratios: np.ndarray) -> tuple[float, float]:
    """Summarise only a fully measurable streamline.

    A single outside/nonfinite sample makes the streamline summary
    unmeasurable. In particular, dropping invalid samples before a percentile
    or mean would turn partial evidence into an apparently low-support score.
    """
    values = np.asarray(ratios, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        return float("nan"), float("nan")
    return float(np.percentile(values, 5.0)), float(values.mean())


def _trilinear(vol: np.ndarray, affine: np.ndarray, points_mm: np.ndarray) -> np.ndarray:
    vol = np.asarray(vol, dtype=np.float64)
    if vol.ndim != 3:
        raise ValueError(f"peak volume must be 3-D, got {vol.shape}")
    validate_affine(affine, label="peak affine")
    inv = np.linalg.inv(np.asarray(affine, dtype=np.float64))
    pts = np.asarray(points_mm, dtype=np.float64)
    vox = (pts @ inv[:3, :3].T) + inv[:3, 3]
    valid = _fractional_in_bounds(vox, vol.shape)
    out = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    if valid.any():
        valid_vox = np.clip(vox[valid], 0.0, np.asarray(vol.shape, dtype=float) - 1.0)
        out[valid] = map_coordinates(vol, valid_vox.T, order=1, mode="nearest")
    return out


def _fractional_in_bounds(vox: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Whether fractional voxel coordinates can be sampled without padding."""
    coords = np.asarray(vox, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"voxel coordinates must be (N,3), got {coords.shape}")
    upper = np.asarray(shape, dtype=np.float64) - 1.0
    eps = 1e-8  # inverse-affine roundoff at an exact boundary only
    return np.isfinite(coords).all(axis=1) & np.all(coords >= -eps, axis=1) & np.all(
        coords <= upper + eps, axis=1,
    )


def segment_ratios(
    line_mm: np.ndarray,
    sh_img: np.ndarray,
    sh_affine: np.ndarray,
    peak_img: np.ndarray,
    peak_affine: np.ndarray,
) -> np.ndarray:
    """Per-SEGMENT amp/peak ratio along a source polyline.

    Evaluated at segment midpoints in the segment's unit direction; scale-
    invariant (amp and peak share the fODF's scale). Where the local peak is
    below PEAK_FLOOR the ratio is 0.0. Out-of-FOV or nonfinite sampled values
    remain ``NaN`` (unmeasurable), never an edge score or fabricated zero.
    """
    pts = np.asarray(line_mm, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[-1] != 3 or pts.shape[0] < 2:
        return np.empty((0,), dtype=np.float64)
    if not np.isfinite(pts).all():
        raise ValueError("non-finite vertex in streamline")
    sh = np.asarray(sh_img, dtype=np.float64)
    peak_map = np.asarray(peak_img, dtype=np.float64)
    if sh.ndim != 4 or peak_map.ndim != 3:
        raise FidelityRefusal(
            f"fidelity source dimensions invalid: SH {sh.shape}, peak {peak_map.shape}"
        )
    if sh.shape[:3] != peak_map.shape:
        raise FidelityRefusal(
            f"fidelity source grid shape mismatch: SH {sh.shape[:3]} vs peak {peak_map.shape}"
        )
    try:
        sh_aff = validate_affine(sh_affine, label="SH affine")
        peak_aff = validate_affine(peak_affine, label="peak affine")
    except ValueError as exc:
        raise FidelityRefusal(str(exc)) from exc
    if not np.allclose(sh_aff, peak_aff, atol=1e-3, rtol=0.0):
        raise FidelityRefusal("SH/peak affine mismatch: support grids must agree")
    segs = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(segs, axis=1)
    keep = seg_len > 0
    if not keep.any():
        return np.empty((0,), dtype=np.float64)
    segs = segs[keep]
    mids = (pts[:-1][keep] + pts[1:][keep]) / 2.0
    dirs = segs / np.linalg.norm(segs, axis=1, keepdims=True)
    amp = amplitude_along(sh, sh_aff, mids, dirs)
    peak = _trilinear(peak_map, peak_aff, mids)
    finite_peak = peak_map[np.isfinite(peak_map)]
    if finite_peak.size == 0:
        raise FidelityRefusal("peak image has no finite values")
    floor = max(float(np.max(finite_peak)) * PEAK_FLOOR_REL, _PEAK_ABS_GUARD)
    ratios = np.full_like(amp, np.nan)
    finite_sample = np.isfinite(amp) & np.isfinite(peak)
    low_support = finite_sample & (peak <= floor)
    ratios[low_support] = 0.0
    ok = finite_sample & (peak > floor)
    ratios[ok] = amp[ok] / peak[ok]
    return ratios


def summaries_for_bank(
    tck_path,
    sh_path,
    peak_path,
    *,
    provenance,
    sh_load_path=None,
    assume_unknown_spatial_units_mm: bool = False,
) -> dict:
    """Per-streamline support summaries for one bank, provenance-gated.

    Refuses (FidelityRefusal) when provenance is absent, lacks fod_sha256, or
    the FOD file on disk does not hash to it — a ratio computed against the
    wrong FOD is worse than no ratio.

    NIfTI source images must declare millimetre units. A legacy unknown-unit
    image is accepted only when the caller explicitly passes
    ``assume_unknown_spatial_units_mm=True`` (normally derived from the case
    manifest); the assumption is never inferred from pixdim.

    The sha pin is ALWAYS on sh_path (the manifest FOD, possibly .mif, which
    nibabel cannot read). sh_load_path, when given, is a nibabel-readable
    conversion of that same volume. It must match an independent source read
    in both geometry and coefficients; an arbitrary scratch image is refused.
    """
    import hashlib

    import nibabel as nib

    if provenance is None:
        raise FidelityRefusal("bank has no provenance — run the backfill first")
    fod_sha = None
    if isinstance(provenance, dict):
        fod_sha = provenance.get("fod_sha256")
    else:
        fod_sha = getattr(provenance, "fod_sha256", None)
    if not fod_sha:
        raise FidelityRefusal("provenance lacks fod_sha256 — cannot pin the FOD")
    h = hashlib.sha256()
    with open(sh_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != fod_sha:
        raise FidelityRefusal(
            f"FOD sha mismatch: manifest {fod_sha[:12]}… vs disk {actual[:12]}…"
        )

    sh_nii = _verified_sh_image(sh_path, sh_load_path)
    peak_nii = nib.load(str(peak_path))
    try:
        sh_grid = grid_from_image(
            sh_nii,
            source="SH source",
            require_mm=True,
            assume_unknown_spatial_units_mm=assume_unknown_spatial_units_mm,
            require_authoritative_affine=True,
        )
        peak_grid = grid_from_image(
            peak_nii,
            source="peak source",
            require_mm=True,
            assume_unknown_spatial_units_mm=assume_unknown_spatial_units_mm,
            require_authoritative_affine=True,
        )
        assert_grids_match(sh_grid, peak_grid, context="SH/peak grid")
    except ValueError as exc:
        raise FidelityRefusal(str(exc)) from exc
    sh_img = sh_nii.get_fdata()
    peak_img = peak_nii.get_fdata()
    tck = nib.streamlines.load(str(tck_path))

    p5: list[float] = []
    mean: list[float] = []
    for s in tck.streamlines:
        r = segment_ratios(
            np.asarray(s, dtype=np.float64),
            sh_img, sh_nii.affine, peak_img, peak_nii.affine,
        )
        p5_value, mean_value = _summarise_ratios(r)
        # n=1 segment: percentile IS the lone sample — a location, not a tail
        # claim; consumers must not read p5 as a distribution there. Any
        # outside/nonfinite sample remains NaN in both summaries.
        p5.append(p5_value)
        mean.append(mean_value)
    return {
        "n": len(p5),
        "p5_ratio": np.asarray(p5, dtype=np.float32),
        "mean_ratio": np.asarray(mean, dtype=np.float32),
        "fod_sha256": fod_sha,
    }


# ── E1/S3 sidecars: (bank sha, ordinal)-keyed summaries, exact join or refuse ─

# Schema 2 (S-05): binds the sidecar to the actual SH-load and peak files used
# to compute it (sh_load_sha256/peak_sha256), not only the pinned FOD/bank
# hashes. A converted .mif->.nii scratch copy could previously change without
# any sidecar consumer noticing. Advancing the schema means an old schema-1
# sidecar refuses outright (below) — it is never silently reinterpreted or
# re-signed under the new contract.
SIDECAR_SCHEMA = 2
# Precomputed thresholds 0.05..0.60 step 0.05 — the signed operating point
# must be ON this grid (asserted at load), so changing R never re-walks banks.
R_GRID = np.round(np.arange(1, 13) * 0.05, 2).astype(np.float32)
# Unsigned-pilot defaults, used ONLY while no signed operating-point sheet
# exists (see read_operating_point). Both sit on R_GRID / MIN_FRAC_GRID.
# Pilot language only.
DEFAULT_R = 0.30
DEFAULT_MIN_FRAC = 0.70
# Published min_frac sweep grid (fidelity_sweep writes sweep.csv over it); a
# signed min_frac must be one of these so the signature points at a row the
# owner actually saw.
MIN_FRAC_GRID = (0.5, 0.7, 0.8, 0.9, 0.95)

SHEET_RELPATH = Path("docs") / "qc" / "OPERATING-POINT-fidelity.md"


@dataclass(frozen=True)
class OperatingPoint:
    """The (R, min_frac) the viewer marks low-support against.

    ``source`` is "signed" when it came from an owner-signed sheet, "pilot"
    when it is the unsigned fallback. The viewer labels the chip from this,
    so a pilot point can never masquerade as a signed one.
    """

    R: float
    min_frac: float
    source: str  # "signed" | "pilot"
    approved_by: str | None = None
    date: str | None = None
    sweep_csv_sha256: str | None = None

    @property
    def signed(self) -> bool:
        return self.source == "signed"


PILOT_OPERATING_POINT = OperatingPoint(DEFAULT_R, DEFAULT_MIN_FRAC, "pilot")


def parse_sheet(text: str) -> dict[str, str]:
    """``key: value`` lines of an OPERATING-POINT sheet (fidelity_sweep.sheet_template).

    Blank values are returned as "" so a half-filled sheet is visible to the
    caller. Prose lines and headings are ignored.
    """
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*?)\s*$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_operating_point(case_root, *, sheet_path=None, sweep_path=None) -> OperatingPoint:
    """Signed sheet → its (R, min_frac); no sheet or unsigned → pilot fallback.

    Fail closed on a signed sheet that cannot be honoured: R off R_GRID,
    min_frac off MIN_FRAC_GRID, missing date, or ``sweep_csv_sha256`` that no
    longer matches ``fidelity/sweep.csv`` (a signature over a stale or absent
    sweep is not a signature). Those raise FidelityRefusal rather than
    silently serving pilot values under a signed name.
    """
    case_root = Path(case_root)
    sheet = Path(sheet_path) if sheet_path is not None else case_root / SHEET_RELPATH
    if not sheet.is_file():
        return PILOT_OPERATING_POINT
    fields = parse_sheet(sheet.read_text())
    who = fields.get("approved_by", "").strip()
    if not who:
        return PILOT_OPERATING_POINT
    date = fields.get("date", "").strip()
    if not date:
        raise FidelityRefusal(f"{sheet}: approved_by set but date blank")
    try:
        r = float(fields.get("R", ""))
        mf = float(fields.get("min_frac", ""))
    except ValueError:
        raise FidelityRefusal(f"{sheet}: signed sheet needs numeric R and min_frac")
    r_hits = np.where(np.isclose(R_GRID, r, atol=1e-6))[0]
    if r_hits.size != 1:
        raise FidelityRefusal(f"{sheet}: signed R {r} is not on R_GRID")
    if not any(abs(mf - g) < 1e-9 for g in MIN_FRAC_GRID):
        raise FidelityRefusal(f"{sheet}: signed min_frac {mf} is not on MIN_FRAC_GRID")
    sha = fields.get("sweep_csv_sha256", "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise FidelityRefusal(f"{sheet}: signed sheet has no sweep_csv_sha256")
    csv_path = Path(sweep_path) if sweep_path is not None else case_root / "fidelity" / "sweep.csv"
    if not csv_path.is_file():
        raise FidelityRefusal(f"{sheet}: signed over sweep.csv that is absent")
    actual = _sha256_file(csv_path)
    if actual != sha:
        raise FidelityRefusal(
            f"{sheet}: sweep.csv drifted since signing ({actual[:12]} != {sha[:12]})"
        )
    return OperatingPoint(
        R=round(float(R_GRID[int(r_hits[0])]), 2), min_frac=mf, source="signed",
        approved_by=who, date=date, sweep_csv_sha256=sha,
    )

FLAG_CROSSES_LESION = 1  # bit0
FLAG_CROSSES_CAVITY = 2  # bit1
FLAG_PROVENANCE_INCOMPLETE = 4  # bit2


class SidecarData:
    """Loaded sidecar rows; construct only via load_sidecar/rows_for."""

    __slots__ = ("schema", "bank_sha256", "fod_sha256", "sh_load_sha256",
                 "peak_sha256", "R_grid", "frac_ge", "p5_ratio", "n_segments",
                 "flags", "n", "ratios_present")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw[k])


def build_sidecar(
    tck_path,
    sh_path,
    peak_path,
    *,
    provenance,
    lesion_mask=None,
    cavity_mask=None,
    grid=None,
    sh_load_path=None,
    assume_unknown_spatial_units_mm: bool = False,
) -> dict:
    """Compute one bank's sidecar arrays from SOURCE geometry.

    Provenance-incomplete banks get flags bit2 and NO ratio arrays — the
    refusal is recorded, never guessed around. Complete banks go through the
    same SHA and physical-unit gates as summaries_for_bank. Unknown NIfTI units
    require the explicit opt-in ``assume_unknown_spatial_units_mm=True``.
    """
    import nibabel as nib

    tck = nib.streamlines.load(str(tck_path))
    lines = [np.asarray(s, dtype=np.float64) for s in tck.streamlines]
    n = len(lines)

    bank_sha = None
    if isinstance(provenance, dict):
        bank_sha = provenance.get("bank_sha256")
        fod_sha = provenance.get("fod_sha256")
    else:
        bank_sha = getattr(provenance, "bank_sha256", None)
        fod_sha = getattr(provenance, "fod_sha256", None) if provenance else None

    flags = np.zeros((n,), dtype=np.uint8)
    if grid is not None:
        from .traversal import segment_voxel_hits

        for i, line in enumerate(lines):
            if lesion_mask is not None and segment_voxel_hits(line, lesion_mask, grid):
                flags[i] |= FLAG_CROSSES_LESION
            if cavity_mask is not None and segment_voxel_hits(line, cavity_mask, grid):
                flags[i] |= FLAG_CROSSES_CAVITY

    if provenance is None or not fod_sha or not bank_sha:
        flags |= FLAG_PROVENANCE_INCOMPLETE
        return {
            "schema": SIDECAR_SCHEMA,
            "bank_sha256": bank_sha or "",
            "fod_sha256": "",
            "sh_load_sha256": "",
            "peak_sha256": "",
            "R_grid": R_GRID.copy(),
            "frac_ge": np.empty((0, len(R_GRID)), dtype=np.float32),
            "p5_ratio": np.empty((0,), dtype=np.float32),
            "n_segments": np.empty((0,), dtype=np.uint16),
            "flags": flags,
            "ratios_present": False,
        }

    # Identity gate: the bank file itself must hash to the manifest claim —
    # trust-the-manifest identity was a reviewed bypass (Grok batch-4 #4).
    import hashlib as _hashlib

    h = _hashlib.sha256()
    with open(tck_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    actual_bank = h.hexdigest()
    if actual_bank != bank_sha:
        raise FidelityRefusal(
            f"bank sha mismatch: manifest {str(bank_sha)[:12]}… vs disk "
            f"{actual_bank[:12]}…"
        )

    summ = summaries_for_bank(
        tck_path, sh_path, peak_path,
        provenance=provenance,
        sh_load_path=sh_load_path,
        assume_unknown_spatial_units_mm=assume_unknown_spatial_units_mm,
    )
    if summ["n"] != n:
        raise FidelityRefusal("streamline count changed between reads")

    # Bind the sidecar to the ACTUAL files read for amplitude/peak data (S-05):
    # sh_load_path may be a converted nibabel-readable scratch copy of a .mif
    # FOD, never itself sha-pinned above. Recording its live bytes here means
    # a later swap of that conversion is detectable even though the pinned
    # sh_path (fod_sha256) never changed.
    from .evidence_identity import sha256_file as _sha256_file_live

    sh_load_sha = _sha256_file_live(sh_load_path if sh_load_path is not None else sh_path)
    peak_sha = _sha256_file_live(peak_path)

    sh_nii = _verified_sh_image(sh_path, sh_load_path)
    peak_nii = nib.load(str(peak_path))
    try:
        sh_grid = grid_from_image(
            sh_nii,
            source="SH source",
            require_mm=True,
            assume_unknown_spatial_units_mm=assume_unknown_spatial_units_mm,
            require_authoritative_affine=True,
        )
        peak_grid = grid_from_image(
            peak_nii,
            source="peak source",
            require_mm=True,
            assume_unknown_spatial_units_mm=assume_unknown_spatial_units_mm,
            require_authoritative_affine=True,
        )
        assert_grids_match(sh_grid, peak_grid, context="SH/peak grid")
    except ValueError as exc:
        raise FidelityRefusal(str(exc)) from exc
    sh_img = sh_nii.get_fdata()
    peak_img = peak_nii.get_fdata()
    frac = np.full((n, len(R_GRID)), np.nan, dtype=np.float32)
    n_segments = np.zeros((n,), dtype=np.uint16)
    for i, line in enumerate(lines):
        r = segment_ratios(line, sh_img, sh_nii.affine, peak_img, peak_nii.affine)
        n_segments[i] = min(r.shape[0], 65535)
        if r.shape[0] and np.isfinite(r).all():
            frac[i] = (r[:, None] >= R_GRID[None, :]).mean(axis=0)
        # else: NaN row — unmeasurable, not zero-support
    return {
        "schema": SIDECAR_SCHEMA,
        "bank_sha256": bank_sha,
        "fod_sha256": summ["fod_sha256"],
        "sh_load_sha256": sh_load_sha,
        "peak_sha256": peak_sha,
        "R_grid": R_GRID.copy(),
        "frac_ge": frac,
        "p5_ratio": summ["p5_ratio"],
        "n_segments": n_segments,
        "flags": flags,
        "ratios_present": True,
    }


def save_sidecar(path, data: dict) -> None:
    """Deterministic npz: rebuilds are byte-identical on the same platform.

    Determinism levers: ZIP_STORED (no zlib version drift), fixed timestamps,
    pinned create_system, SHAs stored as uint8 bytes (no <U/>U endian split),
    sorted member order. Residual platform dependence: numpy array byte order
    (little-endian on every supported host) — cross-platform identity is NOT
    claimed.
    """
    import io
    import zipfile

    def _sha_bytes(s) -> np.ndarray:
        return np.frombuffer(str(s).encode("ascii"), dtype=np.uint8).copy()

    arrays = {
        "schema": np.asarray(data["schema"], dtype=np.int64),
        "bank_sha256": _sha_bytes(data["bank_sha256"]),
        "fod_sha256": _sha_bytes(data["fod_sha256"]),
        "sh_load_sha256": _sha_bytes(data.get("sh_load_sha256") or ""),
        "peak_sha256": _sha_bytes(data.get("peak_sha256") or ""),
        "R_grid": np.asarray(data["R_grid"], dtype=np.float32),
        "frac_ge": np.asarray(data["frac_ge"], dtype=np.float32),
        "p5_ratio": np.asarray(data["p5_ratio"], dtype=np.float32),
        "n_segments": np.asarray(data["n_segments"], dtype=np.uint16),
        "flags": np.asarray(data["flags"], dtype=np.uint8),
        "ratios_present": np.asarray(bool(data["ratios_present"])),
    }
    with zipfile.ZipFile(str(path), "w", zipfile.ZIP_STORED) as z:
        for k in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, arrays[k], allow_pickle=False)
            zi = zipfile.ZipInfo(k + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.create_system = 3  # pinned: unix, not host-dependent
            zi.external_attr = 0o644 << 16
            z.writestr(zi, buf.getvalue())


def load_sidecar(path, **expected) -> SidecarData:
    import zipfile
    try:
        return _load_sidecar_checked(path, **expected)
    except (OSError, KeyError, ValueError, IndexError, UnicodeError, zipfile.BadZipFile) as exc:
        raise FidelityRefusal("malformed or unreadable fidelity sidecar") from exc


def _load_sidecar_checked(path, *, expected_bank_sha: str,
                 expected_fod_sha: str | None = None,
                 expected_sh_load_sha: str | None = None,
                 expected_peak_sha: str | None = None) -> SidecarData:
    """Load + verify a sidecar. Any mismatch refuses — never a partial join.

    expected_fod_sha: pass the CURRENT bytes hash of the pinned FOD file at
    serve time (not a declared manifest string) so a sidecar built against a
    replaced FOD cannot join (Sol batch-4 / S-05).

    expected_sh_load_sha/expected_peak_sha: optional binding (schema 2) to
    the actual SH-conversion and peak files used to BUILD the sidecar. Pass
    these when the caller can recompute current bytes for those files too;
    omitting them (None) skips that check without weakening the mandatory
    schema/bank/fod gates above.

    A schema-1 sidecar (built before this binding existed) always refuses
    here — schema advancement is an explicit refusal, never a silent
    reinterpretation of old data under the new contract.
    """
    with np.load(str(path), allow_pickle=False) as z:
        schema = int(z["schema"])
        if schema != SIDECAR_SCHEMA:
            raise FidelityRefusal(
                f"sidecar schema {schema} != {SIDECAR_SCHEMA} — "
                "rebuild with the current prep_fidelity before serving"
            )
        bank_sha = bytes(z["bank_sha256"]).decode("ascii")
        if bank_sha != expected_bank_sha:
            raise FidelityRefusal(
                f"sidecar bank sha mismatch: {bank_sha[:12]}… vs expected "
                f"{expected_bank_sha[:12]}…"
            )
        r_grid = z["R_grid"].astype(np.float32)
        if not np.array_equal(r_grid, R_GRID):
            raise FidelityRefusal("sidecar R_grid differs from the code grid")
        fod_sha = bytes(z["fod_sha256"]).decode("ascii")
        if expected_fod_sha is not None and fod_sha != expected_fod_sha:
            raise FidelityRefusal(
                f"sidecar FOD sha mismatch: {fod_sha[:12]}… vs expected "
                f"{expected_fod_sha[:12]}…"
            )
        sh_load_sha = bytes(z["sh_load_sha256"]).decode("ascii")
        if expected_sh_load_sha is not None and sh_load_sha != expected_sh_load_sha:
            raise FidelityRefusal(
                f"sidecar SH-load sha mismatch: {sh_load_sha[:12]}… vs "
                f"expected {expected_sh_load_sha[:12]}…"
            )
        peak_sha = bytes(z["peak_sha256"]).decode("ascii")
        if expected_peak_sha is not None and peak_sha != expected_peak_sha:
            raise FidelityRefusal(
                f"sidecar peak sha mismatch: {peak_sha[:12]}… vs expected "
                f"{expected_peak_sha[:12]}…"
            )
        frac_ge = z["frac_ge"]
        p5 = z["p5_ratio"]
        n_segments = z["n_segments"]
        flags = z["flags"]
        ratios_present = bool(z["ratios_present"])
        if flags.ndim != 1 or flags.dtype != np.dtype('uint8') or (flags & ~np.uint8(7)).any():
            raise FidelityRefusal("sidecar flags must be a vector of known uint8 flags")
        n = int(flags.shape[0])
        if ratios_present and (frac_ge.shape != (n, len(R_GRID)) or p5.shape != (n,)
                               or n_segments.shape != (n,)):
            raise FidelityRefusal(
                f"sidecar array length mismatch: flags n={n}, "
                f"frac_ge {frac_ge.shape}, p5 {p5.shape}"
            )
        if not ratios_present and (frac_ge.shape[0] != 0 or p5.shape[0] != 0):
            raise FidelityRefusal("ratios_present=False but ratio arrays non-empty")
        if ratios_present and (not np.issubdtype(frac_ge.dtype, np.floating)
                               or not np.issubdtype(p5.dtype, np.floating)
                               or not re.fullmatch(r'[0-9a-f]{64}', sh_load_sha)
                               or not re.fullmatch(r'[0-9a-f]{64}', peak_sha)):
            raise FidelityRefusal("sidecar support arrays or source digests are invalid")
        if ratios_present:
            # An all-NaN fraction row paired with NaN p5 is the builder's
            # explicit unmeasurable state. Other invalid values are corruption.
            unknown = np.isnan(frac_ge).all(axis=1) & np.isnan(p5)
            measured = np.isfinite(frac_ge).all(axis=1) & np.isfinite(p5)
            valid_domain = ((frac_ge >= 0) & (frac_ge <= 1)).all(axis=1) & (p5 >= 0)
            if not np.all(unknown | (measured & valid_domain)):
                raise FidelityRefusal("sidecar support values are invalid")
            if n_segments.dtype != np.dtype('uint16') or np.any(measured & (n_segments == 0)):
                raise FidelityRefusal("sidecar segment counts are invalid")
        return SidecarData(
            schema=schema, bank_sha256=bank_sha, fod_sha256=fod_sha,
            sh_load_sha256=sh_load_sha, peak_sha256=peak_sha,
            R_grid=r_grid, frac_ge=frac_ge, p5_ratio=p5,
            n_segments=n_segments, flags=flags, n=n,
            ratios_present=ratios_present,
        )


def rows_for(sidecar: SidecarData, ordinals: np.ndarray) -> SidecarData:
    """Exact row selection in the caller's order; out-of-range refuses."""
    idx = np.asarray(ordinals, dtype=np.int64)
    if idx.size and (idx.min() < 0 or idx.max() >= sidecar.n):
        raise FidelityRefusal(
            f"ordinal out of range for sidecar n={sidecar.n}: "
            f"[{idx.min()}, {idx.max()}]"
        )
    return SidecarData(
        schema=sidecar.schema,
        bank_sha256=sidecar.bank_sha256,
        fod_sha256=sidecar.fod_sha256,
        sh_load_sha256=sidecar.sh_load_sha256,
        peak_sha256=sidecar.peak_sha256,
        R_grid=sidecar.R_grid,
        frac_ge=sidecar.frac_ge[idx] if sidecar.ratios_present else sidecar.frac_ge,
        p5_ratio=sidecar.p5_ratio[idx] if sidecar.ratios_present else sidecar.p5_ratio,
        n_segments=(sidecar.n_segments[idx] if sidecar.ratios_present
                    else sidecar.n_segments),
        flags=sidecar.flags[idx],
        n=int(idx.size),
        ratios_present=sidecar.ratios_present,
    )
