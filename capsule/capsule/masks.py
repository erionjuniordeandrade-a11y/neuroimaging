"""Masks on the common grid: display-only render masks (CT/MR head, MR brain, MR vessels) and RTSTRUCT lesions.

Render masks never change voxel values; the viewer uses them only to hide voxels outside the
mask in its 3D render copy of one volume. Hole filling below only closes regions fully enclosed
by the mask (per axial slice, then in 3D); it cannot grow the mask outward and never changes a
scalar value.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Callable

import numpy as np
import pydicom
import SimpleITK as sitk

from .resample import affine_ras


DEFAULT_BRAIN_VOLUME_GATE = (900.0, 1800.0)


def _image(array: np.ndarray, grid: sitk.Image) -> sitk.Image:
    image = sitk.GetImageFromArray(array.astype(np.uint8))
    image.CopyInformation(grid)
    return image


def largest_component(mask: np.ndarray, grid: sitk.Image) -> np.ndarray:
    if not mask.any():
        return mask.astype(bool)
    labels = sitk.GetArrayFromImage(sitk.RelabelComponent(sitk.ConnectedComponent(_image(mask, grid))))
    return labels == 1


def fill_enclosed(mask: np.ndarray, grid: sitk.Image) -> np.ndarray:
    """Fill holes per slice along array axis 0 (axial for an axial reference), then in 3D."""
    filled = mask.astype(np.uint8).copy()
    for k in range(filled.shape[0]):
        if filled[k].any():
            filled[k] = sitk.GetArrayFromImage(sitk.BinaryFillhole(sitk.GetImageFromArray(filled[k])))
    return sitk.GetArrayFromImage(sitk.BinaryFillhole(_image(filled, grid))).astype(bool)


def _opening(mask: np.ndarray, grid: sitk.Image, radius_mm: float) -> np.ndarray:
    radius = [max(1, int(round(radius_mm / s))) for s in grid.GetSpacing()]
    return sitk.GetArrayFromImage(sitk.BinaryMorphologicalOpening(_image(mask, grid), radius, sitk.sitkBall)).astype(bool)


def signal_outside_brain_fraction(values: np.ndarray, brain: np.ndarray, grid: sitk.Image,
                                  margin_mm: float = 3.0, level: float = 0.1) -> float:
    """Voxels beyond a margin around the brain with signal above `level` x the brain median, per brain voxel.
    ~0 on a brain-extracted MR (nothing but background outside the brain); scalp and skull give tens of %.
    Unlike the head-mask volume, this does not depend on log-Otsu finding the head (it fails on some T2)."""
    brain = brain.astype(bool)
    if not brain.any():
        return 0.0
    radius = [max(1, int(round(margin_mm / s))) for s in grid.GetSpacing()]
    near = sitk.GetArrayFromImage(sitk.BinaryDilate(_image(brain, grid), radius, sitk.sitkBall)).astype(bool)
    background = float(np.min(values))
    cut = background + level * (float(np.median(values[brain])) - background)
    return float(np.count_nonzero((values > cut) & ~near) / np.count_nonzero(brain))


def ct_head(hu: np.ndarray, grid: sitk.Image, opening_mm: float = 3.0) -> np.ndarray:
    """Head without table: HU > -500, 3 mm ball opening, largest component, filled."""
    head = largest_component(_opening(hu > -500, grid, opening_mm), grid)
    return fill_enclosed(head, grid)


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    histogram, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    weight = np.cumsum(histogram).astype(float)
    total = weight[-1]
    mean = np.cumsum(histogram * centers)
    between = (mean[-1] * weight - mean * total) ** 2 / np.maximum(weight * (total - weight), 1e-12)
    return float(centers[np.argmax(between[:-1])])


def mr_head_threshold(values: np.ndarray) -> float:
    """Otsu on log intensity of the positive voxels: separates noise/air from all tissue, so dark-skin
    sequences (T2) keep the scalp; zero padding outside the acquired field of view is ignored."""
    positive = values[values > 0]
    return float(np.expm1(otsu_threshold(np.log1p(positive)))) if positive.size else 0.0


def mr_head(values: np.ndarray, grid: sitk.Image, opening_mm: float = 2.0) -> np.ndarray:
    """Head on MR: log-Otsu foreground, opened (detaches frame fiducials), largest component, filled."""
    foreground = values > mr_head_threshold(values)
    return fill_enclosed(largest_component(_opening(foreground, grid, opening_mm), grid), grid)


# SynthStrip (FreeSurfer 7.4 model) is run from its Python script under uv with pinned
# packages because the bundled fspython is x86-only on this machine.
SYNTHSTRIP_PACKAGES = ["torch==2.14.0", "surfa==0.6.3", "numpy==1.26.4"]


def vessel_mask(values: np.ndarray, brain: np.ndarray, head: np.ndarray, grid: sitk.Image,
                threshold: float = 10.0, margin_mm: float = 4.0, min_voxels: int = 40,
                sigmas_mm: tuple[float, ...] = (0.8, 1.2, 1.8)) -> tuple[np.ndarray, dict]:
    """Bright tubular structures (enhancing vessels, sinuses) in the brain plus a margin, display only.

    Multiscale Hessian objectness (SimpleITK ObjectnessMeasure, lines, bright) on the volume divided by
    its in-brain median x 100, so the threshold does not depend on scanner units; components smaller
    than min_voxels are dropped. A non-contrast T1 yields almost nothing, which is the honest answer.
    """
    started = time.time()
    brain = brain.astype(bool)
    record = {"method": f"hessian-objectness(sigmas={list(sigmas_mm)}mm)>{threshold}+cc>={min_voxels}",
              "margin_mm": margin_mm}
    if not brain.any():
        return np.zeros_like(brain), {**record, "seconds": 0.0, "components": 0}
    median = float(np.median(values[brain])) or 1.0
    image = sitk.GetImageFromArray((values.astype(np.float32) / median) * 100.0)
    image.CopyInformation(grid)
    radius = [max(1, int(round(margin_mm / s))) for s in grid.GetSpacing()]
    region = sitk.GetArrayFromImage(sitk.BinaryDilate(_image(brain, grid), radius, sitk.sitkBall)).astype(bool)
    region &= head.astype(bool)
    objectness = None
    for sigma in sigmas_mm:
        smoothed = sitk.SmoothingRecursiveGaussian(image, sigma)
        measure = sitk.GetArrayFromImage(sitk.ObjectnessMeasure(smoothed, objectDimension=1, brightObject=True,
                                                                 scaleObjectnessMeasure=True, alpha=0.5, beta=0.5, gamma=5.0))
        objectness = measure if objectness is None else np.maximum(objectness, measure)
    candidate = (objectness > threshold) & region
    labels = sitk.RelabelComponent(sitk.ConnectedComponent(_image(candidate, grid), True), minimumObjectSize=min_voxels)
    array = sitk.GetArrayFromImage(labels)
    return array > 0, {**record, "seconds": round(time.time() - started, 2), "components": int(array.max())}


def _cta_statistics(mask: np.ndarray, native: np.ndarray, grid: sitk.Image) -> dict:
    labels = sitk.GetArrayFromImage(sitk.ConnectedComponent(_image(mask, grid), True))
    sizes = np.bincount(labels.ravel())[1:]
    components = int(len(sizes))
    largest = float(sizes.max() / mask.sum()) if mask.any() else 0.0
    bone = _image(native > 200.0, grid)
    distance = sitk.GetArrayFromImage(sitk.SignedMaurerDistanceMap(
        bone, insideIsPositive=False, squaredDistance=False, useImageSpacing=True))
    voxel_ml = float(np.prod(grid.GetSpacing())) / 1000.0
    near_bone = np.abs(distance) <= 2.0
    return {"components": components, "largest_component_fraction": round(largest, 3),
            "ml_within_2mm_of_bone": round(float(np.count_nonzero(mask & near_bone)) * voxel_ml, 3)}


def cta_vessel_mask_r7(contrast: np.ndarray, native: np.ndarray, head: np.ndarray, grid: sitk.Image,
                       enhancement_hu: float = 60.0, contrast_hu: float = 120.0, bone_hu: float = 200.0,
                       bone_margin_mm: float = 1.25, sigma_mm: float = 0.5,
                       min_mm3: float = 100.0) -> tuple[np.ndarray, dict]:
    """The r7 bone-exclusion method, retained unchanged as a comparison control."""
    started = time.time()
    spacing = float(grid.GetSpacing()[0])
    record = {"method": f"bone-masked-subtraction(enh>{enhancement_hu:g}HU,ct>{contrast_hu:g}HU,"
                        f"bone>{bone_hu:g}HU+{bone_margin_mm:g}mm,gauss{sigma_mm:g}mm)+cc>={min_mm3:g}mm3"}

    def smooth(values: np.ndarray) -> np.ndarray:
        image = sitk.GetImageFromArray(values.astype(np.float32))
        image.CopyInformation(grid)
        return sitk.GetArrayFromImage(sitk.SmoothingRecursiveGaussian(image, sigma_mm))

    after, before = smooth(contrast), smooth(native)
    radius = [max(1, int(round(bone_margin_mm / s))) for s in grid.GetSpacing()]
    bone = sitk.GetArrayFromImage(sitk.BinaryDilate(_image(before > bone_hu, grid), radius, sitk.sitkBall)).astype(bool)
    candidate = (after - before > enhancement_hu) & (after > contrast_hu) & ~bone & head.astype(bool)
    min_voxels = max(1, int(round(min_mm3 / spacing ** 3)))
    labels = sitk.RelabelComponent(sitk.ConnectedComponent(_image(candidate, grid), True), minimumObjectSize=min_voxels)
    array = sitk.GetArrayFromImage(labels)
    mask = array > 0
    return mask, {**record, "seconds": round(time.time() - started, 2), **_cta_statistics(mask, native, grid)}


def cta_vessel_mask(contrast: np.ndarray, native: np.ndarray, head: np.ndarray, grid: sitk.Image,
                    enhancement_hu: float = 60.0, contrast_hu: float = 120.0, bone_hu: float = 200.0,
                    radius_voxels: int = 1, sigma_mm: float = 0.5, min_mm3: float = 100.0) -> tuple[np.ndarray, dict]:
    """r7 bone-masked subtraction united with a min-max subtraction that reaches into bone canals.

    Min-max core: the angiogram eroded (local minimum over a box of `radius_voxels`) must exceed the native CT
    dilated (local maximum over the same box) by `enhancement_hu`. A bone edge shifted by up to `radius_voxels`
    between phases cannot pass, in either direction, so no bone exclusion zone is needed and a lumen touching a
    canal wall or the inner table is kept. The core is regrown by the same box into voxels that still enhance over
    the 1-voxel native maximum by half the threshold. The r7 mask (which keeps small branches away from bone) is
    added, and components below `min_mm3` are dropped. Swapping the phases yields nothing on both halves.
    """
    started = time.time()
    spacing = float(grid.GetSpacing()[0])
    box = [int(radius_voxels)] * 3
    r7, _ = cta_vessel_mask_r7(contrast, native, head, grid, enhancement_hu=enhancement_hu,
                               contrast_hu=contrast_hu, bone_hu=bone_hu, sigma_mm=sigma_mm, min_mm3=min_mm3)

    def image(values: np.ndarray) -> sitk.Image:
        result = sitk.GetImageFromArray(values.astype(np.float32))
        result.CopyInformation(grid)
        return result

    after_min = sitk.GetArrayFromImage(sitk.GrayscaleErode(
        sitk.SmoothingRecursiveGaussian(image(contrast), sigma_mm), box, sitk.sitkBox))
    before_max = sitk.GetArrayFromImage(sitk.GrayscaleDilate(
        sitk.SmoothingRecursiveGaussian(image(native), sigma_mm), box, sitk.sitkBox))
    before_max1 = sitk.GetArrayFromImage(sitk.GrayscaleDilate(image(native), [1, 1, 1], sitk.sitkBox))
    inside = head.astype(bool) & (contrast > contrast_hu)
    core = (after_min - before_max > enhancement_hu) & inside
    grown = sitk.GetArrayFromImage(sitk.BinaryDilate(_image(core, grid), box, sitk.sitkBox)).astype(bool)
    # Not clipped to the head mask: the core already lies inside it, and the head mask can miss a canal wall
    # voxel (on the pilot CTA that 0.02 mL gap cut the right ICA's rise into the skull base from 27.6 to 5.6 mm).
    grown &= (contrast > contrast_hu) & (contrast - before_max1 > enhancement_hu / 2.0)
    min_voxels = max(1, int(round(min_mm3 / spacing ** 3)))
    labels = sitk.RelabelComponent(sitk.ConnectedComponent(_image(r7 | grown, grid), True), minimumObjectSize=min_voxels)
    mask = sitk.GetArrayFromImage(labels) > 0
    record = {"method": f"r7 + minmax-subtraction(box{radius_voxels}vox,enh>{enhancement_hu:g}HU,"
                        f"regrow>{enhancement_hu / 2.0:g}HU,ct>{contrast_hu:g}HU,gauss{sigma_mm:g}mm)"
                        f"+cc>={min_mm3:g}mm3",
              "seconds": round(time.time() - started, 2)}
    return mask, {**record, **_cta_statistics(mask, native, grid)}


def _freesurfer_home() -> Path:
    return Path(os.environ.get("FREESURFER_HOME", Path.home() / "freesurfer"))


def _fsl_home() -> Path:
    return Path(os.environ.get("FSLDIR", Path.home() / "fsl"))


def _synthstrip_command(source: Path, target: Path) -> list[str] | None:
    home = _freesurfer_home()
    script = home / "python" / "scripts" / "mri_synthstrip"
    uv = shutil.which("uv")
    if not script.is_file() or not (home / "models" / "synthstrip.1.pt").is_file() or not uv:
        return None
    command = [uv, "run", "--no-project", "--python", "3.12"]
    for package in SYNTHSTRIP_PACKAGES:
        command += ["--with", package]
    return command + ["python", str(script), "-i", str(source), "-m", str(target)]


def _bet_command(source: Path, target: Path) -> list[str] | None:
    bet = _fsl_home() / "bin" / "bet"
    if not bet.is_file():
        return None
    return [str(bet), str(source), str(target.with_name("bet")), "-m", "-f", "0.4", "-R"]


class BrainMaskQCError(RuntimeError):
    """No brain-mask method produced a candidate within the requested volume gate."""

    def __init__(self, attempts: list[dict], qc: dict):
        self.attempts = attempts
        self.qc = qc
        super().__init__("brain mask failed the volume gate")


def _volume_gate(volume_gate: tuple[float, float] | None) -> tuple[float, float] | None:
    if volume_gate is None:
        return None
    minimum, maximum = (float(value) for value in volume_gate)
    if not (0 < minimum <= maximum):
        raise ValueError("brain volume gate must be a positive MIN,MAX range")
    return minimum, maximum


def brain_mask_from_file(path: str | Path, grid: sitk.Image, method: str = "synthstrip", *,
                         volume_gate: tuple[float, float] | None = DEFAULT_BRAIN_VOLUME_GATE) -> tuple[np.ndarray, dict]:
    """Gate a brain mask a pipeline already computed (e.g. the DWI ACT SynthStrip mask) instead of re-running it.

    Same resample, volume and QC record as brain_mask; the attempt is marked source "file".
    """
    if method not in {"synthstrip", "bet"}:
        raise ValueError("brain mask method must be synthstrip or bet")
    gate = _volume_gate(volume_gate)
    gate_ml = list(gate) if gate is not None else None
    mask = sitk.ReadImage(str(path))
    if mask.GetDimension() != 3:
        raise ValueError("--brain-mask-file must be a 3D image")
    resampled = sitk.Resample(mask, grid, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    array = sitk.GetArrayFromImage(resampled) > 0
    volume_ml = round(float(array.sum()) * float(np.prod(grid.GetSpacing())) / 1000.0, 3)
    verdict = "FAIL" if not array.any() else \
        "NOT_GATED" if gate is None else "PASS" if gate[0] <= volume_ml <= gate[1] else "FAIL"
    attempts = [{"method": method, "source": "file", "volume_ml": volume_ml, "verdict": verdict}]
    qc = {"gate_ml": gate_ml, "attempts": attempts, "verdict": verdict}
    if verdict == "FAIL":
        raise BrainMaskQCError(attempts, qc)
    return array, {"method": method, "seconds": 0.0,
                   "packages": SYNTHSTRIP_PACKAGES if method == "synthstrip" else None, "qc": qc}


def brain_mask(values: np.ndarray, grid: sitk.Image, method: str = "synthstrip", *,
               volume_gate: tuple[float, float] | None = DEFAULT_BRAIN_VOLUME_GATE,
               command_runner: Callable[[list[str]], subprocess.CompletedProcess] | None = None) -> tuple[np.ndarray, dict]:
    """Brain (with CSF) render mask on the grid via SynthStrip, falling back to FSL BET.

    Returns the first nonempty in-gate candidate and its ordered QC record.
    """
    if method not in {"synthstrip", "bet"}:
        raise ValueError("brain mask method must be synthstrip or bet")
    gate = _volume_gate(volume_gate)
    order = {"synthstrip": ["synthstrip", "bet"], "bet": ["bet"]}[method]
    image = sitk.GetImageFromArray(values.astype(np.float32))
    image.CopyInformation(grid)
    attempts: list[dict] = []
    gate_ml = list(gate) if gate is not None else None
    voxel_ml = float(np.prod(grid.GetSpacing())) / 1000.0
    with tempfile.TemporaryDirectory(prefix="case-capsule-brain-") as temporary:
        work = Path(temporary)
        source = work / "input.nii.gz"
        sitk.WriteImage(image, str(source))
        for name in order:
            target = work / f"{name}_mask.nii.gz"
            command = (_synthstrip_command if name == "synthstrip" else _bet_command)(source, target)
            if command is None:
                attempts.append({"method": name, "volume_ml": None, "verdict": "FAIL"})
                continue
            env = dict(os.environ, FREESURFER_HOME=str(_freesurfer_home())) if name == "synthstrip" else \
                dict(os.environ, FSLDIR=str(_fsl_home()), FSLOUTPUTTYPE="NIFTI_GZ")
            started = time.monotonic()
            run = command_runner(command) if command_runner else \
                subprocess.run(command, capture_output=True, text=True, env=env, check=False)
            seconds = round(time.monotonic() - started, 1)
            produced = target if name == "synthstrip" else work / "bet_mask.nii.gz"
            if run.returncode or not produced.is_file():
                attempts.append({"method": name, "volume_ml": None, "verdict": "FAIL"})
                continue
            mask = sitk.ReadImage(str(produced))
            resampled = sitk.Resample(mask, grid, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
            array = sitk.GetArrayFromImage(resampled) > 0
            volume_ml = round(float(array.sum()) * voxel_ml, 3)
            if not array.any():
                attempts.append({"method": name, "volume_ml": volume_ml, "verdict": "FAIL"})
                continue
            verdict = "NOT_GATED" if gate is None else "PASS" if gate[0] <= volume_ml <= gate[1] else "FAIL"
            attempts.append({"method": name, "volume_ml": volume_ml, "verdict": verdict})
            if verdict == "FAIL":
                continue
            qc = {"gate_ml": gate_ml, "attempts": attempts, "verdict": verdict}
            return array, {"method": name, "seconds": seconds,
                           "packages": SYNTHSTRIP_PACKAGES if name == "synthstrip" else None,
                           "qc": qc}
    qc = {"gate_ml": gate_ml, "attempts": attempts, "verdict": "FAIL"}
    raise BrainMaskQCError(attempts, qc)


# ---------------------------------------------------------------- RTSTRUCT


@dataclass
class Structure:
    name: str
    referenced_series: set[str]
    contours: list[np.ndarray]  # each (n, 3) LPS mm, CLOSED_PLANAR


def read_rtstruct(path: str | Path, roi: str | None = None) -> Structure:
    ds = pydicom.dcmread(path)
    if str(getattr(ds, "Modality", "")) != "RTSTRUCT":
        raise ValueError("--rtstruct is not an RTSTRUCT")
    names = {int(item.ROINumber): str(item.ROIName) for item in ds.StructureSetROISequence}
    if roi is None:
        candidates = [n for n in names.values() if not n.startswith("*")]
        if len(candidates) != 1:
            # VS-SEG convention: the tumour ROI is "AN" (acoustic neuroma) or "TV".
            candidates = [n for n in candidates if n.upper() in {"AN", "TV", "GTV", "TUMOUR", "TUMOR"}]
        if len(candidates) != 1:
            raise ValueError(f"--rtstruct-roi required; ROIs: {sorted(names.values())}")
        roi = candidates[0]
    numbers = [number for number, name in names.items() if name == roi]
    if len(numbers) != 1:
        raise ValueError(f"ROI {roi!r} not found; ROIs: {sorted(names.values())}")
    refs = set()
    for frame in ds.get("ReferencedFrameOfReferenceSequence", []):
        for study in frame.get("RTReferencedStudySequence", []):
            for series in study.get("RTReferencedSeriesSequence", []):
                refs.add(str(series.SeriesInstanceUID))
    contours = []
    for item in ds.ROIContourSequence:
        if int(item.ReferencedROINumber) != numbers[0]:
            continue
        for contour in item.get("ContourSequence", []):
            if str(contour.ContourGeometricType) != "CLOSED_PLANAR":
                raise ValueError(f"unsupported contour type {contour.ContourGeometricType}")
            contours.append(np.asarray(contour.ContourData, dtype=float).reshape(-1, 3))
    if not contours:
        raise ValueError(f"ROI {roi!r} has no contours")
    return Structure(roi, refs, contours)


def _inside_polygon(px: np.ndarray, py: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Even-odd rule for points (px, py) against a closed polygon (n, 2)."""
    inside = np.zeros(px.shape, dtype=bool)
    x0, y0 = polygon[-1]
    for x1, y1 in polygon:
        crosses = (y1 > py) != (y0 > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = (x0 - x1) * (py - y1) / (y0 - y1) + x1
        inside ^= crosses & (px < x_at)
        x0, y0 = x1, y1
    return inside


def rasterize(structure: Structure, image: sitk.Image) -> np.ndarray:
    """Rasterize on the native image grid (array [k, j, i]); multiple contours on a slice XOR (holes)."""
    size = image.GetSize()
    mask = np.zeros(size[::-1], dtype=bool)
    for contour in structure.contours:
        index = np.array([image.TransformPhysicalPointToContinuousIndex(tuple(p)) for p in contour])
        k = np.round(index[:, 2])
        if np.max(np.abs(index[:, 2] - k)) > 0.1 or len(set(k)) != 1:
            raise ValueError("contour is not on an image slice (IPP/IOP mismatch)")
        k = int(k[0])
        if not 0 <= k < size[2]:
            continue
        i0, j0 = np.floor(index[:, :2].min(axis=0)).astype(int)
        i1, j1 = np.ceil(index[:, :2].max(axis=0)).astype(int)
        i0, j0 = max(i0, 0), max(j0, 0)
        i1, j1 = min(i1, size[0] - 1), min(j1, size[1] - 1)
        jj, ii = np.mgrid[j0:j1 + 1, i0:i1 + 1]
        mask[k, j0:j1 + 1, i0:i1 + 1] ^= _inside_polygon(ii.astype(float), jj.astype(float), index[:, :2])
    return mask


def planar_volume_ml(structure: Structure, image: sitk.Image) -> float:
    """Independent estimate: sum of shoelace areas (signed XOR not applied) × slice spacing."""
    direction = np.asarray(image.GetDirection()).reshape(3, 3)
    normal = direction[:, 2]
    area = 0.0
    for contour in structure.contours:
        # area vector of a planar polygon, projected on the slice normal
        cross = np.cross(contour, np.roll(contour, -1, axis=0)).sum(axis=0) / 2.0
        area += abs(float(cross @ normal))
    return area * image.GetSpacing()[2] / 1000.0


def contour_centroid_lps(structure: Structure) -> np.ndarray:
    return np.concatenate(structure.contours).mean(axis=0)


def to_grid(mask: np.ndarray, image: sitk.Image, grid: sitk.Image, transform: sitk.Transform) -> np.ndarray:
    native = sitk.GetImageFromArray(mask.astype(np.float32))
    native.CopyInformation(image)
    return sitk.GetArrayFromImage(sitk.Resample(native, grid, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)) >= 0.5


def voxel_ras(grid: sitk.Image, indices_kji: np.ndarray) -> np.ndarray:
    """RAS mm of array indices (n, 3) given as [k, j, i]."""
    affine = np.asarray(affine_ras(grid))
    ijk = indices_kji[:, ::-1].astype(float)
    return ijk @ affine[:3, :3].T + affine[:3, 3]


def ras_axis_grid(grid: sitk.Image, axis: int) -> np.ndarray:
    """RAS coordinate `axis` (0=x R, 1=y A, 2=z S) of every grid voxel, shaped like the array."""
    affine = np.asarray(affine_ras(grid))
    nx, ny, nz = grid.GetSize()
    k, j, i = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    row = affine[axis]
    return (row[0] * i + row[1] * j + row[2] * k + row[3]).astype(np.float32)


def posterior_disconnected_fraction(mask: np.ndarray, hu: np.ndarray, grid: sitk.Image, slab_mm: float = 15.0) -> dict:
    """Table check on the production grid, in RAS world coordinates (oblique grids allowed).

    Slab = voxels whose RAS y lies within `slab_mm` of the most posterior voxel of the plain
    HU > -500 threshold (the table, when present). The skull component is the connected component
    of `mask` holding the most HU > 300 voxels. Returns the fraction of `mask` voxels in the slab
    that are not in that component (0 = nothing detached from the skull, e.g. no table).
    """
    y = ras_axis_grid(grid, 1)
    plain = hu > -500
    slab = y < float(y[plain].min()) + slab_mm
    labels = sitk.GetArrayFromImage(sitk.ConnectedComponent(_image(mask, grid)))
    bone_counts = np.bincount(labels[(hu > 300) & mask], minlength=labels.max() + 1)
    bone_counts[0] = 0
    skull = labels == int(np.argmax(bone_counts))
    in_slab = mask & slab
    detached = in_slab & ~skull
    return {"slab_voxels": int(in_slab.sum()), "detached_voxels": int(detached.sum()),
            "fraction": float(detached.sum() / in_slab.sum()) if in_slab.any() else 0.0}


def grid_from_manifest(manifest: dict) -> sitk.Image:
    """Rebuild the capsule grid geometry (for checks on a written capsule)."""
    ras = np.asarray(manifest["grid"]["affine_ras"], dtype=float)
    lps = np.diag([-1.0, -1.0, 1.0, 1.0]) @ ras
    spacing = np.asarray(manifest["grid"]["spacing_mm"], dtype=float)
    grid = sitk.Image([int(v) for v in manifest["grid"]["dims"]], sitk.sitkUInt8)
    grid.SetSpacing(spacing.tolist())
    grid.SetDirection((lps[:3, :3] / spacing).flatten().tolist())
    grid.SetOrigin(lps[:3, 3].tolist())
    return grid
