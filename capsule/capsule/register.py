"""Rigid, fixed-to-moving image registration for capsule resampling."""

from __future__ import annotations

from typing import Any

import numpy as np
import SimpleITK as sitk


_HISTOGRAM_BINS = 50
_SAMPLING_FRACTION = 0.20
_SHRINK_FACTORS = (4, 2, 1)
_SMOOTHING_SIGMAS = (2.0, 1.0, 0.0)


def _validate_images(fixed: sitk.Image, moving: sitk.Image) -> None:
    if not isinstance(fixed, sitk.Image) or not isinstance(moving, sitk.Image):
        raise TypeError("fixed and moving must be SimpleITK Images")
    if fixed.GetDimension() != 3 or moving.GetDimension() != 3:
        raise ValueError("rigid registration requires 3-D images")
    if min(fixed.GetSize()) < 8 or min(moving.GetSize()) < 8:
        raise ValueError("registration images must have at least 8 voxels per axis")


def _configure_metric(method: sitk.ImageRegistrationMethod) -> None:
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=_HISTOGRAM_BINS)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(_SAMPLING_FRACTION, seed=1729)
    method.SetInterpolator(sitk.sitkLinear)


def _metric_at(fixed: sitk.Image, moving: sitk.Image, transform: sitk.Transform) -> float:
    """Evaluate the Mattes metric on the full-resolution images."""
    method = sitk.ImageRegistrationMethod()
    _configure_metric(method)
    # Score every voxel: the optimizer's random sampling is too noisy to rank
    # candidates whose true metrics differ in the third decimal.
    method.SetMetricSamplingStrategy(method.NONE)
    method.SetInitialTransform(transform, inPlace=False)
    return float(method.MetricEvaluate(fixed, moving))


def _optimize(
    fixed: sitk.Image,
    moving: sitk.Image,
    initial: sitk.Euler3DTransform,
) -> tuple[sitk.Transform, float, str]:
    method = sitk.ImageRegistrationMethod()
    _configure_metric(method)
    method.SetOptimizerAsGradientDescentLineSearch(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=12,
        maximumStepSizeInPhysicalUnits=2.0,
    )
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel(shrinkFactors=_SHRINK_FACTORS)
    method.SetSmoothingSigmasPerLevel(smoothingSigmas=_SMOOTHING_SIGMAS)
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(initial, inPlace=False)
    result = method.Execute(fixed, moving)
    # Execute returns the transform used by Resample: fixed physical points
    # map to corresponding moving physical points.
    return result, float(method.GetMetricValue()), str(method.GetOptimizerStopConditionDescription())


def _clone_euler(transform: sitk.Transform) -> sitk.Euler3DTransform:
    """Copy an Euler transform without depending on the wrapped transform type."""
    clone = sitk.Euler3DTransform()
    clone.SetFixedParameters(transform.GetFixedParameters())
    clone.SetParameters(transform.GetParameters())
    return clone


def _homogeneous(transform: sitk.Transform) -> np.ndarray:
    origin = np.asarray(transform.TransformPoint((0.0, 0.0, 0.0)), dtype=float)
    matrix = np.eye(4, dtype=float)
    matrix[:3, 3] = origin
    for axis in range(3):
        point = np.zeros(3, dtype=float)
        point[axis] = 1.0
        transformed = np.asarray(transform.TransformPoint(tuple(point)), dtype=float)
        matrix[:3, axis] = transformed - origin
    return matrix


def moving_to_fixed_ras(transform: sitk.Transform) -> list[list[float]]:
    """Return the inverse transform as a homogeneous moving→fixed RAS matrix.

    SimpleITK physical coordinates use DICOM LPS. The inverse of the returned
    fixed→moving transform is conjugated by the LPS↔RAS axis conversion.
    """
    try:
        inverse = transform.GetInverse()
    except RuntimeError as exc:
        raise ValueError("registration transform is not invertible") from exc
    lps = _homogeneous(inverse)
    lps_to_ras = np.diag((-1.0, -1.0, 1.0, 1.0))
    return (lps_to_ras @ lps @ lps_to_ras).tolist()


def register(
    fixed: sitk.Image,
    moving: sitk.Image,
) -> tuple[sitk.Transform, float, dict[str, Any]]:
    """Register ``moving`` to ``fixed`` with a rigid Euler3D transform.

    The returned transform maps fixed physical coordinates to moving physical
    coordinates, so it can be supplied directly to ``sitk.Resample(moving,
    fixed, transform, ...)``. The metric is Mattes mutual information; lower
    metric values are better. Identity and geometry-centred starts are both
    optimized, with a moments-centred retry if the geometry run fails or
    regresses; the best full-sample metric wins.
    """
    _validate_images(fixed, moving)
    # CT arrives as int16 and MR as uint16/float; SimpleITK's initializer and
    # metrics require both inputs to share one pixel type.
    fixed = sitk.Cast(fixed, sitk.sitkFloat32)
    moving = sitk.Cast(moving, sitk.sitkFloat32)

    attempts: list[dict[str, Any]] = []
    candidates: list[tuple[float, sitk.Transform, str]] = []

    def attempt(name: str, initial: sitk.Euler3DTransform) -> float | None:
        """Optimize from one initializer; a failed start is recorded, not fatal."""
        record: dict[str, Any] = {"initializer": name}
        attempts.append(record)
        try:
            record["initial_metric"] = _metric_at(fixed, moving, initial)
            result, reported, stop = _optimize(fixed, moving, initial)
            final = _metric_at(fixed, moving, result)
        except RuntimeError as exc:
            record["error"] = str(exc).strip().splitlines()[-1][:200]
            return None
        record.update(optimizer_reported_metric=reported, evaluated_metric=final, optimizer_stop=stop)
        candidates.append((final, result, name))
        return final

    # Identity (scanner geometry) is the right start when both series share a
    # frame of reference; centring a partial-coverage slab (e.g. an IAC T2)
    # on the whole-head centre would push it off its true position.
    identity = sitk.Euler3DTransform()
    identity.SetCenter(fixed.TransformContinuousIndexToPhysicalPoint(
        [(size - 1) / 2.0 for size in fixed.GetSize()]))
    attempt("identity", identity)

    geometry = _clone_euler(sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    ))
    geometry_final = attempt("geometry", geometry)

    # Mattes MI is minimized. If the geometry-centred run failed or regressed,
    # retry from the independent moments initializer.
    geometry_initial = attempts[-1].get("initial_metric")
    if geometry_final is None or geometry_initial is None or geometry_final > geometry_initial:
        try:
            moments = _clone_euler(sitk.CenteredTransformInitializer(
                fixed, moving, sitk.Euler3DTransform(),
                sitk.CenteredTransformInitializerFilter.MOMENTS,
            ))
        except RuntimeError as exc:
            attempts.append({"initializer": "moments", "error": str(exc).strip().splitlines()[-1][:200]})
        else:
            attempt("moments", moments)

    if not candidates:
        raise ValueError("rigid registration failed from every initializer ("
                         + "; ".join(a.get("error", "") for a in attempts)
                         + "); if the series share the scanner frame of reference, rebuild with --no-register")

    metric, transform, selected = min(candidates, key=lambda candidate: candidate[0])
    metadata: dict[str, Any] = {
        "direction": "fixed_to_moving",
        "transform_type": transform.GetName(),
        "parameters": [float(value) for value in transform.GetParameters()],
        "fixed_parameters": [float(value) for value in transform.GetFixedParameters()],
        "metric_name": "MattesMutualInformation",
        "metric_value": float(metric),
        "initializer": selected,
        "attempts": attempts,
        "moving_to_fixed_ras": moving_to_fixed_ras(transform),
    }
    return transform, float(metric), metadata
