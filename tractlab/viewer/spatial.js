/**
 * Pure spatial contracts for the voxel-axis MPR canvases.
 *
 * Affines are flattened row-major 4x4 IJK-to-world matrices.  The viewer may
 * display rotated, reflected, and anisotropic grids; a sheared grid is refused
 * because a rectangular canvas cannot represent its in-plane parallelogram
 * without silently changing distances.
 */

export class SpatialGeometryError extends Error {
  constructor(message) {
    super(message);
    this.name = 'SpatialGeometryError';
  }
}

const AXES = {
  ax: {fixedAxis: 2, uAxis: 0, vAxis: 1},
  cor: {fixedAxis: 1, uAxis: 0, vAxis: 2},
  sag: {fixedAxis: 0, uAxis: 1, vAxis: 2},
};

const HOMOGENEOUS_TOLERANCE = 1e-8;
// NIfTI qform reconstruction and eight-decimal header serialization can leave
// a few parts-per-million of non-orthogonality. This is still far below a
// display pixel; material shear remains a hard error below.
const ORTHOGONALITY_TOLERANCE = 5e-6;
const SINGULAR_TOLERANCE = 1e-10;

function fail(message) {
  throw new SpatialGeometryError(message);
}

function finiteNumber(value, name) {
  if (!Number.isFinite(value)) fail(`${name} must be finite`);
  return Number(value);
}

function numericVector(value, length, name) {
  if (value == null || typeof value.length !== 'number' || value.length !== length) {
    fail(`${name} must contain ${length} values`);
  }
  return Array.from(value, (item, index) => finiteNumber(item, `${name}[${index}]`));
}

function shape3(shape) {
  const values = numericVector(shape, 3, 'shape');
  for (const [index, size] of values.entries()) {
    if (!Number.isInteger(size) || size <= 0) {
      fail(`shape[${index}] must be a positive integer`);
    }
  }
  return values;
}

function affine16(affine) {
  const values = numericVector(affine, 16, 'affine');
  if (
    Math.abs(values[12]) > HOMOGENEOUS_TOLERANCE
    || Math.abs(values[13]) > HOMOGENEOUS_TOLERANCE
    || Math.abs(values[14]) > HOMOGENEOUS_TOLERANCE
    || Math.abs(values[15] - 1) > HOMOGENEOUS_TOLERANCE
  ) {
    fail('affine must have homogeneous final row [0, 0, 0, 1]');
  }
  return values;
}

function column(affine, axis) {
  return [affine[axis], affine[4 + axis], affine[8 + axis]];
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function norm(vector) {
  return Math.hypot(vector[0], vector[1], vector[2]);
}

function scale(vector, factor) {
  return [vector[0] * factor, vector[1] * factor, vector[2] * factor];
}

function negate(vector) {
  return scale(vector, -1);
}

function determinant3(affine) {
  return (
    affine[0] * (affine[5] * affine[10] - affine[6] * affine[9])
    - affine[1] * (affine[4] * affine[10] - affine[6] * affine[8])
    + affine[2] * (affine[4] * affine[9] - affine[5] * affine[8])
  );
}

function inverseLinear(affine) {
  const det = determinant3(affine);
  const axes = [column(affine, 0), column(affine, 1), column(affine, 2)];
  const scaleProduct = norm(axes[0]) * norm(axes[1]) * norm(axes[2]);
  if (!Number.isFinite(det) || !Number.isFinite(scaleProduct)
      || scaleProduct === 0 || Math.abs(det) <= SINGULAR_TOLERANCE * scaleProduct) {
    fail('affine spatial transform is singular or degenerate');
  }
  const invDet = 1 / det;
  return [
    (affine[5] * affine[10] - affine[6] * affine[9]) * invDet,
    (affine[2] * affine[9] - affine[1] * affine[10]) * invDet,
    (affine[1] * affine[6] - affine[2] * affine[5]) * invDet,
    (affine[6] * affine[8] - affine[4] * affine[10]) * invDet,
    (affine[0] * affine[10] - affine[2] * affine[8]) * invDet,
    (affine[2] * affine[4] - affine[0] * affine[6]) * invDet,
    (affine[4] * affine[9] - affine[5] * affine[8]) * invDet,
    (affine[1] * affine[8] - affine[0] * affine[9]) * invDet,
    (affine[0] * affine[5] - affine[1] * affine[4]) * invDet,
  ];
}

function validateAffine(affine, {rejectShear}) {
  const values = affine16(affine);
  const axes = [column(values, 0), column(values, 1), column(values, 2)];
  const spacingMm = axes.map(norm);
  inverseLinear(values);
  if (rejectShear) {
    for (let first = 0; first < axes.length; first += 1) {
      for (let second = first + 1; second < axes.length; second += 1) {
        const cosine = Math.abs(dot(axes[first], axes[second]) / (spacingMm[first] * spacingMm[second]));
        if (cosine > ORTHOGONALITY_TOLERANCE) {
          fail('affine contains shear; rectangular MPR display would distort physical distances');
        }
      }
    }
  }
  return {affine: values, axisVectors: axes, spacingMm, determinant: determinant3(values)};
}

/** Validate geometry required to draw an undistorted rectangular MPR canvas. */
export function validateMprGeometry(shape, affine) {
  const checkedShape = shape3(shape);
  return {shape: checkedShape, ...validateAffine(affine, {rejectShear: true})};
}

/** Convert a fractional voxel IJK coordinate to world millimetres. */
export function voxelToWorld(affine, voxel) {
  const checkedAffine = validateAffine(affine, {rejectShear: false}).affine;
  const [i, j, k] = numericVector(voxel, 3, 'voxel');
  return [
    checkedAffine[0] * i + checkedAffine[1] * j + checkedAffine[2] * k + checkedAffine[3],
    checkedAffine[4] * i + checkedAffine[5] * j + checkedAffine[6] * k + checkedAffine[7],
    checkedAffine[8] * i + checkedAffine[9] * j + checkedAffine[10] * k + checkedAffine[11],
  ];
}

/** Convert a world-mm point to a fractional voxel IJK coordinate. */
export function worldToVoxel(affine, world) {
  const checkedAffine = validateAffine(affine, {rejectShear: false}).affine;
  const inverse = inverseLinear(checkedAffine);
  const [x, y, z] = numericVector(world, 3, 'world');
  const dx = x - checkedAffine[3];
  const dy = y - checkedAffine[7];
  const dz = z - checkedAffine[11];
  return [
    inverse[0] * dx + inverse[1] * dy + inverse[2] * dz,
    inverse[3] * dx + inverse[4] * dy + inverse[5] * dz,
    inverse[6] * dx + inverse[7] * dy + inverse[8] * dz,
  ];
}

/**
 * Describe one voxel-axis MPR plane.  `u` grows rightward and `v` grows
 * downward in canvas coordinates.  `cornersWorld` are voxel-boundary corners;
 * `centerCornersWorld` are centers of the four extreme displayed voxels.
 */
export function planeFrame({shape, affine, axis, slice}) {
  const geometry = validateMprGeometry(shape, affine);
  const definition = AXES[axis];
  if (!definition) fail(`axis must be one of ${Object.keys(AXES).join(', ')}`);
  if (!Number.isInteger(slice) || slice < 0 || slice >= geometry.shape[definition.fixedAxis]) {
    fail(`slice for ${axis} is outside the volume`);
  }
  const width = geometry.shape[definition.uAxis];
  const height = geometry.shape[definition.vAxis];
  const physicalWidthMm = width * geometry.spacingMm[definition.uAxis];
  const physicalHeightMm = height * geometry.spacingMm[definition.vAxis];
  const normal = cross(
    geometry.axisVectors[definition.uAxis],
    geometry.axisVectors[definition.vAxis],
  );
  const normalLength = norm(normal);
  if (!Number.isFinite(normalLength) || normalLength === 0) {
    fail(`plane ${axis} has no physical normal`);
  }
  const frame = {
    ...geometry,
    axis,
    slice,
    fixedAxis: definition.fixedAxis,
    uAxis: definition.uAxis,
    vAxis: definition.vAxis,
    width,
    height,
    physicalWidthMm,
    physicalHeightMm,
    cssAspect: physicalWidthMm / physicalHeightMm,
    uVectorWorld: geometry.axisVectors[definition.uAxis],
    vVectorWorld: geometry.axisVectors[definition.vAxis],
    normalWorld: scale(normal, 1 / normalLength),
  };
  frame.cornersWorld = {
    topLeft: planeVoxelToWorld(frame, {u: 0, v: 0, edge: true}),
    topRight: planeVoxelToWorld(frame, {u: width, v: 0, edge: true}),
    bottomRight: planeVoxelToWorld(frame, {u: width, v: height, edge: true}),
    bottomLeft: planeVoxelToWorld(frame, {u: 0, v: height, edge: true}),
  };
  // Raw canvas row order. With a vertically flipped texture, this is also the
  // [BL, BR, TR, TL] world-corner order expected by the 3D slice plane.
  frame.cornerLoopWorld = [
    frame.cornersWorld.topLeft,
    frame.cornersWorld.topRight,
    frame.cornersWorld.bottomRight,
    frame.cornersWorld.bottomLeft,
  ];
  frame.centerCornersWorld = {
    topLeft: planeVoxelToWorld(frame, {u: 0, v: 0}),
    topRight: planeVoxelToWorld(frame, {u: width - 1, v: 0}),
    bottomRight: planeVoxelToWorld(frame, {u: width - 1, v: height - 1}),
    bottomLeft: planeVoxelToWorld(frame, {u: 0, v: height - 1}),
  };
  return frame;
}

/** Map a plane pixel coordinate (or edge coordinate) to world millimetres. */
export function planeVoxelToWorld(frame, {u, v, edge = false}) {
  if (!frame || !Array.isArray(frame.affine)) fail('plane frame is required');
  const uu = finiteNumber(u, 'u') - (edge ? 0.5 : 0);
  const vv = finiteNumber(v, 'v') - (edge ? 0.5 : 0);
  const voxel = [0, 0, 0];
  voxel[frame.fixedAxis] = frame.slice;
  voxel[frame.uAxis] = uu;
  voxel[frame.vAxis] = vv;
  return voxelToWorld(frame.affine, voxel);
}

/** Map a world-mm point back into a plane's voxel coordinate system. */
export function worldToPlaneVoxel(frame, world) {
  if (!frame || !Array.isArray(frame.affine)) fail('plane frame is required');
  const voxel = worldToVoxel(frame.affine, world);
  return {
    voxel,
    u: voxel[frame.uAxis],
    v: voxel[frame.vAxis],
    sliceOffset: voxel[frame.fixedAxis] - frame.slice,
  };
}

/** Letterbox a physical-aspect MPR image inside a CSS rectangle. */
export function fitContentRect(rect, aspect) {
  const left = finiteNumber(rect?.left, 'rect.left');
  const top = finiteNumber(rect?.top, 'rect.top');
  const width = finiteNumber(rect?.width, 'rect.width');
  const height = finiteNumber(rect?.height, 'rect.height');
  const checkedAspect = finiteNumber(aspect, 'aspect');
  if (width <= 0 || height <= 0 || checkedAspect <= 0) {
    fail('content rectangle and aspect must be positive');
  }
  if (width / height > checkedAspect) {
    const contentWidth = height * checkedAspect;
    return {left: left + (width - contentWidth) / 2, top, width: contentWidth, height};
  }
  const contentHeight = width / checkedAspect;
  return {left, top: top + (height - contentHeight) / 2, width, height: contentHeight};
}

/**
 * Invert a CSS pointer through a fitted MPR content rect.  Returns null for
 * letterbox bars and outer edges instead of clamping a paint point to a face.
 */
export function canvasPointToVoxel({clientX, clientY}, contentRect, frame, {flipY = false} = {}) {
  const x = finiteNumber(clientX, 'clientX');
  const y = finiteNumber(clientY, 'clientY');
  const left = finiteNumber(contentRect?.left, 'contentRect.left');
  const top = finiteNumber(contentRect?.top, 'contentRect.top');
  const width = finiteNumber(contentRect?.width, 'contentRect.width');
  const height = finiteNumber(contentRect?.height, 'contentRect.height');
  if (!frame || !Number.isInteger(frame.width) || !Number.isInteger(frame.height)
      || width <= 0 || height <= 0) {
    fail('valid content rect and plane frame are required');
  }
  if (x < left || x >= left + width || y < top || y >= top + height) return null;
  const xFraction = (x - left) / width;
  const yFraction = (y - top) / height;
  const pixelU = Math.floor(xFraction * frame.width);
  const rawV = Math.floor(yFraction * frame.height);
  const pixelV = flipY ? frame.height - 1 - rawV : rawV;
  const continuousU = xFraction * frame.width - 0.5;
  const continuousV = (flipY ? (1 - yFraction) : yFraction) * frame.height - 0.5;
  const voxel = [0, 0, 0];
  const continuous = [0, 0, 0];
  voxel[frame.fixedAxis] = frame.slice;
  voxel[frame.uAxis] = pixelU;
  voxel[frame.vAxis] = pixelV;
  continuous[frame.fixedAxis] = frame.slice;
  continuous[frame.uAxis] = continuousU;
  continuous[frame.vAxis] = continuousV;
  return {
    i: voxel[0],
    j: voxel[1],
    k: voxel[2],
    u: pixelU,
    v: pixelV,
    voxel,
    continuous,
    world: voxelToWorld(frame.affine, voxel),
    continuousWorld: voxelToWorld(frame.affine, continuous),
  };
}

/** Convenience wrapper for a pointer whose caller only needs world mm. */
export function canvasPointToWorld(pointer, contentRect, frame, options) {
  return canvasPointToVoxel(pointer, contentRect, frame, options)?.world ?? null;
}

/** Convert a physical brush radius into the canvas ellipse radii for one MPR plane. */
export function brushRadiusPixels(frame, radiusMm) {
  const checkedRadius = finiteNumber(radiusMm, 'radiusMm');
  if (checkedRadius <= 0) fail('radiusMm must be positive');
  return {
    uPx: checkedRadius / frame.spacingMm[frame.uAxis],
    vPx: checkedRadius / frame.spacingMm[frame.vAxis],
  };
}

function worldDirectionCode(vector) {
  const checked = numericVector(vector, 3, 'direction');
  const length = norm(checked);
  if (length === 0) fail('direction must be non-zero');
  const labels = [
    {positive: 'R', negative: 'L'},
    {positive: 'A', negative: 'P'},
    {positive: 'S', negative: 'I'},
  ];
  const parts = checked
    .map((component, axis) => ({component, axis, magnitude: Math.abs(component) / length}))
    .filter(({magnitude}) => magnitude >= 0.25)
    .sort((a, b) => b.magnitude - a.magnitude)
    .map(({component, axis}) => (component >= 0 ? labels[axis].positive : labels[axis].negative));
  if (parts.length) return parts.join('');
  const axis = checked.reduce((best, component, index) => (
    Math.abs(component) > Math.abs(checked[best]) ? index : best
  ), 0);
  return checked[axis] >= 0 ? labels[axis].positive : labels[axis].negative;
}

/** Derive edge glyphs from the signed affine directions, including reflections. */
export function orientationLabels(frame) {
  if (!frame?.uVectorWorld || !frame?.vVectorWorld) fail('plane frame is required');
  return {
    left: worldDirectionCode(negate(frame.uVectorWorld)),
    right: worldDirectionCode(frame.uVectorWorld),
    top: worldDirectionCode(negate(frame.vVectorWorld)),
    bottom: worldDirectionCode(frame.vVectorWorld),
  };
}
