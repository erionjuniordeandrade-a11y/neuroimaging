import assert from 'node:assert/strict';
import test from 'node:test';

import {
  SpatialGeometryError,
  brushRadiusPixels,
  canvasPointToVoxel,
  fitContentRect,
  orientationLabels,
  planeFrame,
  planeVoxelToWorld,
  validateMprGeometry,
  voxelToWorld,
  worldToPlaneVoxel,
  worldToVoxel,
} from '../../viewer/spatial.js';

const reflectedAffine = [
  -2, 0, 0, 10,
  0, 3, 0, 20,
  0, 0, 4, 30,
  0, 0, 0, 1,
];

test('planeFrame preserves anisotropic reflected physical extents and world corners', () => {
  const frame = planeFrame({
    shape: [5, 6, 7], affine: reflectedAffine, axis: 'ax', slice: 3,
  });

  assert.equal(frame.width, 5);
  assert.equal(frame.height, 6);
  assert.equal(frame.physicalWidthMm, 10);
  assert.equal(frame.physicalHeightMm, 18);
  assert.equal(frame.cssAspect, 10 / 18);
  assert.deepEqual(frame.cornersWorld.topLeft, [11, 18.5, 42]);
  assert.deepEqual(frame.cornersWorld.topRight, [1, 18.5, 42]);
  assert.deepEqual(frame.cornersWorld.bottomRight, [1, 36.5, 42]);
  assert.deepEqual(frame.cornerLoopWorld, [
    frame.cornersWorld.topLeft,
    frame.cornersWorld.topRight,
    frame.cornersWorld.bottomRight,
    frame.cornersWorld.bottomLeft,
  ]);
  assert.deepEqual(orientationLabels(frame), {
    left: 'R', right: 'L', top: 'P', bottom: 'A',
  });
});

test('voxel, world, and plane helpers round trip on rotated reflected axes', () => {
  const affine = [
    0, 3, 0, 5,
    -2, 0, 0, -7,
    0, 0, 4, 11,
    0, 0, 0, 1,
  ];
  const voxel = [1.25, 2.5, 3.75];
  const world = voxelToWorld(affine, voxel);
  assert.deepEqual(worldToVoxel(affine, world), voxel);

  const frame = planeFrame({shape: [10, 11, 12], affine, axis: 'cor', slice: 4});
  const planeWorld = planeVoxelToWorld(frame, {u: 2.25, v: 6.5});
  const back = worldToPlaneVoxel(frame, planeWorld);
  assert.deepEqual(back.voxel, [2.25, 4, 6.5]);
  assert.equal(back.u, 2.25);
  assert.equal(back.v, 6.5);
  assert.equal(back.sliceOffset, 0);
});

test('letterboxed canvas pointers map only content pixels to matching voxel centers', () => {
  const frame = planeFrame({
    shape: [100, 50, 10],
    affine: [2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 1],
    axis: 'ax',
    slice: 4,
  });
  const content = fitContentRect({left: 10, top: 20, width: 300, height: 300}, frame.cssAspect);
  assert.deepEqual(content, {left: 10, top: 57.5, width: 300, height: 225});
  assert.equal(canvasPointToVoxel({clientX: 160, clientY: 50}, content, frame), null);

  const hit = canvasPointToVoxel({clientX: 160, clientY: 170}, content, frame);
  assert.deepEqual([hit.i, hit.j, hit.k], [50, 25, 4]);
  assert.deepEqual(hit.world, voxelToWorld(frame.affine, [50, 25, 4]));
  assert.deepEqual(hit.continuous, [49.5, 24.5, 4]);
  const flipped = canvasPointToVoxel(
    {clientX: 160, clientY: 102.5}, content, frame, {flipY: true},
  );
  assert.deepEqual([flipped.i, flipped.j, flipped.k], [50, 39, 4]);
  assert.equal(canvasPointToVoxel({clientX: 310, clientY: 170}, content, frame), null);
});

test('brush radii remain in world millimetres on anisotropic planes', () => {
  const frame = planeFrame({
    shape: [20, 20, 20],
    affine: [2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 1],
    axis: 'ax',
    slice: 9,
  });
  assert.deepEqual(brushRadiusPixels(frame, 6), {uPx: 3, vPx: 2});
});

test('MPR validation rejects degenerate, projective, and sheared affines without rejecting reflections', () => {
  assert.doesNotThrow(() => validateMprGeometry([3, 3, 3], reflectedAffine));
  assert.throws(
    () => validateMprGeometry([3, 3, 3], [
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 1,
    ]),
    SpatialGeometryError,
  );
  assert.throws(
    () => validateMprGeometry([3, 3, 3], [
      1, 0.2, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]),
    /shear/i,
  );
  assert.throws(
    () => validateMprGeometry([3, 3, 3], [
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0.25, 1,
    ]),
    /homogeneous/i,
  );
});

test('MPR accepts numerical near-orthogonality residue but rejects material shear', () => {
  const numericalResidue = [
    1, 2.2e-6, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ];
  assert.doesNotThrow(() => validateMprGeometry([185, 185, 109], numericalResidue));
  assert.throws(
    () => validateMprGeometry([185, 185, 109], [
      1, 5.1e-6, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]),
    /shear/i,
  );
});
