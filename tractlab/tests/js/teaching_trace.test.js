import test from "node:test";
import assert from "node:assert/strict";

import {
  buildTeachingTrace,
  disposeTeachingTrace,
  setTeachingTraceEvidenceMode,
  setTeachingTraceTime,
  teachingTracePhase,
  setTeachingTraceSelection,
} from "../../viewer/teaching_trace.js";

const flat = new Float32Array([
  0, 0, 0,  1, 0, 0,  3, 0, 0,
  0, 1, 0,  0, 1, 0,  0, 1, 0,
]);

test("teaching trace uses normalized cumulative arc length per streamline", () => {
  const trace = buildTeachingTrace({
    flat,
    lineCount: 2,
    pointsPerLine: 3,
    lowSupportMask: new Uint8Array([0, 1]),
    hasEvidence: true,
    color: "#c9a84c",
  });

  assert.equal(trace.type, "LineSegments");
  assert.equal(trace.geometry.index.count, 8);
  const progress = [...trace.geometry.getAttribute("progress").array];
  assert.ok(Math.abs(progress[0] - 0) < 1e-6);
  assert.ok(Math.abs(progress[1] - 1 / 3) < 1e-6);
  assert.ok(Math.abs(progress[2] - 1) < 1e-6);
  assert.deepEqual(progress.slice(3), [0, 0.5, 1]);
  assert.deepEqual(
    [...trace.geometry.getAttribute("lowSupport").array],
    [0, 0, 0, 1, 1, 1],
  );
  assert.equal(trace.userData.directionNeutral, true);
  assert.equal(trace.material.depthTest, false);
  disposeTeachingTrace(trace);
});

test("trace phase moves out and back rather than encoding a tract direction", () => {
  assert.ok(Math.abs(teachingTracePhase(0)-0.2)<1e-8);
  assert.ok(teachingTracePhase(1) > 0);
  assert.ok(teachingTracePhase(100) >= 0.2);
  assert.ok(teachingTracePhase(100) <= 0.9);

  const trace = buildTeachingTrace({ flat, lineCount: 2, pointsPerLine: 3 });
  setTeachingTraceTime(trace, 1);
  assert.equal(trace.material.uniforms.uPhase.value, teachingTracePhase(1));
  assert.match(trace.material.fragmentShader, /abs\(vProgress \* 2\.0 - 1\.0\)/);
  disposeTeachingTrace(trace);
});

test('dense traces are bounded and a pinned row overrides the sample',()=>{
  const trace=buildTeachingTrace({flat:new Float32Array(900*2*3),lineCount:900,pointsPerLine:2});
  assert.equal(trace.material.uniforms.uStride.value,38);
  setTeachingTraceSelection(trace,17);assert.equal(trace.material.uniforms.uSelectedLine.value,17);
  setTeachingTraceSelection(trace);assert.equal(trace.material.uniforms.uSelectedLine.value,-1);
  for(let t=0;t<20;t+=.1){const phase=teachingTracePhase(t);assert.ok(phase>=.2-1e-8 && phase<=.9+1e-8);}
  disposeTeachingTrace(trace);
});

test("teaching overlay follows the existing evidence mode", () => {
  const trace = buildTeachingTrace({
    flat,
    lineCount: 2,
    pointsPerLine: 3,
    lowSupportMask: new Uint8Array([0, 1]),
    hasEvidence: true,
  });
  setTeachingTraceEvidenceMode(trace, { hideLowSupport: true, onlyLowSupport: false });
  assert.equal(trace.material.uniforms.uEvidenceMode.value, 1);
  setTeachingTraceEvidenceMode(trace, { hideLowSupport: false, onlyLowSupport: true });
  assert.equal(trace.material.uniforms.uEvidenceMode.value, 2);
  setTeachingTraceEvidenceMode(trace, { hideLowSupport: false, onlyLowSupport: false });
  assert.equal(trace.material.uniforms.uEvidenceMode.value, 0);
  disposeTeachingTrace(trace);
});

test("large displayed banks use a 32-bit segment index", () => {
  const lineCount = 1100;
  const pointsPerLine = 64;
  const largeFlat = new Float32Array(lineCount * pointsPerLine * 3);
  for (let line = 0; line < lineCount; line += 1) {
    for (let point = 0; point < pointsPerLine; point += 1) {
      largeFlat[(line * pointsPerLine + point) * 3] = point;
    }
  }
  const trace = buildTeachingTrace({ flat: largeFlat, lineCount, pointsPerLine });
  assert.ok(trace.geometry.index.array instanceof Uint32Array);
  disposeTeachingTrace(trace);
});
