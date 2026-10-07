import * as THREE from "./vendor/three.module.js";

/*
 * Arc-length attribute + single-pass pulse technique adapted from human-brain:
 * https://github.com/amyleesterling/human-brain
 * Copyright (c) 2026 Amy Sterling, used under the MIT License.
 * MIT License
 *
 * Copyright (c) 2026 Amy Sterling
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * TractLab deliberately changes the source's one-way pulse into two mirrored
 * heads that travel out and back. The animation demonstrates polyline
 * continuity only; it does not encode anatomical or physiological direction.
 */

const TRACE_RADIANS_PER_SECOND = 1.35;

const VERTEX_SHADER = `
  attribute float progress;
  attribute float lowSupport;
  attribute float lineIndex;
  varying float vProgress;
  varying float vLowSupport;
  varying float vLineIndex;

  void main() {
    vProgress = progress;
    vLowSupport = lowSupport;
    vLineIndex = lineIndex;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const FRAGMENT_SHADER = `
  uniform vec3 uColor;
  uniform float uPhase;
  uniform float uEvidenceMode;
  uniform float uHasEvidence;
  uniform float uSelectedLine;
  uniform float uStride;
  varying float vProgress;
  varying float vLowSupport;
  varying float vLineIndex;

  void main() {
    if (uSelectedLine >= 0.0) {
      if (abs(vLineIndex - uSelectedLine) > 0.1) discard;
    } else if (mod(vLineIndex, uStride) > 0.1) discard;
    if (uHasEvidence > 0.5) {
      if (uEvidenceMode > 0.5 && uEvidenceMode < 1.5 && vLowSupport > 0.5) discard;
      if (uEvidenceMode > 1.5 && vLowSupport < 0.5) discard;
    }

    // Mirroring creates simultaneous heads on the two streamline halves.
    // uPhase oscillates out and back; neither endpoint is treated as origin.
    float mirrored = abs(vProgress * 2.0 - 1.0);
    float distanceToHead = abs(mirrored - uPhase);
    float head = 1.0 - smoothstep(0.0, 0.09, distanceToHead);
    float halo = 1.0 - smoothstep(0.0, 0.26, distanceToHead);
    float evidenceWeight = vLowSupport > 0.5 ? 0.45 : 1.0;
    // Rest stays dim so stacked streamlines do not clip to white; the two
    // heads are the only bright part, matching the reference pulse idea.
    float alpha = (0.04 + head * 0.9 + halo * 0.12) * evidenceWeight;
    vec3 light = uColor * (0.35 + head * 0.65 + halo * 0.15);
    gl_FragColor = vec4(light, alpha);
  }
`;

function validateShape(flat, lineCount, pointsPerLine) {
  if (!(flat instanceof Float32Array)) throw new TypeError("flat must be a Float32Array");
  if (!Number.isInteger(lineCount) || lineCount < 1) {
    throw new RangeError("lineCount must be a positive integer");
  }
  if (!Number.isInteger(pointsPerLine) || pointsPerLine < 2) {
    throw new RangeError("pointsPerLine must be an integer of at least 2");
  }
  const expected = lineCount * pointsPerLine * 3;
  if (flat.length !== expected) {
    throw new RangeError(`flat has ${flat.length} values; expected ${expected}`);
  }
}

function buildProgress(flat, lineCount, pointsPerLine) {
  const progress = new Float32Array(lineCount * pointsPerLine);
  const cumulative = new Float64Array(pointsPerLine);
  for (let line = 0; line < lineCount; line += 1) {
    const pointOffset = line * pointsPerLine;
    cumulative[0] = 0;
    for (let point = 1; point < pointsPerLine; point += 1) {
      const a = (pointOffset + point - 1) * 3;
      const b = (pointOffset + point) * 3;
      const dx = flat[b] - flat[a];
      const dy = flat[b + 1] - flat[a + 1];
      const dz = flat[b + 2] - flat[a + 2];
      cumulative[point] = cumulative[point - 1] + Math.hypot(dx, dy, dz);
    }
    const total = cumulative[pointsPerLine - 1];
    for (let point = 0; point < pointsPerLine; point += 1) {
      progress[pointOffset + point] = total > 0
        ? cumulative[point] / total
        : point / (pointsPerLine - 1);
    }
  }
  return progress;
}

function buildSegmentIndex(lineCount, pointsPerLine, vertexCount) {
  const IndexArray = vertexCount > 65535 ? Uint32Array : Uint16Array;
  const index = new IndexArray(lineCount * (pointsPerLine - 1) * 2);
  let cursor = 0;
  for (let line = 0; line < lineCount; line += 1) {
    const offset = line * pointsPerLine;
    for (let point = 0; point < pointsPerLine - 1; point += 1) {
      index[cursor++] = offset + point;
      index[cursor++] = offset + point + 1;
    }
  }
  return index;
}

function buildLowSupport(lineCount, pointsPerLine, lowSupportMask) {
  const support = new Float32Array(lineCount * pointsPerLine);
  if (!lowSupportMask) return support;
  for (let line = 0; line < lineCount; line += 1) {
    if (!lowSupportMask[line]) continue;
    support.fill(1, line * pointsPerLine, (line + 1) * pointsPerLine);
  }
  return support;
}

export function teachingTracePhase(seconds) {
  const safeSeconds = Number.isFinite(seconds) ? seconds : 0;
  return 0.55 - 0.35 * Math.cos(safeSeconds * TRACE_RADIANS_PER_SECOND);
}

export function buildTeachingTrace({
  flat,
  lineCount,
  pointsPerLine,
  lowSupportMask = null,
  hasEvidence = false,
  color = "#ffd36b",
}) {
  validateShape(flat, lineCount, pointsPerLine);
  const vertexCount = lineCount * pointsPerLine;
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(flat, 3));
  geometry.setAttribute("progress", new THREE.BufferAttribute(
    buildProgress(flat, lineCount, pointsPerLine),
    1,
  ));
  geometry.setAttribute("lowSupport", new THREE.BufferAttribute(
    buildLowSupport(lineCount, pointsPerLine, lowSupportMask),
    1,
  ));
  const lineIndices=new Float32Array(vertexCount);
  for(let i=0;i<lineCount;i++)lineIndices.fill(i,i*pointsPerLine,(i+1)*pointsPerLine);
  geometry.setAttribute('lineIndex',new THREE.BufferAttribute(lineIndices,1));
  geometry.setIndex(new THREE.BufferAttribute(
    buildSegmentIndex(lineCount, pointsPerLine, vertexCount),
    1,
  ));
  geometry.computeBoundingSphere();

  const material = new THREE.ShaderMaterial({
    uniforms: {
      uColor: { value: new THREE.Color(color) },
      uPhase: { value: teachingTracePhase(0) },
      uSelectedLine: {value:-1},
      uStride: {value:Math.max(1,Math.ceil(lineCount/24))},
      uEvidenceMode: { value: 0 },
      uHasEvidence: { value: hasEvidence ? 1 : 0 },
    },
    vertexShader: VERTEX_SHADER,
    fragmentShader: FRAGMENT_SHADER,
    transparent: true,
    blending: THREE.NormalBlending,
    // Centerline sits inside the tube mesh. Depth-testing it makes the pulse
    // invisible; Teaching ghosts the tubes and draws this pass on top.
    depthTest: false,
    depthWrite: false,
    toneMapped: false,
  });
  const trace = new THREE.LineSegments(geometry, material);
  trace.name = "teaching-direction-neutral-trace";
  trace.renderOrder = 4;
  trace.userData.directionNeutral = true;
  trace.userData.displayOnly = true;
  trace.raycast = () => {};
  return trace;
}

export function setTeachingTraceTime(trace, seconds) {
  const uniform = trace?.material?.uniforms?.uPhase;
  if(uniform) uniform.value = teachingTracePhase(seconds);
}

export function setTeachingTraceSelection(trace,displayIndex=null){
  const uniform=trace?.material?.uniforms?.uSelectedLine;
  if(uniform)uniform.value=Number.isInteger(displayIndex) && displayIndex>=0?displayIndex:-1;
}

export function setTeachingTraceEvidenceMode(
  trace,
  { hideLowSupport = false, onlyLowSupport = false } = {},
) {
  const uniform = trace?.material?.uniforms?.uEvidenceMode;
  if (!uniform) return;
  uniform.value = onlyLowSupport ? 2 : (hideLowSupport ? 1 : 0);
}

export function disposeTeachingTrace(trace) {
  trace?.geometry?.dispose?.();
  trace?.material?.dispose?.();
}
