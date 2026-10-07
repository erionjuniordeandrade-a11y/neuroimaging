/**
 * Material presets for the workstation scene.
 *
 * Tube geometry stays CPU-baked and stock Three.js material chunks stay in
 * charge of PBR lighting and clipping. Shader hooks provide display radius
 * adjustment and the non-analytic hatch for a recorded proximity floor.
 */

const MATERIAL_STATE = "__tractlabTubeMaterialState";
const RADIUS_STATE = "__tractlabTubeRadius";

function validRadius(value) {
  if (!Number.isFinite(value) || value < 0) throw new TypeError('tube radius must be finite and non-negative');
  return value;
}

/** Compose after other material hooks. Baked radial normals preserve centres. */
export function configureTubeRadius(material, {bakedRadius, radius = bakedRadius}) {
  validRadius(bakedRadius); validRadius(radius);
  const uniform = {value: radius - bakedRadius};
  const before = material.onBeforeCompile, cacheKey = material.customProgramCacheKey;
  materialUserData(material)[RADIUS_STATE] = {bakedRadius, uniform};
  material.onBeforeCompile = function(shader, renderer) {
    before?.call(this, shader, renderer);
    shader.uniforms.uTractlabRadiusDelta = uniform;
    shader.vertexShader = `uniform float uTractlabRadiusDelta;\n${shader.vertexShader}`.replace(
      '#include <begin_vertex>',
      '#include <begin_vertex>\ntransformed += normal * uTractlabRadiusDelta;'
    );
  };
  material.customProgramCacheKey = function() {
    return `${cacheKey?.call(this) || ''}|tractlab-radius-v1`;
  };
  material.needsUpdate = true;
}

/** Updates an existing GPU uniform, including before the first compilation. */
export function setTubeRadius(material, radius) {
  validRadius(radius);
  const state = material?.userData?.[RADIUS_STATE];
  if (state) state.uniform.value = radius - state.bakedRadius;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function finiteFloor(value) {
  if (value == null || typeof value === "boolean" || (typeof value === "string" && value.trim() === "")) {
    return null;
  }
  const floor = Number(value);
  return Number.isFinite(floor) && floor >= 0 ? floor : null;
}

function shaderAttributeName(name) {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) {
    throw new TypeError("tube distance attribute must be a GLSL identifier");
  }
  return name;
}

function materialUserData(material) {
  if (!material.userData) material.userData = {};
  return material.userData;
}

function savedMaterialState(material) {
  const userData = materialUserData(material);
  if (!userData[MATERIAL_STATE]) {
    userData[MATERIAL_STATE] = {
      onBeforeCompile: material.onBeforeCompile,
      customProgramCacheKey: material.customProgramCacheKey,
    };
  }
  return userData[MATERIAL_STATE];
}

function restoreBaseShaderHooks(material, state) {
  material.onBeforeCompile = state.onBeforeCompile;
  material.customProgramCacheKey = state.customProgramCacheKey;
}

function installSubfloorHatch(material, state, { attributeName, floorMm, neutral }) {
  const uniforms = {
    uTractlabSubfloorMm: { value: floorMm },
    uTractlabSubfloorNeutral: { value: neutral },
  };
  material.onBeforeCompile = function tractlabSubfloorCompile(shader, renderer) {
    state.onBeforeCompile?.call(this, shader, renderer);
    shader.uniforms.uTractlabSubfloorMm = uniforms.uTractlabSubfloorMm;
    shader.uniforms.uTractlabSubfloorNeutral = uniforms.uTractlabSubfloorNeutral;
    if (!shader.vertexShader.includes("vTractlabProximity")) {
      shader.vertexShader = `attribute float ${attributeName};\nvarying float vTractlabProximity;\n${shader.vertexShader}`;
      shader.vertexShader = shader.vertexShader.replace(
        "#include <begin_vertex>",
        `#include <begin_vertex>\nvTractlabProximity = ${attributeName};`
      );
    }
    if (!shader.fragmentShader.includes("uTractlabSubfloorMm")) {
      shader.fragmentShader = `varying float vTractlabProximity;\nuniform float uTractlabSubfloorMm;\nuniform vec3 uTractlabSubfloorNeutral;\n${shader.fragmentShader}`;
      shader.fragmentShader = shader.fragmentShader.replace(
        "#include <color_fragment>",
        `
#include <color_fragment>
float tractlabSubfloorMask = 1.0 - step(uTractlabSubfloorMm, vTractlabProximity);
float tractlabHatch = step(0.5, fract((gl_FragCoord.x + gl_FragCoord.y) * 0.18));
vec3 tractlabHatchedNeutral = mix(
  uTractlabSubfloorNeutral * 0.64,
  uTractlabSubfloorNeutral,
  tractlabHatch
);
diffuseColor.rgb = mix(diffuseColor.rgb, tractlabHatchedNeutral, tractlabSubfloorMask);
`
      );
    }
  };
  material.customProgramCacheKey = () => {
    const inherited = typeof state.customProgramCacheKey === "function"
      ? state.customProgramCacheKey.call(material)
      : "";
    return `${inherited}|tractlab-subfloor-band-v1:${attributeName}`;
  };
}

/**
 * Makes every tract mesh an opaque, depth-writing surface. Colour remains the
 * caller's vertex attribute; this helper does not assign risk-like hues or
 * use brightness/opacity to encode evidence.
 */
export function configTubeMaterial(material, {
  THREE,
  caseMap = false,
  lowSupport = false,
  recovery = false,
  geomFloorMm = null,
  attributeName = "lesionDistance",
  neutralSubfloorRgb = [0.68, 0.71, 0.75],
  emissiveIntensity = 0.12,
} = {}) {
  if (!material || typeof material !== "object") {
    throw new TypeError("configTubeMaterial requires a material");
  }
  const userData = materialUserData(material);
  const state = savedMaterialState(material);
  restoreBaseShaderHooks(material, state);

  material.transparent = false;
  material.opacity = 1;
  material.depthWrite = true;
  material.depthTest = true;
  material.premultipliedAlpha = false;
  if (THREE?.NormalBlending !== undefined) material.blending = THREE.NormalBlending;
  if (typeof material.emissiveIntensity === "number") {
    material.emissiveIntensity = clamp(Number(emissiveIntensity) || 0.12, 0, 0.16);
  }

  userData.renderRole = "tube-evidence";
  userData.depthPolicy = "opaque";
  userData.evidenceClass = lowSupport ? "low-support" : (recovery ? "recovery" : "supported");

  const floor = caseMap ? finiteFloor(geomFloorMm) : null;
  userData.proximityFloorMm = floor;
  userData.proximityBand = caseMap ? (floor == null ? "unavailable" : "sub-floor-hatched") : "off";
  if (floor != null) {
    installSubfloorHatch(material, state, {
      attributeName: shaderAttributeName(attributeName),
      floorMm: floor,
      neutral: neutralSubfloorRgb,
    });
  }
  material.needsUpdate = true;
  return material;
}

/** Context shells remain visible but cannot write the depth used by tube AO. */
export function configContextMaterial(material, {
  THREE,
  opacity = 0.08,
  renderOrder,
} = {}) {
  if (!material || typeof material !== "object") {
    throw new TypeError("configContextMaterial requires a material");
  }
  material.transparent = true;
  material.opacity = clamp(Number(opacity) || 0, 0, 1);
  material.depthTest = true;
  material.depthWrite = false;
  material.premultipliedAlpha = false;
  if (THREE?.NormalBlending !== undefined) material.blending = THREE.NormalBlending;
  if (renderOrder !== undefined) material.renderOrder = renderOrder;
  materialUserData(material).renderRole = "context-shell";
  material.needsUpdate = true;
  return material;
}

/**
 * Removes only a direct, geometry-sharing additive glow child. It leaves other
 * descendants alone and never disposes the shared tube geometry.
 */
export function removeAdditiveTubeGlow(mesh, { THREE, dispose = true } = {}) {
  if (!mesh?.children?.length) return 0;
  const additive = THREE?.AdditiveBlending;
  const removals = mesh.children.filter((child) => {
    const sharedTubeGeometry = child?.geometry === mesh.geometry;
    const isGlow = child?.userData?.renderRole === "tube-glow" ||
      (additive !== undefined && child?.material?.blending === additive);
    return sharedTubeGeometry && isGlow;
  });
  for (const child of removals) {
    mesh.remove?.(child);
    if (dispose) child.material?.dispose?.();
  }
  return removals.length;
}

/** Configure a primary tube mesh and remove the legacy additive glow child. */
export function configTubeMesh(mesh, options = {}) {
  if (!mesh?.material) throw new TypeError("configTubeMesh requires a mesh with a material");
  configTubeMaterial(mesh.material, options);
  removeAdditiveTubeGlow(mesh, options);
  return mesh;
}
