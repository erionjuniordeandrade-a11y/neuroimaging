/**
 * Small, dependency-free Three.js render pipeline for the workstation view.
 *
 * It preserves the scene's stock PBR materials, renders them into an MSAA
 * target, derives a restrained screen-space AO term from that target's depth,
 * and composites the result once.  The AO target is deliberately half-scale:
 * it is a depth cue, never an analytic overlay.
 */

const DEFAULT_QUALITY = Object.freeze({
  maxPixelRatio: 2,
  maxPixels: 2_250_000,
  interactiveMaxPixelRatio: 1.25,
  interactiveMaxPixels: 900_000,
  minPixelRatio: 0.5,
});

const AO_VERTEX_SHADER = /* glsl */ `
varying vec2 vTractlabUv;
void main() {
  vTractlabUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

// The AO pass intentionally compares only nearby depth samples. It does not
// infer anatomy, change tube colour, or create a pseudo-measurement.
const AO_FRAGMENT_SHADER = /* glsl */ `
uniform sampler2D tTractlabDepth;
uniform vec2 uTractlabDepthResolution;
uniform float uTractlabNear;
uniform float uTractlabFar;
uniform float uTractlabTanHalfFov;
uniform float uTractlabAspect;
uniform float uTractlabRadiusMm;
uniform float uTractlabBiasMm;
uniform float uTractlabStrength;
uniform float uTractlabTightPx;
varying vec2 vTractlabUv;

float tractlabViewDepth(float depth) {
  float viewZ = (uTractlabNear * uTractlabFar) /
    ((uTractlabFar - uTractlabNear) * depth - uTractlabFar);
  return -viewZ;
}

float tractlabOccludes(vec2 uv, float centreDepth) {
  float rawDepth = texture2D(tTractlabDepth, clamp(uv, 0.0, 1.0)).x;
  if (rawDepth >= 0.99999) return 0.0;
  float neighbourDepth = tractlabViewDepth(rawDepth);
  float delta = centreDepth - neighbourDepth;
  float withinRadius = 1.0 - smoothstep(uTractlabBiasMm, uTractlabRadiusMm, delta);
  return step(uTractlabBiasMm, delta) * withinRadius;
}

void main() {
  float rawDepth = texture2D(tTractlabDepth, vTractlabUv).x;
  if (rawDepth >= 0.99999) {
    gl_FragColor = vec4(1.0);
    return;
  }

  float centreDepth = tractlabViewDepth(rawDepth);
  float safeDepth = max(centreDepth, 0.001);
  float radiusY = clamp(
    uTractlabRadiusMm / (2.0 * safeDepth * max(uTractlabTanHalfFov, 0.001)),
    0.00075,
    0.075
  );
  vec2 radiusUv = vec2(radiusY / max(uTractlabAspect, 0.001), radiusY);

  // Two rings: the mm ring separates bundles; the tight pixel ring gives a
  // contact shadow where a filament crosses one just behind it (tubes are
  // ~1.4 px, so a mm-only ring mostly samples background).
  vec2 tightUv = uTractlabTightPx / uTractlabDepthResolution;
  float occlusion = 0.0;
  for (int i = 0; i < 8; i++) {
    float a = float(i) * 0.78539816;
    vec2 dir = vec2(cos(a), sin(a));
    occlusion += tractlabOccludes(vTractlabUv + radiusUv * dir, centreDepth);
    vec2 tdir = vec2(cos(a + 0.39269908), sin(a + 0.39269908));
    occlusion += tractlabOccludes(vTractlabUv + tightUv * tdir, centreDepth);
  }
  float ao = 1.0 - clamp((occlusion / 16.0) * uTractlabStrength * 2.0, 0.0, 0.55);
  gl_FragColor = vec4(vec3(ao), 1.0);
}
`;

// The context overlay (glass hull) is premultiplied by NormalBlending onto a
// transparent target, so it lays over the AO-shaded scene without taking AO.
const COMPOSITE_FRAGMENT_SHADER = /* glsl */ `
uniform sampler2D tTractlabColour;
uniform sampler2D tTractlabAo;
uniform sampler2D tTractlabOverlay;
uniform float uTractlabOverlayOn;
varying vec2 vTractlabUv;

void main() {
  vec4 colour = texture2D(tTractlabColour, vTractlabUv);
  float ao = texture2D(tTractlabAo, vTractlabUv).r;
  vec4 overlay = texture2D(tTractlabOverlay, vTractlabUv) * uTractlabOverlayOn;
  gl_FragColor = vec4(colour.rgb * ao * (1.0 - overlay.a) + overlay.rgb, colour.a);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`;

function clampStrength(value) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(0, Math.min(1, number)) : 0.62;
}

function finitePositive(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : fallback;
}

function positiveInteger(value, fallback) {
  return Math.max(1, Math.round(finitePositive(value, fallback)));
}

function deviceDpr() {
  return finitePositive(globalThis.devicePixelRatio, 1);
}

function createVector2(THREE, x, y) {
  if (typeof THREE.Vector2 === "function") return new THREE.Vector2(x, y);
  return {
    x,
    y,
    set(nextX, nextY) {
      this.x = nextX;
      this.y = nextY;
      return this;
    },
  };
}

function setVector2(vector, x, y) {
  if (typeof vector?.set === "function") vector.set(x, y);
  else if (vector) {
    vector.x = x;
    vector.y = y;
  }
}

function perspectiveTanHalfFov(camera) {
  const fov = finitePositive(camera?.fov, 40);
  return Math.tan((fov * Math.PI) / 360);
}

function maxSamples(renderer) {
  if (!renderer?.capabilities?.isWebGL2) return 0;
  const supported = Math.floor(finitePositive(renderer.capabilities.maxSamples, 4));
  return Math.max(0, Math.min(4, supported));
}

function supportsDepthSampling(renderer) {
  return Boolean(
    renderer?.capabilities?.isWebGL2 ||
    renderer?.extensions?.has?.("WEBGL_depth_texture")
  );
}

/**
 * Resolves a renderer scale from the CSS viewport and a strict pixel budget.
 * The ratio is lowered during interaction, then restores on the next settled
 * render. It is a presentation-quality policy only; it does not alter data.
 */
export function resolveRenderQuality({
  width = 1,
  height = 1,
  devicePixelRatio = deviceDpr(),
  interacting = false,
  maxPixelRatio = DEFAULT_QUALITY.maxPixelRatio,
  maxPixels = DEFAULT_QUALITY.maxPixels,
  interactiveMaxPixelRatio = DEFAULT_QUALITY.interactiveMaxPixelRatio,
  interactiveMaxPixels = DEFAULT_QUALITY.interactiveMaxPixels,
  minPixelRatio = DEFAULT_QUALITY.minPixelRatio,
} = {}) {
  const cssWidth = positiveInteger(width, 1);
  const cssHeight = positiveInteger(height, 1);
  const dpr = finitePositive(devicePixelRatio, 1);
  const cap = interacting
    ? Math.min(finitePositive(maxPixelRatio, 2), finitePositive(interactiveMaxPixelRatio, 1.25))
    : finitePositive(maxPixelRatio, 2);
  const budget = interacting
    ? finitePositive(interactiveMaxPixels, DEFAULT_QUALITY.interactiveMaxPixels)
    : finitePositive(maxPixels, DEFAULT_QUALITY.maxPixels);
  const floor = Math.min(finitePositive(minPixelRatio, 0.5), cap);
  const budgetRatio = Math.sqrt(budget / (cssWidth * cssHeight));
  const pixelRatio = Math.max(floor, Math.min(dpr, cap, budgetRatio));
  const drawWidth = Math.max(1, Math.round(cssWidth * pixelRatio));
  const drawHeight = Math.max(1, Math.round(cssHeight * pixelRatio));

  return {
    width: cssWidth,
    height: cssHeight,
    drawWidth,
    drawHeight,
    pixelRatio,
    pixelCount: drawWidth * drawHeight,
    interacting: Boolean(interacting),
    budgetExceededAtFloor: drawWidth * drawHeight > budget,
  };
}

function resolveColourTarget(renderer, THREE) {
  const floatExtension = renderer.extensions?.has?.("EXT_color_buffer_float")
    ? "EXT_color_buffer_float"
    : (renderer.extensions?.has?.("EXT_color_buffer_half_float") ? "EXT_color_buffer_half_float" : null);
  if (floatExtension && THREE.HalfFloatType !== undefined) {
    return { type: THREE.HalfFloatType, hdr: true, name: "half-float", extension: floatExtension };
  }
  return {
    type: THREE.UnsignedByteType,
    hdr: false,
    name: "unsigned-byte-fallback",
    extension: null,
  };
}

function createTarget(THREE, width, height, {
  depth = false,
  samples = 0,
  type = THREE.UnsignedByteType,
  colorSpace = THREE.LinearSRGBColorSpace,
} = {}) {
  const depthTexture = depth
    ? new THREE.DepthTexture(
      width,
      height,
      samples > 0 ? THREE.UnsignedIntType : (THREE.UnsignedShortType ?? THREE.UnsignedIntType)
    )
    : null;
  if (depthTexture) depthTexture.format = THREE.DepthFormat;
  const target = new THREE.WebGLRenderTarget(width, height, {
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    format: THREE.RGBAFormat,
    type,
    colorSpace,
    depthBuffer: depth,
    stencilBuffer: false,
    resolveDepthBuffer: depth,
    depthTexture,
    samples,
  });
  target.texture.generateMipmaps = false;
  target.depthTexture = depthTexture;
  target.samples = samples;
  return target;
}

/**
 * Creates the render pipeline without mutating scene objects. The caller owns
 * scene/camera/lights; this object owns only its targets, fullscreen meshes,
 * and shader materials.
 */
export function createRenderPipeline(renderer, {
  THREE,
  scene,
  camera,
  ao = true,
  aoScale = 0.5,
  aoRadiusMm = 12,
  aoBiasMm = 0.8,
  aoStrength = 0.62,
  aoTightPx = 5,
  overlayLayer = 1,
  ...qualityOptions
} = {}) {
  if (!renderer || typeof renderer.render !== "function") {
    throw new TypeError("createRenderPipeline requires a Three.js renderer");
  }
  if (!scene || !camera) {
    throw new TypeError("createRenderPipeline requires scene and camera");
  }
  if (!THREE) {
    throw new TypeError("createRenderPipeline requires the vendored THREE namespace");
  }

  const canUseAo = Boolean(
    ao &&
    supportsDepthSampling(renderer) &&
    typeof renderer.setRenderTarget === "function" &&
    typeof THREE.WebGLRenderTarget === "function" &&
    typeof THREE.DepthTexture === "function" &&
    typeof THREE.Scene === "function" &&
    typeof THREE.OrthographicCamera === "function" &&
    typeof THREE.PlaneGeometry === "function" &&
    typeof THREE.ShaderMaterial === "function" &&
    typeof THREE.Mesh === "function"
  );
  const resolvedAoScale = Math.max(0.25, Math.min(1, finitePositive(aoScale, 0.5)));
  const samples = canUseAo ? maxSamples(renderer) : 0;
  const colourTarget = canUseAo
    ? resolveColourTarget(renderer, THREE)
    : { type: null, hdr: false, name: "direct-render", extension: null };
  let disposed = false;
  let interacting = false;
  let width = 1;
  let height = 1;
  let dpr = deviceDpr();
  let quality = resolveRenderQuality({ ...qualityOptions, width, height, devicePixelRatio: dpr });

  let sceneTarget = null;
  let aoTarget = null;
  let fullscreenGeometry = null;
  let fullscreenCamera = null;
  let aoScene = null;
  let compositeScene = null;
  let aoMaterial = null;
  let compositeMaterial = null;
  let overlayTarget = null;
  let overlayDrawn = false;
  // Shared with overlay materials: they discard fragments behind the resolved
  // scene depth, so the hull never paints over a tube that sits in front of it.
  const overlayUniforms = {
    tTractlabSceneDepth: { value: null },
    uTractlabSceneSize: { value: createVector2(THREE, 1, 1) },
    uTractlabSceneDepthOn: { value: 0 },
  };

  if (canUseAo) {
    sceneTarget = createTarget(THREE, 1, 1, {
      depth: true,
      samples,
      type: colourTarget.type,
    });
    aoTarget = createTarget(THREE, 1, 1, { type: THREE.UnsignedByteType });
    overlayTarget = createTarget(THREE, 1, 1, {
      depth: true,
      samples,
      type: colourTarget.type,
    });
    overlayUniforms.tTractlabSceneDepth.value = sceneTarget.depthTexture;
    fullscreenGeometry = new THREE.PlaneGeometry(2, 2);
    fullscreenCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    const depthResolution = createVector2(THREE, 1, 1);
    aoMaterial = new THREE.ShaderMaterial({
      uniforms: {
        tTractlabDepth: { value: sceneTarget.depthTexture },
        uTractlabDepthResolution: { value: depthResolution },
        uTractlabNear: { value: finitePositive(camera.near, 1) },
        uTractlabFar: { value: finitePositive(camera.far, 4000) },
        uTractlabTanHalfFov: { value: perspectiveTanHalfFov(camera) },
        uTractlabAspect: { value: finitePositive(camera.aspect, 1) },
        uTractlabRadiusMm: { value: finitePositive(aoRadiusMm, 12) },
        uTractlabBiasMm: { value: Math.max(0.01, finitePositive(aoBiasMm, 0.8)) },
        uTractlabStrength: { value: clampStrength(aoStrength) },
        uTractlabTightPx: { value: finitePositive(aoTightPx, 5) },
      },
      vertexShader: AO_VERTEX_SHADER,
      fragmentShader: AO_FRAGMENT_SHADER,
      depthTest: false,
      depthWrite: false,
      toneMapped: false,
    });
    compositeMaterial = new THREE.ShaderMaterial({
      uniforms: {
        tTractlabColour: { value: sceneTarget.texture },
        tTractlabAo: { value: aoTarget.texture },
        tTractlabOverlay: { value: overlayTarget.texture },
        uTractlabOverlayOn: { value: 0 },
      },
      vertexShader: AO_VERTEX_SHADER,
      fragmentShader: COMPOSITE_FRAGMENT_SHADER,
      depthTest: false,
      depthWrite: false,
      toneMapped: true,
    });
    aoScene = new THREE.Scene();
    compositeScene = new THREE.Scene();
    aoScene.add(new THREE.Mesh(fullscreenGeometry, aoMaterial));
    compositeScene.add(new THREE.Mesh(fullscreenGeometry, compositeMaterial));
  }

  function updateCameraUniforms() {
    if (!aoMaterial) return;
    aoMaterial.uniforms.uTractlabNear.value = finitePositive(camera.near, 1);
    aoMaterial.uniforms.uTractlabFar.value = finitePositive(camera.far, 4000);
    aoMaterial.uniforms.uTractlabTanHalfFov.value = perspectiveTanHalfFov(camera);
    aoMaterial.uniforms.uTractlabAspect.value = finitePositive(camera.aspect, width / height);
  }

  function applySize() {
    quality = resolveRenderQuality({
      ...qualityOptions,
      width,
      height,
      devicePixelRatio: dpr,
      interacting,
    });
    renderer.setPixelRatio?.(quality.pixelRatio);
    renderer.setSize?.(width, height, false);
    if (!canUseAo) return;

    sceneTarget.setSize(quality.drawWidth, quality.drawHeight);
    const aoWidth = Math.max(1, Math.ceil(quality.drawWidth * resolvedAoScale));
    const aoHeight = Math.max(1, Math.ceil(quality.drawHeight * resolvedAoScale));
    aoTarget.setSize(aoWidth, aoHeight);
    overlayTarget.setSize(quality.drawWidth, quality.drawHeight);
    setVector2(overlayUniforms.uTractlabSceneSize.value, quality.drawWidth, quality.drawHeight);
    setVector2(aoMaterial.uniforms.uTractlabDepthResolution.value, quality.drawWidth, quality.drawHeight);
    updateCameraUniforms();
  }

  function diagnostics() {
    const renderInfo = renderer.info?.render;
    const aoLive = canUseAo && !disposed && aoTarget !== null;
    return {
      ...quality,
      aoEnabled: aoLive,
      aoScale: aoLive ? resolvedAoScale : 0,
      aoWidth: aoLive ? aoTarget.width : 0,
      aoHeight: aoLive ? aoTarget.height : 0,
      samples: aoLive ? samples : 0,
      passes: aoLive ? (overlayDrawn ? 4 : 3) : 1,
      overlayDrawn: aoLive && overlayDrawn,
      hdrColorTarget: aoLive && colourTarget.hdr,
      colorTargetType: aoLive ? colourTarget.name : "direct-render",
      colorTargetExtension: aoLive ? colourTarget.extension : null,
      disposed,
      renderer: renderInfo ? {
        calls: renderInfo.calls,
        triangles: renderInfo.triangles,
        points: renderInfo.points,
        lines: renderInfo.lines,
      } : null,
    };
  }

  function resize(nextWidth, nextHeight, nextDpr = dpr) {
    if (disposed) return diagnostics();
    width = positiveInteger(nextWidth, width);
    height = positiveInteger(nextHeight, height);
    dpr = finitePositive(nextDpr, dpr);
    applySize();
    return diagnostics();
  }

  function setInteracting(nextInteracting) {
    if (disposed) return diagnostics();
    const next = Boolean(nextInteracting);
    if (next !== interacting) {
      interacting = next;
      applySize();
    }
    return diagnostics();
  }

  // Lights live on layer 0; the overlay pass must see them too. Returns true
  // when a visible mesh is on the overlay layer.
  function prepareOverlayLayer() {
    if (typeof scene.traverse !== "function" || !camera.layers) return false;
    let found = false;
    scene.traverse((object) => {
      if (object.isLight) object.layers?.enable?.(overlayLayer);
    });
    scene.traverseVisible?.((object) => {
      if ((object.isMesh || object.isPoints) && object.layers?.isEnabled?.(overlayLayer)) found = true;
    });
    return found;
  }

  function render() {
    if (disposed) return false;
    if (!canUseAo) {
      // No depth texture: overlay meshes draw in the main pass, in normal
      // transparent order, without the scene-depth discard.
      overlayUniforms.uTractlabSceneDepthOn.value = 0;
      const mask = camera.layers?.mask;
      camera.layers?.enable?.(overlayLayer);
      try {
        renderer.render(scene, camera);
      } finally {
        if (mask !== undefined) camera.layers.mask = mask;
      }
      return true;
    }

    const previousTarget = renderer.getRenderTarget?.() ?? null;
    const previousAutoClear = renderer.autoClear;
    const renderInfo = renderer.info;
    const previousInfoAutoReset = renderInfo?.autoReset;
    try {
      renderer.autoClear = false;
      // Three resets renderer.info before every render by default. Keep that
      // reset at the start of this pipeline frame, then retain all three pass
      // counts until the composite has completed.
      if (previousInfoAutoReset === true) renderInfo.reset?.();
      if (renderInfo) renderInfo.autoReset = false;
      // frameBrain and named views can change clipping/focal geometry after a
      // resize; AO must use the camera actually used for this scene pass.
      updateCameraUniforms();
      const hasOverlay = prepareOverlayLayer();
      const cameraMask = camera.layers?.mask;
      renderer.setRenderTarget(sceneTarget);
      renderer.clear(true, true, true);
      camera.layers?.disable?.(overlayLayer);
      try {
        renderer.render(scene, camera);
      } finally {
        if (cameraMask !== undefined) camera.layers.mask = cameraMask;
      }

      renderer.setRenderTarget(aoTarget);
      renderer.clear(true, true, true);
      renderer.render(aoScene, fullscreenCamera);

      overlayDrawn = hasOverlay;
      compositeMaterial.uniforms.uTractlabOverlayOn.value = hasOverlay ? 1 : 0;
      if (hasOverlay) {
        // Separate MSAA target: three invalidates a multisampled colour buffer
        // after each resolve, so the scene target cannot take a second pass.
        const background = scene.background;
        const clearAlpha = renderer.getClearAlpha?.();
        overlayUniforms.uTractlabSceneDepthOn.value = 1;
        renderer.setRenderTarget(overlayTarget);
        renderer.setClearAlpha?.(0);
        renderer.clear(true, true, true);
        scene.background = null;
        camera.layers.set(overlayLayer);
        try {
          renderer.render(scene, camera);
        } finally {
          scene.background = background;
          camera.layers.mask = cameraMask;
          if (clearAlpha !== undefined) renderer.setClearAlpha(clearAlpha);
        }
      }

      renderer.setRenderTarget(previousTarget);
      renderer.clear(true, true, true);
      renderer.render(compositeScene, fullscreenCamera);
      return true;
    } finally {
      renderer.setRenderTarget(previousTarget);
      renderer.autoClear = previousAutoClear;
      if (renderInfo) renderInfo.autoReset = previousInfoAutoReset;
    }
  }

  function dispose() {
    if (disposed) return;
    disposed = true;
    sceneTarget?.dispose();
    aoTarget?.dispose();
    overlayTarget?.dispose();
    aoMaterial?.dispose();
    compositeMaterial?.dispose();
    fullscreenGeometry?.dispose();
    sceneTarget = null;
    aoTarget = null;
    overlayTarget = null;
    aoMaterial = null;
    compositeMaterial = null;
    fullscreenGeometry = null;
    aoScene = null;
    compositeScene = null;
    fullscreenCamera = null;
  }

  // Test seam: a strength of 0 must yield an AO-free frame so an off/on diff can prove the pass.
  function setAoStrength(value) {
    if (aoMaterial) aoMaterial.uniforms.uTractlabStrength.value = clampStrength(value);
  }

  const pipeline = { resize, setInteracting, render, dispose, setAoStrength, overlayLayer, overlayUniforms };
  Object.defineProperty(pipeline, "diagnostics", { enumerable: true, get: diagnostics });
  return pipeline;
}
