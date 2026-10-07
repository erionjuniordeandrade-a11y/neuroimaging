import test from "node:test";
import assert from "node:assert/strict";

import {
  createRenderPipeline,
  resolveRenderQuality,
} from "../../viewer/render_pipeline.js";
import {
  configContextMaterial,
  configTubeMesh,
  configTubeMaterial,
} from "../../viewer/scene_materials.js";

function mockThree() {
  const resources = { targets: [], materials: [], geometries: [] };

  class WebGLRenderTarget {
    constructor(width, height, options) {
      this.width = width;
      this.height = height;
      this.options = options;
      this.texture = { generateMipmaps: true };
      this.depthTexture = null;
      this.samples = 0;
      this.disposed = false;
      resources.targets.push(this);
    }

    setSize(width, height) {
      this.width = width;
      this.height = height;
    }

    dispose() {
      this.disposed = true;
    }
  }

  class DepthTexture {
    constructor(width, height, type) {
      this.image = { width, height };
      this.type = type;
    }
  }

  class Scene {
    constructor() {
      this.children = [];
    }

    add(child) {
      this.children.push(child);
    }
  }

  class OrthographicCamera {}

  class PlaneGeometry {
    constructor() {
      this.disposed = false;
      resources.geometries.push(this);
    }

    dispose() {
      this.disposed = true;
    }
  }

  class ShaderMaterial {
    constructor(options) {
      Object.assign(this, options);
      this.disposed = false;
      resources.materials.push(this);
    }

    dispose() {
      this.disposed = true;
    }
  }

  class Mesh {
    constructor(geometry, material) {
      this.geometry = geometry;
      this.material = material;
    }
  }

  return {
    WebGLRenderTarget,
    DepthTexture,
    Scene,
    OrthographicCamera,
    PlaneGeometry,
    ShaderMaterial,
    Mesh,
    LinearFilter: "linear",
    RGBAFormat: "rgba",
    HalfFloatType: "half",
    UnsignedByteType: "ubyte",
    UnsignedIntType: "uint",
    UnsignedShortType: "ushort",
    DepthFormat: "depth",
    NormalBlending: "normal",
    AdditiveBlending: "additive",
    resources,
  };
}

function mockRenderer({ withInfo = false, extensions = [] } = {}) {
  const state = { target: null, renders: [], clears: 0, size: null, pixelRatio: null };
  const renderer = {
    capabilities: { isWebGL2: true, maxSamples: 4 },
    extensions: { has(name) { return extensions.includes(name); } },
    autoClear: true,
    getRenderTarget() { return state.target; },
    setRenderTarget(target) { state.target = target; },
    setPixelRatio(value) { state.pixelRatio = value; },
    setSize(width, height, updateStyle) { state.size = [width, height, updateStyle]; },
    clear() { state.clears += 1; },
    render(scene, camera) {
      state.renders.push([scene, camera]);
      if (this.info) this.info.render.calls += 1;
    },
    state,
  };
  if (withInfo) {
    renderer.info = {
      autoReset: true,
      resetCalls: 0,
      render: { calls: 0, triangles: 0, points: 0, lines: 0 },
      reset() {
        this.resetCalls += 1;
        this.render.calls = 0;
        this.render.triangles = 0;
        this.render.points = 0;
        this.render.lines = 0;
      },
    };
  }
  return renderer;
}

test("quality policy uses native DPR when settled and caps it while interacting", () => {
  const settled = resolveRenderQuality({
    width: 561,
    height: 507,
    devicePixelRatio: 3,
  });
  const interacting = resolveRenderQuality({
    width: 561,
    height: 507,
    devicePixelRatio: 3,
    interacting: true,
  });

  assert.equal(settled.pixelRatio, 2);
  assert.equal(settled.drawWidth, 1122);
  assert.equal(settled.drawHeight, 1014);
  assert.equal(interacting.pixelRatio, 1.25);
  assert.ok(interacting.pixelCount < settled.pixelCount);
});

test("pixel budget lowers DPR for large viewports without lowering it below the policy floor", () => {
  const quality = resolveRenderQuality({
    width: 1920,
    height: 1080,
    devicePixelRatio: 2,
  });

  assert.ok(quality.pixelRatio < 2);
  assert.ok(quality.pixelRatio >= 0.5);
  assert.ok(quality.pixelCount <= 2_250_000);
});

test("pipeline renders a depth-driven half-resolution AO composite and disposes owned resources", () => {
  const THREE = mockThree();
  const renderer = mockRenderer({ extensions: ["EXT_color_buffer_float"] });
  const scene = { kind: "clinical-scene" };
  const camera = { near: 1, far: 4000 };
  const pipeline = createRenderPipeline(renderer, { THREE, scene, camera });

  const diagnostics = pipeline.resize(561, 507, 3);
  assert.equal(diagnostics.aoEnabled, true);
  assert.equal(diagnostics.samples, 4);
  assert.equal(diagnostics.hdrColorTarget, true);
  assert.equal(diagnostics.colorTargetType, "half-float");
  assert.equal(diagnostics.aoWidth, 561);
  assert.equal(diagnostics.aoHeight, 507);
  assert.deepEqual(renderer.state.size, [561, 507, false]);
  assert.equal(THREE.resources.targets[0].options.type, "half");
  assert.equal(THREE.resources.targets[0].options.resolveDepthBuffer, true);

  camera.near = 12;
  camera.far = 800;
  camera.fov = 50;
  camera.aspect = 1.5;

  assert.equal(pipeline.render(), true);
  assert.equal(renderer.state.renders.length, 3, "scene, AO, and composite passes");
  assert.equal(THREE.resources.materials[0].uniforms.uTractlabNear.value, 12);
  assert.equal(THREE.resources.materials[0].uniforms.uTractlabFar.value, 800);
  assert.equal(THREE.resources.materials[0].uniforms.uTractlabAspect.value, 1.5);

  pipeline.setInteracting(true);
  assert.equal(pipeline.diagnostics.pixelRatio, 1.25);

  pipeline.dispose();
  assert.equal(pipeline.render(), false);
  assert.equal(pipeline.diagnostics.disposed, true);
  assert.equal(pipeline.diagnostics.aoEnabled, false);
  assert.equal(pipeline.diagnostics.aoWidth, 0);
  assert.ok(THREE.resources.targets.every((target) => target.disposed));
  assert.ok(THREE.resources.materials.every((material) => material.disposed));
  assert.ok(THREE.resources.geometries.every((geometry) => geometry.disposed));
});

function mockLayers(mask = 1) {
  return {
    mask,
    set(layer) { this.mask = 1 << layer; },
    enable(layer) { this.mask |= 1 << layer; },
    disable(layer) { this.mask &= ~(1 << layer); },
    isEnabled(layer) { return (this.mask & (1 << layer)) !== 0; },
  };
}

test("translucent context renders in its own overlay pass after AO, without the background, onto a transparent clear", () => {
  const THREE = mockThree();
  const renderer = mockRenderer({ extensions: ["EXT_color_buffer_float"] });
  let clearAlpha = 1;
  const alphaAtClear = [];
  renderer.getClearAlpha = () => clearAlpha;
  renderer.setClearAlpha = (value) => { clearAlpha = value; };
  const plainClear = renderer.clear;
  renderer.clear = function () { alphaAtClear.push([this.getRenderTarget(), clearAlpha]); plainClear(); };
  const light = { isLight: true, layers: mockLayers() };
  const hull = { isMesh: true, layers: mockLayers(1 << 1) };
  const tube = { isMesh: true, layers: mockLayers() };
  const background = { kind: "gradient" };
  const scene = {
    background,
    traverse(fn) { [light, hull, tube].forEach(fn); },
    traverseVisible(fn) { [light, hull, tube].forEach(fn); },
  };
  const camera = { near: 1, far: 4000, layers: mockLayers() };
  const seen = [];
  const plainRender = renderer.render;
  renderer.render = function (sceneArg, cameraArg) {
    if (sceneArg === scene) seen.push({ mask: camera.layers.mask, background: scene.background, target: this.getRenderTarget() });
    plainRender.call(this, sceneArg, cameraArg);
  };
  const pipeline = createRenderPipeline(renderer, { THREE, scene, camera });
  pipeline.resize(400, 300, 1);
  pipeline.render();

  const [sceneTarget, , overlayTarget] = THREE.resources.targets;
  assert.equal(renderer.state.renders.length, 4, "scene, AO, overlay, composite");
  assert.deepEqual(seen.map((s) => s.mask), [1, 2], "main pass excludes layer 1; overlay pass is layer 1 only");
  assert.equal(seen[0].target, sceneTarget);
  assert.equal(seen[1].target, overlayTarget);
  assert.equal(seen[1].background, null, "overlay must not repaint the background");
  assert.equal(scene.background, background);
  assert.equal(camera.layers.mask, 1, "camera layers restored");
  assert.ok(light.layers.isEnabled(1), "lights reach the overlay pass");
  assert.deepEqual(alphaAtClear.find(([target]) => target === overlayTarget), [overlayTarget, 0]);
  assert.equal(clearAlpha, 1, "clear alpha restored");
  assert.equal(pipeline.diagnostics.passes, 4);
  assert.equal(pipeline.overlayUniforms.uTractlabSceneDepthOn.value, 1);
  assert.equal(pipeline.overlayUniforms.tTractlabSceneDepth.value, sceneTarget.depthTexture);
  const composite = THREE.resources.materials[1];
  assert.equal(composite.uniforms.uTractlabOverlayOn.value, 1);

  hull.layers.set(0);
  pipeline.render();
  assert.equal(composite.uniforms.uTractlabOverlayOn.value, 0, "no overlay content: composite ignores the stale target");
  assert.equal(pipeline.diagnostics.passes, 3);
});

test("AO strength 0 is honoured (not replaced by the default)", () => {
  const THREE = mockThree();
  const pipeline = createRenderPipeline(mockRenderer(), {
    THREE, scene: { kind: "clinical-scene" }, camera: { near: 1, far: 4000 }, aoStrength: 0,
  });
  assert.equal(THREE.resources.materials[0].uniforms.uTractlabStrength.value, 0);
  pipeline.setAoStrength(0.45);
  assert.equal(THREE.resources.materials[0].uniforms.uTractlabStrength.value, 0.45);
});

test("pipeline reports an explicit unsigned-byte fallback when HDR render targets are unavailable", () => {
  const THREE = mockThree();
  const pipeline = createRenderPipeline(mockRenderer(), {
    THREE,
    scene: { kind: "clinical-scene" },
    camera: { near: 1, far: 4000 },
  });

  pipeline.resize(561, 507, 2);

  assert.equal(pipeline.diagnostics.hdrColorTarget, false);
  assert.equal(pipeline.diagnostics.colorTargetType, "unsigned-byte-fallback");
  assert.equal(THREE.resources.targets[0].options.type, "ubyte");
});

test("pipeline preserves full-frame renderer diagnostics across its three passes", () => {
  const THREE = mockThree();
  const renderer = mockRenderer({ withInfo: true });
  const pipeline = createRenderPipeline(renderer, {
    THREE,
    scene: { kind: "clinical-scene" },
    camera: { near: 1, far: 4000 },
  });
  pipeline.resize(561, 507, 2);

  pipeline.render();

  assert.equal(renderer.info.resetCalls, 1);
  assert.equal(renderer.info.autoReset, true);
  assert.equal(pipeline.diagnostics.renderer.calls, 3);
});

test("tube materials remain opaque depth-writing evidence and add an explicit sub-floor hatch only for a recorded floor", () => {
  const material = {
    transparent: true,
    opacity: 0.35,
    depthWrite: false,
    depthTest: false,
    blending: "additive",
    emissiveIntensity: 0.55,
    userData: {},
  };
  const THREE = { NormalBlending: "normal" };

  configTubeMaterial(material, {
    THREE,
    caseMap: true,
    lowSupport: true,
    geomFloorMm: 2.8,
  });

  assert.equal(material.transparent, false);
  assert.equal(material.opacity, 1);
  assert.equal(material.depthWrite, true);
  assert.equal(material.depthTest, true);
  assert.equal(material.blending, "normal");
  assert.equal(material.userData.renderRole, "tube-evidence");
  assert.equal(material.userData.proximityFloorMm, 2.8);
  assert.ok(material.emissiveIntensity <= 0.16);

  const shader = {
    uniforms: {},
    vertexShader: "void main() {\n#include <begin_vertex>\n}",
    fragmentShader: "void main() {\n#include <color_fragment>\n// lighting begins\n#include <opaque_fragment>\n}",
  };
  material.onBeforeCompile(shader);
  assert.match(shader.vertexShader, /attribute float lesionDistance/);
  assert.match(shader.fragmentShader, /uTractlabSubfloorMm/);
  assert.match(shader.fragmentShader, /tractlabHatch/);
  assert.match(shader.fragmentShader, /#include <color_fragment>[\s\S]*tractlabSubfloorMask[\s\S]*\/\/ lighting begins/);
  assert.match(shader.fragmentShader, /1\.0 - step\(uTractlabSubfloorMm, vTractlabProximity\)/);
  assert.doesNotMatch(shader.fragmentShader, /smoothstep\(/);
});

test("unknown proximity floors keep the material opaque but do not invent a sub-floor band", () => {
  const material = { userData: {} };
  configTubeMaterial(material, { THREE: { NormalBlending: "normal" }, caseMap: true });

  assert.equal(material.userData.proximityFloorMm, null);
  assert.equal(material.userData.proximityBand, "unavailable");
  assert.equal(material.transparent, false);
  assert.equal(material.depthWrite, true);
});

test("context shells stay translucent but never contribute a milky depth occluder", () => {
  const material = { userData: {}, depthWrite: true, transparent: false, opacity: 1 };
  configContextMaterial(material, { THREE: { NormalBlending: "normal" }, opacity: 0.08 });

  assert.equal(material.transparent, true);
  assert.equal(material.opacity, 0.08);
  assert.equal(material.depthWrite, false);
  assert.equal(material.depthTest, true);
  assert.equal(material.blending, "normal");
  assert.equal(material.userData.renderRole, "context-shell");
});

test("tube mesh configuration removes only the legacy shared-geometry additive glow", () => {
  const geometry = {};
  const glowMaterial = { blending: "additive", disposeCalled: false, dispose() { this.disposeCalled = true; } };
  const glow = { geometry, material: glowMaterial, userData: {} };
  const mesh = {
    geometry,
    material: { userData: {} },
    children: [glow],
    remove(child) { this.children = this.children.filter((entry) => entry !== child); },
  };

  configTubeMesh(mesh, { THREE: { NormalBlending: "normal", AdditiveBlending: "additive" } });

  assert.equal(mesh.children.length, 0);
  assert.equal(glowMaterial.disposeCalled, true);
  assert.equal(mesh.material.blending, "normal");
});
