/** Bounded lighting comparison. Materials and anatomical colours are untouched. */
// Chosen after matched overview/anterior captures in direction and distance.
export const DEFAULT_LIGHTING = 'environment';

// A neutral, local studio capture, using the pinned Three r185 PMREM API.
// No downloaded HDR, scene background change or environment geometry in the case.
function studioEnvironment(THREE, renderer) {
  const room = new THREE.Scene();
  const geometry = new THREE.BoxGeometry(1, 1, 1);
  const materials = [];
  const box = (position, scale, radiance, side = THREE.FrontSide) => {
    const material = new THREE.MeshBasicMaterial({
      color: new THREE.Color(...radiance), side, toneMapped: false,
    });
    materials.push(material);
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(...position); mesh.scale.set(...scale); room.add(mesh);
  };
  box([0, 0, 0], [20, 20, 20], [.18, .18, .18], THREE.BackSide);
  box([0, -9, 4], [12, .1, 7], [5, 5, 5]);
  box([-9, 1, 2], [.1, 8, 12], [2, 2, 2]);
  box([7, 4, 9], [5, 6, .1], [3, 3, 3]);
  const generator = new THREE.PMREMGenerator(renderer);
  try { return generator.fromScene(room, .04, .1, 40, {size: 256}); }
  finally {
    generator.dispose(); geometry.dispose(); materials.forEach(m => m.dispose());
  }
}

export function createSceneLighting(THREE, renderer, scene, initial = DEFAULT_LIGHTING) {
  const hemisphere = new THREE.HemisphereLight(0xeef2f8, 0x1a2030, 1.0);
  const key = new THREE.DirectionalLight(0xfff6e8, 1.45); key.position.set(120, -160, 200);
  const fill = new THREE.DirectionalLight(0xb8d4ff, .7); fill.position.set(-100, 80, 40);
  const rim = new THREE.DirectionalLight(0x9ab0ff, .65); rim.position.set(-140, 120, -80);
  const ambient = new THREE.AmbientLight(0xffffff, .28);
  const lights = [hemisphere, key, fill, rim, ambient]; scene.add(...lights);
  const previous = {environment: scene.environment, intensity: scene.environmentIntensity};
  let environment = null, mode = null, disposed = false;
  function setMode(next) {
    if (disposed) throw new Error('Lighting has been disposed');
    if (!['direct', 'environment'].includes(next)) throw new TypeError('Unknown lighting mode');
    if (next === mode) return;
    if (next === 'environment' && !environment) {
      if (!renderer.extensions.has('EXT_color_buffer_float')) throw new Error('Environment lighting needs floating-point colour targets');
      environment = studioEnvironment(THREE, renderer);
    }
    mode = next;
    for (const light of lights) light.visible = mode === 'direct' || light === key;
    key.intensity = mode === 'direct' ? 1.45 : .8;
    scene.environment = mode === 'environment' ? environment.texture : previous.environment;
    scene.environmentIntensity = mode === 'environment' ? .6 : previous.intensity;
  }
  // Keep the established direct rig on devices without a usable PMREM target.
  try { setMode(initial); } catch (error) {
    if (initial !== 'environment') throw error;
    setMode('direct');
  }
  return {
    setMode,
    get mode() { return mode; },
    dispose() {
      if (disposed) return;
      disposed = true; scene.remove(...lights); environment?.dispose();
      scene.environment = previous.environment; scene.environmentIntensity = previous.intensity;
    },
  };
}
