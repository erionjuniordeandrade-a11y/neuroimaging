// Pure, jsdom-free helpers for Proposal C (doubt in context): the hull veil
// tri-state and the ghosted-supported-tube material constants. No THREE.js
// import here — callers translate {side, opacity, depthWrite, points} into
// their own material calls so this module stays testable in plain node.

// The glass shell has no additive points. Solid is deliberately opaque:
// translucent context must never replace the anatomical depth in the AO pass.
// Translucent states draw one nearest front layer (depth pre-pass), so ghost
// is a rim-lit glass outline: alpha = opacity at the silhouette, opacity ×
// HULL_RIM_FLOOR face-on. Double-sided ghosting stacked every fold into milk.
const HULL_RIM_FLOOR = 0.12;
const HULL_VEIL_PARAMS = {
  ghost: { opacity: 0.14, side: 'front', depthWrite: false, points: false, transparent: true },
  mid: { opacity: 0.28, side: 'front', depthWrite: false, points: false, transparent: true },
  solid: { opacity: 1, side: 'front', depthWrite: true, points: false, transparent: false },
};

/**
 * @param {string} state 'ghost' | 'mid' | 'solid'
 * @returns {{opacity:number, side:'front'|'double', depthWrite:boolean, points:boolean}}
 */
function hullVeilParams(state) {
  const p = HULL_VEIL_PARAMS[state];
  if (!p) throw new Error(`unknown hull veil state: ${state}`);
  return { ...p };
}

// FreeSurfer pial cortex (Mid veil, when the case has recon-all): one neutral
// grey ramp by sulcal depth, gyral crowns light and sulcal fundi dark, so the
// folds read without lighting tricks. Grey on purpose: colour is reserved for
// evidence. The voxel hull is hidden while the pial shows (two shells = milk).
// Rim glass like the ghost hull: opacity at the silhouette, opacity ×
// PIAL_RIM_FLOOR face-on, so tracts behind the crowns stay legible.
const PIAL_MID_OPACITY = 0.5;
const PIAL_RIM_FLOOR = 0.3;
const PIAL_GYRUS = [0.9, 0.92, 0.95];
const PIAL_SULCUS = [0.26, 0.29, 0.34];

/** FreeSurfer ?h.sulc (mm; positive = deep) -> linear RGB in [0,1]. */
function pialShade(sulc) {
  const x = Math.min(1, Math.max(0, (Number(sulc) + 4) / 10));
  const t = Number.isFinite(x) ? x * x * (3 - 2 * x) : 0.5;
  return PIAL_GYRUS.map((g, i) => g + (PIAL_SULCUS[i] - g) * t);
}

// Ghost material for the supported-tube mesh under "Only low-support".
// ⛔ Owner ruling (2026-09-01, follow-up to the first landing): vertexColors
// on the ghost let the DEC rainbow accumulate through overdraw and read as a
// full coloured tract in the dense trunk core — colour is reserved for
// evidence, so the ghost must be shape only, colourless. Fixed grey at
// --mute (#6e747e), no vertexColors. Low enough that the low-support
// thread's own luminance stays clearly above it even where many ghost tubes
// overlap (measured; see tests/js/veil.test.js and the capture report).
const GHOST_OPACITY = 0.02;
const GHOST_COLOR = '#6e747e';

/**
 * @returns {{opacity:number, color:string, vertexColors:boolean, transparent:boolean, depthWrite:boolean, emissive:number}}
 */
function ghostMaterialParams() {
  return {
    opacity: GHOST_OPACITY,
    color: GHOST_COLOR,
    vertexColors: false,
    transparent: true,
    depthWrite: false,
    emissive: 0x000000,
  };
}

// Spotlight for the low-support mesh (L.meshLow) while "Only low-support"
// is on: the marked threads are now the SUBJECT, not a footnote next to
// full-bright supported tubes, so they get pulled up from their default
// dim stroke-grammar weight (buildTubeMesh: opacity 0.35, emissiveIntensity
// 0.12) toward near-solid. Radius (×0.3) and the desaturated tint are the
// stroke grammar and are untouched — this only mutates the two brightness
// knobs on the mesh's EXISTING material, never rebuilds geometry or colour.
// Measured (local-contrast method, mean thread px vs mean trunk-baseline
// px): 0.95/0.5 -> ratio 2.47; 0.95/0.8 -> ratio 2.49 (median/p90-based
// reads ~2.7). Owner target was >=3.0; emissiveIntensity has an owner
// ceiling of 0.8 and opacity is already near 1, so 0.8 is the best
// available within the allowed range, reported rather than tuned to pass.
const SPOTLIGHT_OPACITY = 0.95;
const SPOTLIGHT_EMISSIVE_INTENSITY = 0.8;

/**
 * @returns {{opacity:number, emissiveIntensity:number}}
 */
function spotlightLowSupportParams() {
  return {
    opacity: SPOTLIGHT_OPACITY,
    emissiveIntensity: SPOTLIGHT_EMISSIVE_INTENSITY,
  };
}

export {
  hullVeilParams, HULL_RIM_FLOOR, ghostMaterialParams, GHOST_OPACITY, GHOST_COLOR,
  spotlightLowSupportParams, SPOTLIGHT_OPACITY, SPOTLIGHT_EMISSIVE_INTENSITY,
  pialShade, PIAL_MID_OPACITY, PIAL_RIM_FLOOR,
};
