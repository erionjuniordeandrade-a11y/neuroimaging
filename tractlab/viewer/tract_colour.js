/**
 * Pure tract colour helpers (no DOM / no Three.js).
 * Direction / distance / solid modes for tube vertex colours.
 */

/** Distance→lesion ramp saturates at this mm (matches UI legend). */
export const DIST_MAX = 25.0;

export function lerp3(a, b, t) {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

export function sampleStops(stops, t) {
  t = Math.max(0, Math.min(1, t));
  if (stops.length === 1) return stops[0];
  const x = t * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  return lerp3(stops[i], stops[i + 1], f);
}

// Viridis (Matplotlib), reversed: nearer is lighter. No hazard-red encoding.
export const PROXIMITY_STOPS = [
  [.993248,.906157,.143936],[.741388,.873449,.149561],[.477504,.821444,.318195],
  [.266941,.748751,.440573],[.134692,.658636,.517649],[.127568,.566949,.550556],
  [.163625,.471133,.558148],[.206756,.371758,.553117],[.253935,.265254,.529983],
  [.282623,.140926,.457517],[.267004,.004874,.329415],
];
export const UNRESOLVED_COLOR = [.46,.50,.55];
const srgbToLinear = c => c <= .04045 ? c / 12.92 : ((c + .055) / 1.055) ** 2.4;
const linearToSrgb = c => c <= .0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - .055;

/** Below this distance (mm) the proximity ramp is untouched. From here to
 * DIST_MAX the far end desaturates toward FAR_FADE_TARGET_SRGB so that
 * distal fibres — ~80% of the geometry — stop outshouting the near-lesion
 * band. Exported so the legend can annotate the fade onset later. */
export const FAR_FADE_START_MM = 14.0;

/** Desaturated cool graphite the far end blends toward: sRGB #4d5a70, still
 * a shade above the scene background (#0d1117). */
export const FAR_FADE_TARGET_SRGB = [0x4d / 255, 0x5a / 255, 0x70 / 255];
const FAR_FADE_TARGET_LINEAR = FAR_FADE_TARGET_SRGB.map(srgbToLinear);
const FAR_FADE_TARGET_LUMINANCE =
  .2126 * FAR_FADE_TARGET_LINEAR[0] + .7152 * FAR_FADE_TARGET_LINEAR[1] + .0722 * FAR_FADE_TARGET_LINEAR[2];

const smoothstep01 = (x) => {
  const t = Math.max(0, Math.min(1, x));
  return t * t * (3 - 2 * t);
};

/** Distance scale shared by geometry and legend. Recorded floors are applied separately. */
export function heatColor(d, distMax = DIST_MAX) {
  let linear = sampleStops(PROXIMITY_STOPS, Number(d) / distMax).map(srgbToLinear);
  // Luminance of the raw stop, BEFORE any chroma blend below — this is the
  // value the strictly-decreasing ramp is built on, and it is what the
  // gain lift targets. Capturing it first and rescaling the fade target to
  // match it means the blend below can only move hue/chroma, never y: the
  // luminance ordering is therefore untouched by the far-end fade.
  const y = .2126 * linear[0] + .7152 * linear[1] + .0722 * linear[2];
  const fade = smoothstep01((Number(d) - FAR_FADE_START_MM) / (distMax - FAR_FADE_START_MM));
  if (fade > 0 && FAR_FADE_TARGET_LUMINANCE > 0) {
    const targetAtY = FAR_FADE_TARGET_LINEAR.map((c) => c * (y / FAR_FADE_TARGET_LUMINANCE));
    linear = lerp3(linear, targetAtY, fade);
  }
  // Lift in linear light: preserves chromaticity and strictly ordered luminance.
  // This is the same display-adapted ramp in the tubes and the legend.
  const gain = (.055 + .945 * y) / y;
  return linear.map(c => linearToSrgb(c * gain));
}

export function proximityColor(distance, floorMm) {
  if(!Number.isFinite(distance)||!Number.isFinite(floorMm)||floorMm<0||distance<floorMm) return [...UNRESOLVED_COLOR];
  return heatColor(distance);
}

/** DEC-style |R| |A| |S| with sat/gamma boost for dark UI. */
export function dirColor(tx, ty, tz) {
  let r = Math.abs(tx);
  let g = Math.abs(ty);
  let b = Math.abs(tz);
  const m = Math.max(r, g, b, 1e-6);
  r /= m;
  g /= m;
  b /= m;
  const gamma = 0.82;
  r = Math.pow(r, gamma);
  g = Math.pow(g, gamma);
  b = Math.pow(b, gamma);
  const mean = (r + g + b) / 3;
  const sat = 1.35;
  r = mean + (r - mean) * sat;
  g = mean + (g - mean) * sat;
  b = mean + (b - mean) * sat;
  const lift = 0.06;
  return [
    Math.min(1, Math.max(0, r + lift)),
    Math.min(1, Math.max(0, g + lift)),
    Math.min(1, Math.max(0, b + lift)),
  ];
}

/** rgb floats (0-1 each) → "#rrggbb", clamped and rounded. Used to turn a
 * tube-colour triple into a CSS custom-property value (chip family swatch). */
export function rgbToHex(rgb) {
  const toByte = (v) => Math.max(0, Math.min(255, Math.round(Number(v) * 255)));
  const hex = (v) => toByte(v).toString(16).padStart(2, "0");
  return `#${hex(rgb[0])}${hex(rgb[1])}${hex(rgb[2])}`;
}

/** Bundle accent for solid colour mode. */
export function solidColorForBundle(role, label) {
  const s = `${role || ""} ${label || ""}`.toLowerCase();
  // Recovery overlay — neutral grey: no named family owns grey, so it can
  // never be mistaken for a bundle accent (esp. CST gold in Solid mode)
  if (s.includes("recovery")) return [0.6, 0.6, 0.58];
  if (s.includes("cst")) return [0.95, 0.78, 0.28];
  if (s.includes("fat") || s.includes("aslant")) return [0.2, 0.88, 0.78];
  if (s.includes("slf")) return [0.72, 0.48, 1.0];
  if (s.includes("ifof")) return [0.35, 0.65, 1.0]; // sky blue
  if (s.includes("uf") || s.includes("uncinate")) return [1.0, 0.45, 0.55]; // rose
  if (s.includes("cing")) return [0.55, 0.9, 0.45]; // lime
  // optic radiation before forceps (which contains "or" as substring)
  if (s.includes("true_or") || s.includes("optic") || /\bor\b/.test(s) || s.includes("meyer")) {
    return [0.93, 0.3, 0.78]; // fuchsia-magenta — moved off amber (near-identical to CST gold on sidebar chips)
  }
  if (s.includes("forceps major") || s.includes("occipital")) return [0.55, 0.78, 1.0];
  if (s.includes("forceps") || s.includes("callos") || s.includes("cc")) {
    return [1.0, 0.72, 0.55];
  }
  return [0.7, 0.82, 1.0];
}

/** Same hue, saturation × 0.22 and value × 0.85, opacity 0.35 — reads as a
 * dim thread of the bundle's colour, distinct from recovery grey. Paired with
 * the thin low-support tube radius in buildTubeMesh: evidence class must be
 * legible in the stroke itself, not only via the hide toggle. */
export function lowSupportColor(rgb) {
  const r = Number(rgb[0]);
  const g = Number(rgb[1]);
  const b = Number(rgb[2]);
  const mean = (r + g + b) / 3;
  const sat = 0.30;
  const value = 0.55;
  return {
    rgb: [
      (mean + (r - mean) * sat) * value,
      (mean + (g - mean) * sat) * value,
      (mean + (b - mean) * sat) * value,
    ],
    opacity: 0.35,
  };
}
