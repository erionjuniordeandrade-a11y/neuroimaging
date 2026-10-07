export const DISPLAY_MODES = Object.freeze({
  CLINICAL: "clinical",
  PRESENTER: "presenter",
  TEACHING: "teaching",
});

const VALID_MODES = new Set(Object.values(DISPLAY_MODES));

const MODE_COPY = Object.freeze({
  [DISPLAY_MODES.CLINICAL]: Object.freeze({
    label: "Clinical",
    chipText: "",
    note: "Evidence-first display. Presentation glow and animated trace are off.",
  }),
  [DISPLAY_MODES.PRESENTER]: Object.freeze({
    label: "Presenter",
    chipText: "presentation · evidence dimmed",
    note: "Static tract detail with subdued cortical context. Evidence controls and provenance remain live.",
  }),
  [DISPLAY_MODES.TEACHING]: Object.freeze({
    label: "Teaching",
    chipText: "teaching · symmetric · display only",
    note: "A symmetric gold trace follows a sample of displayed streamlines, or the pinned line. It is display-only and does not show axonal direction, neural conduction, functional necessity, or surgical safety.",
  }),
});

function normaliseMode(mode) {
  return VALID_MODES.has(mode) ? mode : DISPLAY_MODES.CLINICAL;
}

/**
 * Decode display-only URL state. Teaching wins if both legacy-compatible flags
 * are present because it is the more specific mode and already includes the
 * Presenter treatment.
 */
export function displayModeFromSearch(search = "") {
  const params = new URLSearchParams(String(search).replace(/^\?/, ""));
  const profile=params.get('profile');
  if(profile==='evidence') return DISPLAY_MODES.CLINICAL;
  if(profile==='presenter') return DISPLAY_MODES.PRESENTER;
  if(profile==='teaching') return DISPLAY_MODES.TEACHING;
  if (params.get("teaching") === "1") return DISPLAY_MODES.TEACHING;
  if (params.get("presentation") === "1") return DISPLAY_MODES.PRESENTER;
  return DISPLAY_MODES.CLINICAL;
}

/** Return immutable display semantics without touching evidence state. */
export function displayModeSettings(mode, { reducedMotion = false, paused: userPaused = false } = {}) {
  const id = normaliseMode(mode);
  const presentation = id !== DISPLAY_MODES.CLINICAL;
  const teaching = id === DISPLAY_MODES.TEACHING;
  const copy = MODE_COPY[id];
  const paused = teaching && Boolean(reducedMotion || userPaused);
  // Additive copies saturate dense bundles. Presenter uses ordinary surface
  // lighting; only the bounded Teaching traces add a display accent.
  const tubeEmissiveScale = teaching ? 0.10 : (presentation ? 0.38 : 1);
  const tubeOpacity = teaching ? 0.08 : (presentation ? 0.88 : 1);
  const glowOpacity = 0;
  return Object.freeze({
    id,
    label: copy.label,
    chipText: copy.chipText,
    note: paused ? `${copy.note} Animation is paused${reducedMotion ? ' for reduced motion' : ''}.` : copy.note,
    presentation,
    teaching,
    glow: false,
    constellation: presentation,
    animateTrace: teaching && !paused,
    directionNeutral: teaching,
    tubeEmissiveScale,
    tubeOpacity,
    glowOpacity,
    constellationSize: teaching ? 1.2 : (presentation ? 1.8 : 1.1),
  });
}

/**
 * Return canonical display-mode query state while preserving every unrelated
 * deep-link parameter. Clinical is represented by absence of either flag.
 */
export function searchForDisplayMode(search, mode) {
  const params = new URLSearchParams(String(search || "").replace(/^\?/, ""));
  params.delete("presentation");
  params.delete("teaching");
  const id = normaliseMode(mode);
  params.set('profile',id===DISPLAY_MODES.CLINICAL?'evidence':id);
  const encoded = params.toString();
  return encoded ? `?${encoded}` : "";
}
