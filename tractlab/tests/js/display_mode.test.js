import test from "node:test";
import assert from "node:assert/strict";

import {
  DISPLAY_MODES,
  displayModeFromSearch,
  displayModeSettings,
  searchForDisplayMode,
} from "../../viewer/display_mode.js";

test("clinical is the safe default for absent or invalid URL state", () => {
  assert.equal(displayModeFromSearch(""), DISPLAY_MODES.CLINICAL);
  assert.equal(displayModeFromSearch("?presentation=0&teaching=0"), DISPLAY_MODES.CLINICAL);
  assert.equal(displayModeFromSearch("?mode=cinematic"), DISPLAY_MODES.CLINICAL);
});

test("teaching takes precedence over the backward-compatible presentation flag", () => {
  assert.equal(displayModeFromSearch("?presentation=1"), DISPLAY_MODES.PRESENTER);
  assert.equal(displayModeFromSearch("?teaching=1"), DISPLAY_MODES.TEACHING);
  assert.equal(
    displayModeFromSearch("?presentation=1&teaching=1"),
    DISPLAY_MODES.TEACHING,
  );
});

test("teaching inherits presentation but its trace is symmetric and direction-neutral", () => {
  const settings = displayModeSettings(DISPLAY_MODES.TEACHING);
  assert.equal(settings.presentation, true);
  assert.equal(settings.glow, false);
  assert.equal(settings.constellation, true);
  assert.equal(settings.teaching, true);
  assert.equal(settings.animateTrace, true);
  assert.equal(settings.directionNeutral, true);
  assert.match(settings.note, /symmetric/i);
  assert.match(settings.note, /does not show axonal direction/i);
  assert.match(settings.note, /conduction/i);
});

test("Presenter cannot accumulate additive glow; Teaching reserves contrast for traces", () => {
  const clinical = displayModeSettings(DISPLAY_MODES.CLINICAL);
  const presenter = displayModeSettings(DISPLAY_MODES.PRESENTER);
  const teaching = displayModeSettings(DISPLAY_MODES.TEACHING);
  assert.equal(clinical.tubeOpacity, 1);
  assert.equal(clinical.tubeEmissiveScale, 1);
  assert.equal(clinical.glowOpacity, 0);
  assert.ok(presenter.tubeOpacity < clinical.tubeOpacity);
  assert.ok(presenter.tubeEmissiveScale < clinical.tubeEmissiveScale);
  assert.equal(presenter.glowOpacity, 0);
  assert.equal(presenter.glow, false);
  assert.ok(teaching.tubeOpacity < presenter.tubeOpacity);
  assert.ok(teaching.tubeEmissiveScale < presenter.tubeEmissiveScale);
  assert.ok(teaching.constellationSize > 1.1);
});

test("reduced motion pauses Teaching without changing its evidence meaning", () => {
  const settings = displayModeSettings(DISPLAY_MODES.TEACHING, { reducedMotion: true });
  assert.equal(settings.teaching, true);
  assert.equal(settings.animateTrace, false);
  assert.equal(settings.directionNeutral, true);
  assert.match(settings.note, /paused/i);
});

test("URL synchronisation preserves unrelated state and emits one canonical mode flag", () => {
  assert.equal(
    searchForDisplayMode("?bank=AF_L&presentation=1&teaching=1", DISPLAY_MODES.CLINICAL),
    "?bank=AF_L&profile=evidence",
  );
  assert.equal(
    searchForDisplayMode("?bank=AF_L&teaching=1", DISPLAY_MODES.PRESENTER),
    "?bank=AF_L&profile=presenter",
  );
  assert.equal(
    searchForDisplayMode("?bank=AF_L&presentation=1", DISPLAY_MODES.TEACHING),
    "?bank=AF_L&profile=teaching",
  );
});

test('canonical profile wins over legacy flags; manual pause preserves evidence meaning',()=>{
  assert.equal(displayModeFromSearch('?profile=evidence&teaching=1'),'clinical');
  assert.equal(displayModeSettings('teaching',{paused:true}).animateTrace,false);
});
