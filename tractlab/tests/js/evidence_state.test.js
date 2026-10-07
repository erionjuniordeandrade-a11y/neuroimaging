import test from 'node:test';
import assert from 'node:assert/strict';
import { preflightState, proximityFromHeaders, derivationState, underlayHint } from '../../viewer/evidence_state.js';
import { provenanceChips } from '../../viewer/provenance.js';

test('any evidence drift invalidates a signed preflight, including unchanged-verdict byte drift', () => {
  const signed = {schema:2, approved_by:'synthetic reviewer'};
  assert.equal(preflightState(signed, []), 'signed');
  for (const drift of [['gradients'], ['evidence:shells'], ['schema'], ['derivation']]) {
    assert.equal(preflightState(signed, drift), 'stale');
  }
  assert.equal(preflightState({...signed, schema:1}, []), 'stale');
  assert.equal(preflightState(null, []), 'absent');
});

test('unavailable preflight and derivation never claim absence or a three-mm floor', () => {
  const state = derivationState(null);
  assert.equal(state.status, 'unavailable');
  const chips = provenanceChips({space:state.floorLabel, preflightState:'unavailable'});
  assert.equal(chips[0].text, 'DWI space, floor unavailable');
  assert.equal(chips[1].text, 'Preflight unavailable');
  assert.doesNotMatch(underlayHint('t1', state), /no topup|no reverse-PE|3 mm/);
});

test('corrected derivation supplies T1 copy without uncorrected claims', () => {
  const state = derivationState({active:'rpe',kind:'rpe_pair',floor_label:'reverse-PE corrected — residual uncertainty unquantified (delta QC unsigned)'});
  assert.match(underlayHint('t1', state), /reverse-PE corrected/);
  assert.doesNotMatch(underlayHint('t1', state), /no topup|no reverse-PE|3 mm/);
});

test('sub-floor distances retain raw proximity and caution rather than asserting a minimum distance', () => {
  for (const raw of ['0', '1']) {
    const p = proximityFromHeaders(new Headers({'X-clearanceP5':raw,'X-clearanceP5Display':'>=3','X-clearanceFloorMm':'3'}));
    assert.equal(p.rawP5, Number(raw));
    assert.equal(p.status, 'below_floor');
    assert.equal(p.near, true);
    assert.equal(p.p5, '<3');
    assert.doesNotMatch(p.p5, />=|≥/);
  }
});

test('missing floor refuses a numeric interpretation; refusal and no measure are distinct', () => {
  assert.equal(proximityFromHeaders(new Headers({'X-clearanceP5':'1'})).status, 'unavailable');
  assert.equal(proximityFromHeaders(new Headers({'X-clearanceRefusal':'missing uncertainty record'})).status, 'unavailable');
  assert.equal(proximityFromHeaders(new Headers()).status, 'absent');
  const p = proximityFromHeaders(new Headers({'X-clearanceP5':'7.8','X-clearanceFloorMm':'4'}));
  assert.equal(p.p5, '7.8');
  assert.equal(p.near, false);
});
