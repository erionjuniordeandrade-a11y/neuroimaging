/** Served evidence states. Transport failure never supplies missing case facts. */
export function preflightState(stored, drift = []) {
  if (!Array.isArray(drift)) return 'unavailable';
  if (!stored) return 'absent';
  if (stored.schema !== 2 || drift.length) return 'stale';
  return typeof stored.approved_by === 'string' && stored.approved_by.trim()
    ? 'signed' : 'unsigned';
}

export function derivationState(payload) {
  if (!payload || typeof payload.floor_label !== 'string' || !payload.floor_label.trim()) {
    return Object.freeze({status:'unavailable', active:null, kind:null,
      floorLabel:'Geometric evidence unavailable'});
  }
  return Object.freeze({status:'available', active:payload.active ?? null,
    kind:payload.kind ?? null, floorLabel:payload.floor_label,
    unitAssumption:payload.spatial_unit_assumption ?? null});
}

export function underlayHint(name, derivation) {
  if (name === 't1') return `T1 anatomy · ${derivation?.floorLabel || 'geometric evidence unavailable'} · aid only`;
  return ({b0:'DWI b0', fa:'FA map — anisotropy, not tracking cutoff',
    dec:'DEC — R/L=red, A/P=green, S/I=blue (approx)'})[name] || '';
}

function numberOrNull(value) {
  if (value == null || String(value).trim() === '') return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}
const distanceText = n => n >= 10 ? n.toFixed(0) : n.toFixed(1);

export function proximityFromHeaders(headers, ctx = {}) {
  const get = key => (typeof headers?.get === 'function' ? headers.get(key) : headers?.[key]) ?? '';
  const rawP5 = numberOrNull(get('X-clearanceP5'));
  const rawP50 = numberOrNull(get('X-clearanceP50'));
  const floor = numberOrNull(get('X-clearanceFloorMm'));
  const refusal = get('X-clearanceRefusal');
  const numericalBound=numberOrNull(get('X-clearanceNumericalBoundMm'));
  const status = refusal || (rawP5 !== null && (rawP5 < 0 || !(floor > 0)))
    ? 'unavailable' : rawP5 === null ? 'absent' : rawP5 < floor ? 'below_floor' : 'measured';
  const shown = n => n === null || !(floor > 0) ? '' : n < floor ? `<${floor}` : distanceText(n);
  return Object.freeze({status, rawP5, rawP50, floor, numericalBound, near:status === 'below_floor',
    p5:status === 'measured' || status === 'below_floor' ? shown(rawP5) : '',
    p50:shown(rawP50), refusal:String(refusal),
    nShow:String(get('X-nDisplayed') || ctx.nShow || ''),
    nFull:String(get('X-nAnalyticFull') || get('X-nReturned') || ''),
    eligible:String(get('X-clearanceEligible')), population:String(get('X-clearancePopulation')),
    nAnalytic:String(get('X-nAnalytic')), sift:get('X-hasSift2') === '1',
  });
}
