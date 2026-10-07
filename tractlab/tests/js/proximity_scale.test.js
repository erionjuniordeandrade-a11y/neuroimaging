import test from 'node:test';
import assert from 'node:assert/strict';
import {heatColor,proximityColor,UNRESOLVED_COLOR,FAR_FADE_START_MM} from '../../viewer/tract_colour.js';
const luminance=rgb=>rgb.map(x=>x<=.04045?x/12.92:((x+.055)/1.055)**2.4).reduce((n,x,i)=>n+x*[.2126,.7152,.0722][i],0);
const BACKGROUND_SRGB=[0x0d/255,0x11/255,0x17/255]; // scene background #0d1117
test('distance magnitude follows monotonically decreasing luminance across the scale',()=>{
  let previous=Infinity;
  for(let d=0;d<=25;d+=.1){const current=luminance(heatColor(d));assert.ok(current<=previous);previous=current;}
});
test('distal fibres retain a readable luminance floor without clipping the distance ramp',()=>{
  assert.ok(luminance(heatColor(25))>=.07);
  for(let d=0;d<=25;d+=.05)assert.ok(heatColor(d).every(v=>v>=0&&v<=1),'ramp stays within sRGB gamut');
});

// --- Far-end recede (2026-09-11): distal fibres (~80% of geometry, 14-25mm)
// desaturate toward a cool graphite so the near-lesion band stays the
// loudest thing on screen. See viewer/tract_colour.js FAR_FADE_START_MM.

test('luminance is strictly monotone decreasing across 0..25mm sampled every 0.5mm (linear light)',()=>{
  let previous=Infinity;
  for(let d=0;d<=25;d+=.5){
    const current=luminance(heatColor(d));
    assert.ok(current<previous,`luminance at ${d}mm (${current}) should be strictly less than at previous sample (${previous})`);
    previous=current;
  }
});

test('far end (25mm) is low-chroma and still lighter than the scene background',()=>{
  const far=heatColor(25);
  const chroma=Math.max(...far)-Math.min(...far);
  assert.ok(chroma<0.16,`chroma at 25mm (${chroma}) should be below 0.16`);
  assert.ok(luminance(far)>luminance(BACKGROUND_SRGB),'25mm colour should read lighter than the #0d1117 background');
});

test('ramp below FAR_FADE_START_MM is untouched: 10mm matches the pre-fade implementation to 1e-3',()=>{
  // Recorded from heatColor(10) before the far-end fade was added.
  const before=[0.144480,0.695457,0.547196];
  const after=heatColor(10);
  assert.ok(FAR_FADE_START_MM>10,'10mm sample must sit below the fade onset for this pin to be meaningful');
  for(let i=0;i<3;i++)assert.ok(Math.abs(after[i]-before[i])<1e-3,`channel ${i}: ${after[i]} vs recorded ${before[i]}`);
});
test('unresolved and missing per-layer floors never encode spurious precision',()=>{
  for(const d of [0,1,2.99])assert.deepEqual(proximityColor(d,3),UNRESOLVED_COLOR);
  assert.notDeepEqual(proximityColor(3,3),UNRESOLVED_COLOR);
  for(const floor of [undefined,null,NaN,-1])assert.deepEqual(proximityColor(10,floor),UNRESOLVED_COLOR);
  assert.deepEqual(proximityColor(5,6),UNRESOLVED_COLOR);
  assert.notDeepEqual(proximityColor(5,3),UNRESOLVED_COLOR);
});
