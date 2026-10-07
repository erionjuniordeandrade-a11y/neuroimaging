import test from 'node:test';
import assert from 'node:assert/strict';
import {toggleBundle,MAX_BUNDLES,BUNDLE_TINTS,tintForIndex,tintCss,filterBundles,selectionCaption} from '../../viewer/bundle_picker.js';

test('toggle adds in pick order and removes without reordering the rest',()=>{
  let s=[];
  s=toggleBundle(s,'CST_L');s=toggleBundle(s,'AF_R');s=toggleBundle(s,'OR_L');
  assert.deepEqual(s,['CST_L','AF_R','OR_L']);
  s=toggleBundle(s,'AF_R');
  assert.deepEqual(s,['CST_L','OR_L']);
});

test('toggle refuses to add past the limit and returns the same array identity',()=>{
  const full=Array.from({length:MAX_BUNDLES},(_,i)=>`B${i}`);
  const out=toggleBundle(full,'EXTRA');
  assert.equal(out,full);
  assert.equal(out.length,MAX_BUNDLES);
  // removing still works at the limit
  assert.equal(toggleBundle(full,'B0').length,MAX_BUNDLES-1);
});

test('tints cycle in pick order and render as 6-digit css hex',()=>{
  assert.equal(tintForIndex(0),BUNDLE_TINTS[0]);
  assert.equal(tintForIndex(BUNDLE_TINTS.length),BUNDLE_TINTS[0]);
  assert.match(tintCss(tintForIndex(1)),/^#[0-9a-f]{6}$/);
});

test('filter matches id or label, case-insensitively; empty query keeps all',()=>{
  const bundles=[{id:'CST_L'},{id:'AF_R'},{id:'OR_L'}];
  const label=b=>({CST_L:'Corticospinal · left',AF_R:'Arcuate · right',OR_L:'Optic radiation · left'})[b.id];
  assert.deepEqual(filterBundles(bundles,'',label),bundles);
  assert.deepEqual(filterBundles(bundles,'arcu',label).map(b=>b.id),['AF_R']);
  assert.deepEqual(filterBundles(bundles,'_l',label).map(b=>b.id),['CST_L','OR_L']);
});

test('caption reports count and the limit',()=>{
  assert.equal(selectionCaption(0),`none · up to ${MAX_BUNDLES}`);
  assert.equal(selectionCaption(3),`3 / ${MAX_BUNDLES}`);
  assert.equal(selectionCaption(MAX_BUNDLES),`${MAX_BUNDLES} / ${MAX_BUNDLES} · limit reached`);
});
