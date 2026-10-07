import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {decodeAtlasLabels,regionIdentity,decodeAtlasBundle,atlasSelectionFromSearch} from '../../viewer/atlas_data.js';
const root=new URL('../../viewer/atlas/',import.meta.url);
const json=async p=>JSON.parse(await readFile(new URL(p,root),'utf8'));
const bytes=async p=>{const b=await readFile(new URL(p,root));return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);};
test('public atlas assets match the pinned manifest bytes',async()=>{
  const m=await json('manifest.json');
  for(const a of m.assets)assert.equal(createHash('sha256').update(await readFile(new URL(a.path,root))).digest('hex'),a.sha256,a.path);
});
test('cortical labels preserve hemisphere-specific identities and reject a mismatched mesh',async()=>{
  const m=await json('surface.json'),b=await bytes('surface-labels.bin');
  const labels=decodeAtlasLabels(m,b,[32492,32492]);
  assert.equal(labels.L.length,32492);assert(labels.L.includes(8));assert(labels.R.includes(8));
  assert.equal(regionIdentity(m,'L',8).raw,'L_4_ROI');assert.equal(regionIdentity(m,'R',8).raw,'R_4_ROI');
  assert.throws(()=>decodeAtlasLabels(m,b,[32491,32493]));
  assert.throws(()=>decodeAtlasLabels(m,b.slice(0,-2),[32492,32492]));
});
test('atlas tract byte offsets decode their documented side and inferior/superior extent',async()=>{
  const m=await json('tracts.json'),b=await bytes('tracts.bin');
  const left=decodeAtlasBundle(m.bundles.find(x=>x.id==='CST_L'),b).flat();
  const right=decodeAtlasBundle(m.bundles.find(x=>x.id==='CST_R'),b).flat();
  assert(Math.max(...left.map(p=>p[0]))<5);assert(Math.min(...right.map(p=>p[0]))>-5);
  assert(Math.min(...left.map(p=>p[2]))< -30);assert(Math.max(...left.map(p=>p[2]))>50);
  assert.throws(()=>decodeAtlasBundle({...m.bundles[0],offset:b.byteLength-2},b));
});
test('atlas links validate their selection and always restore paused',()=>{
  assert.deepEqual(atlasSelectionFromSearch('?hemi=R&area=8&playing=1&profile=presenter'),{hemi:'R',area:8,profile:'presenter',playing:false});
  assert.equal(atlasSelectionFromSearch('?area=-1&hemi=invalid').area,8);
});
