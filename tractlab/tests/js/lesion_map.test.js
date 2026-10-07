import test from 'node:test';
import assert from 'node:assert/strict';
import {existsSync} from 'node:fs';

test('default lesion map and owned measurements',async t=>{
  const url=new URL('../../viewer/lesion_map.js',import.meta.url);
  assert.ok(existsSync(url),'lesion map is not implemented');
  const {lesionOrigin,initialMapBanks,layerMeasurement}=await import(url);
  const trio=['bank_cst_r','bank_fat_r','bank_slf3_r'];
  const catalog=[...trio,'bank_cst_l'].map(id=>({id}));
  await t.test('cold lesion start loads the trio; explicit URL and no-lesion fallback remain honest',()=>{
    assert.deepEqual(initialMapBanks(catalog,true,'','bank_cst_r'),trio);
    assert.deepEqual(initialMapBanks(catalog,true,'?bank=bank_cst_l','bank_cst_r'),['bank_cst_l']);
    assert.deepEqual(initialMapBanks(catalog,true,'?bank=','bank_cst_r'),[]);
    assert.deepEqual(initialMapBanks(catalog,false,'','bank_cst_r'),['bank_cst_r']);
    assert.deepEqual(initialMapBanks(catalog,false,'',null),[]);
  });
  await t.test('camera and slices share the mask centroid transformed through the recorded affine',()=>{
    const mask=new Uint8Array(12); mask[0]=mask[11]=1;
    const origin=lesionOrigin(mask,[3,2,2],[0,-2,0,10,3,0,0,20,0,0,4,30,0,0,0,1]);
    assert.deepEqual(origin.voxel,[1,.5,.5]);
    assert.deepEqual(origin.world,[9,23,32]);
    assert.equal(lesionOrigin(new Uint8Array(12),[3,2,2],[]),null);
    assert.equal(lesionOrigin(mask,[3,2,3],[]),null);
  });
  const layer=(id,p5='8.7',floor='3')=>({bankId:id,tract:{id,short:id,source:'bank'},sourcePopulation:`bank:${id}`,lineCount:100,
    responseHeaders:new Headers({'X-clearanceP5':p5,'X-clearanceP5Display':'>=3','X-clearanceFloorMm':floor,'X-nAnalytic':'200','X-nAnalyticFull':'300','X-clearancePopulation':'random_sample'})});
  await t.test('each layer owns its raw p5, floor and analytic population',()=>{
    const a=layer(trio[0]),b=layer(trio[1],'12');
    const first=layerMeasurement(a,`bank:${trio[0]}`);
    assert.equal(first.value,'8.7 mm');
    assert.equal(layerMeasurement(b,`bank:${trio[1]}`).value,'12 mm');
    assert.deepEqual(layerMeasurement(a,`bank:${trio[0]}`),first);
    assert.equal(first.analytic,200); assert.equal(first.full,300);
    assert.equal(first.population,'random_sample');
    assert.equal(layerMeasurement(a,`bank:${trio[1]}`).state,'identity-unavailable');
  });
  await t.test('unknown, refused and sub-floor measurements never become reassuring millimetres',()=>{
    for(const floor of ['', '0','-1','NaN']) assert.equal(layerMeasurement(layer(trio[0],'8.7',floor),`bank:${trio[0]}`).state,'unknown-floor');
    const sub=layerMeasurement(layer(trio[0],'0.4'),`bank:${trio[0]}`);
    assert.equal(sub.state,'below-floor'); assert.doesNotMatch(sub.value,/0\.4|≥|>=/);
    const refused=layer(trio[0]); refused.responseHeaders.set('X-clearanceRefusal','<img src=x onerror=alert(1)>');
    assert.equal(layerMeasurement(refused,`bank:${trio[0]}`).state,'refused');
    const missing=layer(trio[0],'');
    assert.equal(layerMeasurement(missing,`bank:${trio[0]}`).state,'unavailable');
  });
});

test("recorded p5 shows the server's display string, never a re-rounded header", async () => {
  const {layerMeasurement}=await import(new URL('../../viewer/lesion_map.js',import.meta.url).href);
  const mk=(p5,display)=>({sourcePopulation:'bank:b',bankId:'b',tract:{id:'b',short:'B'},lineCount:10,
    responseHeaders:new Headers({'X-clearanceP5':p5,'X-clearanceP5Display':display,'X-clearanceFloorMm':'3','X-nAnalytic':'200','X-nAnalyticFull':'200','X-clearancePopulation':'full'})});
  assert.equal(layerMeasurement(mk('6.35','6.4'),'bank:b').value,'6.4 mm');
  assert.equal(layerMeasurement(mk('6.35',''),'bank:b').value,'6.3 mm');
  assert.equal(layerMeasurement(mk('12.4','12'),'bank:b').value,'12 mm');
  assert.equal(layerMeasurement(mk('2.5','>=3'),'bank:b').state,'below-floor');
});
