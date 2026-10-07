import assert from 'node:assert/strict';
import {chromium} from 'playwright';
const preview=process.argv.includes('--preview');
const browser=await chromium.launch({headless:true,channel:'chrome'});
const page=await browser.newPage({viewport:{width:1157,height:556},deviceScaleFactor:2.5});
const errors=[];page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
try{
  await page.goto('http://127.0.0.1:18993/#test');await page.waitForFunction(()=>window.__tractlabTest?.mapReady,undefined,{timeout:120000});
  const identity=()=>page.evaluate(()=>({map:window.__tractlabTest.map.layers,values:[...document.querySelectorAll('.map-value')].map(e=>e.textContent)}));
  const original=await identity();
  if(!preview)assert.equal(await page.locator('.mpr figure').evaluateAll(figures=>figures.every(f=>f.lastElementChild.tagName==='FIGCAPTION')),true,'slice captions retain figure semantics');
  if(preview){
    for(const radius of ['0.14','0.22','0.28']){
      const input=page.locator('#fibre').locator('..').locator('input[type=number]');await input.fill(radius);await input.press('Tab');await page.waitForTimeout(220);
      await page.screenshot({path:`/private/tmp/tractlab-detail-radius-${radius}.png`,scale:'css'});
      assert.deepEqual(await identity(),original,'radius preserves measurements and source populations');
    }
  }else{
    const before=await page.evaluate(()=>({...window.__tractlabTest.map.slices}));
    await page.locator('#expand-cor').click();
    assert.equal(await page.locator('#cax').isVisible(),false);assert.equal(await page.locator('#ccor').isVisible(),true);
    const expanded=await page.locator('#ccor').boundingBox();assert.ok(expanded.width>600&&expanded.height>190,'one-click coronal inspection has meaningful space');
    await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
    const painted=await page.locator('#ccor').evaluate(c=>{const p=c.getContext('2d').getImageData(Math.floor(c.width*.4),Math.floor(c.height*.3),Math.floor(c.width*.2),Math.floor(c.height*.4)).data;return p.filter((v,i)=>i%4!==3&&v>40).length>p.length*.03;});
    assert.equal(painted,true,'expanded slice contains anatomical pixels');
    assert.deepEqual(await page.evaluate(()=>window.__tractlabTest.map.slices),before);
    await page.screenshot({path:'/private/tmp/tractlab-detail-expanded.png',scale:'css'});
    await page.locator('#expand-cor').click();assert.equal(await page.locator('#cax').isVisible(),true);
    assert.deepEqual(await identity(),original,'slice inspection preserves measurement/source identity');
    const target=await page.evaluate(()=>window.__tractlabPickProbe.projectTarget());await page.mouse.click(target.x,target.y);
    await page.waitForFunction(()=>window.__tractlabTest.selected);
    const selected=await page.evaluate(()=>({key:window.__tractlabTest.selected.layerKey,focus:window.__tractlabTest.focusLayerKey,source:document.getElementById('streamlineInspector').dataset.sourcePopulation,inside:!!document.querySelector('#panelTracts #streamlineInspector')}));
    assert.equal(selected.key,selected.focus);assert.equal(selected.key,selected.source);assert.equal(selected.inside,true,'fibre details live beside tract controls');
    assert.equal(await page.locator('#streamlineInspector').isVisible(),true);
    assert.match(await page.locator('#mapScale').textContent(),/Selected streamline/);
    await page.screenshot({path:'/private/tmp/tractlab-detail-inspector.png',scale:'css'});
    await page.locator('#btnClearStreamline').click();assert.equal(await page.locator('#streamlineInspector').isVisible(),false);
    assert.equal(await page.evaluate(()=>window.__tractlabTest.selected),null);
    await page.screenshot({path:'/private/tmp/tractlab-detail-final.png',scale:'css'});
    await page.locator('#expand-cor').click();await page.locator('#btnPaintMode').click();
    const paintBox=await page.locator('#ccor').boundingBox();await page.locator('#ccor').click({position:{x:paintBox.width/2,y:paintBox.height/2}});
    // Painting starts the existing exploratory workflow; named banks must not
    // inherit its new ROI identity. Expansion/inspection above does preserve them.
    await page.waitForFunction(()=>window.__tractlabTest.map.layers.length===0,undefined,{timeout:5000});
    assert.deepEqual((await identity()).values,[],'painting invalidates previously displayed named banks');
    await page.locator('#btnUndoPaint').click();assert.equal(await page.locator('#btnUndoPaint').isDisabled(),true);
  }
  assert.deepEqual(errors,[]);console.log(JSON.stringify({preview,identity:original,errors}));
}finally{await browser.close();}
