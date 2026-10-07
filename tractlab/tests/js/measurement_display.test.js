import test from 'node:test';
import assert from 'node:assert/strict';
import {layerMeasurement} from '../../viewer/lesion_map.js';

const measurement=(p5,display)=>layerMeasurement({sourcePopulation:'bank:example',bankId:'example',tract:{id:'example'},lineCount:12,
  responseHeaders:new Headers({'X-clearanceP5':p5,'X-clearanceP5Display':display,'X-clearanceFloorMm':'3'})},'bank:example');

test('display rounding is accepted only when consistent with the numeric header',()=>{
  assert.equal(measurement('6.35','6.4').value,'6.4 mm');
  assert.equal(measurement('10.5','10').value,'10 mm');
  assert.equal(measurement('10.5','11').value,'11 mm');
  for(const display of ['999','0','2.9','<b>6.4</b>','Infinity'])
    assert.equal(measurement('6.35',display).value,'6.3 mm',display);
});

test('server sub-floor marker survives numeric header rounding onto the floor',()=>{
  assert.equal(measurement('3.00','>=3').state,'below-floor');
  assert.equal(measurement('3.00','3.0').value,'3.0 mm');
  assert.equal(measurement('2.99','3.0').state,'below-floor');
});
