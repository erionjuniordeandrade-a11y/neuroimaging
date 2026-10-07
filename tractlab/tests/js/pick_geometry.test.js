import assert from 'node:assert/strict';
import test from 'node:test';
import {streamlineIndexFromTubeFace} from '../../viewer/pick_geometry.js';

test('tube face layout maps a 3-line, 4-ring, 6-side indexed tube',()=>{
  const lines=3,k=4,radial=6,indices=[];
  for(let line=0;line<lines;line++)for(let ring=0;ring<k-1;ring++)for(let side=0;side<radial;side++){
    const base=line*k*radial+ring*radial,next=(side+1)%radial;
    indices.push(base+side,base+radial+side,base+next,
      base+next,base+radial+side,base+radial+next);
  }
  for(let face=0;face<indices.length/3;face++){
    const expected=Math.floor(indices[face*3]/(k*radial));
    assert.equal(streamlineIndexFromTubeFace(face,lines,k,radial),expected);
  }
  assert.equal(streamlineIndexFromTubeFace(indices.length/3,lines,k,radial),null);
  assert.equal(streamlineIndexFromTubeFace(-1,lines,k,radial),null);
});
