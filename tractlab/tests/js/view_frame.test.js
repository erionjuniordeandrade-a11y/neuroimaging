import {test} from 'node:test';
import assert from 'node:assert/strict';
import {fitViewFrame,projectFramePoint} from '../../viewer/view_frame.js';

const points=[[-30,-40,-100],[50,35,45],[-30,35,45],[50,-40,-100],[0,0,0]];
for(const [width,height] of [[865,307],[865,515],[850,400],[1350,750]]){
  test(`home frame keeps lesion and all displayed bounds clear of controls at ${width}×${height}`,()=>{
    const insets={left:220,right:140,top:64,bottom:20};
    const frame=fitViewFrame({points,target:[0,0,0],direction:[2.4,-.4,.45],up:[0,0,1],width,height,insets});
    assert.ok(Number.isFinite(frame.distance)&&frame.distance>0);
    for(const point of points){
      const p=projectFramePoint(point,frame);
      assert.ok(p.x>=insets.left-1e-6&&p.x<=width-insets.right+1e-6,`x=${p.x}`);
      assert.ok(p.y>=insets.top-1e-6&&p.y<=height-insets.bottom+1e-6,`y=${p.y}`);
    }
    assert.deepEqual(frame.target,[0,0,0],'fitting must retain lesion as the orbit origin');
  });
}
test('a point-sized subject and degenerate camera input fail predictably',()=>{
  const frame=fitViewFrame({points:[[0,0,0]],target:[0,0,0],direction:[1,0,0],up:[0,0,1],width:600,height:400});
  assert.ok(Number.isFinite(frame.distance));
  assert.throws(()=>fitViewFrame({points:[],target:[0,0,0],direction:[0,0,0],up:[0,0,1],width:600,height:400}));
});
