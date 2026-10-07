import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from '../../viewer/vendor/three.module.js';
import CameraControls from '../../viewer/vendor/camera-controls.js';
import {restoreCameraPose} from '../../viewer/camera_pose.js';

globalThis.DOMRect ||= class DOMRect {constructor(){this.x=this.y=0;this.width=this.height=1;}};
CameraControls.install({THREE});

test('saved position, target and up survive camera-controls restore and idle updates',()=>{
  const camera=new THREE.PerspectiveCamera(40,1,1,4000),controls=new CameraControls(camera);
  const poses=[
    {position:[37.5,-19.2,83.1],target:[2,3,5],up:[0,0,1]},
    {position:[0,0,100],target:[0,0,0],up:[0,1,0]},
    {position:[31.123,12.52,42.71],target:[7.91,2.1,-13.4],up:[0,0,1]},
  ];
  for(const pose of poses){
    restoreCameraPose(camera,controls,pose.position,pose.target,pose.up);
    for(let i=0;i<4;i++)assert.equal(controls.update(1/60),false,'settled pose cannot drift');
    const actual={position:camera.position.toArray(),target:controls.getTarget().toArray(),up:camera.up.toArray()};
    for(const key of Object.keys(pose))for(let axis=0;axis<3;axis++)
      assert.ok(Math.abs(actual[key][axis]-pose[key][axis])<1e-10,`${key}[${axis}] changed`);
  }
  controls.dispose();
});
