/** Set the complete saved pose in camera-controls' orbit state and the camera. */
export function restoreCameraPose(camera,controls,position,target,up){
  const p=position.toArray?.()||position,t=target.toArray?.()||target,u=up.toArray?.()||up;
  camera.up.fromArray(u);controls.updateCameraUp();
  controls.setLookAt(...p,...t,false);controls.stop();controls.update(0);
}
