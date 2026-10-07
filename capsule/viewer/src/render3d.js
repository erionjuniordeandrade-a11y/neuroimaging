// WebGL2 ray marcher. int16/uint16 payloads become R32F textures after slope/intercept.
// R32F preserves signed HU and scaled MR values in one sampler3D shader.
const VERTEX_SHADER=`#version 300 es
in vec2 aPos;out vec2 vUV;void main(){vUV=aPos*.5+.5;gl_Position=vec4(aPos,0.,1.);}`;
const FRAGMENT_SHADER=`#version 300 es
precision highp float;precision highp sampler3D;
in vec2 vUV;out vec4 frag;
uniform sampler3D uVolume;uniform sampler3D uMask;
uniform vec3 uCross;uniform vec2 uWindow;uniform vec2 uAngles;uniform float uZoom;
uniform float uThreshold;uniform float uOpacity;uniform int uMode;uniform int uCut;uniform int uPlanes;
vec3 rotate(vec3 p){float cy=cos(uAngles.x),sy=sin(uAngles.x),cp=cos(uAngles.y),sp=sin(uAngles.y);p=vec3(cy*p.x+sy*p.z,p.y,-sy*p.x+cy*p.z);return vec3(p.x,cp*p.y-sp*p.z,sp*p.y+cp*p.z);}
void main(){vec2 uv=(vUV-.5)*vec2(1.2,1.2)/uZoom;vec3 origin=rotate(vec3(uv,.0))+.5;vec3 dir=rotate(vec3(0.,0.,1.));vec3 color=vec3(.025,.039,.060);float alpha=0.;float maximum=0.;
  for(int s=0;s<112;s++){vec3 p=origin+dir*(float(s)-56.)/63.;
    if(any(lessThan(p,vec3(0.)))||any(greaterThan(p,vec3(1.))))continue;
    if(uCut==1&&p.z>uCross.z)continue;if(uCut==2&&p.y>uCross.y)continue;if(uCut==3&&p.x>uCross.x)continue;
    float val=texture(uVolume,p).r;vec4 mask=texture(uMask,p);float g=clamp((val-uWindow.x)/max(1.,uWindow.y)+.5,0.,1.);
    if(uMode==2){maximum=max(maximum,g);continue;}
    float a=0.;vec3 col=vec3(g);
    if(uMode==0){if(val>uThreshold){a=uOpacity*.23;col=mix(vec3(.55,.65,.72),vec3(1.,.85,.57),g);}}
    else{if(val>-.55e3&&val<uThreshold){a=uOpacity*.065;col=vec3(.45,.58,.65);}else if(val>=uThreshold){a=uOpacity*.20;col=vec3(.94,.82,.60);}}
    if(mask.a>.1){a=max(a,uOpacity*.55);col=mix(col,mask.rgb,.85);}
    if(uPlanes==1&&min(min(abs(p.x-uCross.x),abs(p.y-uCross.y)),abs(p.z-uCross.z))<.003){a=max(a,.025);col=mix(col,vec3(.79,.66,.30),.7);}
    color+=(1.-alpha)*a*col;alpha+=(1.-alpha)*a;if(alpha>.98)break;
  }
  if(uMode==2)color=vec3(maximum);frag=vec4(color,1.);
}`;
function compileShader(gl,type,source){const shader=gl.createShader(type);gl.shaderSource(shader,source);gl.compileShader(shader);if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(shader));return shader}
function initGL(){
  const gl=$('three-canvas').getContext('webgl2',{preserveDrawingBuffer:true,antialias:false});if(!gl)throw Error('WebGL2 unavailable');state.gl=gl;
  const program=gl.createProgram();gl.attachShader(program,compileShader(gl,gl.VERTEX_SHADER,VERTEX_SHADER));gl.attachShader(program,compileShader(gl,gl.FRAGMENT_SHADER,FRAGMENT_SHADER));gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));
  const buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,1,1]),gl.STATIC_DRAW);gl.useProgram(program);const loc=gl.getAttribLocation(program,'aPos');gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,2,gl.FLOAT,false,0,0);
  state.glData={program,volumeTextures:new Map(),maskTexture:null,maskDirty:true};
  for(const [id,vol] of state.volumes){const tex=gl.createTexture();gl.bindTexture(gl.TEXTURE_3D,tex);gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_WRAP_R,gl.CLAMP_TO_EDGE);const values=new Float32Array(vol.data.length);for(let i=0;i<values.length;i++)values[i]=vol.data[i]*vol.meta.slope+vol.meta.intercept;gl.pixelStorei(gl.UNPACK_ALIGNMENT,1);gl.texImage3D(gl.TEXTURE_3D,0,gl.R32F,...state.manifest.grid.dims,0,gl.RED,gl.FLOAT,values);state.glData.volumeTextures.set(id,tex)}
  const mask=gl.createTexture();gl.bindTexture(gl.TEXTURE_3D,mask);for(const param of [gl.TEXTURE_MIN_FILTER,gl.TEXTURE_MAG_FILTER])gl.texParameteri(gl.TEXTURE_3D,param,gl.NEAREST);for(const param of [gl.TEXTURE_WRAP_S,gl.TEXTURE_WRAP_T,gl.TEXTURE_WRAP_R])gl.texParameteri(gl.TEXTURE_3D,param,gl.CLAMP_TO_EDGE);state.glData.maskTexture=mask;
  $('gpu-badge').textContent=gl.getParameter(gl.RENDERER)||'WebGL2';
}
function uploadMaskTexture(){const gl=state.gl,{maskTexture}=state.glData,total=state.manifest.grid.dims.reduce((a,b)=>a*b,1),rgba=new Uint8Array(total*4);for(const meta of visibleMasks()){const bytes=state.masks.get(meta.id),hex=meta.color||'#E4572E',rgb=[1,3,5].map(i=>parseInt(hex.slice(i,i+2),16));for(let i=0;i<total;i++)if(bytes[i]){const n=i*4;rgba[n]=rgb[0];rgba[n+1]=rgb[1];rgba[n+2]=rgb[2];rgba[n+3]=255}}gl.bindTexture(gl.TEXTURE_3D,maskTexture);gl.texImage3D(gl.TEXTURE_3D,0,gl.RGBA8,...state.manifest.grid.dims,0,gl.RGBA,gl.UNSIGNED_BYTE,rgba);state.glData.maskDirty=false}
function render3D(){
  const gl=state.gl;if(!gl||!state.base)return;const canvas=$('three-canvas'),box=canvas.getBoundingClientRect();if(!box.width||!box.height)return;
  const w=Math.min(300,Math.max(1,Math.round(box.width))),h=Math.min(250,Math.max(1,Math.round(box.height)));if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h}
  if(state.glData.maskDirty)uploadMaskTexture();const p=state.glData.program;gl.viewport(0,0,w,h);gl.useProgram(p);gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_3D,state.glData.volumeTextures.get(state.base));gl.uniform1i(gl.getUniformLocation(p,'uVolume'),0);gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_3D,state.glData.maskTexture);gl.uniform1i(gl.getUniformLocation(p,'uMask'),1);
  const q=rasToVoxel(...state.crosshair),d=state.manifest.grid.dims;gl.uniform3f(gl.getUniformLocation(p,'uCross'),q[0]/(d[0]-1),q[1]/(d[1]-1),q[2]/(d[2]-1));gl.uniform2f(gl.getUniformLocation(p,'uWindow'),state.window.center,state.window.width);gl.uniform2f(gl.getUniformLocation(p,'uAngles'),state.camera.yaw,state.camera.pitch);gl.uniform1f(gl.getUniformLocation(p,'uZoom'),state.zoom.three);gl.uniform1f(gl.getUniformLocation(p,'uThreshold'),state.threshold3d);gl.uniform1f(gl.getUniformLocation(p,'uOpacity'),state.opacity3d);gl.uniform1i(gl.getUniformLocation(p,'uMode'),{bone:0,skin:1,mip:2}[state.mode3d]);gl.uniform1i(gl.getUniformLocation(p,'uCut'),{off:0,axial:1,coronal:2,sagittal:3}[state.cut3d]);gl.uniform1i(gl.getUniformLocation(p,'uPlanes'),state.planes3d?1:0);gl.drawArrays(gl.TRIANGLE_STRIP,0,4);
}
