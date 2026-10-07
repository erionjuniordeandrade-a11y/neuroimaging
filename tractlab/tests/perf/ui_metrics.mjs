// Browser-side audit of visible, enabled HTML text; no case text is retained.
export function visibleUiMetrics(){
  const ctx=document.createElement('canvas').getContext('2d',{willReadFrequently:true});ctx.canvas.width=ctx.canvas.height=1;
  const rgba=css=>{ctx.clearRect(0,0,1,1);ctx.fillStyle=css;ctx.fillRect(0,0,1,1);const p=ctx.getImageData(0,0,1,1).data;return [p[0],p[1],p[2],p[3]/255];};
  const over=(a,b)=>{const alpha=a[3]+b[3]*(1-a[3]);return alpha?[0,1,2].map(i=>(a[i]*a[3]+b[i]*b[3]*(1-a[3]))/alpha).concat(alpha):[0,0,0,0];};
  const lum=c=>c.slice(0,3).map(x=>{x/=255;return x<=.04045?x/12.92:((x+.055)/1.055)**2.4;}).reduce((s,x,i)=>s+x*[.2126,.7152,.0722][i],0);
  const visible=el=>{
    if(!el.checkVisibility({checkOpacity:true,checkVisibilityCSS:true}))return false;
    let r=el.getBoundingClientRect(),box={left:Math.max(0,r.left),right:Math.min(innerWidth,r.right),top:Math.max(0,r.top),bottom:Math.min(innerHeight,r.bottom)};
    for(let p=el.parentElement;p;p=p.parentElement){const s=getComputedStyle(p);if(/hidden|auto|scroll|clip/.test(s.overflow)){const b=p.getBoundingClientRect();box={left:Math.max(box.left,b.left),right:Math.min(box.right,b.right),top:Math.max(box.top,b.top),bottom:Math.min(box.bottom,b.bottom)};}}
    return box.right>box.left&&box.bottom>box.top;
  };
  const metrics={textNodes:0,contrastFailures:[],smallText:[],smallTargets:[],tabStops:0};
  const tree=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT);
  while(tree.nextNode()){
    const node=tree.currentNode,el=node.parentElement;if(!node.textContent.trim()||!el||el.closest('svg,script,style,button:disabled,input:disabled')||!visible(el))continue;
    const style=getComputedStyle(el),size=parseFloat(style.fontSize),label=el.id||el.className||el.tagName;
    const ancestry=[];for(let p=el;p;p=p.parentElement)ancestry.unshift(p);
    let bg=[13,17,23,1];for(const p of ancestry){const s=getComputedStyle(p);bg=over(rgba(s.backgroundColor),bg);}
    let fg=rgba(style.color);for(const p of ancestry)fg[3]*=Number(getComputedStyle(p).opacity);
    fg=over(fg,bg);const a=lum(fg),b=lum(bg),ratio=(Math.max(a,b)+.05)/(Math.min(a,b)+.05);
    const required=size>=24||(size>=18.66&&Number(style.fontWeight)>=700)?3:4.5;
    metrics.textNodes++;
    if(ratio<required-.03)metrics.contrastFailures.push({element:label,size,ratio:+ratio.toFixed(2)});
    if(size<12)metrics.smallText.push({element:label,size});
  }
  for(const el of document.querySelectorAll('button,input,select,summary,a[href],[tabindex]')){
    if(el.disabled||!visible(el))continue;
    if(el.tabIndex>=0)metrics.tabStops++;
    const r=el.getBoundingClientRect();if(r.width<32||r.height<32)metrics.smallTargets.push({element:el.id||el.tagName,width:r.width,height:r.height});
  }
  return metrics;
}
