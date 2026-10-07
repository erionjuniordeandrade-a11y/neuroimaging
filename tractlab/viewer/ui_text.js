/** Preserve only the viewer's small numeric/label markup vocabulary.
 * All attributes and elements from metadata are inert. Never insert parsed
 * nodes directly: build fresh text / allowed presentation nodes instead.
 */
export function setHudMarkup(target, value) {
  const template=document.createElement('template');template.innerHTML=String(value??'');
  const classes=new Set(['metric','val','badge','tag']);
  function copy(node){
    if(node.nodeType===Node.TEXT_NODE)return document.createTextNode(node.textContent);
    if(node.nodeType!==Node.ELEMENT_NODE)return document.createTextNode('');
    if(!['B','U','SPAN','BR','SMALL'].includes(node.tagName))return document.createTextNode(node.outerHTML);
    const out=document.createElement(node.tagName.toLowerCase());
    for(const c of node.classList)if(classes.has(c))out.classList.add(c);
    const op=node.getAttribute('data-op');if(['signed','pilot'].includes(op))out.dataset.op=op;
    for(const child of node.childNodes)out.append(copy(child));return out;
  }
  target.replaceChildren(...[...template.content.childNodes].map(copy));
}
