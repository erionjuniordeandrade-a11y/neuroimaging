/**
 * Pure helpers for the atlas multi-bundle picker. No DOM, no three.js, so node tests can
 * import it directly and atlas_scene.js can share the same limits and tints.
 *
 * The picker owns the *order* of the primary set: the scene assigns tints by that order
 * (first picked = first tint), so a chip can show the same swatch the canvas draws.
 */
export const MAX_BUNDLES=12;
export const BUNDLE_TINTS=Object.freeze([0xe3bf82,0x7cbbc4,0xb39abd,0x8fd0a0,0xe2a1c2,0xa9c4e8]);

/** Toggle `id` in an ordered selection. Adding past `max` is refused (returns the same
 * array identity so callers can detect the refusal). Never reorders existing picks. */
export function toggleBundle(selected,id,max=MAX_BUNDLES){
  if(selected.includes(id))return selected.filter(x=>x!==id);
  if(selected.length>=max)return selected;
  return [...selected,id];
}

/** Tint (hex int) the scene will use for the primary bundle at `index`. */
export function tintForIndex(index){return BUNDLE_TINTS[index%BUNDLE_TINTS.length];}
export const tintCss=hex=>`#${hex.toString(16).padStart(6,'0')}`;

/** Case-insensitive filter over id + label; empty query keeps everything. */
export function filterBundles(bundles,query,label=b=>b.id){
  const q=(query||'').trim().toLowerCase();
  if(!q)return bundles;
  return bundles.filter(b=>b.id.toLowerCase().includes(q)||label(b).toLowerCase().includes(q));
}

/** Short caption for the picker header. */
export function selectionCaption(count,max=MAX_BUNDLES){
  if(!count)return `none · up to ${max}`;
  return count>=max?`${count} / ${max} · limit reached`:`${count} / ${max}`;
}
