/** Bind follow-up operations to the displayed layer, never a last-loaded fallback. */
export function analyticIdentity(layer){
  if(!layer)return null;
  if(layer.bankId&&layer.sourcePopulation===`bank:${layer.bankId}`)return {bankId:layer.bankId};
  const resultId=layer.responseHeaders?.get?.('X-resultId');
  return typeof resultId==='string'&&resultId.length?{resultId}:null;
}

export function responseMatchesIdentity(headers,identity){
  if(!identity)return false;
  return identity.bankId?headers.get('X-bankId')===identity.bankId:headers.get('X-resultId')===identity.resultId;
}
