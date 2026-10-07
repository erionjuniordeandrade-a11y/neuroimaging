/** Tube indices are emitted line-major, with two triangles per radial quad. */
export function streamlineIndexFromTubeFace(faceIndex, lineCount, k, radial) {
  if(![faceIndex,lineCount,k,radial].every(Number.isInteger)
    || faceIndex<0 || lineCount<1 || k<2 || radial<3) return null;
  const facesPerLine=(k-1)*radial*2;
  const index=Math.floor(faceIndex/facesPerLine);
  return index<lineCount?index:null;
}
