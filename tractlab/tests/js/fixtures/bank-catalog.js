// Invented metadata only. No case, image or streamline files are accessed.
export const syntheticBankManifest = {
  case_id:'synthetic-test',
  inputs:Object.fromEntries(['cst','slf1','slf3','or','fat'].flatMap(family=>
    ['l','r'].map(side=>[`bank_${family}_${side}`,{
      role:`true_${family}`,path:`tracts/${family}_${side}.tck`,n_streamlines:7,
    }]))),
};
