# Third-party notices

## human-brain tract pulse technique

The normalized arc-length attribute and GPU pulse technique in
`viewer/teaching_trace.js` are adapted from
[human-brain](https://github.com/amyleesterling/human-brain). TractLab uses a
new symmetric, oscillating trace so the display does not encode tract
direction.

MIT License

Copyright (c) 2026 Amy Sterling

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Reference atlas package

The separate atlas viewer includes public reference assets prepared by Amy
Sterling, pinned at human-brain commit
`e02876655d0216995340a9c153d37d49ff649078`. These assets have their upstream
data terms, separate from the software MIT license. The vertex correspondence
and nearest-corner picking technique are adapted from her MIT-licensed
`js/brain-surface.js`. No H01 assets or copied lesson prose are included.

- HCP S1200 cortex and paired HCP-MMP1 labels: [HCP Open Access Data Use
  Terms](viewer/atlas/licenses/hcp-data-use-terms.txt). Van Essen et al., 2013,
  doi:10.1016/j.neuroimage.2013.05.041; Glasser et al., 2016,
  doi:10.1038/nature18933. The compressed vertex order is preserved; only the
  Glasser block is extracted from the label container.
- HCP1065 sampled tract geometry: Yeh, 2022,
  doi:10.1038/s41467-022-32595-4, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
  This binary and its metadata remain under CC BY-SA 4.0. The sample has at
  most 220 streamlines per bundle, resampled to 28 points by the reference
  preparation pipeline. [Upstream terms](https://brain.labsolver.org/hcp_trk_atlas.html).
- Melbourne Subcortex scale 1 geometry: Tian et al., 2020,
  doi:10.1038/s41593-020-00711-6, [Melbourne license](viewer/atlas/licenses/melbourne-subcortex.txt).
  Label surfaces were smoothed by the reference pipeline; thalamic subdivisions
  are merged. No precise nuclei or biological routes are inferred.
- Inferior template context derived from the MNI152NLin2009cAsym brain mask:
  Fonov et al., 2011, doi:10.1016/j.neuroimage.2010.07.033,
  [MNI permission notice](viewer/atlas/licenses/mni-template-license.txt).
  This is a mask remainder, not a named cerebellar or brainstem segmentation.

Data were provided in part by the Human Connectome Project, WU-Minn Consortium
(Principal Investigators: David Van Essen and Kamil Ugurbil; 1U54MH091657)
funded by the 16 NIH Institutes and Centers that support the NIH Blueprint for
Neuroscience Research; and by the McDonnell Center for Systems Neuroscience
at Washington University.

The exact source/output hashes and modifications are recorded in
`viewer/atlas/manifest.json`. Different reference constructions share an MNI
display convention; this juxtaposition supplies no cross-atlas quantitative
registration or patient-to-atlas mapping.

## Three.js and Draco

Three.js is pinned in package-lock.json. Its MIT license and bundled Draco
decoder license are restored with the runtime by `npm ci`. Runtime files are
served locally; no external CDN or telemetry is used.
