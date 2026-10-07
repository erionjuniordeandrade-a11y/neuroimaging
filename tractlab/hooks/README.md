# hooks/

`pre-commit` refuses to let patient imaging enter this repo. It blocks
imaging/surface/transform extensions (`.nii`, `.nii.gz`, `.mif`, `.tck`,
`.dcm`, `.mgz`, ...), rendered rasters (`.png`, `.jpg`, `.tif`, ...),
tabular data (`.csv`/`.tsv`), case-scoped JSON other than `manifest.json`
and `preflight.json`, and any staged blob over 1 MB.

Install it once per clone or worktree: `scripts/install_hooks.sh`. That
points `core.hooksPath` at this tracked directory instead of the untracked
`.git/hooks/`.

It is fail-closed by design: if it cannot determine what is staged or the
size of a staged object, it refuses the commit rather than letting it
through. The size check only (not the content checks) can be waived
per-command with `TRACTLAB_ALLOW_LARGE=1 git commit ...`.
