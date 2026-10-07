# Case Capsule

One offline HTML file per imaging case: CT and MRI co-registered into one space,
viewed and annotated by the surgeon, explained to the patient in pt-BR.
Not a diagnostic device. See `SPEC.md`.

`capsule dwi` runs resumable preprocessing, T1/FLAIR alignment, registration
checks, profile-specific CSD, probabilistic tractography, bundle QC, and
`build-nifti`. `--profile full` uses SS3T and 1.5M seeds per bundle;
`--profile fast` uses two-tissue MSMT-CSD and 750k seeds. Orientation and
registration checks run before tracking. The resulting tracts are unreviewed
anatomical aids, not diagnostic findings. Use `--dry-run` to inspect the argv
plan and `--check-tools` to verify the local toolchain.
Bundle QC reports descriptive sanity checks for streamline count, midline
crossings, and wrong-hemisphere centroids. It does not assess tract validity.
