# TractLab for macOS

Run `app/macos/build.sh` from this repository to build
`app/macos/build/TractLab.app`. The app uses this checkout by default; set
`TRACTLAB_REPO` before launch to select another repository checkout. It starts
the local Python server with `~/fsl/bin/python` and keeps case data under
`TRACTLAB_CASES_ROOT` or `~/Library/Application Support/TractLab/cases/`.

The bundle is ad-hoc signed for local use. The build script does not install or
launch the app.
