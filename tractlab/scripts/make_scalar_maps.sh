#!/usr/bin/env bash
# Derive FA and MD (MRtrix ADC) for along-tract profiles on the demo case.
#
# Source: cases/demo-leipzig-sub-010005/work/ss3t/dwi_preproc.mif
# Mask:   manifest inputs.mask (nifti/mask_up.nii.gz) regridded onto the
#         dwi_preproc grid with nearest-neighbour. NOT dwi2mask — that
#         brain-extraction drops brainstem and zeros inferior CST FA.
# Output: work/profiles/{mask,dt,fa,md}_casemask.*  (native DWI grid)
#         Old dwi2mask products (fa.nii.gz, dt.mif, dwi_preproc_mask.mif)
#         are never read. --force deletes casemask products and rebuilds.
#
# Manifest keys the profile route reads (in order):
#   FA: inputs.fa, then inputs.profile_fa
#   MD: inputs.md, then inputs.profile_md
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: make_scalar_maps.sh [--case-root <dir>] [--force]
  --case-root <dir>  case directory containing manifest.json
                      (default: cases/demo-leipzig-sub-010005)
  --force             delete and rebuild casemask products
  -h, --help          show this help and exit
USAGE
}

NTHREADS="${NTHREADS:-8}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CASE_ROOT_ARG=""
FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --case-root)
      [[ $# -ge 2 ]] || { echo "missing value for --case-root" >&2; usage >&2; exit 2; }
      CASE_ROOT_ARG="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -n "$CASE_ROOT_ARG" ]]; then
  [[ -d "$CASE_ROOT_ARG" ]] || { echo "no such case-root directory: $CASE_ROOT_ARG" >&2; exit 2; }
  ROOT="$(cd "$CASE_ROOT_ARG" && pwd)"
else
  ROOT="$REPO/cases/demo-leipzig-sub-010005"
fi
if [[ ! -f "$ROOT/manifest.json" ]]; then
  echo "missing manifest.json under case root: $ROOT" >&2
  exit 2
fi
MRTRIX="${MRTRIX:-$HOME/mrtrix3/bin}"
PY="${TRACTLAB_PYTHON:-$HOME/fsl/bin/python}"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

# Resolves case_id/case_root/active-lineage mask+DWI paths + the profiles
# output dir, all in one refusal-first pass (ADR-0007): fails closed on a
# missing/invalid case_id or case_root (never "<unknown>"), on a manifest
# whose own case_root does not identify $ROOT (or a location inside it), on
# derivation.validate() failure, and on a missing/absent active-lineage mask.
CASE_INFO="$("$PY" "$REPO/scripts/_case_scalar_maps_inputs.py" "$ROOT")"
CASE_ID=""; CASE_ROOT_CANON=""; ACTIVE_DERIVATION=""; MASK_SRC=""; DWI=""; OUT_DIR=""
while IFS='=' read -r key value; do
  case "$key" in
    case_id) CASE_ID="$value" ;;
    case_root) CASE_ROOT_CANON="$value" ;;
    active_derivation) ACTIVE_DERIVATION="$value" ;;
    mask_path) MASK_SRC="$value" ;;
    dwi_path) DWI="$value" ;;
    out_dir) OUT_DIR="$value" ;;
  esac
done <<<"$CASE_INFO"

OUT="${OUT:-$OUT_DIR}"

echo "case_id: $CASE_ID"
echo "case_root: $CASE_ROOT_CANON"
if [[ -n "$ACTIVE_DERIVATION" ]]; then
  echo "active_derivation: $ACTIVE_DERIVATION (scoped outputs)"
fi

mkdir -p "$OUT"

if [[ ! -f "$DWI" ]]; then
  echo "missing $DWI" >&2
  exit 2
fi
command -v "$MRTRIX/dwi2tensor" >/dev/null
command -v "$MRTRIX/tensor2metric" >/dev/null
command -v "$MRTRIX/mrgrid" >/dev/null

MASK_REGRID="$OUT/mask_casemask.mif"
DT="$OUT/dt_casemask.mif"
FA="$OUT/fa_casemask.nii.gz"
MD="$OUT/md_casemask.nii.gz"

if [[ "$FORCE" -eq 1 ]]; then
  rm -f "$MASK_REGRID" "$DT" "$FA" "$MD" "$OUT/casemask.sources.json"
  echo "FORCE: removed casemask products"
fi

DECIDE_ARGS=("$OUT" "$DWI" "$MASK_SRC")
if [[ "$FORCE" -eq 1 ]]; then
  DECIDE_ARGS+=(--force)
fi
ACTION="$("$PY" -m tractlab.scalar_maps decide "${DECIDE_ARGS[@]}")"
if [[ "$ACTION" == "reuse" ]]; then
  echo "SKIP products match sidecar (dwi+mask hashes unchanged)"
else
  echo "BUILD casemask FA/MD (action=$ACTION)"
fi

run() {
  local out="$1"; shift
  if [[ -e "$out" ]]; then
    echo "SKIP exists: $out"
    return 0
  fi
  echo "RUN $*"
  "$@" -force -nthreads "$NTHREADS"
}

if [[ "$ACTION" != "reuse" ]]; then
  run "$MASK_REGRID" "$MRTRIX/mrgrid" "$MASK_SRC" regrid "$MASK_REGRID" \
    -template "$DWI" -interp nearest
  run "$DT" "$MRTRIX/dwi2tensor" "$DWI" "$DT" -mask "$MASK_REGRID"
  if [[ ! -e "$FA" || ! -e "$MD" ]]; then
    echo "RUN tensor2metric -fa -adc (casemask)"
    "$MRTRIX/tensor2metric" "$DT" \
      -fa "$FA" -adc "$MD" \
      -force -nthreads "$NTHREADS"
  fi
  "$PY" -m tractlab.scalar_maps write "$OUT" "$DWI" "$MASK_SRC"
fi

export OUT MASK_SRC MASK_REGRID FA MD
"$PY" - <<'PY'
import hashlib, os

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

src = os.environ["MASK_SRC"]
regrid = os.environ["MASK_REGRID"]
print(f"mask_source path={src} sha256={sha256(src)}")
print(f"mask_regrid path={os.path.basename(regrid)} sha256={sha256(regrid)}")
for key in ("FA", "MD"):
    path = os.environ[key]
    st = os.stat(path)
    print(f"{os.path.basename(path)} bytes={st.st_size} sha256={sha256(path)}")
PY
echo "OK $OUT"
