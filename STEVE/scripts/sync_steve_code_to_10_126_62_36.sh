#!/usr/bin/env bash
set -euo pipefail

# Sync /data/OuXiaoyu/STEVE_CODE to the target server while avoiding the large
# NYCTaxi_TDS experiment tree.  Only the pure AGCRN pretrained checkpoint needed
# by FPEM pretrained-invariant experiments is copied from that tree.

SRC=${SRC:-/data/OuXiaoyu/STEVE_CODE}
TARGET=${TARGET:-OuXiaoyu@10.126.62.36}
TARGET_BASE=${TARGET_BASE:-/data/OuXiaoyu}
TARGET_DIR=${TARGET_DIR:-${TARGET_BASE}/STEVE_CODE}
CKPT_REL=${CKPT_REL:-STEVE/experiments/NYCTaxi_TDS/pure_agcrn_seed2024/best_val_model.pth}
PROXY_JUMP=${PROXY_JUMP:-}
DRY_RUN=${DRY_RUN:-false}

truthy() {
  case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

if [ ! -d "$SRC" ]; then
  echo "[sync-steve] source directory not found: $SRC" >&2
  exit 2
fi

if [ ! -f "${SRC}/${CKPT_REL}" ]; then
  echo "[sync-steve] required checkpoint not found: ${SRC}/${CKPT_REL}" >&2
  exit 2
fi

SSH_CMD=(ssh)
if [ -n "$PROXY_JUMP" ]; then
  SSH_CMD+=(-J "$PROXY_JUMP")
fi
RSYNC_RSH="${SSH_CMD[*]}"
RSYNC_FLAGS=(-az --human-readable --info=progress2)
if truthy "$DRY_RUN"; then
  RSYNC_FLAGS+=(--dry-run)
fi

echo "[sync-steve] source: $SRC"
echo "[sync-steve] target: ${TARGET}:${TARGET_DIR}"
echo "[sync-steve] excluding: STEVE/experiments/NYCTaxi_TDS/***"
echo "[sync-steve] including checkpoint only: $CKPT_REL"

"${SSH_CMD[@]}" "$TARGET" "mkdir -p '${TARGET_DIR}' '${TARGET_DIR}/$(dirname "$CKPT_REL")'"

rsync "${RSYNC_FLAGS[@]}" \
  -e "$RSYNC_RSH" \
  --exclude "STEVE/experiments/NYCTaxi_TDS/***" \
  "${SRC}/" \
  "${TARGET}:${TARGET_DIR}/"

rsync "${RSYNC_FLAGS[@]}" \
  -e "$RSYNC_RSH" \
  "${SRC}/${CKPT_REL}" \
  "${TARGET}:${TARGET_DIR}/$(dirname "$CKPT_REL")/"

if ! truthy "$DRY_RUN"; then
  echo "[sync-steve] verifying target checkpoint and NYCTaxi_TDS contents"
  "${SSH_CMD[@]}" "$TARGET" "
    set -e
    ls -lh '${TARGET_DIR}/${CKPT_REL}'
    sha256sum '${TARGET_DIR}/${CKPT_REL}'
    find '${TARGET_DIR}/STEVE/experiments/NYCTaxi_TDS' -mindepth 1 -maxdepth 2 -print | sort
  "
fi

echo "[sync-steve] done"
