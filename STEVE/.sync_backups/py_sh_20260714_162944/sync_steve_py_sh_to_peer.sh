#!/usr/bin/env bash
set -euo pipefail

# One-way STEVE source sync to the peer server.
#
# Scope is intentionally narrow:
#   - sync only *.py and *.sh
#   - never delete peer-only files
#   - exclude data, experiments, logs, caches, and git metadata
#   - keep timestamped backups of overwritten peer files by default
#
# Usage:
#   bash sync_steve_py_sh_to_peer.sh
#   DRY_RUN=true bash sync_steve_py_sh_to_peer.sh
#   PEER=OuXiaoyu@host PEER_DIR=/data/OuXiaoyu/STEVE_CODE/STEVE bash sync_steve_py_sh_to_peer.sh

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/STEVE_CODE/STEVE}"
SYNC_KEY="${SYNC_KEY:-$HOME/.ssh/id_steve_sync_rsa}"
DRY_RUN="${DRY_RUN:-false}"
BACKUP="${BACKUP:-true}"
RUN_COMPILE_CHECK="${RUN_COMPILE_CHECK:-true}"
RUN_BASH_N_CHECK="${RUN_BASH_N_CHECK:-true}"

host_name="$(hostname 2>/dev/null || true)"
host_ips="$(hostname -I 2>/dev/null || true)"

if [ -z "${PEER:-}" ]; then
  case " ${host_name} ${host_ips} " in
    *" gpu-39 "*|*" 211.71.76.25 "*)
      PEER="OuXiaoyu@10.126.62.36"
      ;;
    *" Lin-AI-28 "*|*" 10.126.62.36 "*)
      PEER="OuXiaoyu@211.71.76.25"
      ;;
    *)
      echo "[ERROR] Cannot auto-detect peer from host=${host_name} ips=${host_ips}" >&2
      echo "[ERROR] Please set PEER=OuXiaoyu@host explicitly." >&2
      exit 2
      ;;
  esac
fi

PEER_DIR="${PEER_DIR:-/data/OuXiaoyu/STEVE_CODE/STEVE}"

truthy() {
  case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

if [ ! -d "$PROJECT_DIR" ]; then
  echo "[ERROR] PROJECT_DIR not found: $PROJECT_DIR" >&2
  exit 2
fi
if [ ! -f "$SYNC_KEY" ]; then
  echo "[ERROR] SSH sync key not found: $SYNC_KEY" >&2
  echo "[ERROR] Expected passwordless SSH to be configured before running sync." >&2
  exit 2
fi

ssh_cmd=(
  ssh
  -i "$SYNC_KEY"
  -o IdentitiesOnly=yes
  -o BatchMode=yes
  -o ConnectTimeout=15
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=4
  -o StrictHostKeyChecking=no
)

echo "[sync-steve] local_host=${host_name}"
echo "[sync-steve] project=${PROJECT_DIR}"
echo "[sync-steve] peer=${PEER}:${PEER_DIR}"
echo "[sync-steve] scope=*.py,*.sh only; excludes data/ experiments/ logs/ caches/ .git/"

"${ssh_cmd[@]}" "$PEER" "mkdir -p '$PEER_DIR'"

rsync_args=(
  -avh
  --itemize-changes
  --prune-empty-dirs
  --no-owner
  --no-group
  --exclude=.git/***
  --exclude=.sync_backups/***
  --exclude=experiments/***
  --exclude=data/***
  --exclude=screen_logs/***
  --exclude=auto_rerun_logs/***
  --exclude=__pycache__/***
  --exclude='*.pyc'
  --exclude='*.pth'
  --exclude='*.pt'
  --exclude='*.npz'
  --exclude='*.npy'
  --exclude='*.log'
  --include='*/'
  --include='*.py'
  --include='*.sh'
  --exclude='*'
)

if truthy "$DRY_RUN"; then
  rsync_args+=(--dry-run)
  echo "[sync-steve] DRY_RUN=true; no files will be changed."
fi

if truthy "$BACKUP" && ! truthy "$DRY_RUN"; then
  backup_dir="${PEER_DIR}/.sync_backups/py_sh_$(date +%Y%m%d_%H%M%S)"
  rsync_args+=(--backup --backup-dir="$backup_dir")
  echo "[sync-steve] backup_dir=${backup_dir}"
fi

RSYNC_RSH="$(printf '%q ' "${ssh_cmd[@]}")" \
  rsync "${rsync_args[@]}" "${PROJECT_DIR}/" "${PEER}:${PEER_DIR}/"

if truthy "$DRY_RUN"; then
  echo "[sync-steve] dry-run finished."
  exit 0
fi

if truthy "$RUN_BASH_N_CHECK"; then
  echo "[sync-steve] remote bash -n check"
  "${ssh_cmd[@]}" "$PEER" "cd '$PEER_DIR' && find scripts -maxdepth 1 -type f -name '*.sh' -print0 2>/dev/null | xargs -0 -r -n1 bash -n"
fi

if truthy "$RUN_COMPILE_CHECK"; then
  echo "[sync-steve] remote python compile check"
  "${ssh_cmd[@]}" "$PEER" "cd '$PEER_DIR' && python -m compileall -q run_tds_nyctaxi.py train.py test.py models scripts"
fi

echo "[sync-steve] done"
