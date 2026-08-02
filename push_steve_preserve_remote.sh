#!/bin/sh
set -eu

OWNER="${OWNER:-fffffffuuuuuccckkk}"
REPO_NAME="${REPO_NAME:-steve}"
BRANCH="${BRANCH:-main}"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SOURCE_DIR="${SOURCE_DIR:-${SCRIPT_DIR}}"
WORK_DIR="${WORK_DIR:-/tmp/${REPO_NAME}-preserve-remote-push}"
REMOTE_URL="${REMOTE_URL:-git@github.com:${OWNER}/${REPO_NAME}.git}"
GIT_AUTHOR_NAME_VALUE="${GIT_AUTHOR_NAME:-OuXiaoyu}"
GIT_AUTHOR_EMAIL_VALUE="${GIT_AUTHOR_EMAIL:-ouxiaoyu@example.com}"
HOST_SHORT=$(hostname -s 2>/dev/null || hostname)
DRY_RUN="${DRY_RUN:-false}"
MAX_PUSH_RETRIES="${MAX_PUSH_RETRIES:-3}"
LOCK_DIR="/tmp/${REPO_NAME}-preserve-remote-push.lock"

case "${1:-}" in
  --dry-run)
    DRY_RUN=true
    shift
    ;;
  --help|-h)
    cat <<EOF
Usage: $0 [--dry-run]

Clone GitHub first, overlay this server's sanitized files without deletion,
then commit and push normally. Remote-only files are preserved.

Environment overrides: SOURCE_DIR WORK_DIR OWNER REPO_NAME BRANCH
                       COMMIT_MESSAGE GIT_AUTHOR_NAME GIT_AUTHOR_EMAIL
EOF
    exit 0
    ;;
  "") ;;
  *)
    printf '[push-steve-preserve] Unknown argument: %s\n' "$1" >&2
    exit 2
    ;;
esac

if [ "$#" -ne 0 ]; then
  printf '[push-steve-preserve] Too many arguments.\n' >&2
  exit 2
fi

log() {
  printf '[push-steve-preserve] %s\n' "$*"
}

truthy() {
  case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

cleanup_lock() {
  rmdir "${LOCK_DIR}" 2>/dev/null || true
}

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  log "Another preserve-remote push is already running: ${LOCK_DIR}"
  exit 2
fi
trap cleanup_lock EXIT HUP INT TERM

if [ ! -d "${SOURCE_DIR}" ]; then
  log "Source directory does not exist: ${SOURCE_DIR}"
  exit 2
fi
if ! command -v git >/dev/null 2>&1 || ! command -v rsync >/dev/null 2>&1; then
  log "Both git and rsync are required."
  exit 2
fi
case "${WORK_DIR}" in
  /tmp/*) ;;
  *)
    log "Refusing non-/tmp WORK_DIR: ${WORK_DIR}"
    exit 2
    ;;
esac
if [ "${WORK_DIR}" = "/tmp" ] || [ "${WORK_DIR}" = "/tmp/" ]; then
  log "Refusing unsafe WORK_DIR: ${WORK_DIR}"
  exit 2
fi

# gpu-39 has an old OpenSSH client, so it uses the existing Paramiko Git
# transport. The single-GPU server reaches GitHub through gpu-39 as a jump host
# because its direct GitHub HTTPS/SSH route is intercepted or unavailable.
TRANSPORT_MODE=native
PARAMIKO_TRANSPORT="${SSH_TRANSPORT:-${SOURCE_DIR}/.push-tools/git_ssh_paramiko.py}"
JUMP_HOST="${GITHUB_JUMP_HOST:-OuXiaoyu@211.71.76.25}"
if [ -x "${PARAMIKO_TRANSPORT}" ]; then
  TRANSPORT_MODE=paramiko
elif [ "${HOST_SHORT}" = "insis-cyy-4090" ]; then
  TRANSPORT_MODE=jump
fi

git_remote() {
  case "${TRANSPORT_MODE}" in
    paramiko)
      env PYTHONWARNINGS=ignore \
        GIT_SSH="${PARAMIKO_TRANSPORT}" GIT_SSH_VARIANT=ssh \
        git "$@"
      ;;
    jump)
      env GIT_SSH_COMMAND="ssh -J ${JUMP_HOST} -o BatchMode=yes" git "$@"
      ;;
    *) git "$@" ;;
  esac
}

verify_remote() {
  log "Checking GitHub repository via transport=${TRANSPORT_MODE}"
  if ! git_remote ls-remote "${REMOTE_URL}" >/dev/null 2>&1; then
    log "Cannot access ${REMOTE_URL}; no push was attempted."
    exit 3
  fi
}

clone_remote_base() {
  log "Cloning current ${OWNER}/${REPO_NAME}:${BRANCH} as the base"
  rm -rf "${WORK_DIR}"
  if ! git_remote clone --no-tags --single-branch --branch "${BRANCH}" \
      "${REMOTE_URL}" "${WORK_DIR}"; then
    log "Clone failed; no local files were uploaded and no remote files changed."
    exit 3
  fi
}

overlay_local_files() {
  log "Overlaying local files without --delete: ${SOURCE_DIR}"
  rsync -a \
    --exclude='.git' \
    --exclude='.git/' \
    --exclude='.push-tools/' \
    --exclude='STEVE/data/' \
    --exclude='data/' \
    --exclude='*.7z' \
    --exclude='*.zip' \
    --exclude='*.tar' \
    --exclude='*.tar.*' \
    --exclude='*.pt' \
    --exclude='*.pth' \
    --exclude='*.ckpt' \
    --exclude='*.bin' \
    --exclude='*.npz' \
    --exclude='*.npy' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='*.lock' \
    --exclude='.DS_Store' \
    "${SOURCE_DIR}/" "${WORK_DIR}/"
}

verify_overlay() {
  log "Checking for blocked artifacts"
  blocked=$(find "${WORK_DIR}" -path "${WORK_DIR}/.git" -prune -o -type f \( \
    -iname '*.7z' -o -iname '*.zip' -o -iname '*.tar' -o -iname '*.tar.*' -o \
    -iname '*.pt' -o -iname '*.pth' -o -iname '*.ckpt' -o -iname '*.bin' -o \
    -iname '*.npz' -o -iname '*.npy' -o -iname '*.pyc' \
  \) -print)
  if [ -n "${blocked}" ]; then
    log "Blocked artifacts found; aborting:"
    printf '%s\n' "${blocked}"
    exit 4
  fi
  if find "${WORK_DIR}" \( -path '*/STEVE/data/*' -o -path '*/data/*' \) \
      -print -quit | grep -q .; then
    log "A data directory is present in the prepared tree; aborting."
    exit 4
  fi
}

prepare_commit() {
  cd "${WORK_DIR}"
  git config user.name "${GIT_AUTHOR_NAME_VALUE}"
  git config user.email "${GIT_AUTHOR_EMAIL_VALUE}"
  git add -A

  deleted=$(git diff --cached --name-status --diff-filter=D || true)
  if [ -n "${deleted}" ]; then
    log "Deletion detected, refusing to continue:"
    printf '%s\n' "${deleted}"
    exit 5
  fi

  if git diff --cached --quiet; then
    log "No local changes to push; remote-only files remain untouched."
    return 1
  fi

  log "Prepared changes (deletions are forbidden):"
  git status --short
  if truthy "${DRY_RUN}"; then
    log "Dry run complete; nothing was committed or pushed."
    return 2
  fi

  message="${COMMIT_MESSAGE:-Update STEVE from ${HOST_SHORT}}"
  git commit -m "${message}" >/dev/null
  return 0
}

push_commit() {
  attempt=1
  while [ "${attempt}" -le "${MAX_PUSH_RETRIES}" ]; do
    log "Pushing normally (never --force), attempt ${attempt}/${MAX_PUSH_RETRIES}"
    if git_remote push origin "${BRANCH}"; then
      log "Done: https://github.com/${OWNER}/${REPO_NAME}"
      return 0
    fi

    if [ "${attempt}" -ge "${MAX_PUSH_RETRIES}" ]; then
      break
    fi
    log "Remote changed during this run; rebasing before retry"
    if ! git_remote pull --rebase origin "${BRANCH}"; then
      log "Rebase conflict. Nothing was force-pushed; inspect ${WORK_DIR}."
      return 6
    fi
    attempt=$((attempt + 1))
  done
  log "Push failed without force. Prepared repository remains at ${WORK_DIR}."
  return 6
}

main() {
  log "host=${HOST_SHORT} source=${SOURCE_DIR}"
  verify_remote
  clone_remote_base
  overlay_local_files
  verify_overlay

  prepare_status=0
  prepare_commit || prepare_status=$?
  case "${prepare_status}" in
    0) push_commit ;;
    1|2) return 0 ;;
    *) return "${prepare_status}" ;;
  esac
}

main
