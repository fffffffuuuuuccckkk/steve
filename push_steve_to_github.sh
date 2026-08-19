#!/bin/sh
set -eu

OWNER="${OWNER:-fffffffuuuuuccckkk}"
REPO_NAME="${REPO_NAME:-steve}"
PRIVATE="${PRIVATE:-false}"
SOURCE_DIR="${SOURCE_DIR:-/data/OuXiaoyu/STEVE_CODE}"
WORK_DIR="${WORK_DIR:-/tmp/${REPO_NAME}-sanitized-push}"
BRANCH="${BRANCH:-main}"
REMOTE_URL="git@github.com:${OWNER}/${REPO_NAME}.git"
API_URL="https://api.github.com/user/repos"
SSH_TRANSPORT="${SSH_TRANSPORT:-${SOURCE_DIR}/.push-tools/git_ssh_paramiko.py}"
CREATE_REPO_IF_MISSING="${CREATE_REPO_IF_MISSING:-false}"

log() {
  printf '[push-steve] %s\n' "$*"
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

repo_exists() {
  PYTHONWARNINGS=ignore GIT_SSH="${SSH_TRANSPORT}" GIT_SSH_VARIANT=ssh \
    git ls-remote "${REMOTE_URL}" >/dev/null 2>&1
}

create_repo_with_gh() {
  if have_cmd gh && gh auth status >/dev/null 2>&1; then
    log "Creating GitHub repo with gh: ${OWNER}/${REPO_NAME}"
    gh repo create "${OWNER}/${REPO_NAME}" --source "${WORK_DIR}" --remote origin \
      $( [ "${PRIVATE}" = "true" ] && printf '%s' '--private' || printf '%s' '--public' ) \
      --confirm >/dev/null 2>&1 || true
    return 0
  fi
  return 1
}

create_repo_with_token() {
  token="${GITHUB_TOKEN:-${GH_TOKEN:-}}"
  if [ -z "${token}" ]; then
    return 1
  fi
  log "Creating GitHub repo with GitHub API: ${OWNER}/${REPO_NAME}"
  payload=$(printf '{"name":"%s","private":%s,"auto_init":false}' "${REPO_NAME}" "${PRIVATE}")
  status=$(curl -sS -o /tmp/create_repo_${REPO_NAME}.json -w '%{http_code}' \
    -H "Authorization: Bearer ${token}" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    -d "${payload}" \
    "${API_URL}" || true)
  if [ "${status}" = "201" ] || [ "${status}" = "422" ]; then
    rm -f /tmp/create_repo_${REPO_NAME}.json
    return 0
  fi
  log "GitHub API create failed with HTTP ${status}:"
  sed -n '1,120p' /tmp/create_repo_${REPO_NAME}.json || true
  rm -f /tmp/create_repo_${REPO_NAME}.json
  return 1
}

create_repo_with_netrc() {
  if [ ! -f "${HOME}/.netrc" ]; then
    return 1
  fi
  log "Trying to create GitHub repo with ~/.netrc credentials: ${OWNER}/${REPO_NAME}"
  payload=$(printf '{"name":"%s","private":%s,"auto_init":false}' "${REPO_NAME}" "${PRIVATE}")
  status=$(curl -n -sS -o /tmp/create_repo_${REPO_NAME}.json -w '%{http_code}' \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    -d "${payload}" \
    "${API_URL}" || true)
  if [ "${status}" = "201" ] || [ "${status}" = "422" ]; then
    rm -f /tmp/create_repo_${REPO_NAME}.json
    return 0
  fi
  log "~/.netrc repo creation unavailable, HTTP ${status}."
  rm -f /tmp/create_repo_${REPO_NAME}.json
  return 1
}

prepare_clean_copy() {
  log "Preparing sanitized copy: ${WORK_DIR}"
  rm -rf "${WORK_DIR}"
  mkdir -p "${WORK_DIR}"
  rsync -a --delete \
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
    --exclude='.DS_Store' \
    "${SOURCE_DIR}/" "${WORK_DIR}/"

  cat > "${WORK_DIR}/.gitignore" <<'EOF'
# Data and compressed artifacts
data/
STEVE/data/
*.7z
*.zip
*.tar
*.tar.*
*.npz
*.npy

# Model weights and binary checkpoints
*.pt
*.pth
*.ckpt
*.bin

# Python/cache/local clutter
__pycache__/
*.pyc
.DS_Store
EOF
}

verify_clean_copy() {
  log "Checking sanitized copy for blocked artifacts"
  blocked=$(find "${WORK_DIR}" -type f \( \
    -iname '*.7z' -o -iname '*.zip' -o -iname '*.tar' -o -iname '*.tar.*' -o \
    -iname '*.pt' -o -iname '*.pth' -o -iname '*.ckpt' -o -iname '*.bin' -o \
    -iname '*.npz' -o -iname '*.npy' -o -iname '*.pyc' \
  \) -print)
  if [ -n "${blocked}" ]; then
    log "Blocked artifacts remain:"
    printf '%s\n' "${blocked}"
    exit 2
  fi
  if find "${WORK_DIR}" -path '*/STEVE/data/*' -print -quit | grep -q .; then
    log "Data directory remains under STEVE/data; aborting."
    exit 2
  fi
}

# push_repo() {
#   cd "${WORK_DIR}"
#   rm -rf .git
#   git init -b "${BRANCH}" >/dev/null
#   git config user.name "${GIT_AUTHOR_NAME:-OuXiaoyu}"
#   git config user.email "${GIT_AUTHOR_EMAIL:-ouxiaoyu@example.com}"
#   git add .
#   git commit -m "${COMMIT_MESSAGE:-Upload sanitized STEVE code and logs}" >/dev/null
#   git remote add origin "${REMOTE_URL}"
#   log "Pushing to ${REMOTE_URL}"
#   PYTHONWARNINGS=ignore GIT_SSH="${SSH_TRANSPORT}" GIT_SSH_VARIANT=ssh \
#     git push -u origin "${BRANCH}" --force
# }
push_repo() {
  cd "${WORK_DIR}"
  rm -rf .git
  
  # 1. 兼容旧版本 Git：先初始化，再手动切分支
  git init >/dev/null
  git checkout -b "${BRANCH}" >/dev/null 2>&1 || git branch -M "${BRANCH}"
  
  git config user.name "${GIT_AUTHOR_NAME:-OuXiaoyu}"
  git config user.email "${GIT_AUTHOR_EMAIL:-ouxiaoyu@example.com}"
  git add .
  git commit -m "${COMMIT_MESSAGE:-Upload sanitized STEVE code and logs}" >/dev/null
  git remote add origin "${REMOTE_URL}"
  log "Pushing to ${REMOTE_URL}"
  PYTHONWARNINGS=ignore GIT_SSH="${SSH_TRANSPORT}" GIT_SSH_VARIANT=ssh \
    git push -u origin "${BRANCH}" --force
}

main() {
  if ! have_cmd rsync; then
    log "rsync is required but not found."
    exit 2
  fi
  if [ ! -x "${SSH_TRANSPORT}" ]; then
    log "Git SSH transport is missing or not executable: ${SSH_TRANSPORT}"
    exit 2
  fi
  prepare_clean_copy
  verify_clean_copy

  if [ "${CREATE_REPO_IF_MISSING}" = "true" ] && ! repo_exists; then
    create_repo_with_gh || create_repo_with_token || create_repo_with_netrc || true
  fi

  if ! repo_exists; then
    cat >&2 <<EOF
[push-steve] The repository exists, but this server's SSH key is not authorized by GitHub.

Add the public key below at https://github.com/settings/ssh/new, then rerun:
EOF
    cat "${HOME}/.ssh/id_rsa.pub" >&2
    cat >&2 <<EOF

The sanitized copy is ready at:
  ${WORK_DIR}
EOF
    exit 4
  fi

  push_repo
  log "Done: https://github.com/${OWNER}/${REPO_NAME}"
}

main "$@"
