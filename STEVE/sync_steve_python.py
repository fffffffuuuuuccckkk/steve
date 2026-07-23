#!/bin/sh
""":"
exec /data/OuXiaoyu/miniconda3/envs/basicts/bin/python "$0" "$@"
":"""
"""One-way sync of STEVE Python sources from the current server to its peer."""

import argparse
import fcntl
import os
import posixpath
import socket
import stat
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path("/data/OuXiaoyu/STEVE_CODE/STEVE")
REMOTE_ROOT = "/data/OuXiaoyu/STEVE_CODE/STEVE"
PEER_BY_HOSTNAME = {
    "gpu-39": "211.71.72.121",       # multi-GPU -> single-GPU
    "insis-cyy-4090": "211.71.76.25",  # single-GPU -> multi-GPU
}

# These trees contain generated artifacts, datasets, caches, or repository metadata.
# The allow-list below already limits transfers to *.py; these exclusions add another
# hard boundary so that Python-looking files inside output/data trees are ignored too.
EXCLUDED_DIR_NAMES = {
    ".git",
    ".idea",
    ".pytest_cache",
    ".vscode",
    "__pycache__",
    "artifacts",
    "cache",
    "checkpoints",
    "data",
    "datasets",
    "experiments",
    "logs",
    "outputs",
    "results",
    "runs",
    "wandb",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use the current server as the source and copy only STEVE *.py files "
            "to the other server. No destination files are deleted."
        )
    )
    parser.add_argument("--peer", help="Override the automatically detected peer IP")
    parser.add_argument("--user", default="OuXiaoyu", help="SSH user (default: OuXiaoyu)")
    parser.add_argument("--port", type=int, default=22, help="SSH port (default: 22)")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without uploading")
    parser.add_argument("--force", action="store_true", help="Upload every allowed Python file")
    return parser.parse_args()


def detect_peer(override: str | None) -> tuple[str, str]:
    hostname = socket.gethostname().split(".", 1)[0].lower()
    if override:
        return hostname, override
    try:
        return hostname, PEER_BY_HOSTNAME[hostname]
    except KeyError as exc:
        known = ", ".join(sorted(PEER_BY_HOSTNAME))
        raise RuntimeError(
            f"Unknown host {hostname!r}; expected one of: {known}. Use --peer IP."
        ) from exc


def iter_python_files(root: Path):
    for current, dirs, files in os.walk(root, topdown=True, followlinks=False):
        dirs[:] = sorted(
            directory
            for directory in dirs
            if directory not in EXCLUDED_DIR_NAMES
            and not (Path(current) / directory).is_symlink()
        )
        for filename in sorted(files):
            if filename.endswith(".py"):
                path = Path(current) / filename
                if path.is_file() and not path.is_symlink():
                    yield path


def ensure_remote_dir(sftp, directory: str, cache: set[str]) -> None:
    if directory in cache or directory == "/":
        return
    missing: list[str] = []
    cursor = directory
    while cursor not in cache and cursor != "/":
        try:
            attrs = sftp.stat(cursor)
            if not stat.S_ISDIR(attrs.st_mode):
                raise RuntimeError(f"Remote path is not a directory: {cursor}")
            cache.add(cursor)
            break
        except OSError:
            missing.append(cursor)
            cursor = posixpath.dirname(cursor)
    for item in reversed(missing):
        sftp.mkdir(item, mode=0o755)
        cache.add(item)


def same_file(local_stat: os.stat_result, remote_stat) -> bool:
    return (
        local_stat.st_size == remote_stat.st_size
        and int(local_stat.st_mtime) == int(remote_stat.st_mtime)
    )


def upload_atomic(sftp, local: Path, remote: str, local_stat: os.stat_result) -> None:
    temp = f"{remote}.steve_py_sync_tmp_{os.getpid()}"
    try:
        sftp.put(str(local), temp, confirm=True)
        sftp.chmod(temp, stat.S_IMODE(local_stat.st_mode))
        sftp.utime(temp, (int(local_stat.st_atime), int(local_stat.st_mtime)))
        try:
            sftp.posix_rename(temp, remote)
        except (AttributeError, OSError):
            try:
                sftp.remove(remote)
            except OSError:
                pass
            sftp.rename(temp, remote)
        sftp.utime(remote, (int(local_stat.st_atime), int(local_stat.st_mtime)))
    finally:
        try:
            sftp.remove(temp)
        except OSError:
            pass


def sync_with_rsync(args: argparse.Namespace, peer: str, files: list[Path]) -> tuple[int, int]:
    relative_files = [path.relative_to(PROJECT_ROOT).as_posix() for path in files]
    command = [
        "rsync",
        "-a",
        "--from0",
        "--files-from=-",
        "--itemize-changes",
        "--no-implied-dirs",
        "--omit-dir-times",
    ]
    if args.dry_run:
        command.append("--dry-run")
    if args.force:
        command.append("--ignore-times")
    command.extend(
        [
            f"{PROJECT_ROOT}/",
            f"{args.user}@{peer}:{REMOTE_ROOT}/",
        ]
    )
    payload = b"\0".join(path.encode("utf-8") for path in relative_files) + b"\0"
    result = subprocess.run(
        command,
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = result.stdout.decode("utf-8", errors="replace")
    if output:
        print(output, end="" if output.endswith("\n") else "\n")
    if result.returncode != 0:
        raise RuntimeError(f"rsync failed with exit code {result.returncode}")
    changed = sum(
        1
        for line in output.splitlines()
        if line.startswith((">f", "<f")) and line.rstrip().endswith(".py")
    )
    return changed, len(files) - changed


def sync_with_paramiko(args: argparse.Namespace, peer: str, files: list[Path]) -> tuple[int, int]:
    try:
        import paramiko
    except ImportError as exc:
        raise RuntimeError(
            "Paramiko is missing. Install it in basicts with: python -m pip install paramiko"
        ) from exc

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        peer,
        port=args.port,
        username=args.user,
        timeout=15,
        banner_timeout=15,
        auth_timeout=15,
        allow_agent=True,
        look_for_keys=True,
    )

    uploaded = 0
    unchanged = 0
    try:
        with client.open_sftp() as sftp:
            root_stat = sftp.stat(REMOTE_ROOT)
            if not stat.S_ISDIR(root_stat.st_mode):
                raise RuntimeError(f"Remote STEVE path is not a directory: {REMOTE_ROOT}")
            known_dirs = {REMOTE_ROOT}

            for local in files:
                relative = local.relative_to(PROJECT_ROOT).as_posix()
                remote = posixpath.join(REMOTE_ROOT, relative)
                local_stat = local.stat()
                try:
                    remote_stat = sftp.stat(remote)
                except OSError:
                    remote_stat = None

                if not args.force and remote_stat is not None and same_file(local_stat, remote_stat):
                    unchanged += 1
                    continue

                action = "WOULD_COPY" if args.dry_run else "COPY"
                print(f"[{action}] {relative}")
                if not args.dry_run:
                    ensure_remote_dir(sftp, posixpath.dirname(remote), known_dirs)
                    upload_atomic(sftp, local, remote, local_stat)
                uploaded += 1
    finally:
        client.close()
    return uploaded, unchanged


def main() -> int:
    args = parse_args()
    if not PROJECT_ROOT.is_dir():
        raise RuntimeError(f"Local STEVE directory does not exist: {PROJECT_ROOT}")

    hostname, peer = detect_peer(args.peer)
    lock_path = Path("/tmp/steve_python_sync.lock")
    with lock_path.open("w", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another STEVE Python sync is already running locally") from exc

        files = list(iter_python_files(PROJECT_ROOT))
        print(f"[sync-steve-py] source={hostname}:{PROJECT_ROOT}")
        print(f"[sync-steve-py] destination={args.user}@{peer}:{REMOTE_ROOT}")
        print(f"[sync-steve-py] allowed_files={len(files)} dry_run={args.dry_run}")

        # The single-GPU host has a modern OpenSSH client, so rsync is fastest there.
        # gpu-39 uses Paramiko because its old system SSH client cannot negotiate the
        # single-GPU host's modern host-key algorithms.
        if hostname == "insis-cyy-4090":
            uploaded, unchanged = sync_with_rsync(args, peer, files)
        else:
            uploaded, unchanged = sync_with_paramiko(args, peer, files)

        verb = "would_copy" if args.dry_run else "copied"
        print(f"[sync-steve-py] done {verb}={uploaded} unchanged={unchanged} deleted=0")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[sync-steve-py] ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
