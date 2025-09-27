from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Tuple

import httpx

from rag_common.config import AppConfig
from rag_common.logging import configure_logging

logger = logging.getLogger(__name__)


def _git_env(deploy_key: Path) -> dict:
    return {
        "GIT_SSH_COMMAND": f"ssh -i {deploy_key} -o StrictHostKeyChecking=no",
        **os.environ,
    }


def ensure_mirror(git_url: str, repo: str, repo_root: Path, deploy_key: Path) -> Path:
    target = repo_root / f"{repo}.git"
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    full_url = f"{git_url}/{repo}.git"
    cmd = ["git", "clone", "--mirror", full_url, str(target)]
    logger.info("cloning mirror", extra={"repo": repo, "event": "poller.clone"})
    subprocess.check_call(cmd, env=_git_env(deploy_key))
    return target


def fetch_updates(repo_dir: Path, deploy_key: Path) -> str:
    cmd = ["git", "--git-dir", str(repo_dir), "fetch", "--prune"]
    subprocess.check_call(cmd, env=_git_env(deploy_key))
    rev = subprocess.check_output(["git", "--git-dir", str(repo_dir), "rev-parse", "HEAD"])
    return rev.decode().strip()


def branch_head(repo_dir: Path, branch: str) -> str | None:
    try:
        rev = subprocess.check_output(["git", "--git-dir", str(repo_dir), "rev-parse", branch])
        return rev.decode().strip()
    except subprocess.CalledProcessError:
        return None


def trigger_reindex(indexer_url: str, repo: str, branch: str, timeout: int) -> None:
    with httpx.Client(base_url=indexer_url, timeout=timeout) as client:
        response = client.post("/reindex", json={"repo": repo, "branch": branch})
        if response.status_code >= 300:
            logger.error(
                "reindex request failed",
                extra={"repo": repo, "branch": branch, "status": response.status_code, "body": response.text},
            )
        else:
            logger.info("reindex triggered", extra={"repo": repo, "branch": branch, "event": "poller.reindex"})


def main() -> None:
    configure_logging()
    config_path = os.getenv("CONFIG_PATH", "config.yaml")
    config = AppConfig.load(config_path)
    indexer_url = os.getenv("INDEXER_URL", "http://indexer:8000")
    timeout = config.limits.http_timeout_sec

    last_seen: Dict[Tuple[str, str], str | None] = {}

    while True:
        for project in config.projects:
            mirror = ensure_mirror(
                config.gitlab.url,
                project.repo,
                config.services.repo_root,
                config.gitlab.deploy_key_path,
            )
            try:
                fetch_updates(mirror, config.gitlab.deploy_key_path)
            except subprocess.CalledProcessError as exc:
                logger.error(
                    "git fetch failed",
                    extra={"repo": project.repo, "error": exc.returncode, "event": "poller.fetch"},
                )
                continue
            for branch in project.branches:
                head = branch_head(mirror, branch)
                if head is None:
                    logger.warning("branch missing", extra={"repo": project.repo, "branch": branch})
                    continue
                key = (project.repo, branch)
                if last_seen.get(key) != head:
                    trigger_reindex(indexer_url, project.repo, branch, timeout)
                    last_seen[key] = head
        time.sleep(config.gitlab.poll_interval_sec)


if __name__ == "__main__":
    main()
