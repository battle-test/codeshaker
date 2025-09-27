from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


logger = logging.getLogger(__name__)


class GitError(RuntimeError):
    pass


@dataclass
class GitChange:
    status: str
    path: str


class GitRepo:
    def __init__(self, git_dir: Path) -> None:
        self.git_dir = git_dir

    def _run(self, *args: str) -> str:
        cmd = ["git", "--git-dir", str(self.git_dir), *args]
        logger.debug("running git", extra={"event": "git.exec", "cmd": cmd})
        try:
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as exc:
            raise GitError(exc.output.decode()) from exc
        return output.decode().strip()

    def current_commit(self, branch: str) -> Optional[str]:
        try:
            return self._run("rev-parse", branch)
        except GitError:
            return None

    def list_changes(self, base: Optional[str], head: str) -> List[GitChange]:
        if base:
            args = ["diff-tree", "--no-commit-id", "--name-status", "-r", base, head]
        else:
            args = ["ls-tree", "-r", "--name-only", head]
        try:
            output = self._run(*args)
        except GitError as exc:
            logger.error("git diff failed", extra={"event": "git.diff", "error": str(exc)})
            raise
        changes: List[GitChange] = []
        if not output:
            return changes
        for line in output.splitlines():
            if base:
                status, path = line.split("\t", 1)
            else:
                status, path = "A", line
            changes.append(GitChange(status=status, path=path))
        return changes

    def file_blob(self, commit: str, path: str) -> Tuple[str, bytes]:
        blob = self._run("ls-tree", commit, path)
        if not blob:
            raise GitError(f"missing blob for {path}")
        file_sha = blob.split()[2]
        content = subprocess.check_output(
            ["git", "--git-dir", str(self.git_dir), "show", f"{commit}:{path}"]
        )
        return file_sha, content

    def remove_deleted(self, branch: str, base: Optional[str], head: str) -> List[str]:
        if not base:
            return []
        output = self._run("diff-tree", "--no-commit-id", "--name-status", "-r", base, head)
        removed = []
        for line in output.splitlines():
            status, path = line.split("\t", 1)
            if status == "D":
                removed.append(path)
        return removed


class StateManager:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._state: Dict[str, Dict[str, str]] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._state = json.loads(self.path.read_text("utf-8"))
            except json.JSONDecodeError:
                logger.warning("state file corrupt, starting fresh", extra={"event": "state.load"})
                self._state = {}

    def save(self) -> None:
        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
        os.replace(tmp_path, self.path)

    def get(self, repo: str, branch: str) -> Optional[str]:
        return self._state.get(repo, {}).get(branch)

    def set(self, repo: str, branch: str, commit: str) -> None:
        self._state.setdefault(repo, {})[branch] = commit
        self.save()
