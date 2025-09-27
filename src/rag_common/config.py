from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import List, Optional

import yaml


@dataclasses.dataclass
class GitLabConfig:
    url: str
    deploy_key_path: Path
    poll_interval_sec: int


@dataclasses.dataclass
class ProjectConfig:
    repo: str
    branches: List[str]
    denylist: List[str]


@dataclasses.dataclass
class ChunkingConfig:
    max_chars: int
    overlap: int


@dataclasses.dataclass
class ServicesConfig:
    meilisearch_url: str
    qdrant_url: str
    tei_url: str
    rerank_url: str
    blob_root: Path
    repo_root: Path
    state_path: Path


@dataclasses.dataclass
class MCPConfig:
    host: str
    port: int
    enforce_meta_tool: bool
    server_description: str
    classification_enabled: bool


@dataclasses.dataclass
class LimitsConfig:
    http_timeout_sec: int
    max_parallel_index_tasks: int


@dataclasses.dataclass
class AppConfig:
    gitlab: GitLabConfig
    projects: List[ProjectConfig]
    chunking: ChunkingConfig
    services: ServicesConfig
    mcp: MCPConfig
    limits: LimitsConfig

    @staticmethod
    def load(path: os.PathLike[str] | str) -> "AppConfig":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        gitlab = GitLabConfig(
            url=data["gitlab"]["url"],
            deploy_key_path=Path(data["gitlab"]["deploy_key_path"]),
            poll_interval_sec=int(data["gitlab"].get("poll_interval_sec", 30)),
        )

        projects = [
            ProjectConfig(
                repo=entry["repo"],
                branches=list(entry.get("branches", [])),
                denylist=list(entry.get("denylist", [])),
            )
            for entry in data.get("projects", [])
        ]

        chunking = ChunkingConfig(
            max_chars=int(data["chunking"]["max_chars"]),
            overlap=int(data["chunking"].get("overlap", 0)),
        )

        services = ServicesConfig(
            meilisearch_url=data["services"]["meilisearch_url"],
            qdrant_url=data["services"]["qdrant_url"],
            tei_url=data["services"]["tei_url"],
            rerank_url=data["services"]["rerank_url"],
            blob_root=Path(data["services"]["blob_root"]),
            repo_root=Path(data["services"]["repo_root"]),
            state_path=Path(data["services"]["state_path"]),
        )

        mcp_cfg = data.get("mcp", {})
        mcp = MCPConfig(
            host=mcp_cfg.get("host", "0.0.0.0"),
            port=int(mcp_cfg.get("port", 8765)),
            enforce_meta_tool=bool(mcp_cfg.get("enforce_meta_tool", False)),
            server_description=mcp_cfg.get(
                "server_description",
                "Git is the only source of truth. Always call MCP tools before answering.",
            ),
            classification_enabled=bool(mcp_cfg.get("classification_enabled", True)),
        )

        limits_data = data.get("limits", {})
        limits = LimitsConfig(
            http_timeout_sec=int(limits_data.get("http_timeout_sec", 10)),
            max_parallel_index_tasks=int(limits_data.get("max_parallel_index_tasks", 4)),
        )

        return AppConfig(
            gitlab=gitlab,
            projects=projects,
            chunking=chunking,
            services=services,
            mcp=mcp,
            limits=limits,
        )

    def find_project(self, repo: str) -> Optional[ProjectConfig]:
        for project in self.projects:
            if project.repo == repo:
                return project
        return None
