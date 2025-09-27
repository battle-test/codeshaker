from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, List, Optional, Set
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_common.chunking import Chunk, batched, is_textual, iter_chunks
from rag_common.clients import HttpEmbeddingClient, ensure_qdrant_collection
from rag_common.config import AppConfig
from rag_common.git import GitRepo, StateManager
from rag_common.logging import configure_logging

from meilisearch import Client as MeiliClient
from meilisearch.errors import MeiliSearchApiError
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from datetime import datetime

logger = logging.getLogger(__name__)


class ReindexRequest(BaseModel):
    repo: str = Field(..., min_length=1)
    branch: str = Field(..., min_length=1)


class ReindexTask(BaseModel):
    task_id: str
    repo: str
    branch: str
    status: str
    detail: Optional[str] = None


@dataclass
class IndexerContext:
    config: AppConfig
    state: StateManager
    meili: MeiliClient
    qdrant: QdrantClient
    embedding_client: HttpEmbeddingClient


def load_config() -> AppConfig:
    config_path = os.getenv("CONFIG_PATH", "config.yaml")
    return AppConfig.load(config_path)


def create_app() -> FastAPI:
    configure_logging()
    config = load_config()
    state = StateManager(config.services.state_path)
    meili = MeiliClient(config.services.meilisearch_url)
    try:
        meili.create_index("code_chunks", {"primaryKey": "id"})
    except MeiliSearchApiError as exc:
        if getattr(exc, "error_code", "") != "index_already_exists":
            logger.warning("meilisearch index init failed", extra={"event": "meili.init", "error": str(exc)})

    qdrant = QdrantClient(url=config.services.qdrant_url, timeout=config.limits.http_timeout_sec)
    embedding_client = HttpEmbeddingClient(config.services.tei_url, config.limits.http_timeout_sec)

    probe = embedding_client.embed(["dimension probe"])
    if not probe:
        raise RuntimeError("embedding service returned empty vector")
    vector_probe = probe[0]
    ensure_qdrant_collection(qdrant, vector_size=len(vector_probe))

    ctx = IndexerContext(
        config=config,
        state=state,
        meili=meili,
        qdrant=qdrant,
        embedding_client=embedding_client,
    )

    app = FastAPI()
    tasks: Dict[str, ReindexTask] = {}

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> dict:
        return {"status": "ok"}

    @app.post("/reindex", status_code=202)
    async def reindex(req: ReindexRequest, background: BackgroundTasks) -> dict:
        project = ctx.config.find_project(req.repo)
        if not project:
            raise HTTPException(status_code=404, detail="unknown repo")
        if req.branch not in project.branches:
            raise HTTPException(status_code=403, detail="branch not allowed")
        task_id = str(uuid4())
        tasks[task_id] = ReindexTask(task_id=task_id, repo=req.repo, branch=req.branch, status="queued")
        background.add_task(_run_reindex, ctx, req.repo, req.branch, tasks[task_id])
        return {"accepted": True, "task_id": task_id}

    @app.get("/tasks/{task_id}")
    async def task_status(task_id: str) -> ReindexTask:
        task = tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="task not found")
        return task

    return app


def _run_reindex(ctx: IndexerContext, repo_name: str, branch: str, task: ReindexTask) -> None:
    task.status = "running"
    git_dir = ctx.config.services.repo_root / f"{repo_name}.git"
    repo = GitRepo(git_dir)
    head = repo.current_commit(branch)
    if not head:
        task.status = "failed"
        task.detail = "branch missing"
        return
    previous = ctx.state.get(repo_name, branch)
    try:
        changes = repo.list_changes(previous, head)
    except Exception as exc:  # noqa: BLE001
        logger.exception("change detection failed", extra={"repo": repo_name, "branch": branch})
        task.status = "failed"
        task.detail = str(exc)
        return

    try:
        _process_changes(ctx, repo_name, branch, repo, head, previous, changes)
    except Exception as exc:  # noqa: BLE001
        logger.exception("indexing failed", extra={"repo": repo_name, "branch": branch})
        task.status = "failed"
        task.detail = str(exc)
        return

    ctx.state.set(repo_name, branch, head)
    task.status = "finished"


def _process_changes(
    ctx: IndexerContext,
    repo_name: str,
    branch: str,
    repo: GitRepo,
    head: str,
    previous: Optional[str],
    changes: List,
) -> None:
    project = ctx.config.find_project(repo_name)
    assert project
    additions: List[tuple[str, Chunk, str]] = []
    deletions: Set[str] = set()
    refresh: Set[str] = set()

    for change in changes:
        path = change.path
        if change.status == "D":
            deletions.add(path)
            continue
        if any(fnmatch(path, pattern) for pattern in project.denylist):
            continue
        if not is_textual(path):
            continue
        try:
            file_sha, content = repo.file_blob(head, path)
        except Exception:
            logger.warning("failed to read blob", extra={"path": path, "repo": repo_name, "branch": branch})
            continue
        text = content.decode("utf-8", errors="ignore")
        refresh.add(path)
        for chunk in iter_chunks(text, ctx.config.chunking.max_chars, ctx.config.chunking.overlap):
            additions.append((path, chunk, file_sha))
        _write_blob(ctx, repo_name, branch, path, text)

    purge_paths = list(deletions.union(refresh))
    if purge_paths:
        _delete_documents(ctx, repo_name, branch, purge_paths)

    if not additions:
        return

    _index_chunks(ctx, repo_name, branch, head, additions)


def _write_blob(ctx: IndexerContext, repo: str, branch: str, path: str, content: str) -> None:
    root = ctx.config.services.blob_root / repo / branch
    safe_path = Path(path)
    dest = root / safe_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")


def _delete_documents(ctx: IndexerContext, repo: str, branch: str, paths: List[str]) -> None:
    index = ctx.meili.index("code_chunks")
    # Meilisearch does not support prefix delete, fetch matching docs
    to_delete = []
    for path in paths:
        search = index.search("", {"filter": f"repo = '{repo}' AND branch = '{branch}' AND path = '{path}'", "limit": 1000})
        for hit in search.get("hits", []):
            to_delete.append(hit["id"])
    if to_delete:
        unique = list(set(to_delete))
        index.delete_documents(unique)
        ctx.qdrant.delete(collection_name="code_chunks", points_selector=qmodels.PointIdsList(points=unique))


def _index_chunks(
    ctx: IndexerContext,
    repo: str,
    branch: str,
    commit_sha: str,
    additions: List[tuple[str, Chunk, str]],
) -> None:
    index = ctx.meili.index("code_chunks")
    points: List[qmodels.PointStruct] = []
    documents = []
    batch_size = 32
    for batch in batched(additions, batch_size):
        texts = [chunk.content for (_, chunk, _) in batch]
        embeddings = ctx.embedding_client.embed(texts)
        for (path, chunk, file_sha), vector in zip(batch, embeddings):
            doc_id = chunk.doc_id(repo, branch, path, file_sha)
            documents.append(
                {
                    "id": doc_id,
                    "repo": repo,
                    "branch": branch,
                    "path": path,
                    "chunk_id": chunk.chunk_id,
                    "start": chunk.start,
                    "end": chunk.end,
                    "content": chunk.content,
                    "preview": chunk.preview,
                    "file_sha": file_sha,
                    "commit_sha": commit_sha,
                    "lang": Path(path).suffix.lstrip("."),
                    "updated_at": datetime.utcnow().isoformat() + "Z",
                }
            )
            points.append(
                qmodels.PointStruct(
                    id=doc_id,
                    vector=vector,
                    payload={
                        "repo": repo,
                        "branch": branch,
                        "path": path,
                        "chunk_id": chunk.chunk_id,
                        "start": chunk.start,
                        "end": chunk.end,
                        "preview": chunk.preview,
                        "file_sha": file_sha,
                        "commit_sha": commit_sha,
                    },
                )
            )
    if documents:
        index.add_documents(documents, primary_key="id")
    if points:
        ctx.qdrant.upsert(collection_name="code_chunks", points=qmodels.PointsList(points=points))


app = create_app()
