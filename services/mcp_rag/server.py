from __future__ import annotations

import anyio
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, List

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from meilisearch import Client as MeiliClient
from qdrant_client import QdrantClient
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse

from rag_common.clients import HttpEmbeddingClient, HttpRerankClient, HybridResult, HybridSearch
from rag_common.config import AppConfig
from rag_common.logging import configure_logging

logger = logging.getLogger(__name__)

POLICY_TEXT = (
    "Git is the single source of truth. For ANY code, architecture, configuration, or documentation question, "
    "you MUST call MCP tools (search_code, get_file, grep, explain_symbol) before answering. If content is missing, reply: NOT FOUND IN GIT."
)

CODE_KEYWORDS = {
    "function",
    "class",
    "yaml",
    "helm",
    "k8s",
    "error",
    "traceback",
    "stack",
    "config",
    "docker",
    "kubernetes",
    "terraform",
    "proto",
    "sql",
    "json",
    "ts",
    "tsx",
    "py",
    "java",
    "go",
    "rs",
    "kt",
    "front-end",
    "backend",
}


def load_context() -> tuple[FastMCP, HybridSearch, AppConfig]:
    configure_logging()
    config_path = os.getenv("CONFIG_PATH", "config.yaml")
    config = AppConfig.load(config_path)

    meili = MeiliClient(config.services.meilisearch_url)
    qdrant = QdrantClient(url=config.services.qdrant_url, timeout=config.limits.http_timeout_sec)
    embedding = HttpEmbeddingClient(config.services.tei_url, config.limits.http_timeout_sec)
    rerank_client = HttpRerankClient(config.services.rerank_url, config.limits.http_timeout_sec)
    hybrid = HybridSearch(meili, qdrant, embedding, rerank=rerank_client)

    mcp = FastMCP(
        name="mcp-rag",
        instructions=config.mcp.server_description,
        host=config.mcp.host,
        port=config.mcp.port,
    )
    return mcp, hybrid, config


mcp, hybrid_search, app_config = load_context()


def _set_strict_schema(tool_name: str) -> None:
    tool = mcp._tool_manager._tools.get(tool_name)
    if tool:
        tool.parameters.setdefault("additionalProperties", False)


@mcp.resource("gitops_policy://authority", name="GitOps Policy", mime_type="text/plain")
def gitops_policy_resource() -> str:
    return POLICY_TEXT


@mcp.prompt(name="git_answer", title="Git Answer Policy")
def git_answer_prompt() -> List[dict]:
    return [
        {
            "role": "system",
            "content": (
                "Always call search_code before attempting to answer. Cite repo/path:lines in every response. "
                "If nothing is returned from MCP tools, answer exactly: NOT FOUND IN GIT."
            ),
        }
    ]


@mcp.tool(
    name="gitops_assert",
    description="This workspace is full GitOps. Always call search_code/get_file/grep/explain_symbol before any answer.",
    annotations=ToolAnnotations(title="GitOps enforcement", idempotentHint=True, openWorldHint=False),
)
async def gitops_assert() -> dict:
    ts = datetime.now(timezone.utc).isoformat()
    return {"policy": POLICY_TEXT, "ts": ts}


@mcp.tool(
    name="search_code",
    description="Always call this before answering. Hybrid BM25 + vector search over Git-tracked code.",
    annotations=ToolAnnotations(idempotentHint=True, openWorldHint=False),
)
async def search_code(
    query: Annotated[str, Field(min_length=1)],
    limit: Annotated[int, Field(ge=1, le=64)] = 16,
) -> List[dict]:
    def _run() -> List[HybridResult]:
        return hybrid_search.search(query, limit)

    hits = await anyio.to_thread.run_sync(_run)
    return [
        {
            "score": hit.score,
            "repo": hit.repo,
            "branch": hit.branch,
            "path": hit.path,
            "chunk_id": hit.chunk_id,
            "start": hit.start,
            "end": hit.end,
            "preview": hit.preview,
        }
        for hit in hits
    ]


@mcp.tool(
    name="get_file",
    description="Retrieve the full file content from the Git-backed blob store. Must be called before referencing file contents.",
    annotations=ToolAnnotations(idempotentHint=True, openWorldHint=False),
)
async def get_file(
    path: Annotated[str, Field(min_length=1)],
    repo: Annotated[str, Field(min_length=1)],
    branch: Annotated[str, Field(min_length=1)],
) -> dict:
    blob_path = _safe_blob_path(repo, branch, path)
    if not blob_path.exists():
        raise ValueError("file not found in blob store")

    def _read() -> dict:
        content = blob_path.read_text(encoding="utf-8")
        size = blob_path.stat().st_size
        sha = _sha256_text(content)
        return {"path": path, "size": size, "sha": sha, "content": content}

    return await anyio.to_thread.run_sync(_read)


@mcp.tool(
    name="grep",
    description="Strict Git-backed grep. Always use before stating that a pattern exists in a file.",
    annotations=ToolAnnotations(idempotentHint=True, openWorldHint=False),
)
async def grep(
    path: Annotated[str, Field(min_length=1)],
    pattern: Annotated[str, Field(min_length=1)],
    repo: Annotated[str, Field(min_length=1)],
    branch: Annotated[str, Field(min_length=1)],
    max_lines: Annotated[int, Field(ge=1, le=500)] = 200,
) -> List[dict]:
    blob_path = _safe_blob_path(repo, branch, path)
    if not blob_path.exists():
        raise ValueError("file not found in blob store")

    def _run() -> List[dict]:
        regex = re.compile(pattern)
        matches: List[dict] = []
        with blob_path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle, start=1):
                if regex.search(line):
                    matches.append({"ln": idx, "text": line.rstrip("\n")})
                    if len(matches) >= max_lines:
                        break
        return matches

    return await anyio.to_thread.run_sync(_run)


@mcp.tool(
    name="explain_symbol",
    description="Find symbol definitions or signatures from Git-backed indexes before answering any API question.",
    annotations=ToolAnnotations(idempotentHint=True, openWorldHint=False),
)
async def explain_symbol(
    symbol: Annotated[str, Field(min_length=1)],
    lang: Annotated[str | None, Field(min_length=1)] = None,
    repo: Annotated[str, Field(min_length=1)] = "",
    branch: Annotated[str, Field(min_length=1)] = "",
) -> List[dict]:
    query = symbol if not lang else f"{symbol} {lang}"

    def _run() -> List[dict]:
        results = hybrid_search.search(query, limit=8)
        filtered = []
        for hit in results:
            if repo and hit.repo != repo:
                continue
            if branch and hit.branch != branch:
                continue
            filtered.append(
                {
                    "repo": hit.repo,
                    "branch": hit.branch,
                    "path": hit.path,
                    "ln": hit.start,
                    "snippet": hit.preview,
                }
            )
        return filtered

    return await anyio.to_thread.run_sync(_run)


_set_strict_schema("search_code")
_set_strict_schema("get_file")
_set_strict_schema("grep")
_set_strict_schema("explain_symbol")
_set_strict_schema("gitops_assert")


def _safe_blob_path(repo: str, branch: str, rel_path: str) -> Path:
    root = app_config.services.blob_root / repo / branch
    candidate = (root / rel_path).resolve()
    if not str(candidate).startswith(str(root.resolve())):
        raise ValueError("invalid path request")
    return candidate


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _should_trigger_policy(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(keyword in lowered for keyword in CODE_KEYWORDS)


@mcp.custom_route("/intent", methods=["POST"])
async def classify_intent(request: Request) -> JSONResponse:
    body = await request.json()
    question = body.get("question", "")
    flagged = bool(question and _should_trigger_policy(question))
    return JSONResponse({"is_code_related": flagged, "policy": POLICY_TEXT if flagged else ""})


@mcp.custom_route("/healthz", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


async def main() -> None:
    logger.info("starting MCP server", extra={"host": app_config.mcp.host, "port": app_config.mcp.port})
    await mcp.run_streamable_http_async()


if __name__ == "__main__":
    anyio.run(main)
