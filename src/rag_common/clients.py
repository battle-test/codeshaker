from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence

import httpx
from meilisearch import Client as MeiliClient
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

logger = logging.getLogger(__name__)


class HttpEmbeddingClient:
    def __init__(self, base_url: str, timeout: float) -> None:
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        resp = self._client.post("/embeddings", json={"inputs": list(texts)})
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data")
        embeddings: List[List[float]] = []
        if isinstance(data, list):
            for entry in data:
                embeddings.append(entry.get("embedding", []))
        elif isinstance(data, dict):
            embeddings.append(data.get("embedding", []))
        else:
            raise ValueError("unexpected TEI response format")
        return embeddings

    def close(self) -> None:
        self._client.close()


class HttpRerankClient:
    def __init__(self, base_url: str, timeout: float) -> None:
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def rerank(self, query: str, candidates: Sequence[str]) -> List[dict]:
        resp = self._client.post("/rerank", json={"query": query, "candidates": list(candidates)})
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self._client.close()


@dataclass
class HybridResult:
    id: str
    repo: str
    branch: str
    path: str
    chunk_id: int
    start: int
    end: int
    preview: str
    score: float


class HybridSearch:
    def __init__(
        self,
        meili: MeiliClient,
        qdrant: QdrantClient,
        embeddings: HttpEmbeddingClient,
        rerank: HttpRerankClient | None = None,
    ) -> None:
        self.meili = meili
        self.qdrant = qdrant
        self.embeddings = embeddings
        self.rerank = rerank

    def search(self, query: str, limit: int = 16) -> List[HybridResult]:
        bm25_hits = self._search_meili(query)
        vector_hits = self._search_qdrant(query)
        merged = self._merge_hits(bm25_hits, vector_hits)
        if not merged:
            return []
        if self.rerank:
            try:
                return self._apply_rerank(query, merged, limit)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "rerank failed, fallback to hybrid score",
                    extra={"event": "rerank.error", "error": str(exc)},
                )
        return sorted(merged, key=lambda item: item.score, reverse=True)[:limit]

    def _search_meili(self, query: str) -> List[HybridResult]:
        index = self.meili.index("code_chunks")
        search_result = index.search(query, {"limit": 64})
        hits: List[HybridResult] = []
        for hit in search_result.get("hits", []):
            hits.append(
                HybridResult(
                    id=hit["id"],
                    repo=hit["repo"],
                    branch=hit["branch"],
                    path=hit["path"],
                    chunk_id=int(hit["chunk_id"]),
                    start=int(hit["start"]),
                    end=int(hit["end"]),
                    preview=hit["preview"],
                    score=float(hit.get("_score", 0.0)),
                )
            )
        return hits

    def _search_qdrant(self, query: str) -> List[HybridResult]:
        try:
            [query_vector] = self.embeddings.embed([query])
        except Exception as exc:  # noqa: BLE001
            logger.warning("embedding failed", extra={"event": "embed.error", "error": str(exc)})
            return []
        search_result = self.qdrant.search(
            collection_name="code_chunks",
            query_vector=query_vector,
            limit=64,
            with_payload=True,
        )
        hits: List[HybridResult] = []
        for point in search_result:
            payload = point.payload or {}
            hits.append(
                HybridResult(
                    id=point.id if isinstance(point.id, str) else str(point.id),
                    repo=payload.get("repo", ""),
                    branch=payload.get("branch", ""),
                    path=payload.get("path", ""),
                    chunk_id=int(payload.get("chunk_id", 0)),
                    start=int(payload.get("start", 0)),
                    end=int(payload.get("end", 0)),
                    preview=payload.get("preview", ""),
                    score=float(point.score or 0.0),
                )
            )
        return hits

    def _merge_hits(self, bm25_hits: List[HybridResult], vector_hits: List[HybridResult]) -> List[HybridResult]:
        merged: dict[str, HybridResult] = {}
        for hit in bm25_hits + vector_hits:
            if hit.id in merged:
                merged_hit = merged[hit.id]
                merged_hit.score = max(merged_hit.score, hit.score)
            else:
                merged[hit.id] = HybridResult(**hit.__dict__)
        return list(merged.values())

    def _apply_rerank(self, query: str, hits: List[HybridResult], limit: int) -> List[HybridResult]:
        assert self.rerank is not None
        candidates = [hit.preview for hit in hits]
        try:
            rerank_scores = self.rerank.rerank(query, candidates)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "rerank request failed",
                extra={"event": "rerank.error", "error": str(exc)},
            )
            return sorted(hits, key=lambda item: item.score, reverse=True)[:limit]
        scored = []
        for hit, rerank_hit in zip(hits, rerank_scores):
            hit.score = float(rerank_hit.get("score", hit.score))
            scored.append(hit)
        return sorted(scored, key=lambda item: item.score, reverse=True)[:limit]


def ensure_qdrant_collection(client: QdrantClient, vector_size: int) -> None:
    collections = client.get_collections()
    names = {collection.name for collection in collections.collections}
    if "code_chunks" in names:
        return
    client.create_collection(
        collection_name="code_chunks",
        vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
    )
