from __future__ import annotations

import logging
import os
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder
import torch

from rag_common.logging import configure_logging

logger = logging.getLogger(__name__)


class RerankRequest(BaseModel):
    query: str = Field(..., min_length=1)
    candidates: List[str] = Field(..., min_items=1)


class RerankResponse(BaseModel):
    idx: int
    score: float


class RerankService:
    def __init__(self) -> None:
        configure_logging()
        model_name = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-large")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=device)
        logger.info("rerank model loaded", extra={"model": model_name, "device": device})

    def rerank(self, query: str, candidates: List[str]) -> List[RerankResponse]:
        inputs = [[query, text] for text in candidates]
        scores = self.model.predict(inputs, convert_to_numpy=True, show_progress_bar=False)
        indexed = sorted(enumerate(scores.tolist()), key=lambda item: item[1], reverse=True)
        return [RerankResponse(idx=idx, score=float(score)) for idx, score in indexed]


service = RerankService()
app = FastAPI()


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict:
    return {"status": "ok"}


@app.post("/rerank")
def rerank(payload: RerankRequest) -> List[dict]:
    if not payload.candidates:
        raise HTTPException(status_code=400, detail="no candidates provided")
    results = service.rerank(payload.query, payload.candidates)
    return [result.dict() for result in results]
