"""Vector store — cosine similarity search over case embeddings stored in SQLite."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from sqlalchemy import select, text

from src.core.config import settings
from src.core.logging import get_logger
from src.domain.models import Asset
from src.persistence.database import AsyncSessionLocal

logger = get_logger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarities between a query vector and a matrix of vectors."""
    if matrix.shape[0] == 0:
        return np.array([])
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return np.zeros(matrix.shape[0])
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1.0  # avoid division by zero
    return np.dot(matrix, query) / (norms * query_norm)


class VectorStore:
    """In-memory vector index backed by SQLite for persistence.

    On init, loads all embeddings from DB into memory for fast similarity search.
    New embeddings are written to both memory and DB.
    """

    def __init__(self) -> None:
        self._vectors: dict[str, np.ndarray] = {}  # case_id -> embedding
        self._metadata: dict[str, dict[str, Any]] = {}  # case_id -> metadata
        self._loaded = False

    async def load(self) -> None:
        """Load all stored embeddings from the database into memory."""
        from src.persistence.database import CaseEmbeddingRow

        async with AsyncSessionLocal() as s:
            rows = (await s.execute(select(CaseEmbeddingRow))).scalars().all()
            for row in rows:
                try:
                    vec = np.frombuffer(row.embedding_blob, dtype=np.float32).copy()
                    self._vectors[row.case_id] = vec
                    self._metadata[row.case_id] = {
                        "asset": row.asset,
                        "severity": row.severity,
                        "decision": row.decision,
                        "summary_zh": row.summary_zh,
                        "created_at": row.created_at.isoformat() if row.created_at else None,
                        "content_hash": row.content_hash,
                    }
                except Exception as exc:
                    logger.warning("Failed to load embedding for case %s: %s", row.case_id, exc)
        self._loaded = True
        logger.info("VectorStore loaded %d embeddings", len(self._vectors))

    async def add(
        self,
        case_id: str,
        embedding: np.ndarray,
        *,
        asset: str,
        severity: str | None,
        decision: str | None,
        summary_zh: str,
        content_hash: str,
    ) -> None:
        """Store an embedding in both memory and database."""
        from src.persistence.database import CaseEmbeddingRow

        self._vectors[case_id] = embedding
        self._metadata[case_id] = {
            "asset": asset,
            "severity": severity,
            "decision": decision,
            "summary_zh": summary_zh,
            "created_at": datetime.utcnow().isoformat(),
            "content_hash": content_hash,
        }

        async with AsyncSessionLocal() as s:
            existing = await s.get(CaseEmbeddingRow, case_id)
            blob = embedding.tobytes()
            now = datetime.utcnow()
            if existing:
                existing.embedding_blob = blob
                existing.asset = asset
                existing.severity = severity
                existing.decision = decision
                existing.summary_zh = summary_zh
                existing.content_hash = content_hash
                existing.updated_at = now
            else:
                s.add(CaseEmbeddingRow(
                    case_id=case_id,
                    asset=asset,
                    severity=severity,
                    decision=decision,
                    summary_zh=summary_zh,
                    embedding_blob=blob,
                    embedding_dim=len(embedding),
                    content_hash=content_hash,
                    created_at=now,
                    updated_at=now,
                ))
            await s.commit()

    def search(
        self,
        query_embedding: np.ndarray,
        *,
        top_k: int = 5,
        asset_filter: str | None = None,
        min_similarity: float | None = None,
        exclude_case_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Find top-K most similar cases by cosine similarity.

        Returns list of dicts with keys: case_id, similarity, asset, severity,
        decision, summary_zh, created_at.
        """
        threshold = min_similarity if min_similarity is not None else settings.memory_similarity_threshold

        candidates: list[tuple[str, np.ndarray]] = []
        for case_id, vec in self._vectors.items():
            if exclude_case_ids and case_id in exclude_case_ids:
                continue
            meta = self._metadata.get(case_id, {})
            if asset_filter and meta.get("asset") != asset_filter:
                continue
            candidates.append((case_id, vec))

        if not candidates:
            return []

        case_ids = [c[0] for c in candidates]
        matrix = np.stack([c[1] for c in candidates])
        similarities = cosine_similarity_batch(query_embedding, matrix)

        results = []
        for i, case_id in enumerate(case_ids):
            sim = float(similarities[i])
            if sim < threshold:
                continue
            meta = self._metadata.get(case_id, {})
            results.append({
                "case_id": case_id,
                "similarity": round(sim, 4),
                **meta,
            })

        results.sort(key=lambda r: r["similarity"], reverse=True)
        return results[:top_k]

    def has_embedding(self, case_id: str) -> bool:
        return case_id in self._vectors

    @property
    def size(self) -> int:
        return len(self._vectors)


# Global singleton
_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


async def init_vector_store() -> None:
    """Initialize and load the vector store. Call during app startup."""
    store = get_vector_store()
    if not store._loaded:
        await store.load()
