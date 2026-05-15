"""Embedding client — generate text embeddings via the Ark API (OpenAI-compatible)."""
from __future__ import annotations

import hashlib
import json
import time
from functools import lru_cache
from typing import Any

import numpy as np

from src.core.config import settings
from src.core.logging import get_logger
from src.observability.llm_trace import record_llm_call

logger = get_logger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None:
        from src.core.proxy import build_openai_client
        _client = build_openai_client(service="embedding")
    return _client


def embed_text(text: str) -> np.ndarray:
    """Generate embedding vector for a single text string."""
    t0 = time.perf_counter()
    try:
        response = _get_client().embeddings.create(
            model=settings.embedding_model,
            input=text,
        )
        elapsed = time.perf_counter() - t0
        record_llm_call(
            operation="embedding",
            status="success",
            duration_ms=elapsed * 1000,
            prompt_tokens=getattr(response.usage, "prompt_tokens", 0) if response.usage else 0,
            model=settings.embedding_model,
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        record_llm_call(
            operation="embedding",
            status="error",
            duration_ms=elapsed * 1000,
            model=settings.embedding_model,
        )
        logger.warning("Embedding call failed: %s", exc)
        raise


def embed_batch(texts: list[str]) -> list[np.ndarray]:
    """Generate embedding vectors for a batch of texts."""
    if not texts:
        return []
    t0 = time.perf_counter()
    try:
        response = _get_client().embeddings.create(
            model=settings.embedding_model,
            input=texts,
        )
        elapsed = time.perf_counter() - t0
        record_llm_call(
            operation="embedding_batch",
            status="success",
            duration_ms=elapsed * 1000,
            prompt_tokens=getattr(response.usage, "prompt_tokens", 0) if response.usage else 0,
            model=settings.embedding_model,
        )
        # Sort by index to preserve order
        sorted_data = sorted(response.data, key=lambda d: d.index)
        return [np.array(d.embedding, dtype=np.float32) for d in sorted_data]
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        record_llm_call(
            operation="embedding_batch",
            status="error",
            duration_ms=elapsed * 1000,
            model=settings.embedding_model,
        )
        logger.warning("Batch embedding call failed: %s", exc)
        raise


def build_case_text(case_summary: str, rule_hits_desc: str, asset: str) -> str:
    """Build a unified text representation of a risk case for embedding."""
    return f"[{asset}] {case_summary} | {rule_hits_desc}"


def content_hash(text: str) -> str:
    """Stable hash for deduplication of embedding content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
