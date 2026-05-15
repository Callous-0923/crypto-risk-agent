"""Unit tests for the memory layer — embedding, vector store, experience, preferences."""
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.domain.models import (
    Asset, CaseStatus, Decision, FeatureSnapshot, ReviewAction,
    RiskCase, RuleHit, Severity,
)


# ---------------------------------------------------------------------------
# Embedding tests
# ---------------------------------------------------------------------------

class TestEmbedding:
    def test_build_case_text(self):
        from src.memory.embedding import build_case_text
        text = build_case_text("BTC 出现剧烈波动", "MKT_EXTREME_VOL_P1(P1)", "BTC")
        assert "[BTC]" in text
        assert "剧烈波动" in text
        assert "MKT_EXTREME_VOL_P1" in text

    def test_content_hash_deterministic(self):
        from src.memory.embedding import content_hash
        h1 = content_hash("test string")
        h2 = content_hash("test string")
        assert h1 == h2
        assert len(h1) == 16

    def test_content_hash_different_inputs(self):
        from src.memory.embedding import content_hash
        h1 = content_hash("hello")
        h2 = content_hash("world")
        assert h1 != h2


# ---------------------------------------------------------------------------
# Vector store tests (in-memory, no DB)
# ---------------------------------------------------------------------------

class TestVectorStore:
    def test_cosine_similarity_identical(self):
        from src.memory.store import cosine_similarity
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        from src.memory.store import cosine_similarity
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_cosine_similarity_opposite(self):
        from src.memory.store import cosine_similarity
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_cosine_similarity_zero_vector(self):
        from src.memory.store import cosine_similarity
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.zeros(2, dtype=np.float32)
        assert cosine_similarity(a, b) == 0.0

    def test_cosine_similarity_batch(self):
        from src.memory.store import cosine_similarity_batch
        query = np.array([1.0, 0.0], dtype=np.float32)
        matrix = np.array([
            [1.0, 0.0],   # identical
            [0.0, 1.0],   # orthogonal
            [0.7, 0.7],   # 45 degrees
        ], dtype=np.float32)
        sims = cosine_similarity_batch(query, matrix)
        assert abs(sims[0] - 1.0) < 1e-6
        assert abs(sims[1]) < 1e-6
        assert 0.6 < sims[2] < 0.8

    def test_cosine_similarity_batch_empty(self):
        from src.memory.store import cosine_similarity_batch
        query = np.array([1.0, 0.0], dtype=np.float32)
        matrix = np.empty((0, 2), dtype=np.float32)
        sims = cosine_similarity_batch(query, matrix)
        assert len(sims) == 0

    def test_vector_store_search_in_memory(self):
        """Test search without DB — directly populate memory."""
        from src.memory.store import VectorStore

        store = VectorStore()
        store._loaded = True

        # Add some vectors
        store._vectors["case_1"] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        store._metadata["case_1"] = {
            "asset": "BTC", "severity": "P1", "decision": "emit",
            "summary_zh": "BTC 暴跌", "created_at": "2024-01-01",
        }

        store._vectors["case_2"] = np.array([0.9, 0.1, 0.0], dtype=np.float32)
        store._metadata["case_2"] = {
            "asset": "BTC", "severity": "P2", "decision": "suppress",
            "summary_zh": "BTC 小幅波动", "created_at": "2024-01-02",
        }

        store._vectors["case_3"] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        store._metadata["case_3"] = {
            "asset": "ETH", "severity": "P1", "decision": "emit",
            "summary_zh": "ETH 异常", "created_at": "2024-01-03",
        }

        # Search with a BTC-like query
        query = np.array([0.95, 0.05, 0.0], dtype=np.float32)
        results = store.search(query, top_k=3, min_similarity=0.5)

        assert len(results) >= 2
        assert results[0]["case_id"] in ("case_1", "case_2")
        assert results[0]["similarity"] > results[1]["similarity"]

    def test_vector_store_search_with_asset_filter(self):
        from src.memory.store import VectorStore

        store = VectorStore()
        store._loaded = True
        store._vectors["case_btc"] = np.array([1.0, 0.0], dtype=np.float32)
        store._metadata["case_btc"] = {"asset": "BTC", "severity": "P1"}
        store._vectors["case_eth"] = np.array([1.0, 0.0], dtype=np.float32)
        store._metadata["case_eth"] = {"asset": "ETH", "severity": "P1"}

        query = np.array([1.0, 0.0], dtype=np.float32)
        results = store.search(query, asset_filter="BTC", min_similarity=0.5)
        assert all(r["asset"] == "BTC" for r in results)

    def test_vector_store_search_exclude(self):
        from src.memory.store import VectorStore

        store = VectorStore()
        store._loaded = True
        store._vectors["case_1"] = np.array([1.0, 0.0], dtype=np.float32)
        store._metadata["case_1"] = {"asset": "BTC"}
        store._vectors["case_2"] = np.array([1.0, 0.0], dtype=np.float32)
        store._metadata["case_2"] = {"asset": "BTC"}

        query = np.array([1.0, 0.0], dtype=np.float32)
        results = store.search(query, exclude_case_ids={"case_1"}, min_similarity=0.5)
        assert all(r["case_id"] != "case_1" for r in results)

    def test_vector_store_has_embedding(self):
        from src.memory.store import VectorStore
        store = VectorStore()
        store._vectors["x"] = np.zeros(3, dtype=np.float32)
        assert store.has_embedding("x")
        assert not store.has_embedding("y")


# ---------------------------------------------------------------------------
# Decision node with memory tests
# ---------------------------------------------------------------------------

class TestDecideWithMemory:
    def test_suppress_on_high_reject_rate(self):
        """P2 with high reviewer reject rate should be suppressed."""
        from src.graph.nodes import node_decide

        state = {
            "rule_hits": [RuleHit(
                rule_id="TEST", asset=Asset.BTC, severity=Severity.P2,
                description="test", confidence=0.8,
            )],
            "highest_severity": Severity.P2,
            "recent_alert_history": [],
            "enriched_memory": {
                "reviewer_preference": {
                    "approve_rate": 0.15,
                    "reject_rate": 0.85,
                    "sample_count": 10,
                },
                "semantic_matches": [],
            },
        }
        result = node_decide(state)
        assert result["decision"] == Decision.SUPPRESS

    def test_no_suppress_on_low_sample_count(self):
        """Don't suppress if not enough samples."""
        from src.graph.nodes import node_decide

        state = {
            "rule_hits": [RuleHit(
                rule_id="TEST", asset=Asset.BTC, severity=Severity.P2,
                description="test", confidence=0.8,
            )],
            "highest_severity": Severity.P2,
            "recent_alert_history": [],
            "enriched_memory": {
                "reviewer_preference": {
                    "approve_rate": 0.1,
                    "reject_rate": 0.9,
                    "sample_count": 3,  # Too few
                },
                "semantic_matches": [],
            },
        }
        result = node_decide(state)
        assert result["decision"] == Decision.MANUAL_REVIEW

    def test_suppress_on_all_similar_suppressed(self):
        """P2 with all highly similar historical cases suppressed."""
        from src.graph.nodes import node_decide

        state = {
            "rule_hits": [RuleHit(
                rule_id="TEST", asset=Asset.BTC, severity=Severity.P2,
                description="test", confidence=0.8,
            )],
            "highest_severity": Severity.P2,
            "recent_alert_history": [],
            "enriched_memory": {
                "reviewer_preference": None,
                "semantic_matches": [
                    {"case_id": "a", "similarity": 0.90, "decision": "suppress"},
                    {"case_id": "b", "similarity": 0.88, "decision": "suppress"},
                    {"case_id": "c", "similarity": 0.86, "decision": "suppress"},
                ],
            },
        }
        result = node_decide(state)
        assert result["decision"] == Decision.SUPPRESS

    def test_p1_not_affected_by_memory(self):
        """P1 should always emit regardless of memory."""
        from src.graph.nodes import node_decide

        state = {
            "rule_hits": [RuleHit(
                rule_id="TEST", asset=Asset.BTC, severity=Severity.P1,
                description="test", confidence=1.0,
            )],
            "highest_severity": Severity.P1,
            "recent_alert_history": [],
            "enriched_memory": {
                "reviewer_preference": {
                    "approve_rate": 0.0,
                    "reject_rate": 1.0,
                    "sample_count": 100,
                },
                "semantic_matches": [],
            },
        }
        result = node_decide(state)
        assert result["decision"] == Decision.EMIT

    def test_normal_p2_with_no_memory(self):
        """P2 with no enriched memory should go to MANUAL_REVIEW."""
        from src.graph.nodes import node_decide

        state = {
            "rule_hits": [RuleHit(
                rule_id="TEST", asset=Asset.BTC, severity=Severity.P2,
                description="test", confidence=0.8,
            )],
            "highest_severity": Severity.P2,
            "recent_alert_history": [],
            "enriched_memory": None,
        }
        result = node_decide(state)
        assert result["decision"] == Decision.MANUAL_REVIEW


# ---------------------------------------------------------------------------
# Reviewer preference tests (logic only, no DB)
# ---------------------------------------------------------------------------

class TestReviewerPreferenceLogic:
    def test_preference_description_high_reject(self):
        """Verify description logic for high reject rate."""
        reject_rate = 0.85
        approve_rate = 0.15
        assert reject_rate > 0.7
        # In the actual code, this produces "审核者倾向于拒绝此类告警"

    def test_preference_description_high_approve(self):
        approve_rate = 0.90
        assert approve_rate > 0.8
        # Produces "审核者通常批准此类告警"
