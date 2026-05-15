"""MemoryManager — unified facade for all memory operations.

Used by LangGraph nodes to:
1. Load enriched memory context (short-term + long-term + semantic)
2. Store case embeddings after case creation
3. Retrieve reviewer preferences for decision support
"""
from __future__ import annotations

import json
from typing import Any

from src.core.config import settings
from src.core.logging import get_logger
from src.domain.models import Asset, FeatureSnapshot, RiskCase, RuleHit
from src.observability.llm_trace import run_in_executor_with_context

logger = get_logger(__name__)


class MemoryManager:
    """Stateless facade — all state lives in VectorStore and DB."""

    async def build_enriched_context(
        self,
        asset: Asset,
        snapshot: FeatureSnapshot | None,
        rule_hits: list[RuleHit],
        *,
        short_term_context: dict | None = None,
    ) -> dict[str, Any]:
        """Build a comprehensive memory context combining short-term and long-term memory.

        Returns a dict with:
        - short_term: the original node_load_memory context (recent cases, alerts, fatigue)
        - semantic_matches: top-K semantically similar historical cases
        - experience_insights: distilled patterns relevant to current asset/rules
        - reviewer_preference: learned approve/reject tendencies for this rule pattern
        """
        result: dict[str, Any] = {
            "short_term": short_term_context or {},
            "semantic_matches": [],
            "experience_insights": [],
            "reviewer_preference": None,
        }

        if not settings.memory_enabled:
            return result

        # 1. Semantic search — find similar historical cases
        if snapshot is not None:
            try:
                semantic = await self._semantic_search(asset, snapshot, rule_hits)
                result["semantic_matches"] = semantic
            except Exception as exc:
                logger.warning("Semantic search failed: %s", exc)

        # 2. Experience insights — distilled patterns
        try:
            from src.memory.experience import get_insights
            insights = await get_insights(asset=asset.value, limit=5)
            result["experience_insights"] = insights
        except Exception as exc:
            logger.warning("Experience insight retrieval failed: %s", exc)

        # 3. Reviewer preferences — learned approval patterns
        try:
            hits_json = json.dumps([h.model_dump(mode="json") for h in rule_hits])
            from src.memory.reviewer_preference import get_preference_for_hits
            pref = await get_preference_for_hits(asset.value, hits_json)
            result["reviewer_preference"] = pref
        except Exception as exc:
            logger.warning("Reviewer preference lookup failed: %s", exc)

        return result

    async def _semantic_search(
        self,
        asset: Asset,
        snapshot: FeatureSnapshot,
        rule_hits: list[RuleHit],
    ) -> list[dict[str, Any]]:
        """Search for semantically similar historical cases."""
        from src.memory.embedding import build_case_text, embed_text
        from src.memory.store import get_vector_store

        store = get_vector_store()
        if store.size == 0:
            return []

        # Build query text from current snapshot context
        rule_desc = "; ".join(f"{h.rule_id}({h.severity.value})" for h in rule_hits[:5])
        query_text = build_case_text(
            f"ret_1m={snapshot.ret_1m:.4f} ret_5m={snapshot.ret_5m:.4f} "
            f"liq={snapshot.liq_5m_usd:.0f} oi_delta={snapshot.oi_delta_15m_pct:.4f} "
            f"funding_z={snapshot.funding_z:.2f}",
            rule_desc,
            asset.value,
        )

        query_vec = await run_in_executor_with_context(
            lambda: embed_text(query_text)
        )

        # Search across all assets for cross-asset patterns, but rank same-asset higher
        matches = store.search(
            query_vec,
            top_k=settings.memory_top_k,
            min_similarity=settings.memory_similarity_threshold,
        )

        # Boost same-asset matches in ranking
        for m in matches:
            if m.get("asset") == asset.value:
                m["similarity"] = min(1.0, m["similarity"] * 1.1)

        matches.sort(key=lambda m: m["similarity"], reverse=True)
        return matches[:settings.memory_top_k]

    async def store_case_embedding(self, case: RiskCase) -> None:
        """Generate and store embedding for a completed case."""
        if not settings.memory_enabled:
            return

        from src.memory.embedding import build_case_text, content_hash, embed_text
        from src.memory.store import get_vector_store

        store = get_vector_store()
        if store.has_embedding(case.case_id):
            return

        rule_desc = "; ".join(
            f"{h.rule_id}({h.severity.value})" for h in case.rule_hits[:5]
        )
        text = build_case_text(case.summary_zh, rule_desc, case.asset.value)
        c_hash = content_hash(text)

        try:
            vec = await run_in_executor_with_context(
                lambda: embed_text(text)
            )
            await store.add(
                case.case_id,
                vec,
                asset=case.asset.value,
                severity=case.severity.value if case.severity else None,
                decision=case.decision.value if case.decision else None,
                summary_zh=case.summary_zh[:200],
                content_hash=c_hash,
            )
            logger.debug("Stored embedding for case %s", case.case_id)
        except Exception as exc:
            logger.warning("Failed to store case embedding for %s: %s", case.case_id, exc)


# Global singleton
_manager: MemoryManager | None = None


def get_memory_manager() -> MemoryManager:
    global _manager
    if _manager is None:
        _manager = MemoryManager()
    return _manager
