"""LangGraph orchestrator — SQLite checkpointer for persistent resume."""
from __future__ import annotations

import uuid

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from src.core.config import settings
from src.domain.models import Asset, Decision, FeatureSnapshot
from src.graph.nodes import (
    node_build_case, node_decide, node_human_review,
    node_llm_explain, node_run_rules, node_send_alert,
)
from src.graph.state import RiskState

# SQLite 文件路径从 DATABASE_URL 里提取，fallback 到固定路径
_DB_PATH = settings.database_url.replace("sqlite+aiosqlite:///", "").replace("./", "")


def _route_after_decide(state: RiskState) -> str:
    d = state.get("decision")
    if d == Decision.SUPPRESS:
        return END
    return "build_case"


def _route_after_build(state: RiskState) -> str:
    d = state.get("decision")
    if d == Decision.MANUAL_REVIEW:
        return "await_review"   # interrupt_before 这个节点，图在此暂停
    return "send_alert"


def _route_after_review(state: RiskState) -> str:
    """resume 后：approved → send_alert，rejected → END"""
    if state.get("human_approved"):
        return "send_alert"
    return END


def build_graph() -> StateGraph:
    g = StateGraph(RiskState)

    g.add_node("run_rules", node_run_rules)
    g.add_node("llm_explain", node_llm_explain)
    g.add_node("decide", node_decide)
    g.add_node("build_case", node_build_case)
    g.add_node("await_review", node_human_review)  # interrupt_before 此节点
    g.add_node("send_alert", node_send_alert)

    g.add_edge(START, "run_rules")
    g.add_edge("run_rules", "llm_explain")
    g.add_edge("llm_explain", "decide")
    g.add_conditional_edges("decide", _route_after_decide, {"build_case": "build_case", END: END})
    g.add_conditional_edges("build_case", _route_after_build, {
        "await_review": "await_review",
        "send_alert": "send_alert",
    })
    g.add_conditional_edges("await_review", _route_after_review, {
        "send_alert": "send_alert",
        END: END,
    })
    g.add_edge("send_alert", END)

    return g


# 全局 checkpointer 和 compiled graph，在 init_graph() 里初始化
_checkpointer: AsyncSqliteSaver | None = None
_compiled = None
_db_conn = None  # 持有 aiosqlite 连接，防止被 GC


async def init_graph() -> None:
    """必须在 asyncio 事件循环启动后调用一次，初始化 SQLite checkpointer。"""
    global _checkpointer, _compiled, _db_conn
    # 直接打开 aiosqlite 连接并传给 AsyncSqliteSaver
    _db_conn = await aiosqlite.connect(_DB_PATH)
    _checkpointer = AsyncSqliteSaver(_db_conn)
    await _checkpointer.setup()  # 建 checkpoints / writes 表
    _compiled = build_graph().compile(
        checkpointer=_checkpointer,
        interrupt_before=["await_review"],  # 在 await_review 节点前暂停
    )


async def process_snapshot(snap: FeatureSnapshot) -> None:
    """提交一个快照进入图处理，每次新建独立 thread_id（case_id）。"""
    if _compiled is None:
        raise RuntimeError("Graph not initialized, call init_graph() first")

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial: RiskState = {
        "asset": snap.asset,
        "snapshot": snap,
        "rule_hits": [],
        "highest_severity": None,
        "summary_zh": "",
        "review_guidance": "",
        "decision": None,
        "case": None,
        "alert": None,
        "human_approved": None,
        "human_comment": "",
    }
    await _compiled.ainvoke(initial, config=config)


async def resume_case(case_id: str, approved: bool, comment: str = "") -> None:
    """恢复被 interrupt_before 暂停的 P2 await_review 节点。进程重启后仍有效。"""
    if _compiled is None:
        raise RuntimeError("Graph not initialized, call init_graph() first")

    config = {"configurable": {"thread_id": case_id}}
    # 1. 把审核结果写入 checkpoint state
    await _compiled.aupdate_state(
        config,
        {"human_approved": approved, "human_comment": comment},
        as_node="await_review",
    )
    # 2. 继续执行图（从 await_review 节点往后）
    await _compiled.ainvoke(None, config=config)


def get_compiled_graph():
    return _compiled
