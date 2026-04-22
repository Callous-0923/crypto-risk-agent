"""Minimal tool definitions and dispatchers for graph LLM nodes."""
from __future__ import annotations

import json

from src.domain.models import Asset
from src.persistence.repositories import (
    get_latest_snapshot,
    get_recent_alerts,
    get_recent_snapshots,
    get_rule_version_info,
)

GRAPH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_recent_snapshots",
            "description": "Fetch recent feature snapshots for one asset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "asset": {"type": "string", "enum": [asset.value for asset in Asset]},
                    "n": {"type": "integer", "minimum": 1, "maximum": 20, "default": 10},
                },
                "required": ["asset"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_alerts",
            "description": "Fetch recent alerts for one asset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "asset": {"type": "string", "enum": [asset.value for asset in Asset]},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                },
                "required": ["asset"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sibling_snapshot",
            "description": "Fetch latest snapshots for the other major assets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "asset": {"type": "string", "enum": [asset.value for asset in Asset]},
                },
                "required": ["asset"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_rule_version_info",
            "description": "Fetch the currently active rule version and thresholds.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]


async def call_tool(name: str, args: dict) -> str:
    if name == "get_recent_snapshots":
        asset = Asset(args["asset"])
        n = int(args.get("n", 10))
        snapshots = await get_recent_snapshots(asset, n=n)
        return json.dumps([snapshot.model_dump(mode="json") for snapshot in snapshots], ensure_ascii=False)

    if name == "get_recent_alerts":
        asset = Asset(args["asset"])
        limit = int(args.get("limit", 5))
        alerts = await get_recent_alerts(asset, limit=limit)
        return json.dumps([alert.model_dump(mode="json") for alert in alerts], ensure_ascii=False)

    if name == "get_sibling_snapshot":
        asset = Asset(args["asset"])
        siblings = {}
        for sibling in Asset:
            if sibling == asset:
                continue
            snapshot = await get_latest_snapshot(sibling)
            siblings[sibling.value] = snapshot.model_dump(mode="json") if snapshot else None
        return json.dumps(siblings, ensure_ascii=False)

    if name == "get_rule_version_info":
        return json.dumps(await get_rule_version_info(), ensure_ascii=False)

    raise ValueError(f"Unknown tool: {name}")
