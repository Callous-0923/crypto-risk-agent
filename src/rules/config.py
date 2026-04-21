"""
规则配置版本管理。

职责：
- RuleThresholds: 一份完整的规则阈值配置（可序列化为 JSON 存库）
- RuleRegistry: 运行时单例，持有当前 active 版本，支持热更新
- 提供 load_active / publish_version / list_versions 三个核心操作
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel
from sqlalchemy import select, update

from src.core.config import settings as app_settings
from src.core.logging import get_logger
from src.persistence.database import AsyncSessionLocal, RuleChangeLogRow, RuleVersionRow

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 规则阈值数据结构
# ---------------------------------------------------------------------------

class RuleThresholds(BaseModel):
    """一套完整的规则阈值配置，可版本化存储和回放。"""
    price_change_p1: float = 0.05
    price_change_p2: float = 0.03
    oi_delta_p2: float = 0.10
    liq_usd_p1: float = 50_000_000.0
    funding_z_p2: float = 2.5
    vol_z_spike: float = 3.0          # 波动率 z-score 触发阈值
    cross_source_conflict_pct: float = 0.005  # 价格冲突偏差阈值

    @classmethod
    def from_app_settings(cls) -> "RuleThresholds":
        """从 config.py 读取初始值，用于首次入库。"""
        return cls(
            price_change_p1=app_settings.price_change_p1,
            price_change_p2=app_settings.price_change_p2,
            oi_delta_p2=app_settings.oi_delta_p2,
            liq_usd_p1=app_settings.liq_usd_p1,
            funding_z_p2=app_settings.funding_z_p2,
        )

    def diff(self, other: "RuleThresholds") -> dict[str, dict[str, Any]]:
        """返回与另一版本的差异字段。"""
        result = {}
        for field in self.model_fields:
            old_val = getattr(self, field)
            new_val = getattr(other, field)
            if old_val != new_val:
                result[field] = {"old": old_val, "new": new_val}
        return result


# ---------------------------------------------------------------------------
# 运行时注册表（单例）
# ---------------------------------------------------------------------------

class RuleRegistry:
    """持有当前生效的规则阈值，支持不停机热更新。"""

    def __init__(self) -> None:
        self._active: RuleThresholds = RuleThresholds.from_app_settings()
        self._active_version: str = "default"

    @property
    def thresholds(self) -> RuleThresholds:
        return self._active

    @property
    def active_version(self) -> str:
        return self._active_version

    async def load_active(self) -> None:
        """启动时从数据库加载 active 版本；若无则自动写入 v1。"""
        async with AsyncSessionLocal() as s:
            result = await s.execute(
                select(RuleVersionRow).where(RuleVersionRow.is_active == True)
            )
            row = result.scalar_one_or_none()

        if row is None:
            logger.info("No active rule version found, initializing v1 from app settings")
            await self._init_v1()
        else:
            self._active = RuleThresholds(**row.thresholds)
            self._active_version = row.version_tag
            logger.info("Loaded rule version %s", row.version_tag)

    async def _init_v1(self) -> None:
        """首次启动：把 config.py 里的阈值写入数据库作为 v1。"""
        thresholds = RuleThresholds.from_app_settings()
        async with AsyncSessionLocal() as s:
            s.add(RuleVersionRow(
                version_tag="v1",
                thresholds=thresholds.model_dump(),
                is_active=True,
                created_by="system",
                created_at=datetime.utcnow(),
                description="初始版本，从 config.py 自动生成",
            ))
            await s.commit()
        self._active = thresholds
        self._active_version = "v1"
        logger.info("Initialized rule version v1")

    async def publish_version(
        self,
        new_thresholds: RuleThresholds,
        version_tag: str,
        created_by: str,
        reason: str = "",
    ) -> None:
        """
        发布新版本：
        1. 写入 rule_version 表（is_active=True）
        2. 旧版本置 is_active=False
        3. 写入 rule_change_log（diff）
        4. 热更新运行时 _active
        """
        diff = self._active.diff(new_thresholds)
        if not diff:
            raise ValueError("新阈值与当前版本完全相同，无需发布")

        async with AsyncSessionLocal() as s:
            # 检查 version_tag 不重复
            existing = await s.execute(
                select(RuleVersionRow).where(RuleVersionRow.version_tag == version_tag)
            )
            if existing.scalar_one_or_none():
                raise ValueError(f"版本号 {version_tag} 已存在")

            # 旧版本 deactivate
            await s.execute(
                update(RuleVersionRow)
                .where(RuleVersionRow.is_active == True)
                .values(is_active=False)
            )

            # 写新版本
            s.add(RuleVersionRow(
                version_tag=version_tag,
                thresholds=new_thresholds.model_dump(),
                is_active=True,
                created_by=created_by,
                created_at=datetime.utcnow(),
                description=reason,
            ))

            # 写审计日志
            s.add(RuleChangeLogRow(
                log_id=str(uuid.uuid4()),
                from_version=self._active_version,
                to_version=version_tag,
                changed_by=created_by,
                diff=diff,
                changed_at=datetime.utcnow(),
                reason=reason,
            ))
            await s.commit()

        # 热更新（不重启进程）
        self._active = new_thresholds
        self._active_version = version_tag
        logger.info(
            "Rule version updated: %s → %s, changed fields: %s",
            self._active_version, version_tag, list(diff.keys()),
        )


# 全局单例
registry = RuleRegistry()
