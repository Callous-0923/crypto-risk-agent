"""Normalizer 单元测试 —— 覆盖 dedupe key 生成、事件入队、队列满降级。"""
from __future__ import annotations

import asyncio
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from src.domain.models import Asset, RawEvent


class NormalizerTests(unittest.IsolatedAsyncioTestCase):
    """EventBus 事件归一化与发布测试。"""

    async def asyncSetUp(self):
        import src.ingestion.normalizer as norm

        self._old_bus = norm._event_bus
        norm._event_bus = asyncio.Queue(maxsize=10)

    async def asyncTearDown(self):
        import src.ingestion.normalizer as norm

        norm._event_bus = self._old_bus

    def _make_event(self, **kwargs) -> RawEvent:
        defaults = {
            "asset": Asset.BTC,
            "source": "binance_spot",
            "event_type": "price",
            "event_ts": datetime.now(tz=timezone.utc),
            "payload": {"price": 62000.0},
        }
        defaults.update(kwargs)
        return RawEvent(**defaults)

    async def test_dedupe_key_same_for_same_window(self):
        """同一 30s 窗口内的事件生成相同的 dedupe_key。"""
        from src.ingestion.normalizer import _make_dedupe_key

        t = datetime(2024, 1, 1, 12, 0, 10, tzinfo=timezone.utc)
        e1 = self._make_event(event_ts=t)
        t2 = datetime(2024, 1, 1, 12, 0, 20, tzinfo=timezone.utc)
        e2 = self._make_event(event_ts=t2)

        self.assertEqual(_make_dedupe_key(e1), _make_dedupe_key(e2))

    async def test_dedupe_key_differs_for_diff_window(self):
        """不同 30s 窗口生成不同 dedupe_key。"""
        from src.ingestion.normalizer import _make_dedupe_key

        t1 = datetime(2024, 1, 1, 12, 0, 10, tzinfo=timezone.utc)
        t2 = datetime(2024, 1, 1, 12, 0, 40, tzinfo=timezone.utc)
        e1 = self._make_event(event_ts=t1)
        e2 = self._make_event(event_ts=t2)

        self.assertNotEqual(_make_dedupe_key(e1), _make_dedupe_key(e2))

    async def test_dedupe_key_differs_for_diff_asset(self):
        """不同资产生成不同 dedupe_key。"""
        from src.ingestion.normalizer import _make_dedupe_key

        t = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        e1 = self._make_event(asset=Asset.BTC, event_ts=t)
        e2 = self._make_event(asset=Asset.ETH, event_ts=t)

        self.assertNotEqual(_make_dedupe_key(e1), _make_dedupe_key(e2))

    async def test_dedupe_key_differs_for_diff_source(self):
        """不同数据源生成不同 dedupe_key。"""
        from src.ingestion.normalizer import _make_dedupe_key

        t = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        e1 = self._make_event(source="binance_spot", event_ts=t)
        e2 = self._make_event(source="okx_spot", event_ts=t)

        self.assertNotEqual(_make_dedupe_key(e1), _make_dedupe_key(e2))

    async def test_dedupe_key_is_16_char_hex(self):
        """dedupe_key 长度 = 16。"""
        from src.ingestion.normalizer import _make_dedupe_key

        e = self._make_event()
        key = _make_dedupe_key(e)
        self.assertEqual(len(key), 16)

    async def test_normalize_and_publish_adds_dedupe_key(self):
        """normalize_and_publish 后 event.dedupe_key 非空。"""
        import src.ingestion.normalizer as norm

        e = self._make_event()
        self.assertEqual(e.dedupe_key, "")

        with patch("src.ingestion.normalizer.save_raw_event") as mock_save:
            await norm.normalize_and_publish(e)

        self.assertNotEqual(e.dedupe_key, "")
        self.assertEqual(len(e.dedupe_key), 16)
        mock_save.assert_called_once()

    async def test_normalize_and_publish_pushes_to_event_bus(self):
        """normalize_and_publish 将事件推入 event_bus。"""
        import src.ingestion.normalizer as norm

        e = self._make_event()
        with patch("src.ingestion.normalizer.save_raw_event"):
            await norm.normalize_and_publish(e)

        self.assertEqual(norm._event_bus.qsize(), 1)
        dequeued = norm._event_bus.get_nowait()
        self.assertEqual(dequeued.event_id, e.event_id)

    async def test_normalize_and_publish_handles_db_error_gracefully(self):
        """DB 保存失败不应阻止事件入队。"""
        import src.ingestion.normalizer as norm

        def failing_save(*args, **kwargs):
            raise RuntimeError("DB unavailable")

        e = self._make_event()
        with patch("src.ingestion.normalizer.save_raw_event", side_effect=failing_save):
            await norm.normalize_and_publish(e)

        self.assertEqual(norm._event_bus.qsize(), 1)

    async def test_queue_full_drops_event_gracefully(self):
        """队列满时事件被丢弃，不抛异常。"""
        import src.ingestion.normalizer as norm

        # fill queue
        for _ in range(10):
            norm._event_bus.put_nowait(self._make_event())

        e = self._make_event()
        with patch("src.ingestion.normalizer.save_raw_event"):
            await norm.normalize_and_publish(e)

        self.assertEqual(norm._event_bus.qsize(), 10)

    async def test_event_bus_singleton(self):
        """get_event_bus 返回同一实例。"""
        from src.ingestion.normalizer import get_event_bus

        bus1 = get_event_bus()
        bus2 = get_event_bus()
        self.assertIs(bus1, bus2)


if __name__ == "__main__":
    unittest.main()
