import asyncio
import unittest
from unittest.mock import patch


class Phase8Tests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        import src.api.app as app_mod

        if app_mod.get_agent_status()["running"]:
            await app_mod.stop_ingestion()

    async def test_start_and_stop_ingestion_updates_agent_status(self):
        import src.api.app as app_mod

        started = asyncio.Event()

        async def fake_supervisor():
            started.set()
            try:
                while True:
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                raise

        with patch.object(app_mod, "_run_ingestion_supervisor", fake_supervisor):
            await app_mod.start_ingestion()
            await asyncio.wait_for(started.wait(), timeout=1)
            self.assertTrue(app_mod.get_agent_status()["running"])
            await app_mod.stop_ingestion()
            self.assertFalse(app_mod.get_agent_status()["running"])

    async def test_start_ingestion_is_idempotent_guarded(self):
        import src.api.app as app_mod

        async def fake_supervisor():
            try:
                while True:
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                raise

        with patch.object(app_mod, "_run_ingestion_supervisor", fake_supervisor):
            await app_mod.start_ingestion()
            with self.assertRaises(RuntimeError):
                await app_mod.start_ingestion()


if __name__ == "__main__":
    unittest.main()
