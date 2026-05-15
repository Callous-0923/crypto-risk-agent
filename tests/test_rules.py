"""Unit tests for rule engine — no external dependencies needed."""
import pytest
from datetime import datetime

from src.domain.models import Asset, FeatureSnapshot, Severity
from src.rules.engine import evaluate


def _snap(**kwargs) -> FeatureSnapshot:
    defaults = dict(
        asset=Asset.BTC,
        price=60000.0,
        ret_1m=0.0,
        ret_5m=0.0,
        vol_z_1m=0.0,
        oi_delta_15m_pct=0.0,
        liq_5m_usd=0.0,
        funding_z=0.0,
        source_stale=False,
        cross_source_conflict=False,
    )
    defaults.update(kwargs)
    return FeatureSnapshot(**defaults)


def test_no_hits_on_normal():
    hits = evaluate(_snap(ret_1m=0.001, vol_z_1m=0.5))
    assert hits == []


def test_p1_extreme_price_move():
    hits = evaluate(_snap(ret_1m=0.06))
    assert any(h.rule_id == "MKT_EXTREME_VOL_P1" for h in hits)
    assert any(h.severity == Severity.P1 for h in hits)


def test_p2_moderate_price_move():
    hits = evaluate(_snap(ret_1m=0.035))
    assert any(h.rule_id == "MKT_EXTREME_VOL_P2" for h in hits)


def test_liquidation_cascade():
    hits = evaluate(_snap(liq_5m_usd=60_000_000))
    assert any(h.rule_id == "DERIV_CASCADE_LIQ_P1" for h in hits)


def test_oi_buildup():
    hits = evaluate(_snap(oi_delta_15m_pct=0.12))
    assert any(h.rule_id == "DERIV_OI_BUILDUP" for h in hits)


def test_quality_stale():
    hits = evaluate(_snap(source_stale=True))
    assert any(h.rule_id == "QA_SOURCE_STALE" for h in hits)


def test_quality_conflict():
    hits = evaluate(_snap(cross_source_conflict=True))
    assert any(h.rule_id == "QA_CROSS_SOURCE_CONFLICT" for h in hits)


def test_early_warning_price_drift_before_formal_p2():
    hits = evaluate(_snap(ret_5m=0.01))
    assert any(h.rule_id == "EW_PRICE_DRIFT_5M" for h in hits)
    assert any(h.severity == Severity.P3 for h in hits)


def test_early_warning_does_not_duplicate_formal_p2():
    hits = evaluate(_snap(ret_1m=0.035, ret_5m=0.04))
    assert any(h.rule_id == "MKT_EXTREME_VOL_P2" for h in hits)
    assert not any(h.rule_id.startswith("EW_") for h in hits)
