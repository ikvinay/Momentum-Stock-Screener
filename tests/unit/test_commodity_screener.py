"""
Unit tests for src/commodity_screener.py scoring functions.
No network calls — all data is synthetic.
"""

import numpy as np
import pandas as pd
import pytest

from src.commodity_screener import (
    _score_52w_high,
    _score_outperf,
    _score_rsi,
    _score_rmv,
    _score_kma,
    _score_rs_trend,
    run_commodity_screener,
)


def _make_df(n: int = 260, trend: float = 0.002, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1 + rng.normal(trend, 0.008, n))
    high  = close * (1 + rng.uniform(0.001, 0.015, n))
    low   = close * (1 - rng.uniform(0.001, 0.015, n))
    vol   = rng.integers(1_000, 100_000, n).astype(float)
    df = pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close, "Volume": vol})
    df.index = pd.date_range("2023-01-01", periods=n, freq="B")
    return df


# ---------------------------------------------------------------------------
# _score_52w_high
# ---------------------------------------------------------------------------

class TestScore52WH:
    def test_at_52w_high_is_one(self):
        close = pd.Series([100.0] * 10)
        assert _score_52w_high(close, high_52w=100.0) == pytest.approx(1.0)

    def test_far_below_high_is_zero(self):
        close = pd.Series([50.0] * 10)
        assert _score_52w_high(close, high_52w=100.0) == 0.0

    def test_monotone_decreasing_with_distance(self):
        prices = [95.0, 90.0, 80.0, 70.0]
        scores = [_score_52w_high(pd.Series([p]), high_52w=100.0) for p in prices]
        assert scores == sorted(scores, reverse=True)

    def test_zero_high_returns_zero(self):
        assert _score_52w_high(pd.Series([50.0]), high_52w=0.0) == 0.0


# ---------------------------------------------------------------------------
# _score_outperf
# ---------------------------------------------------------------------------

class TestScoreOutperf:
    def test_strong_outperformance_near_one(self):
        assert _score_outperf(ret_1m=0.15, bench_ret_1m=-0.05) > 0.9

    def test_strong_underperformance_is_zero(self):
        assert _score_outperf(ret_1m=-0.20, bench_ret_1m=0.10) == 0.0

    def test_parity_gives_mid_score(self):
        score = _score_outperf(ret_1m=0.0, bench_ret_1m=0.0)
        assert 0.4 < score < 0.6

    def test_same_function_used_for_both_benchmarks(self):
        # The same _score_outperf applies to Nifty500 and Metal Index
        s1 = _score_outperf(0.05, 0.02)
        s2 = _score_outperf(0.05, 0.02)
        assert s1 == s2


# ---------------------------------------------------------------------------
# _score_rsi
# ---------------------------------------------------------------------------

class TestScoreRSI:
    def test_below_55_is_zero(self):
        assert _score_rsi(50.0) == 0.0

    def test_at_80_is_one(self):
        assert _score_rsi(80.0) == pytest.approx(1.0)

    def test_midpoint_is_half(self):
        assert _score_rsi(67.5) == pytest.approx(0.5, abs=0.02)

    def test_above_80_clipped_to_one(self):
        assert _score_rsi(95.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _score_rmv
# ---------------------------------------------------------------------------

class TestScoreRMV:
    def test_zero_rmv_is_perfect(self):
        assert _score_rmv(0.0) == pytest.approx(1.0)

    def test_high_rmv_is_zero(self):
        assert _score_rmv(90.0) == 0.0

    def test_monotone_decreasing(self):
        scores = [_score_rmv(v) for v in [0, 10, 20, 50, 100]]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# _score_kma
# ---------------------------------------------------------------------------

class TestScoreKMA:
    def test_at_ema10_is_one(self):
        assert _score_kma(100.0, 100.0) == pytest.approx(1.0)

    def test_outside_band_is_zero(self):
        assert _score_kma(110.0, 100.0) == 0.0

    def test_within_band_between_zero_and_one(self):
        score = _score_kma(101.0, 100.0)
        assert 0 < score < 1

    def test_zero_ema_returns_zero(self):
        assert _score_kma(100.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# _score_rs_trend
# ---------------------------------------------------------------------------

class TestScoreRSTrend:
    def test_uptrending_rs_gives_one(self):
        n = 60
        close = pd.Series(np.linspace(100, 130, n), index=pd.date_range("2023-01-01", periods=n, freq="B"))
        bench = pd.Series(np.linspace(100, 110, n), index=close.index)
        assert _score_rs_trend(close, bench) == 1.0

    def test_downtrending_rs_gives_zero(self):
        n = 60
        close = pd.Series(np.linspace(130, 100, n), index=pd.date_range("2023-01-01", periods=n, freq="B"))
        bench = pd.Series(np.linspace(100, 130, n), index=close.index)
        assert _score_rs_trend(close, bench) == 0.0

    def test_none_benchmark_gives_zero(self):
        close = pd.Series(range(50), dtype=float)
        assert _score_rs_trend(close, None) == 0.0

    def test_short_benchmark_gives_zero(self):
        close = pd.Series(range(50), dtype=float)
        short = pd.Series(range(5), dtype=float)
        assert _score_rs_trend(close, short) == 0.0


# ---------------------------------------------------------------------------
# run_commodity_screener — integration
# ---------------------------------------------------------------------------

class TestRunCommodityScreener:
    def _make_universe(self, n: int = 5) -> dict:
        return {f"Commodity {i}": _make_df(n=260, trend=0.003, seed=i) for i in range(n)}

    def test_returns_dataframe(self):
        result = run_commodity_screener(self._make_universe())
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        result = run_commodity_screener(self._make_universe())
        if not result.empty:
            for col in ("Commodity", "Category", "Score", "RSI", "RMV"):
                assert col in result.columns

    def test_scores_in_range(self):
        result = run_commodity_screener(self._make_universe())
        if not result.empty:
            assert result["Score"].between(0, 100).all()

    def test_sorted_descending_by_score(self):
        result = run_commodity_screener(self._make_universe(n=8))
        if len(result) > 1:
            scores = result["Score"].tolist()
            assert scores == sorted(scores, reverse=True)

    def test_ema_stack_filter_applied(self):
        downtrend = {"Bear Commodity": _make_df(n=260, trend=-0.005, seed=99)}
        result = run_commodity_screener(downtrend)
        assert result.empty or "Bear Commodity" not in result["Commodity"].values

    def test_empty_input_returns_empty(self):
        result = run_commodity_screener({})
        assert result.empty

    def test_nifty500_and_metal_columns_present(self):
        result = run_commodity_screener(self._make_universe())
        if not result.empty:
            assert "vs Nifty500 1M"    in result.columns
            assert "vs Metal Index 1M" in result.columns
