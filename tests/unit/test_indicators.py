"""
Unit tests for src/indicators.py — all run on synthetic data (no network).
"""

import numpy as np
import pandas as pd
import pytest

from src.indicators import (
    calculate_ema,
    calculate_rsi,
    add_indicators,
    detect_inside_day,
    detect_vcp,
    calculate_rmv,
    near_10ema_score,
    ema_stack_ok,
    price_above_20ema,
    get_52w_high_low,
    get_performance,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 300, trend: float = 0.001) -> pd.DataFrame:
    """Synthetic uptrending OHLCV DataFrame."""
    rng = np.random.default_rng(42)
    close = 100.0 * np.cumprod(1 + rng.normal(trend, 0.01, n))
    high  = close * (1 + rng.uniform(0.001, 0.02, n))
    low   = close * (1 - rng.uniform(0.001, 0.02, n))
    vol   = rng.integers(100_000, 2_000_000, n).astype(float)
    df = pd.DataFrame({"Open": close * 0.999, "High": high, "Low": low, "Close": close, "Volume": vol})
    df.index = pd.date_range("2023-01-01", periods=n, freq="B")
    return df


def _make_flat_ohlcv(n: int = 300) -> pd.DataFrame:
    """Synthetic flat-price OHLCV."""
    df = pd.DataFrame({
        "Open":   [100.0] * n,
        "High":   [101.0] * n,
        "Low":    [99.0]  * n,
        "Close":  [100.0] * n,
        "Volume": [500_000.0] * n,
    })
    df.index = pd.date_range("2023-01-01", periods=n, freq="B")
    return df


def _last_row(df: pd.DataFrame) -> pd.Series:
    """Return last row of a DataFrame with add_indicators applied."""
    return add_indicators(df).iloc[-1]


# ---------------------------------------------------------------------------
# EMA tests
# ---------------------------------------------------------------------------

class TestCalculateEMA:
    def test_length_matches_input(self):
        s = pd.Series(range(1, 51), dtype=float)
        result = calculate_ema(s, 10)
        assert len(result) == len(s)

    def test_ema_values_between_min_and_max(self):
        s = pd.Series(range(1, 101), dtype=float)
        ema = calculate_ema(s, 20)
        assert ema.min() >= s.min()
        assert ema.max() <= s.max()

    def test_ema_smoothing(self):
        df = _make_ohlcv()
        ema200 = calculate_ema(df["Close"], 200)
        assert ema200.std() < df["Close"].std()

    def test_ema1_converges_quickly(self):
        """EMA with span=1 should equal the raw series (no smoothing)."""
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = calculate_ema(s, 1)
        # span=1 → alpha=1 → each EMA value equals the input value
        pd.testing.assert_series_equal(result, s, check_names=False)


# ---------------------------------------------------------------------------
# RSI tests
# ---------------------------------------------------------------------------

class TestCalculateRSI:
    def test_rsi_bounds(self):
        df = _make_ohlcv()
        rsi = calculate_rsi(df["Close"])
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_falling_price_low_rsi(self):
        s = pd.Series(range(100, 0, -1), dtype=float)
        rsi = calculate_rsi(s)
        assert rsi.iloc[-1] < 30

    def test_flat_price_rsi_near_50(self):
        s = pd.Series([100.0] * 50)
        rsi = calculate_rsi(s)
        assert rsi.dropna().empty or abs(rsi.dropna().iloc[-1] - 50) < 5

    def test_rising_series_rsi_above_50(self):
        """After enough bars, a generally rising series has RSI > 50."""
        rng = np.random.default_rng(99)
        s = pd.Series(100.0 * np.cumprod(1 + rng.normal(0.005, 0.01, 100)))
        rsi = calculate_rsi(s)
        valid = rsi.dropna()
        assert len(valid) > 0 and valid.iloc[-1] > 50


# ---------------------------------------------------------------------------
# EMA stack / price filters (operate on a pre-computed row)
# ---------------------------------------------------------------------------

class TestEMAFilters:
    def test_ema_stack_ok_uptrend(self):
        """Strong uptrend should have EMA10 > EMA20 > EMA50 > EMA200."""
        df = _make_ohlcv(n=300, trend=0.003)
        row = _last_row(df)
        assert ema_stack_ok(row)  # numpy bool — don't use `is True`

    def test_price_above_20ema_uptrend(self):
        df = _make_ohlcv(n=300, trend=0.003)
        row = _last_row(df)
        assert price_above_20ema(row) is True

    def test_price_below_20ema_downtrend(self):
        df = _make_ohlcv(n=300, trend=-0.004)
        row = _last_row(df)
        assert price_above_20ema(row) is False


# ---------------------------------------------------------------------------
# 52W high/low + performance
# ---------------------------------------------------------------------------

class TestHighLowPerf:
    def test_52w_high_gte_low(self):
        df = _make_ohlcv()
        high, low = get_52w_high_low(df)
        assert high >= low

    def test_52w_high_is_high_column_max(self):
        """get_52w_high_low uses the High column, not Close."""
        df = _make_ohlcv()
        high, _ = get_52w_high_low(df)
        assert high == pytest.approx(df["High"].tail(252).max(), rel=1e-6)

    def test_performance_returns_tuple(self):
        df = _make_ohlcv(n=300, trend=0.003)
        result = get_performance(df)
        assert isinstance(result, tuple) and len(result) == 2

    def test_performance_positive_uptrend(self):
        df = _make_ohlcv(n=300, trend=0.003)
        weekly, monthly = get_performance(df)
        assert monthly > 0

    def test_performance_negative_downtrend(self):
        df = _make_ohlcv(n=300, trend=-0.004)
        weekly, monthly = get_performance(df)
        assert monthly < 0


# ---------------------------------------------------------------------------
# Inside Day detection
# ---------------------------------------------------------------------------

class TestInsideDay:
    def _make_with_inside_day(self) -> pd.DataFrame:
        df = _make_ohlcv(n=10)
        prev_high = df["High"].iloc[-2]
        prev_low  = df["Low"].iloc[-2]
        df.loc[df.index[-1], "High"] = prev_high - 0.1
        df.loc[df.index[-1], "Low"]  = prev_low  + 0.1
        return df

    def _make_without_inside_day(self) -> pd.DataFrame:
        df = _make_ohlcv(n=10)
        df.loc[df.index[-1], "High"] = df["High"].iloc[-2] + 5
        df.loc[df.index[-1], "Low"]  = df["Low"].iloc[-2]  + 5
        return df

    def test_detects_inside_day(self):
        assert detect_inside_day(self._make_with_inside_day()) is True

    def test_no_inside_day_on_breakout(self):
        assert detect_inside_day(self._make_without_inside_day()) is False

    def test_requires_at_least_two_bars(self):
        df = _make_ohlcv(n=1)
        assert detect_inside_day(df) is False


# ---------------------------------------------------------------------------
# RMV tests
# ---------------------------------------------------------------------------

class TestRMV:
    def test_rmv_in_range(self):
        df = _make_ohlcv(n=50)
        rmv = calculate_rmv(df)
        assert 0 <= rmv <= 100

    def test_compressed_bar_low_rmv(self):
        df = _make_ohlcv(n=30)
        mid = float(df["Close"].iloc[-1])
        df.loc[df.index[-1], "High"] = mid + 0.01
        df.loc[df.index[-1], "Low"]  = mid - 0.01
        rmv = calculate_rmv(df)
        assert rmv < 15

    def test_nan_on_insufficient_data(self):
        df = _make_ohlcv(n=5)
        rmv = calculate_rmv(df)
        assert np.isnan(rmv)


# ---------------------------------------------------------------------------
# Near-EMA score (takes a pre-computed row with ema_10 and Close)
# ---------------------------------------------------------------------------

class TestNear10EMAScore:
    def _row_at_ema(self, cmp: float, ema10: float) -> pd.Series:
        return pd.Series({"Close": cmp, "ema_10": ema10})

    def test_at_ema_gives_nonzero_score(self):
        # Band is asymmetric (-2.5% to +1.5%); center ≠ EMA, so score < 1.0
        # but price at EMA is inside the band → score must be > 0
        row = self._row_at_ema(100.0, 100.0)
        score = near_10ema_score(row)
        assert score > 0.5

    def test_far_above_ema_gives_zero(self):
        row = self._row_at_ema(115.0, 100.0)  # 15% above — way outside band
        score = near_10ema_score(row)
        assert score == 0.0

    def test_far_below_ema_gives_zero(self):
        row = self._row_at_ema(90.0, 100.0)  # 10% below — outside band
        score = near_10ema_score(row)
        assert score == 0.0
