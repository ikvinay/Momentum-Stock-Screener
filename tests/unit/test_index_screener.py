"""
Unit tests for src/index_screener.py scoring functions.
No network calls.
"""

import numpy as np
import pandas as pd
import pytest

from src.index_screener import (
    _score_52w_high,
    _score_nifty500_outperf,
    _score_rsi,
    _score_rmv,
    _score_kma,
    _score_rs_trend,
    _normalise_nselib_df,
    _index_category,
    run_index_screener,
)


def _make_df(n: int = 260, trend: float = 0.002, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1 + rng.normal(trend, 0.008, n))
    high  = close * (1 + rng.uniform(0.001, 0.015, n))
    low   = close * (1 - rng.uniform(0.001, 0.015, n))
    vol   = rng.integers(1_000_000, 10_000_000, n).astype(float)
    df = pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close, "Volume": vol})
    df.index = pd.date_range("2023-01-01", periods=n, freq="B")
    return df


# ---------------------------------------------------------------------------
# Individual scoring functions
# ---------------------------------------------------------------------------

class TestScore52WH:
    def test_at_52w_high_is_one(self):
        close = pd.Series([100.0] * 10)
        score = _score_52w_high(close, high_52w=100.0)
        assert score == pytest.approx(1.0)

    def test_far_below_high_is_zero(self):
        close = pd.Series([50.0] * 10)
        score = _score_52w_high(close, high_52w=100.0)
        assert score == 0.0

    def test_monotone_decreasing_with_distance(self):
        prices = [95.0, 90.0, 80.0, 70.0]
        scores = [_score_52w_high(pd.Series([p]), high_52w=100.0) for p in prices]
        assert scores == sorted(scores, reverse=True)


class TestScoreNifty500Outperf:
    def test_strong_outperformance_near_one(self):
        score = _score_nifty500_outperf(ret_1m=0.15, n500_ret_1m=-0.05)
        assert score > 0.9

    def test_strong_underperformance_is_zero(self):
        score = _score_nifty500_outperf(ret_1m=-0.20, n500_ret_1m=0.10)
        assert score == 0.0

    def test_parity_gives_mid_score(self):
        score = _score_nifty500_outperf(ret_1m=0.0, n500_ret_1m=0.0)
        assert 0.4 < score < 0.6


class TestScoreRSI:
    def test_below_55_is_zero(self):
        assert _score_rsi(50.0) == 0.0

    def test_at_80_is_one(self):
        assert _score_rsi(80.0) == pytest.approx(1.0)

    def test_midpoint_is_half(self):
        assert _score_rsi(67.5) == pytest.approx(0.5, abs=0.02)

    def test_above_80_clipped_to_one(self):
        assert _score_rsi(95.0) == pytest.approx(1.0)


class TestScoreRMV:
    def test_zero_rmv_is_perfect(self):
        assert _score_rmv(0.0) == pytest.approx(1.0)

    def test_high_rmv_near_zero(self):
        score = _score_rmv(90.0)  # > threshold * 3
        assert score == 0.0

    def test_monotone_decreasing(self):
        scores = [_score_rmv(v) for v in [0, 10, 20, 50, 100]]
        assert scores == sorted(scores, reverse=True)


class TestScoreKMA:
    def test_at_ema10_is_one(self):
        from config import KMA_BAND_PCT
        assert _score_kma(100.0, 100.0) == pytest.approx(1.0)

    def test_outside_band_is_zero(self):
        assert _score_kma(110.0, 100.0) == 0.0  # 10% above — way outside band

    def test_within_band_between_zero_and_one(self):
        score = _score_kma(101.0, 100.0)  # 1% above — inside ±1.5% band
        assert 0 < score < 1


class TestScoreRSTrend:
    def test_uptrending_rs_gives_one(self):
        n = 60
        close  = pd.Series(np.linspace(100, 130, n), index=pd.date_range("2023-01-01", periods=n, freq="B"))
        n500   = pd.Series(np.linspace(100, 110, n), index=close.index)
        assert _score_rs_trend(close, n500) == 1.0

    def test_downtrending_rs_gives_zero(self):
        n = 60
        close = pd.Series(np.linspace(130, 100, n), index=pd.date_range("2023-01-01", periods=n, freq="B"))
        n500  = pd.Series(np.linspace(100, 130, n), index=close.index)
        assert _score_rs_trend(close, n500) == 0.0

    def test_none_benchmark_gives_zero(self):
        close = pd.Series(range(50), dtype=float)
        assert _score_rs_trend(close, None) == 0.0


# ---------------------------------------------------------------------------
# run_index_screener integration
# ---------------------------------------------------------------------------

class TestRunIndexScreener:
    def _make_ohlcv_dict(self, n_indices: int = 5) -> dict:
        data = {}
        for i in range(n_indices):
            data[f"Index {i}"] = _make_df(n=260, trend=0.003, seed=i)
        return data

    def test_returns_dataframe(self):
        ohlcv = self._make_ohlcv_dict()
        result = run_index_screener(ohlcv, nifty500_df=_make_df(n=260, trend=0.001))
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        ohlcv = self._make_ohlcv_dict()
        result = run_index_screener(ohlcv)
        if not result.empty:
            for col in ("Index", "Category", "Score", "RSI", "RMV"):
                assert col in result.columns

    def test_scores_in_range(self):
        ohlcv = self._make_ohlcv_dict()
        result = run_index_screener(ohlcv)
        if not result.empty:
            assert result["Score"].between(0, 100).all()

    def test_ema_stack_filter_applied(self):
        # Downtrending data should fail EMA10 > EMA20 > EMA50
        downtrend = {"Downtrend": _make_df(n=260, trend=-0.005, seed=99)}
        result = run_index_screener(downtrend)
        assert result.empty or "Downtrend" not in result["Index"].values

    def test_empty_input_returns_empty(self):
        result = run_index_screener({})
        assert result.empty

    def test_sorted_descending_by_score(self):
        ohlcv = self._make_ohlcv_dict(n_indices=10)
        result = run_index_screener(ohlcv)
        if len(result) > 1:
            scores = result["Score"].tolist()
            assert scores == sorted(scores, reverse=True)

    def test_rs_rating_column_present_and_in_range(self):
        ohlcv = self._make_ohlcv_dict(n_indices=5)
        result = run_index_screener(ohlcv)
        if not result.empty:
            assert "RS Rating" in result.columns
            assert result["RS Rating"].between(1, 99).all()


# ---------------------------------------------------------------------------
# _normalise_nselib_df
# ---------------------------------------------------------------------------

def _make_nselib_raw(n: int = 30, bad_dates: bool = False) -> pd.DataFrame:
    """Simulate the raw DataFrame returned by nselib.capital_market.index_data()."""
    import numpy as np
    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    # NSE timestamp format: "DD-MON-YYYY"
    timestamps = [d.strftime("%d-%b-%Y").upper() for d in dates]
    if bad_dates:
        timestamps = ["INVALID"] * n
    close = 1000.0 * np.cumprod(1 + rng.normal(0.001, 0.005, n))
    return pd.DataFrame({
        "INDEX_NAME":       ["NIFTY IND DIGITAL"] * n,
        "OPEN_INDEX_VAL":   close * rng.uniform(0.995, 1.005, n),
        "HIGH_INDEX_VAL":   close * rng.uniform(1.001, 1.015, n),
        "LOW_INDEX_VAL":    close * rng.uniform(0.985, 0.999, n),
        "CLOSE_INDEX_VAL":  close,
        "TURN_OVER":        rng.uniform(1000, 50000, n),
        "TRADED_QTY":       rng.integers(1_000_000, 5_000_000, n).astype(float),
        "TIMESTAMP":        timestamps,
    })


class TestNormaliseNselibDf:
    def test_returns_dataframe_with_ohlcv(self):
        raw = _make_nselib_raw(n=30)
        df = _normalise_nselib_df(raw)
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        for col in ("Open", "High", "Low", "Close"):
            assert col in df.columns

    def test_index_is_datetimeindex(self):
        raw = _make_nselib_raw(n=30)
        df = _normalise_nselib_df(raw)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_volume_mapped(self):
        raw = _make_nselib_raw(n=30)
        df = _normalise_nselib_df(raw)
        assert "Volume" in df.columns

    def test_sorted_ascending_by_date(self):
        raw = _make_nselib_raw(n=30)
        df = _normalise_nselib_df(raw)
        assert df.index.is_monotonic_increasing

    def test_none_on_empty_input(self):
        assert _normalise_nselib_df(pd.DataFrame()) is None

    def test_none_on_missing_close(self):
        raw = _make_nselib_raw(n=30).drop(columns=["CLOSE_INDEX_VAL"])
        assert _normalise_nselib_df(raw) is None

    def test_none_when_fewer_than_20_rows(self):
        raw = _make_nselib_raw(n=10)
        assert _normalise_nselib_df(raw) is None

    def test_invalid_dates_dropped(self):
        raw = _make_nselib_raw(n=30, bad_dates=True)
        result = _normalise_nselib_df(raw)
        # All dates are invalid → no rows → should return None
        assert result is None

    def test_values_are_numeric(self):
        raw = _make_nselib_raw(n=30)
        df = _normalise_nselib_df(raw)
        for col in ("Open", "High", "Low", "Close"):
            assert pd.api.types.is_numeric_dtype(df[col])

    def test_non_ohlc_columns_dropped(self):
        raw = _make_nselib_raw(n=30)
        df = _normalise_nselib_df(raw)
        assert "INDEX_NAME" not in df.columns
        assert "TURN_OVER"  not in df.columns


# ---------------------------------------------------------------------------
# _index_category
# ---------------------------------------------------------------------------

class TestIndexCategory:
    def test_broad_market_index(self):
        assert _index_category("Nifty 50") == "Broad Market"

    def test_sectoral_index(self):
        assert _index_category("Nifty Bank") == "Sectoral"

    def test_thematic_index_from_nselib(self):
        assert _index_category("Nifty India Digital") == "Thematic"

    def test_unknown_index_returns_other(self):
        assert _index_category("Some Unknown Index") == "Other"

    def test_all_nse_thematic_indices_classified(self):
        from config import NSE_THEMATIC_INDICES
        for name in NSE_THEMATIC_INDICES:
            cat = _index_category(name)
            assert cat == "Thematic", f"{name!r} got category {cat!r}"
