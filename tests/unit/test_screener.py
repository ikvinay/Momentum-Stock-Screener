"""
Unit tests for src/screener.py — ranking helpers and filter logic.
No network calls; all data is synthetic.
"""

import numpy as np
import pandas as pd
import pytest

from src.screener import _normalize_rs_ratings


def _make_ohlcv(n: int = 260, trend: float = 0.001, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1 + rng.normal(trend, 0.01, n))
    high  = close * (1 + rng.uniform(0.001, 0.02, n))
    low   = close * (1 - rng.uniform(0.001, 0.02, n))
    vol   = rng.integers(100_000, 2_000_000, n).astype(float)
    df = pd.DataFrame({
        "Open":   close * 0.999,
        "High":   high,
        "Low":    low,
        "Close":  close,
        "Volume": vol,
    })
    df.index = pd.date_range("2023-01-01", periods=n, freq="B")
    return df


# ---------------------------------------------------------------------------
# RS Rating normalisation
# ---------------------------------------------------------------------------

class TestNormalizeRSRatings:
    def test_ratings_in_1_99_range(self):
        raw = {"A": 1.5, "B": 2.0, "C": 0.5, "D": 3.0, "E": 1.0}
        ratings = _normalize_rs_ratings(raw)
        for v in ratings.values():
            assert 1 <= v <= 99

    def test_best_score_gets_highest_rating(self):
        raw = {"A": 10.0, "B": 5.0, "C": 1.0}
        ratings = _normalize_rs_ratings(raw)
        assert ratings["A"] > ratings["B"] > ratings["C"]

    def test_single_ticker_gets_50(self):
        raw = {"ONLY": 42.0}
        ratings = _normalize_rs_ratings(raw)
        assert ratings["ONLY"] == 50

    def test_empty_input_returns_empty(self):
        assert _normalize_rs_ratings({}) == {}

    def test_equal_scores_get_equal_ratings(self):
        raw = {"A": 5.0, "B": 5.0, "C": 5.0}
        ratings = _normalize_rs_ratings(raw)
        # All equal → all map to same rating
        assert len(set(ratings.values())) == 1

    def test_output_count_matches_input(self):
        raw = {f"T{i}": float(i) for i in range(20)}
        ratings = _normalize_rs_ratings(raw)
        assert len(ratings) == 20


# ---------------------------------------------------------------------------
# run_screener output schema (synthetic fast test)
# ---------------------------------------------------------------------------

class TestRunScreenerSchema:
    """
    Validates that run_screener() returns a well-formed DataFrame when given
    a small synthetic dataset.  Uses a deliberately weak stock universe so
    at least some stocks pass the hard filters with default config.
    """

    def _make_universe(self, n_stocks: int = 5, n_days: int = 260):
        price_data = {}
        stock_info = {}
        for i in range(n_stocks):
            seed  = i * 7
            df    = _make_ohlcv(n=n_days, trend=0.003, seed=seed)
            ticker = f"STOCK{i:02d}.NS"
            price_data[ticker] = df
            stock_info[ticker] = {
                "sector":    "Technology",
                "industry":  "Software",
                "pe":        20.0 + i,
                "market_cap": 50_000 * 1_00_00_000,  # 50,000 Cr
                "company":   f"Company {i}",
            }
        return price_data, stock_info

    def test_returns_dataframe(self):
        from src.screener import run_screener
        price_data, stock_info = self._make_universe()
        result = run_screener(price_data, stock_info, {}, None)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns_present(self):
        from src.screener import run_screener
        price_data, stock_info = self._make_universe()
        result = run_screener(price_data, stock_info, {}, None)
        if not result.empty:
            for col in ("Symbol", "CMP", "Score", "Sector"):
                assert col in result.columns, f"Missing column: {col}"

    def test_scores_in_valid_range(self):
        from src.screener import run_screener
        price_data, stock_info = self._make_universe()
        result = run_screener(price_data, stock_info, {}, None)
        if not result.empty:
            assert result["Score"].between(0, 100).all()

    def test_sorted_descending_by_score(self):
        from src.screener import run_screener
        price_data, stock_info = self._make_universe()
        result = run_screener(price_data, stock_info, {}, None)
        if len(result) > 1:
            scores = result["Score"].tolist()
            assert scores == sorted(scores, reverse=True)
