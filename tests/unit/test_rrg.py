"""
Unit tests for src/rrg.py — all run on synthetic data, no network access.

Covers:
  _to_weekly     — daily → weekly resampling
  _quadrant      — four-quadrant classification including boundary conditions
  _compute_series — RS-Ratio and RS-Momentum computation against a benchmark
  compute_rrg    — full computation: shape, correctness, edge cases, directionality
"""

import numpy as np
import pandas as pd
import pytest

from src.rrg import (
    _to_weekly,
    _quadrant,
    _compute_series,
    compute_rrg,
    QUADRANT_COLORS,
    QUADRANT_FILL,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ohlcv(n: int = 260, trend: float = 0.002, seed: int = 0) -> pd.DataFrame:
    """Synthetic daily OHLCV with a DatetimeIndex."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1 + rng.normal(trend, 0.01, n))
    high  = close * (1 + rng.uniform(0.001, 0.015, n))
    low   = close * (1 - rng.uniform(0.001, 0.015, n))
    vol   = rng.integers(100_000, 2_000_000, n).astype(float)
    df = pd.DataFrame({
        "Open": close * 0.999, "High": high,
        "Low": low, "Close": close, "Volume": vol,
    })
    df.index = pd.date_range("2023-01-01", periods=n, freq="B")
    return df


def _close(n: int = 260, trend: float = 0.002, seed: int = 0) -> pd.Series:
    """Synthetic daily Close series."""
    rng = np.random.default_rng(seed)
    return pd.Series(
        100.0 * np.cumprod(1 + rng.normal(trend, 0.01, n)),
        index=pd.date_range("2023-01-01", periods=n, freq="B"),
    )


# ---------------------------------------------------------------------------
# _to_weekly
# ---------------------------------------------------------------------------

class TestToWeekly:

    def test_output_has_fewer_rows_than_daily(self):
        assert len(_to_weekly(_ohlcv(n=260))) < 260

    def test_close_column_always_present(self):
        assert "Close" in _to_weekly(_ohlcv()).columns

    def test_all_ohlcv_columns_preserved(self):
        weekly = _to_weekly(_ohlcv())
        for col in ("Open", "High", "Low", "Close", "Volume"):
            assert col in weekly.columns

    def test_index_is_datetimeindex(self):
        assert isinstance(_to_weekly(_ohlcv()).index, pd.DatetimeIndex)

    def test_no_nan_close_rows(self):
        assert not _to_weekly(_ohlcv())["Close"].isna().any()

    def test_weekly_high_gte_highest_daily_high(self):
        """Weekly High must be ≥ any individual daily High in the same period."""
        df = _ohlcv(n=50)
        weekly = _to_weekly(df)
        assert weekly["High"].max() >= df["High"].max() * 0.999

    def test_weekly_low_lte_lowest_daily_low(self):
        df = _ohlcv(n=50)
        weekly = _to_weekly(df)
        assert weekly["Low"].min() <= df["Low"].min() * 1.001

    def test_close_only_dataframe_works(self):
        """DataFrame with only a Close column must not raise."""
        df = pd.DataFrame(
            {"Close": _close().values},
            index=pd.date_range("2023-01-01", periods=260, freq="B"),
        )
        weekly = _to_weekly(df)
        assert len(weekly) > 0
        assert "Close" in weekly.columns

    def test_idempotent_on_already_weekly_data(self):
        """Calling _to_weekly on already-weekly data must not raise."""
        weekly_once = _to_weekly(_ohlcv(n=260))
        weekly_twice = _to_weekly(weekly_once)
        assert len(weekly_twice) <= len(weekly_once)


# ---------------------------------------------------------------------------
# _quadrant
# ---------------------------------------------------------------------------

class TestQuadrant:

    def test_leading_both_above_100(self):
        assert _quadrant(105.0, 103.0) == "Leading"

    def test_weakening_ratio_above_mom_below(self):
        assert _quadrant(105.0, 97.0) == "Weakening"

    def test_lagging_both_below_100(self):
        assert _quadrant(95.0, 97.0) == "Lagging"

    def test_improving_ratio_below_mom_above(self):
        assert _quadrant(95.0, 105.0) == "Improving"

    def test_boundary_100_100_is_leading(self):
        """Exactly (100, 100) falls in Leading by the ≥ 100 condition."""
        assert _quadrant(100.0, 100.0) == "Leading"

    def test_boundary_ratio_100_mom_below_is_weakening(self):
        assert _quadrant(100.0, 99.9) == "Weakening"

    def test_boundary_ratio_below_mom_100_is_improving(self):
        assert _quadrant(99.9, 100.0) == "Improving"

    def test_all_outputs_are_known_quadrant_strings(self):
        valid = {"Leading", "Weakening", "Lagging", "Improving"}
        for ratio, mom in [(110, 110), (110, 90), (90, 90), (90, 110),
                           (100, 100), (100, 99), (99, 100), (99, 99)]:
            assert _quadrant(ratio, mom) in valid

    def test_extreme_outperformer_is_leading(self):
        assert _quadrant(150.0, 130.0) == "Leading"

    def test_extreme_underperformer_is_lagging(self):
        assert _quadrant(50.0, 60.0) == "Lagging"


# ---------------------------------------------------------------------------
# _compute_series
# ---------------------------------------------------------------------------

class TestComputeSeries:

    def test_returns_two_series(self):
        rsr, rsm = _compute_series(_close(200), _close(200, seed=1), 14, 10)
        assert isinstance(rsr, pd.Series) and isinstance(rsm, pd.Series)

    def test_non_empty_with_sufficient_data(self):
        rsr, rsm = _compute_series(_close(200), _close(200, seed=1), 14, 10)
        assert len(rsr) > 0 and len(rsm) > 0

    def test_empty_with_insufficient_data(self):
        """Fewer bars than m + k + 5 → empty series."""
        rsr, rsm = _compute_series(_close(10), _close(10, seed=1), 14, 10)
        assert rsr.empty and rsm.empty

    def test_series_same_length(self):
        rsr, rsm = _compute_series(_close(260, seed=0), _close(260, seed=1), 14, 10)
        assert len(rsr) == len(rsm)

    def test_no_nan_in_output(self):
        rsr, rsm = _compute_series(_close(260), _close(260, seed=1), 14, 10)
        assert not rsr.isna().any()
        assert not rsm.isna().any()

    def test_identical_series_ratio_converges_to_100(self):
        """When close == benchmark, RS-Ratio must converge to ≈ 100."""
        base = _close(260, seed=42)
        rsr, _ = _compute_series(base.copy(), base.copy(), 14, 10)
        assert abs(rsr.iloc[-1] - 100.0) < 0.5

    def test_outperformer_ratio_above_100(self):
        """Stock growing faster than benchmark → RS-Ratio > 100 after warmup."""
        bench = _close(260, trend=0.001, seed=10)
        fast  = _close(260, trend=0.008, seed=10)
        rsr, _ = _compute_series(fast, bench, 14, 10)
        assert rsr.iloc[-1] > 100.0

    def test_underperformer_ratio_below_100(self):
        bench = _close(260, trend=0.003, seed=10)
        slow  = _close(260, trend=-0.002, seed=10)
        rsr, _ = _compute_series(slow, bench, 14, 10)
        assert rsr.iloc[-1] < 100.0

    def test_handles_partial_date_overlap(self):
        """Partially overlapping date ranges must use the intersection."""
        full  = _close(260, seed=0)
        short = _close(150, seed=1)
        short.index = full.index[-150:]  # align at tail
        rsr, rsm = _compute_series(full, short, 14, 10)
        assert len(rsr) > 0

    def test_momentum_positive_when_ratio_accelerating(self):
        """A strongly outperforming stock should have RS-Momentum > 100."""
        bench = _close(300, trend=0.001, seed=99)
        fast  = _close(300, trend=0.009, seed=99)
        _, rsm = _compute_series(fast, bench, 14, 10)
        assert rsm.iloc[-1] > 100.0

    def test_custom_m_k_parameters_accepted(self):
        rsr, rsm = _compute_series(_close(300), _close(300, seed=1), m=20, k=5)
        assert len(rsr) > 0 and len(rsm) > 0


# ---------------------------------------------------------------------------
# compute_rrg — output structure
# ---------------------------------------------------------------------------

class TestComputeRRGShape:

    def _universe(self, n=5, bars=260):
        pd_ = {f"S{i}": _ohlcv(n=bars, trend=0.002 + i * 0.001, seed=i) for i in range(n)}
        bench = _ohlcv(n=bars, trend=0.001, seed=99)
        return pd_, bench

    def test_returns_dataframe(self):
        pd_, bench = self._universe()
        assert isinstance(compute_rrg(pd_, bench), pd.DataFrame)

    def test_has_all_required_columns(self):
        pd_, bench = self._universe()
        result = compute_rrg(pd_, bench)
        for col in ("name", "rs_ratio", "rs_momentum", "quadrant", "color", "tail_x", "tail_y"):
            assert col in result.columns, f"Missing column: {col}"

    def test_one_row_per_security(self):
        pd_, bench = self._universe(n=6)
        assert len(compute_rrg(pd_, bench)) == 6

    def test_tail_lists_have_requested_length(self):
        pd_, bench = self._universe()
        for t in (4, 8, 12):
            result = compute_rrg(pd_, bench, tail=t)
            for _, row in result.iterrows():
                assert len(row["tail_x"]) == t
                assert len(row["tail_y"]) == t

    def test_current_position_equals_last_tail_element(self):
        pd_, bench = self._universe(n=1)
        row = compute_rrg(pd_, bench).iloc[0]
        assert row["tail_x"][-1] == pytest.approx(row["rs_ratio"],    abs=1e-6)
        assert row["tail_y"][-1] == pytest.approx(row["rs_momentum"], abs=1e-6)

    def test_no_nan_in_ratio_or_momentum(self):
        pd_, bench = self._universe()
        result = compute_rrg(pd_, bench)
        assert not result["rs_ratio"].isna().any()
        assert not result["rs_momentum"].isna().any()

    def test_tail_x_and_tail_y_same_length_per_row(self):
        pd_, bench = self._universe()
        result = compute_rrg(pd_, bench, tail=8)
        for _, row in result.iterrows():
            assert len(row["tail_x"]) == len(row["tail_y"])


# ---------------------------------------------------------------------------
# compute_rrg — correctness
# ---------------------------------------------------------------------------

class TestComputeRRGCorrectness:

    def test_quadrant_consistent_with_ratio_and_momentum(self):
        pd_ = {f"S{i}": _ohlcv(n=260, seed=i) for i in range(8)}
        bench = _ohlcv(n=260, seed=99)
        result = compute_rrg(pd_, bench)
        for _, row in result.iterrows():
            expected = _quadrant(row["rs_ratio"], row["rs_momentum"])
            assert row["quadrant"] == expected, (
                f"{row['name']}: rs_ratio={row['rs_ratio']:.2f}, "
                f"rs_momentum={row['rs_momentum']:.2f} → expected {expected}, "
                f"got {row['quadrant']}"
            )

    def test_color_matches_quadrant(self):
        pd_ = {f"S{i}": _ohlcv(n=260, seed=i) for i in range(8)}
        bench = _ohlcv(n=260, seed=99)
        result = compute_rrg(pd_, bench)
        for _, row in result.iterrows():
            assert row["color"] == QUADRANT_COLORS[row["quadrant"]]

    def test_strong_outperformer_leading_or_improving(self):
        bench = _ohlcv(n=260, trend=0.001, seed=5)
        strong = {"Star": _ohlcv(n=260, trend=0.012, seed=5)}
        result = compute_rrg(strong, bench)
        assert result.iloc[0]["quadrant"] in ("Leading", "Improving")

    def test_strong_underperformer_lagging_or_weakening(self):
        bench = _ohlcv(n=260, trend=0.003, seed=5)
        weak = {"Dog": _ohlcv(n=260, trend=-0.006, seed=5)}
        result = compute_rrg(weak, bench)
        assert result.iloc[0]["quadrant"] in ("Lagging", "Weakening")

    def test_outperformer_ratio_above_100(self):
        bench = _ohlcv(n=260, trend=0.001, seed=7)
        fast  = {"Fast": _ohlcv(n=260, trend=0.010, seed=7)}
        result = compute_rrg(fast, bench)
        assert result.iloc[0]["rs_ratio"] > 100.0

    def test_underperformer_ratio_below_100(self):
        bench = _ohlcv(n=260, trend=0.004, seed=7)
        slow  = {"Slow": _ohlcv(n=260, trend=-0.004, seed=7)}
        result = compute_rrg(slow, bench)
        assert result.iloc[0]["rs_ratio"] < 100.0


# ---------------------------------------------------------------------------
# compute_rrg — edge cases
# ---------------------------------------------------------------------------

class TestComputeRRGEdgeCases:

    def test_empty_dict_returns_empty_df(self):
        result = compute_rrg({}, _ohlcv())
        assert result.empty

    def test_none_benchmark_returns_empty_df(self):
        result = compute_rrg({"A": _ohlcv()}, None)
        assert result.empty

    def test_empty_benchmark_returns_empty_df(self):
        result = compute_rrg({"A": _ohlcv()}, pd.DataFrame())
        assert result.empty

    def test_too_short_security_excluded(self):
        """Securities with fewer bars than m+k+5 must be silently skipped."""
        bench = _ohlcv(n=260, seed=99)
        pd_ = {"Short": _ohlcv(n=10), "Long": _ohlcv(n=260)}
        result = compute_rrg(pd_, bench)
        assert "Long"  in result["name"].values
        assert "Short" not in result["name"].values

    def test_single_security_returns_one_row(self):
        result = compute_rrg({"Solo": _ohlcv(n=260)}, _ohlcv(n=260, seed=99))
        assert len(result) == 1

    def test_benchmark_with_nan_does_not_raise(self):
        bench = _ohlcv(n=260, seed=99)
        bench.loc[bench.index[5:15], "Close"] = np.nan
        result = compute_rrg({"A": _ohlcv(n=260)}, bench)
        assert isinstance(result, pd.DataFrame)

    def test_security_with_nan_close_excluded_not_crash(self):
        bench = _ohlcv(n=260, seed=99)
        bad = _ohlcv(n=260)
        bad["Close"] = np.nan
        good = _ohlcv(n=260, seed=1)
        result = compute_rrg({"Bad": bad, "Good": good}, bench)
        assert "Good" in result["name"].values
        assert "Bad"  not in result["name"].values

    def test_all_securities_insufficient_returns_empty_df(self):
        bench = _ohlcv(n=260, seed=99)
        pd_ = {"A": _ohlcv(n=5), "B": _ohlcv(n=8)}
        result = compute_rrg(pd_, bench)
        assert result.empty


# ---------------------------------------------------------------------------
# compute_rrg — weekly resampling
# ---------------------------------------------------------------------------

class TestComputeRRGWeekly:

    def test_weekly_mode_returns_non_empty(self):
        pd_ = {f"S{i}": _ohlcv(n=300, seed=i) for i in range(5)}
        bench = _ohlcv(n=300, seed=99)
        assert not compute_rrg(pd_, bench, weekly=True).empty

    def test_weekly_and_daily_cover_same_securities(self):
        pd_ = {f"S{i}": _ohlcv(n=300, seed=i) for i in range(4)}
        bench = _ohlcv(n=300, seed=99)
        daily  = set(compute_rrg(pd_, bench, weekly=False)["name"])
        weekly = set(compute_rrg(pd_, bench, weekly=True)["name"])
        assert daily == weekly

    def test_weekly_values_differ_from_daily(self):
        """Daily and weekly RRG values should not be identical."""
        pd_ = {"A": _ohlcv(n=300, seed=0)}
        bench = _ohlcv(n=300, seed=99)
        d_ratio = compute_rrg(pd_, bench, weekly=False).iloc[0]["rs_ratio"]
        w_ratio = compute_rrg(pd_, bench, weekly=True).iloc[0]["rs_ratio"]
        assert abs(d_ratio - w_ratio) > 0.01

    def test_weekly_tail_length_respected(self):
        pd_ = {"A": _ohlcv(n=300, seed=0)}
        bench = _ohlcv(n=300, seed=99)
        for tail in (4, 8, 10):
            result = compute_rrg(pd_, bench, tail=tail, weekly=True)
            assert len(result.iloc[0]["tail_x"]) == tail


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------

class TestConstants:

    def test_all_four_quadrants_have_color(self):
        for q in ("Leading", "Weakening", "Lagging", "Improving"):
            assert q in QUADRANT_COLORS

    def test_all_four_quadrants_have_fill(self):
        for q in ("Leading", "Weakening", "Lagging", "Improving"):
            assert q in QUADRANT_FILL

    def test_colors_are_hex_strings(self):
        for q, color in QUADRANT_COLORS.items():
            assert color.startswith("#"), f"{q}: {color!r} is not a hex color"
            assert len(color) == 7, f"{q}: {color!r} is not a 6-digit hex"

    def test_fills_are_rgba_strings(self):
        for q, fill in QUADRANT_FILL.items():
            assert fill.startswith("rgba("), f"{q}: {fill!r} is not rgba"
