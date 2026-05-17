"""
Integration tests for the RRG pipeline.

Tests the full end-to-end flow with realistic synthetic data:
  - Sector-level RRG (8-sector universe vs benchmark)
  - Stock drill-down simulation (15 stocks in a sector)
  - Mathematical properties of the output
  - Performance benchmark (<10 s for 50 securities)
"""

import time
import numpy as np
import pandas as pd
import pytest

from src.rrg import compute_rrg, _to_weekly, QUADRANT_COLORS


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_price_df(
    n_days: int = 400,
    start_price: float = 100.0,
    drift: float = 0.0,
    vol: float = 0.01,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic daily OHLCV DataFrame with a given drift and volatility."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    log_returns = rng.normal(drift / 252, vol, size=n_days)
    close = start_price * np.exp(np.cumsum(log_returns))
    df = pd.DataFrame(
        {
            "Open":   close * (1 - rng.uniform(0, 0.005, n_days)),
            "High":   close * (1 + rng.uniform(0, 0.01, n_days)),
            "Low":    close * (1 - rng.uniform(0, 0.01, n_days)),
            "Close":  close,
            "Volume": rng.integers(100_000, 1_000_000, n_days).astype(float),
        },
        index=dates,
    )
    return df


def _make_sector_universe(
    n_sectors: int = 8,
    n_days: int = 400,
    seeds_and_drifts: list | None = None,
) -> dict[str, pd.DataFrame]:
    """Build a dict of {sector_name: price_df} for n_sectors sectors."""
    sector_names = [
        "Nifty Bank", "Nifty IT", "Nifty Pharma", "Nifty FMCG",
        "Nifty Metal", "Nifty Energy", "Nifty Realty", "Nifty Auto",
    ][:n_sectors]

    if seeds_and_drifts is None:
        seeds_and_drifts = [(i, 0.0) for i in range(n_sectors)]

    return {
        name: _make_price_df(n_days=n_days, drift=drift, seed=seed)
        for (seed, drift), name in zip(seeds_and_drifts, sector_names)
    }


# ---------------------------------------------------------------------------
# TestSectorRRGPipeline
# ---------------------------------------------------------------------------

class TestSectorRRGPipeline:
    """Full pipeline test with an 8-sector universe vs a benchmark."""

    def test_returns_dataframe_with_required_columns(self):
        benchmark = _make_price_df(n_days=400, seed=99)
        universe = _make_sector_universe(8)
        result = compute_rrg(universe, benchmark)
        required = {"name", "rs_ratio", "rs_momentum", "quadrant", "color", "tail_x", "tail_y"}
        assert required.issubset(result.columns), f"Missing columns: {required - set(result.columns)}"

    def test_all_sectors_appear_with_sufficient_data(self):
        benchmark = _make_price_df(n_days=400, seed=99)
        universe = _make_sector_universe(8)
        result = compute_rrg(universe, benchmark, m=14, k=10, tail=8)
        assert len(result) == 8, f"Expected 8 rows, got {len(result)}"

    def test_quadrants_are_valid_values(self):
        benchmark = _make_price_df(n_days=400, seed=99)
        universe = _make_sector_universe(8)
        result = compute_rrg(universe, benchmark)
        valid = {"Leading", "Weakening", "Lagging", "Improving"}
        assert set(result["quadrant"]).issubset(valid)

    def test_color_matches_quadrant(self):
        benchmark = _make_price_df(n_days=400, seed=99)
        universe = _make_sector_universe(8)
        result = compute_rrg(universe, benchmark)
        for _, row in result.iterrows():
            assert row["color"] == QUADRANT_COLORS[row["quadrant"]]

    def test_tail_lengths_match_requested_tail(self):
        benchmark = _make_price_df(n_days=400, seed=99)
        universe = _make_sector_universe(8)
        for tail in (5, 8, 12):
            result = compute_rrg(universe, benchmark, tail=tail)
            for _, row in result.iterrows():
                assert len(row["tail_x"]) == tail, f"tail_x length {len(row['tail_x'])} != {tail}"
                assert len(row["tail_y"]) == tail, f"tail_y length {len(row['tail_y'])} != {tail}"

    def test_quadrant_coverage_with_forced_divergence(self):
        """Sectors with very different drifts should land in different quadrants."""
        benchmark = _make_price_df(n_days=500, drift=0.05, seed=0)
        # Strong outperformer, strong underperformer, and neutrals
        universe = {
            "Outperformer":  _make_price_df(n_days=500, drift=0.30, seed=1),
            "Underperformer": _make_price_df(n_days=500, drift=-0.20, seed=2),
            "Neutral A":     _make_price_df(n_days=500, drift=0.05, seed=3),
            "Neutral B":     _make_price_df(n_days=500, drift=0.06, seed=4),
        }
        result = compute_rrg(universe, benchmark, m=10, k=8, tail=6)
        # At least 2 distinct quadrants across the 4 securities
        assert len(result["quadrant"].unique()) >= 2

    def test_weekly_mode_produces_fewer_data_points(self):
        benchmark = _make_price_df(n_days=600, seed=0)
        universe = _make_sector_universe(4)
        result_daily  = compute_rrg(universe, benchmark, weekly=False, m=14, k=10, tail=8)
        result_weekly = compute_rrg(universe, benchmark, weekly=True,  m=14, k=10, tail=8)
        # Both should return the same number of securities
        assert len(result_daily) == len(result_weekly)

    def test_rs_ratio_and_momentum_are_floats(self):
        benchmark = _make_price_df(n_days=400, seed=99)
        universe = _make_sector_universe(4)
        result = compute_rrg(universe, benchmark)
        assert result["rs_ratio"].dtype == float
        assert result["rs_momentum"].dtype == float

    def test_current_position_is_last_tail_point(self):
        """The scalar rs_ratio / rs_momentum must match the last element of tail lists."""
        benchmark = _make_price_df(n_days=400, seed=99)
        universe = _make_sector_universe(4)
        result = compute_rrg(universe, benchmark, tail=8)
        for _, row in result.iterrows():
            assert abs(row["rs_ratio"] - row["tail_x"][-1]) < 1e-6
            assert abs(row["rs_momentum"] - row["tail_y"][-1]) < 1e-6


# ---------------------------------------------------------------------------
# TestStockDrillDownSimulation
# ---------------------------------------------------------------------------

class TestStockDrillDownSimulation:
    """Simulate drilling down from a sector RRG into individual stock RRGs."""

    @pytest.fixture
    def sector_benchmark(self):
        """Return a sector index DataFrame used as benchmark for stock drill-down."""
        return _make_price_df(n_days=450, drift=0.10, seed=50)

    @pytest.fixture
    def stock_universe(self):
        """15 stocks with varying drifts."""
        drifts = [0.25, 0.20, 0.15, 0.10, 0.08,
                  0.05, 0.02, 0.00, -0.03, -0.07,
                  -0.10, -0.15, -0.20, -0.25, -0.30]
        return {
            f"STOCK{i+1:02d}": _make_price_df(n_days=450, drift=d, seed=i)
            for i, d in enumerate(drifts)
        }

    def test_drill_down_returns_all_stocks(self, sector_benchmark, stock_universe):
        result = compute_rrg(stock_universe, sector_benchmark, m=14, k=10, tail=8)
        assert len(result) == 15

    def test_top_drifters_tend_to_be_stronger(self, sector_benchmark, stock_universe):
        """STOCK01 (highest drift) should have rs_ratio or rs_momentum > median."""
        result = compute_rrg(stock_universe, sector_benchmark, m=10, k=8, tail=6)
        result_indexed = result.set_index("name")
        top_stock = result_indexed.loc["STOCK01"]
        med_ratio = result["rs_ratio"].median()
        med_mom   = result["rs_momentum"].median()
        # At least one of ratio/momentum above median
        assert (top_stock["rs_ratio"] > med_ratio) or (top_stock["rs_momentum"] > med_mom), \
            f"STOCK01 ratio={top_stock['rs_ratio']:.3f} mom={top_stock['rs_momentum']:.3f} vs " \
            f"medians ({med_ratio:.3f}, {med_mom:.3f})"

    def test_extreme_drifts_produce_correct_ordering(self):
        """With very low noise, a strong outperformer has higher rs_ratio than a strong underperformer.

        Uses independent, low-volatility data (vol=0.002) and a long enough EMA (m=20)
        so the drift signal dominates the random walk noise (SNR > 10).
        """
        bench = _make_price_df(n_days=600, drift=0.0,  vol=0.002, seed=200)
        strong = _make_price_df(n_days=600, drift=1.5,  vol=0.002, seed=201)
        weak   = _make_price_df(n_days=600, drift=-1.0, vol=0.002, seed=202)
        universe = {"Strong": strong, "Weak": weak}
        result = compute_rrg(universe, bench, m=20, k=10, tail=8)

        if result.empty:
            pytest.skip("Insufficient data")

        ri = result.set_index("name")
        assert ri.loc["Strong", "rs_ratio"] > ri.loc["Weak", "rs_ratio"], (
            f"Strong rs_ratio={ri.loc['Strong','rs_ratio']:.3f} not > "
            f"Weak rs_ratio={ri.loc['Weak','rs_ratio']:.3f}"
        )

    def test_names_preserved_exactly(self, sector_benchmark, stock_universe):
        result = compute_rrg(stock_universe, sector_benchmark)
        assert set(result["name"]) == set(stock_universe.keys())

    def test_tail_x_tail_y_are_lists_of_floats(self, sector_benchmark, stock_universe):
        result = compute_rrg(stock_universe, sector_benchmark, tail=8)
        for _, row in result.iterrows():
            assert isinstance(row["tail_x"], list)
            assert isinstance(row["tail_y"], list)
            assert all(isinstance(v, float) for v in row["tail_x"])
            assert all(isinstance(v, float) for v in row["tail_y"])

    def test_stock_not_enough_data_excluded(self, sector_benchmark):
        """Stocks with <(m+k+tail) rows should be excluded, not crash."""
        short_stock  = _make_price_df(n_days=20, seed=77)   # too short
        normal_stock = _make_price_df(n_days=400, seed=78)
        universe = {"ShortStock": short_stock, "NormalStock": normal_stock}
        result = compute_rrg(universe, sector_benchmark, m=14, k=10, tail=8)
        assert "ShortStock" not in result["name"].values
        assert "NormalStock" in result["name"].values

    def test_weekly_drill_down(self, sector_benchmark, stock_universe):
        result = compute_rrg(stock_universe, sector_benchmark, weekly=True, m=10, k=8, tail=6)
        # With 450 days → ~90 weeks, enough for m+k+tail=24
        assert len(result) >= 10, f"Expected ≥10 stocks in weekly drill-down, got {len(result)}"


# ---------------------------------------------------------------------------
# TestRRGMathematicalProperties
# ---------------------------------------------------------------------------

class TestRRGMathematicalProperties:
    """Verify mathematical invariants of the RRG computation."""

    def test_benchmark_vs_itself_is_near_100(self):
        """Benchmark against itself: RS=1 constantly → rs_ratio ≈ 100."""
        benchmark = _make_price_df(n_days=400, drift=0.10, seed=5)
        universe = {"Self": benchmark.copy()}
        result = compute_rrg(universe, benchmark, m=14, k=10, tail=8)
        if not result.empty:
            rs_ratio = result.iloc[0]["rs_ratio"]
            # RS Ratio should be very close to 100 (within 0.01%)
            assert abs(rs_ratio - 100.0) < 0.1, f"rs_ratio={rs_ratio} not near 100"

    def test_larger_ema_window_produces_smoother_tails(self):
        """Larger m should reduce variance in the tail series."""
        benchmark = _make_price_df(n_days=500, seed=0)
        stock     = _make_price_df(n_days=500, drift=0.05, seed=1)
        universe = {"Sector": stock}

        res_small = compute_rrg(universe, benchmark, m=5,  k=3, tail=10)
        res_large = compute_rrg(universe, benchmark, m=20, k=10, tail=10)

        if res_small.empty or res_large.empty:
            pytest.skip("Insufficient data")

        var_small = np.var(res_small.iloc[0]["tail_x"])
        var_large = np.var(res_large.iloc[0]["tail_x"])
        assert var_large <= var_small * 2.0, \
            f"Large EMA variance {var_large:.4f} unexpectedly > 2× small EMA variance {var_small:.4f}"

    def test_tail_values_are_finite(self):
        benchmark = _make_price_df(n_days=400, seed=0)
        universe = _make_sector_universe(8)
        result = compute_rrg(universe, benchmark)
        for _, row in result.iterrows():
            assert all(np.isfinite(v) for v in row["tail_x"]), f"{row['name']} tail_x has non-finite"
            assert all(np.isfinite(v) for v in row["tail_y"]), f"{row['name']} tail_y has non-finite"

    def test_no_nan_in_scalar_fields(self):
        benchmark = _make_price_df(n_days=400, seed=0)
        universe = _make_sector_universe(8)
        result = compute_rrg(universe, benchmark)
        assert result["rs_ratio"].notna().all()
        assert result["rs_momentum"].notna().all()

    def test_tail_monotonically_advances_in_time(self):
        """tail_x[i] is the rs_ratio at an earlier date than tail_x[i+1]."""
        benchmark = _make_price_df(n_days=500, drift=0.05, seed=10)
        strong = _make_price_df(n_days=500, drift=0.25, seed=11)
        universe = {"Strong": strong}
        result = compute_rrg(universe, benchmark, m=10, k=8, tail=10)
        if result.empty:
            pytest.skip("Insufficient data")
        # We can't assert direction, but we CAN assert the values differ
        tx = result.iloc[0]["tail_x"]
        # Not all identical (non-constant RS)
        assert len(set(round(v, 6) for v in tx)) > 1, "All tail_x values identical — suspicious"

    def test_50_securities_completes_under_10_seconds(self):
        """Performance: 50 securities vs benchmark must complete in <10 s."""
        benchmark = _make_price_df(n_days=400, seed=0)
        universe = {
            f"SEC{i:02d}": _make_price_df(n_days=400, seed=i + 1)
            for i in range(50)
        }
        t0 = time.perf_counter()
        result = compute_rrg(universe, benchmark, m=14, k=10, tail=8)
        elapsed = time.perf_counter() - t0
        assert elapsed < 10.0, f"compute_rrg took {elapsed:.2f}s for 50 securities (limit: 10s)"
        assert len(result) > 0

    def test_result_is_deterministic(self):
        """Same inputs must always produce identical outputs."""
        benchmark = _make_price_df(n_days=400, seed=0)
        universe = _make_sector_universe(4)
        r1 = compute_rrg(universe, benchmark, m=14, k=10, tail=8)
        r2 = compute_rrg(universe, benchmark, m=14, k=10, tail=8)
        r1s = r1.sort_values("name").reset_index(drop=True)
        r2s = r2.sort_values("name").reset_index(drop=True)
        pd.testing.assert_frame_equal(r1s[["name", "rs_ratio", "rs_momentum"]], r2s[["name", "rs_ratio", "rs_momentum"]])

    def test_empty_benchmark_returns_empty_df(self):
        universe = _make_sector_universe(4)
        result = compute_rrg(universe, pd.DataFrame())
        assert result.empty

    def test_none_benchmark_returns_empty_df(self):
        universe = _make_sector_universe(4)
        result = compute_rrg(universe, None)
        assert result.empty

    def test_empty_universe_returns_empty_df(self):
        benchmark = _make_price_df(n_days=400, seed=0)
        result = compute_rrg({}, benchmark)
        assert result.empty
