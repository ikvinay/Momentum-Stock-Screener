"""
Unit tests for detect_symmetrical_triangle() in src/indicators.py.

All tests use synthetic OHLCV data — no network access required.

Coverage:
  - Valid symmetrical triangle (should detect)
  - Expanding triangle / megaphone (diverging — should NOT detect)
  - Insufficient data (fewer bars than minimum required)
  - Price outside the triangle at current bar (should NOT detect)
  - Triangle at apex — too close to convergence (should NOT detect)
  - Increasing (non-contracting) volume — should NOT detect
  - Highly asymmetric slopes (wedge-like) — should NOT detect
  - Return dict structure is always complete
  - Configurable parameter behaviour
"""

import numpy as np
import pandas as pd
import pytest

from src.indicators import detect_symmetrical_triangle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ohlcv(n: int, close: np.ndarray | None = None, seed: int = 0) -> pd.DataFrame:
    """Minimal OHLCV DataFrame with a business-day DatetimeIndex."""
    rng = np.random.default_rng(seed)
    if close is None:
        close = 100.0 * np.cumprod(1 + rng.normal(0.0, 0.005, n))
    high  = close * (1 + rng.uniform(0.002, 0.01, n))
    low   = close * (1 - rng.uniform(0.002, 0.01, n))
    vol   = rng.integers(500_000, 2_000_000, n).astype(float)
    df = pd.DataFrame({
        "Open":   close * 0.999,
        "High":   high,
        "Low":    low,
        "Close":  close,
        "Volume": vol,
    })
    df.index = pd.date_range("2023-01-01", periods=n, freq="B")
    return df


def _sym_triangle_ohlcv(
    n: int = 120,
    mid: float = 100.0,
    width_start: float = 12.0,
    width_end: float = 2.0,
    n_cycles: int = 5,
    noise_amp: float = 0.05,
) -> pd.DataFrame:
    """
    Symmetrical triangle via alternating zigzag touches of the upper and lower
    trendlines — the way price actually behaves inside a real triangle formation.

    Each cycle: price travels from lower trendline → upper trendline → lower trendline.
    This guarantees clear swing highs at the upper touchpoints and swing lows at the
    lower touchpoints, which is what `detect_symmetrical_triangle` looks for.

    `width_start` / `width_end` are the FULL width (high-to-low span) at bar 0 / bar n-1.
    Defaults produce a 12 → 2 % width contraction, starting width >> SYMTRI_MIN_START_WIDTH_PCT.
    """
    rng = np.random.default_rng(42)
    half_w    = np.linspace(width_start / 2, width_end / 2, n)
    upper_tl  = mid + half_w
    lower_tl  = mid - half_w

    bars_per_half = max(2, n // (2 * n_cycles))
    close = np.empty(n)
    going_up, seg_start = True, 0
    while seg_start < n:
        seg_end = min(seg_start + bars_per_half, n)
        seg_len = seg_end - seg_start
        for j in range(seg_len):
            i   = seg_start + j
            frc = j / max(seg_len - 1, 1)
            close[i] = (lower_tl[i] + frc * (upper_tl[i] - lower_tl[i])
                        if going_up
                        else upper_tl[i] + frc * (lower_tl[i] - upper_tl[i]))
        going_up  = not going_up
        seg_start = seg_end

    close += rng.uniform(-noise_amp, noise_amp, n)
    # High reaches slightly above close toward the upper trendline at touchpoints
    high = np.minimum(close + half_w * 0.3 + rng.uniform(0, 0.02, n), upper_tl + 0.1)
    low  = np.maximum(close - half_w * 0.3 - rng.uniform(0, 0.02, n), lower_tl - 0.1)
    # Volume contracts linearly into the apex (required by SYMTRI_REQUIRE_VOL_CONTRACTION)
    vol_trend = np.linspace(1_800_000, 300_000, n)
    vol_noise = rng.integers(-150_000, 150_000, n).astype(float)
    vol = np.maximum(50_000, vol_trend + vol_noise).astype(float)

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
# Return structure
# ---------------------------------------------------------------------------

class TestReturnStructure:

    def test_always_returns_dict(self):
        df = _ohlcv(10)
        result = detect_symmetrical_triangle(df)
        assert isinstance(result, dict)

    def test_required_keys_present_on_miss(self):
        df = _ohlcv(10)
        result = detect_symmetrical_triangle(df)
        required = {
            "detected", "upper_slope", "upper_intercept",
            "lower_slope", "lower_intercept",
            "upper_pivots", "lower_pivots", "n_slice",
        }
        assert required.issubset(result.keys())

    def test_required_keys_present_on_hit(self):
        df = _sym_triangle_ohlcv(n=80)
        result = detect_symmetrical_triangle(df)
        required = {
            "detected", "upper_slope", "upper_intercept",
            "lower_slope", "lower_intercept",
            "upper_pivots", "lower_pivots", "n_slice",
        }
        assert required.issubset(result.keys())

    def test_not_detected_when_insufficient_data(self):
        df = _ohlcv(8)
        result = detect_symmetrical_triangle(df, swing_window=3, min_touches=2)
        assert result["detected"] is False
        assert result["n_slice"] == 0
        assert result["upper_pivots"] == []
        assert result["lower_pivots"] == []


# ---------------------------------------------------------------------------
# Valid symmetrical triangle — should detect
# ---------------------------------------------------------------------------

class TestValidSymmetricalTriangle:

    def test_detects_clean_pattern_default_params(self):
        """Default params (min_touches=3, lookback=60) detect a genuine clean pattern.
        n_cycles=8 → 7 bars per half-cycle → ≥4 upper and ≥4 lower touchpoints visible
        in the 60-bar window, which satisfies the min_touches=3 requirement."""
        df = _sym_triangle_ohlcv(n=120, n_cycles=8)
        result = detect_symmetrical_triangle(df)   # default lookback=60, min_touches=3
        assert result["detected"] is True

    def test_detects_clean_pattern_explicit_two_touches(self):
        df = _sym_triangle_ohlcv(n=120, n_cycles=5)
        result = detect_symmetrical_triangle(df, lookback=80, min_touches=2, swing_window=3)
        assert result["detected"] is True

    def test_upper_slope_is_negative(self):
        df = _sym_triangle_ohlcv(n=120, n_cycles=8)
        result = detect_symmetrical_triangle(df)
        if result["detected"]:
            assert result["upper_slope"] < 0

    def test_lower_slope_is_positive(self):
        df = _sym_triangle_ohlcv(n=120, n_cycles=8)
        result = detect_symmetrical_triangle(df)
        if result["detected"]:
            assert result["lower_slope"] > 0

    def test_pivot_lists_have_min_touches_entries(self):
        df = _sym_triangle_ohlcv(n=120, n_cycles=8)
        result = detect_symmetrical_triangle(df, min_touches=3)
        if result["detected"]:
            assert len(result["upper_pivots"]) >= 3
            assert len(result["lower_pivots"]) >= 3

    def test_n_slice_within_lookback(self):
        df = _sym_triangle_ohlcv(n=120, n_cycles=8)
        result = detect_symmetrical_triangle(df, lookback=60)
        assert result["n_slice"] <= 60


# ---------------------------------------------------------------------------
# Patterns that must NOT be detected
# ---------------------------------------------------------------------------

class TestNonPatterns:

    def test_insufficient_bars(self):
        df = _ohlcv(15)
        result = detect_symmetrical_triangle(df, lookback=60, min_touches=2, swing_window=3)
        assert result["detected"] is False

    def test_uptrend_no_triangle(self):
        """Steady uptrend — no converging trendlines."""
        n = 100
        close = 100.0 * np.cumprod(1 + np.full(n, 0.003))
        df = _ohlcv(n, close=close)
        result = detect_symmetrical_triangle(df, lookback=60, min_touches=2, swing_window=3)
        assert result["detected"] is False

    def test_downtrend_no_triangle(self):
        """Steady downtrend — no converging trendlines."""
        n = 100
        close = 100.0 * np.cumprod(1 + np.full(n, -0.003))
        df = _ohlcv(n, close=close)
        result = detect_symmetrical_triangle(df, lookback=60, min_touches=2, swing_window=3)
        assert result["detected"] is False

    def test_expanding_pattern_not_detected(self):
        """Megaphone / expanding triangle — trendlines diverge."""
        n = 100
        rng = np.random.default_rng(7)
        half_w = np.linspace(3.0, 12.0, n)
        mid = 100.0
        high  = mid + half_w + rng.uniform(0, 0.2, n)
        low   = mid - half_w - rng.uniform(0, 0.2, n)
        close = mid + rng.uniform(-half_w * 0.4, half_w * 0.4, n)
        vol   = rng.integers(500_000, 2_000_000, n).astype(float)
        df = pd.DataFrame(
            {"Open": close * 0.999, "High": high, "Low": low,
             "Close": close, "Volume": vol},
            index=pd.date_range("2023-01-01", periods=n, freq="B"),
        )
        result = detect_symmetrical_triangle(df, lookback=60, min_touches=2, swing_window=3)
        assert result["detected"] is False

    def test_price_outside_triangle_not_detected(self):
        """Price breaks out above the upper trendline — no longer inside."""
        df = _sym_triangle_ohlcv(n=100).copy()
        df.loc[df.index[-1], "Close"] = df["High"].max() * 1.05
        result = detect_symmetrical_triangle(df, lookback=60, min_touches=2, swing_window=3)
        assert result["detected"] is False

    def test_apex_too_close_not_detected(self):
        """Triangle converges in fewer than SYMTRI_MIN_BARS_TO_APEX bars — should not flag."""
        n = 100
        rng = np.random.default_rng(99)
        half_w = np.linspace(10.0, 0.05, n)   # width is ~zero at last bar
        mid = 100.0
        high  = mid + half_w + rng.uniform(0, 0.02, n)
        low   = mid - half_w - rng.uniform(0, 0.02, n)
        close = np.full(n, mid)
        vol   = rng.integers(500_000, 2_000_000, n).astype(float)
        df = pd.DataFrame(
            {"Open": close * 0.999, "High": high, "Low": low,
             "Close": close, "Volume": vol},
            index=pd.date_range("2023-01-01", periods=n, freq="B"),
        )
        result = detect_symmetrical_triangle(df, lookback=60, min_touches=2, swing_window=3)
        assert result["detected"] is False

    def test_narrow_starting_width_not_detected(self):
        """Pattern whose range is far below SYMTRI_MIN_START_WIDTH_PCT should not be flagged."""
        # width_start=0.5 on a mid=100 → 0.5% starting width << 3% threshold
        df = _sym_triangle_ohlcv(n=120, mid=100.0, width_start=0.5, width_end=0.1, noise_amp=0.002)
        result = detect_symmetrical_triangle(df, lookback=60, min_touches=2, swing_window=3)
        assert result["detected"] is False

    def test_span_too_short_not_detected(self):
        """Pattern that forms over too few bars (< SYMTRI_MIN_SPAN_BARS) should not flag."""
        df = _sym_triangle_ohlcv(n=100)
        # Very large swing_window pushes all pivots into a narrow index range
        result = detect_symmetrical_triangle(df, lookback=20, min_touches=2, swing_window=8)
        # With only 20 bars and swing_window=8, span between pivots < 15 → not detected
        assert result["detected"] is False

    def test_increasing_volume_not_detected(self):
        """Volume expanding into the apex violates the contraction requirement."""
        df = _sym_triangle_ohlcv(n=120, n_cycles=8).copy()
        # Reverse the volume so it increases (expands) toward the apex
        df["Volume"] = df["Volume"].iloc[::-1].values
        result = detect_symmetrical_triangle(df)
        assert result["detected"] is False

    def test_wedge_asymmetric_slopes_not_detected(self):
        """
        A rising wedge where the upper slope is ~5× steeper than the lower slope
        should fail the slope symmetry check and not be labelled a symmetric triangle.
        """
        n = 120
        rng = np.random.default_rng(55)
        mid = 100.0
        # Upper trendline falls steeply, lower barely rises → ratio >> 2.0
        upper = np.linspace(mid + 10.0, mid + 1.0, n)   # slope ≈ -0.075
        lower = np.linspace(mid - 10.0, mid - 9.0, n)   # slope ≈ +0.008
        close = (upper + lower) / 2 + rng.uniform(-0.1, 0.1, n)
        high  = np.minimum(close + 0.5, upper + 0.1)
        low   = np.maximum(close - 0.5, lower - 0.1)
        vol_trend = np.linspace(1_800_000, 300_000, n)
        vol = np.maximum(50_000, vol_trend + rng.integers(-100_000, 100_000, n)).astype(float)
        df = pd.DataFrame(
            {"Open": close * 0.999, "High": high, "Low": low,
             "Close": close, "Volume": vol},
            index=pd.date_range("2023-01-01", periods=n, freq="B"),
        )
        result = detect_symmetrical_triangle(df, lookback=80, min_touches=2, swing_window=3)
        assert result["detected"] is False


# ---------------------------------------------------------------------------
# Configurable parameter behaviour
# ---------------------------------------------------------------------------

class TestConfigurableParameters:

    def test_longer_lookback_uses_more_history(self):
        df = _sym_triangle_ohlcv(n=120)
        r40 = detect_symmetrical_triangle(df, lookback=40, min_touches=2, swing_window=3)
        r90 = detect_symmetrical_triangle(df, lookback=90, min_touches=2, swing_window=3)
        if r40["detected"] and r90["detected"]:
            assert r90["n_slice"] > r40["n_slice"]

    def test_higher_min_touches_harder_to_satisfy(self):
        """Requiring more touchpoints makes detection strictly harder."""
        df = _sym_triangle_ohlcv(n=120)
        r2 = detect_symmetrical_triangle(df, lookback=80, min_touches=2, swing_window=3)
        r5 = detect_symmetrical_triangle(df, lookback=80, min_touches=5, swing_window=3)
        # If 5-touch passes, 2-touch must also pass (stricter ⊆ looser)
        if r5["detected"]:
            assert r2["detected"]

    def test_wider_swing_window_finds_fewer_pivots(self):
        """Larger swing_window means more bars must agree → fewer pivots detected."""
        df = _sym_triangle_ohlcv(n=120)
        r3 = detect_symmetrical_triangle(df, lookback=80, min_touches=2, swing_window=3)
        r7 = detect_symmetrical_triangle(df, lookback=80, min_touches=2, swing_window=7)
        if r3["detected"] and r7["detected"]:
            assert len(r3["upper_pivots"]) >= len(r7["upper_pivots"])

    def test_default_min_touches_3_detects_genuine_pattern(self):
        """The new default min_touches=3 should still detect a clear, wide triangle."""
        df = _sym_triangle_ohlcv(n=120, width_start=14.0, width_end=2.0, noise_amp=0.1)
        result = detect_symmetrical_triangle(df, lookback=80)
        assert result["detected"] is True
