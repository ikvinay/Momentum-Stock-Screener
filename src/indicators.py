"""
Technical indicator calculations.

All functions operate on a DataFrame with columns: Open, High, Low, Close, Volume.
"""

from typing import Optional

import numpy as np
import pandas as pd

from config import (
    EMA_PERIODS, RSI_PERIOD,
    VCP_RUNUP_LOOKBACK, VCP_RUNUP_MIN_PCT, VCP_CONSOL_WEEKS,
    VCP_CONTRACTION_TOLERANCE, VCP_VOL_DRY_UP_RATIO, VCP_MAX_BASE_DEPTH_PCT,
    VOL_SHORT_PERIOD, VOL_LONG_PERIOD, VOL_EXTENDED_PERIOD, VOL_CONTRACTION_RATIO,
    VOL_PRICE_STABILITY_PCT, VOL_DISTRIB_DAY_PCT,
    EMA10_UPPER_PCT, EMA10_LOWER_PCT, KMA_BAND_PCT,
    RMV_LOOKBACK, RMV_TIGHT_THRESHOLD,
    WEEKLY_DAYS, MONTHLY_DAYS,
    IPO_BASE_MAX_DAYS, IPO_BASE_MIN_DAYS, IPO_BASE_SKIP_DAYS,
    IPO_BASE_RANGE_MAX, IPO_BREAKOUT_LOWER, IPO_BREAKOUT_UPPER,
    IPO_VOL_PICKUP_RATIO,
    SYMTRI_LOOKBACK, SYMTRI_MIN_TOUCHES, SYMTRI_SWING_WINDOW, SYMTRI_MIN_BARS_TO_APEX,
    SYMTRI_MIN_SPAN_BARS, SYMTRI_MIN_START_WIDTH_PCT,
    SYMTRI_R2_THRESHOLD, SYMTRI_SLOPE_SYMMETRY_MIN, SYMTRI_SLOPE_SYMMETRY_MAX,
    SYMTRI_REQUIRE_VOL_CONTRACTION,
    ASCTRI_LOOKBACK, ASCTRI_MIN_TOUCHES_RESIST, ASCTRI_MIN_TOUCHES_SUPPORT,
    ASCTRI_SWING_WINDOW, ASCTRI_MIN_SPAN_BARS, ASCTRI_MAX_SPAN_BARS,
    ASCTRI_RESIST_MAX_SLOPE_PCT, ASCTRI_SUPPORT_MIN_SLOPE_PCT, ASCTRI_SUPPORT_R2,
    ASCTRI_VOL_END_RATIO, ASCTRI_RSI_MIN, ASCTRI_RSI_MAX, ASCTRI_REQUIRE_ABOVE_200DMA,
)


# ---------------------------------------------------------------------------
# Core indicators
# ---------------------------------------------------------------------------

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMA columns and RSI column in-place (returns copy)."""
    df = df.copy()
    close = df["Close"]
    for p in EMA_PERIODS:
        df[f"ema_{p}"] = calculate_ema(close, p)
    df["rsi"] = calculate_rsi(close)
    return df


# ---------------------------------------------------------------------------
# Filter checks (operate on the latest row as a pd.Series)
# ---------------------------------------------------------------------------

def ema_stack_ok(row: pd.Series) -> bool:
    """EMA10 > EMA20 > EMA50 > EMA200."""
    return (
        row["ema_10"] > row["ema_20"]
        and row["ema_20"] > row["ema_50"]
        and row["ema_50"] > row["ema_200"]
    )


def price_above_20ema(row: pd.Series) -> bool:
    return float(row["Close"]) > float(row["ema_20"])


# ---------------------------------------------------------------------------
# 52-week stats
# ---------------------------------------------------------------------------

def get_52w_high_low(df: pd.DataFrame) -> tuple[float, float]:
    """Return (52W high, 52W low) using last 252 trading days."""
    window = df.tail(252)
    return float(window["High"].max()), float(window["Low"].min())


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

def get_performance(df: pd.DataFrame) -> tuple[float, float]:
    """Return (weekly_pct, monthly_pct) based on Close prices."""
    close = df["Close"]
    n = len(close)
    weekly = (
        (close.iloc[-1] / close.iloc[-WEEKLY_DAYS] - 1) * 100
        if n > WEEKLY_DAYS else 0.0
    )
    monthly = (
        (close.iloc[-1] / close.iloc[-MONTHLY_DAYS] - 1) * 100
        if n > MONTHLY_DAYS else 0.0
    )
    return round(weekly, 2), round(monthly, 2)


# ---------------------------------------------------------------------------
# VCP (Volatility Contraction Pattern)
# ---------------------------------------------------------------------------

def detect_vcp(df: pd.DataFrame) -> bool:
    """
    Returns True if the stock shows a Volatility Contraction Pattern.

    Improvements over naive range-based detection:
      1. Temporal run-up: low_idx = argmin(Low) then gain = (max High after low_idx
         − low) / low — enforces that the Low preceded the High (directional move,
         not a crash-recovery range artifact).
      2. VCP_CONSOL_WEEKS defaults to 4 (Indian mid/small caps need more coiling).
      3. Tolerance band: each week must be ≥ VCP_CONTRACTION_TOLERANCE tighter than
         the prior week (avoids failing on near-flat weeks in a genuine VCP).
      4. Base depth check: final (tightest) week's range ≤ VCP_MAX_BASE_DEPTH_PCT —
         the stock must have genuinely compressed, not just plateaued at a wide level.
      5. Volume dry-up vs run-up: consolidation average volume compared to run-up
         average volume (< VCP_VOL_DRY_UP_RATIO), not just week-over-week, which
         is the true accumulation signal.
    """
    consol_days = VCP_CONSOL_WEEKS * 5
    min_needed = VCP_RUNUP_LOOKBACK + consol_days
    if len(df) < min_needed:
        return False

    runup_slice = df.iloc[-(min_needed):-consol_days]
    consol_slice = df.tail(consol_days)

    # --- Prior run-up check (temporally ordered: low must come before high) ---
    if len(runup_slice) < 5:
        return False
    low_idx = int(runup_slice["Low"].values.argmin())
    runup_low = float(runup_slice["Low"].iloc[low_idx])
    if runup_low <= 0:
        return False
    runup_high = float(runup_slice["High"].iloc[low_idx:].max())
    if (runup_high - runup_low) / runup_low < VCP_RUNUP_MIN_PCT:
        return False

    runup_avg_vol = float(runup_slice["Volume"].mean())

    # --- Consolidation contraction check ---
    weeks = []
    for i in range(VCP_CONSOL_WEEKS):
        w = consol_slice.iloc[i * 5 : (i + 1) * 5]
        if len(w) == 0:
            continue
        close_avg = w["Close"].mean()
        price_range_pct = (w["High"].max() - w["Low"].min()) / close_avg if close_avg > 0 else 0
        avg_vol = float(w["Volume"].mean())
        weeks.append((price_range_pct, avg_vol))

    if len(weeks) < 2:
        return False

    # Each week must tighten by at least VCP_CONTRACTION_TOLERANCE vs prior week
    range_contracting = all(
        weeks[i + 1][0] <= weeks[i][0] * (1 - VCP_CONTRACTION_TOLERANCE)
        for i in range(len(weeks) - 1)
    )
    if not range_contracting:
        return False

    # Final week must be genuinely tight (absolute depth check)
    if weeks[-1][0] > VCP_MAX_BASE_DEPTH_PCT:
        return False

    # Volume must have dried up relative to the run-up, not just week-over-week
    consol_avg_vol = float(consol_slice["Volume"].mean())
    if runup_avg_vol > 0 and consol_avg_vol / runup_avg_vol >= VCP_VOL_DRY_UP_RATIO:
        return False

    return True


# ---------------------------------------------------------------------------
# Volume contraction
# ---------------------------------------------------------------------------

def detect_volume_contraction(df: pd.DataFrame) -> bool:
    """
    True when recent volume has genuinely dried up in an accumulation context.

    Three-tier volume check:
      Tier 1 (required): mean(Vol[-5:]) / mean(Vol[-20:])  < VOL_CONTRACTION_RATIO
      Tier 2 (required): mean(Vol[-5:]) / mean(Vol[-50:])  < VOL_CONTRACTION_RATIO
                         (only applied when 50 bars of history are available)
      Tier 3 (guard):    no distribution day in the recent window — a bar
                         that closes >VOL_DISTRIB_DAY_PCT below its open on
                         above-average volume signals active selling, not quiet rest.

    Price stability guard:
      If price fell more than VOL_PRICE_STABILITY_PCT over the short window the
      low volume reflects distribution (sellers in control), not accumulation.
    """
    if len(df) < VOL_LONG_PERIOD:
        return False

    short = df.tail(VOL_SHORT_PERIOD)
    recent_vol   = float(short["Volume"].mean())
    hist_vol     = float(df["Volume"].tail(VOL_LONG_PERIOD).mean())

    if hist_vol <= 0:
        return False

    # Tier 1 — 5d vs 20d
    if recent_vol / hist_vol >= VOL_CONTRACTION_RATIO:
        return False

    # Tier 2 — 5d vs 50d (stronger baseline, applied when history allows)
    if len(df) >= VOL_EXTENDED_PERIOD:
        extended_vol = float(df["Volume"].tail(VOL_EXTENDED_PERIOD).mean())
        if extended_vol > 0 and recent_vol / extended_vol >= VOL_CONTRACTION_RATIO:
            return False

    # Price stability — low volume + falling price = distribution, not accumulation
    if len(short) >= 2:
        p_start = float(short["Close"].iloc[0])
        p_end   = float(short["Close"].iloc[-1])
        if p_start > 0 and (p_end - p_start) / p_start < -VOL_PRICE_STABILITY_PCT:
            return False

    # Distribution day filter — any bar down >1% intraday on above-average volume
    for _, row in short.iterrows():
        bar_open  = float(row["Open"])
        bar_close = float(row["Close"])
        bar_vol   = float(row["Volume"])
        if bar_open > 0 and (bar_close - bar_open) / bar_open < -VOL_DISTRIB_DAY_PCT:
            if bar_vol > hist_vol:
                return False

    return True


# ---------------------------------------------------------------------------
# Flag pattern detection
# ---------------------------------------------------------------------------

# Tuneable constants (kept here for co-location with the algorithm)
_FLAG_DAYS        = 15     # Max length of flag consolidation body
_POLE_SCAN_WINDOW = 30     # Bars before flag to scan for the pole (handles variable-length poles)
_POLE_MIN_DAYS    = 3      # Minimum pole length in bars
_POLE_MAX_DAYS    = 15     # Maximum pole length in bars
_BASELINE_DAYS    = 20     # Days before the scan window used for volume baseline
_MIN_POLE_GAIN    = 0.15   # Min gain (temporally-ordered trough→peak) to qualify as a pole
_MIN_VOL_RATIO    = 2.0    # Pole avg volume ≥ 2× baseline avg volume
_MAX_FLAG_RANGE   = 0.12   # Flag body (High−Low)/CMP ≤ 12%
_MAX_RETRACE      = 0.50   # Flag low ≥ pole_peak − 50% of pole move
_FLAG_MAX_SLOPE   = 0.002  # Max flag close slope per bar as fraction of price (rising wedge filter)
_FLAG_BREAKOUT_ZONE = 2/3  # Current close must be in the upper third of the flag range


def detect_flag(df: pd.DataFrame) -> bool:
    """
    Bull Flag Pattern.

    Pole  — located by scanning _POLE_SCAN_WINDOW bars before the flag for the highest
            High (peak); the trough is the lowest Low before that peak within the same
            window. Gain is measured trough→peak (temporally ordered). Pole span must
            be between _POLE_MIN_DAYS and _POLE_MAX_DAYS.
    Flag  — last _FLAG_DAYS bars: tight range, slightly downward/sideways slope
            (upward slope = rising wedge, rejected), declining volume, current close
            in the upper third of the flag range (breakout readiness).

    Timeline (from oldest to newest):
      [...baseline_20d...][...pole-scan_30d...][...flag_15d...] ← today
    """
    needed = _FLAG_DAYS + _POLE_SCAN_WINDOW + _BASELINE_DAYS
    if len(df) < needed:
        return False

    flag_slice = df.tail(_FLAG_DAYS)
    scan_slice = df.iloc[-(_FLAG_DAYS + _POLE_SCAN_WINDOW) : -_FLAG_DAYS]
    base_slice = df.iloc[-(_FLAG_DAYS + _POLE_SCAN_WINDOW + _BASELINE_DAYS) : -(_FLAG_DAYS + _POLE_SCAN_WINDOW)]

    if len(base_slice) < 5:
        return False

    # --- (a)/(b) Locate pole via scan: peak = highest High; trough = lowest Low before peak ---
    scan_highs = scan_slice["High"].to_numpy(dtype=float)
    scan_lows  = scan_slice["Low"].to_numpy(dtype=float)
    peak_pos   = int(scan_highs.argmax())

    if peak_pos < _POLE_MIN_DAYS:
        return False    # not enough room before the peak for a valid pole

    trough_pos   = int(scan_lows[:peak_pos].argmin())
    pole_span    = peak_pos - trough_pos
    if pole_span < _POLE_MIN_DAYS or pole_span > _POLE_MAX_DAYS:
        return False

    peak_price   = float(scan_highs[peak_pos])
    trough_price = float(scan_lows[trough_pos])
    if trough_price <= 0:
        return False

    # (b) Pole gain: trough precedes peak — no crash-recovery artifacts
    if (peak_price - trough_price) / trough_price < _MIN_POLE_GAIN:
        return False

    # --- Pole: volume spike vs baseline ---
    pole_slice = scan_slice.iloc[trough_pos : peak_pos + 1]
    pole_vol   = float(pole_slice["Volume"].mean())
    base_vol   = float(base_slice["Volume"].mean())
    if base_vol <= 0 or pole_vol / base_vol < _MIN_VOL_RATIO:
        return False

    # --- Flag: tight price range ---
    current_price = float(flag_slice["Close"].iloc[-1])
    flag_high     = float(flag_slice["High"].max())
    flag_low      = float(flag_slice["Low"].min())
    if current_price <= 0:
        return False
    if (flag_high - flag_low) / current_price > _MAX_FLAG_RANGE:
        return False

    # --- Flag: limited retracement from pole peak ---
    pole_move         = peak_price - trough_price
    max_allowable_low = peak_price - pole_move * _MAX_RETRACE
    if flag_low < max_allowable_low:
        return False

    # --- Flag: volume declining vs pole ---
    flag_vol = float(flag_slice["Volume"].mean())
    if flag_vol >= pole_vol:
        return False

    # --- (c) Flag slope — upward slope signals a rising wedge, not a bull flag ---
    flag_x = np.arange(len(flag_slice), dtype=float)
    flag_slope, _ = np.polyfit(flag_x, flag_slice["Close"].to_numpy(dtype=float), 1)
    if flag_slope > _FLAG_MAX_SLOPE * current_price:
        return False

    # --- (d) Breakout readiness — current close must be in the upper third of the flag ---
    flag_range = flag_high - flag_low
    if flag_range > 0 and current_price < flag_low + _FLAG_BREAKOUT_ZONE * flag_range:
        return False

    return True


# ---------------------------------------------------------------------------
# Near 10 EMA scoring
# ---------------------------------------------------------------------------

def near_10ema_flag(row: pd.Series) -> bool:
    """True if price is within the -2.5% to +1.5% band around EMA10."""
    ema10 = float(row["ema_10"])
    price = float(row["Close"])
    if ema10 <= 0:
        return False
    pct = (price - ema10) / ema10
    return EMA10_LOWER_PCT <= pct <= EMA10_UPPER_PCT


def near_10ema_score(row: pd.Series) -> float:
    """
    Normalized 0–1 score for proximity to EMA10.
    Returns 1.0 when price is centered in the band, 0.0 when outside.
    """
    ema10 = float(row["ema_10"])
    price = float(row["Close"])
    if ema10 <= 0:
        return 0.0
    pct = (price - ema10) / ema10
    if not (EMA10_LOWER_PCT <= pct <= EMA10_UPPER_PCT):
        return 0.0
    band_center = (EMA10_UPPER_PCT + EMA10_LOWER_PCT) / 2
    half_width = (EMA10_UPPER_PCT - EMA10_LOWER_PCT) / 2
    return max(0.0, 1.0 - abs(pct - band_center) / half_width)


# ---------------------------------------------------------------------------
# IPO Base pattern
# ---------------------------------------------------------------------------

def detect_ipo_base(df: pd.DataFrame) -> tuple[bool, float]:
    """
    IPO Base Pattern — stock breaking out of its post-listing consolidation base.

    Conditions:
      1. Listing age: IPO_BASE_MIN_DAYS ≤ len(df) < IPO_BASE_MAX_DAYS
      2. Base validity: after skipping the first IPO_BASE_SKIP_DAYS, the price
         consolidation range (High-Low / Low) must be ≤ IPO_BASE_RANGE_MAX
      3. Breakout zone: current price is within IPO_BREAKOUT_LOWER to
         IPO_BREAKOUT_UPPER relative to the base high
      4. Volume pickup: recent volume ≥ historical average (accumulation)

    Returns (is_ipo_base: bool, pct_from_base_high: float).
    pct_from_base_high is negative when price is below the base high.
    """
    total_days = len(df)

    if total_days < IPO_BASE_MIN_DAYS or total_days >= IPO_BASE_MAX_DAYS:
        return False, 0.0

    # Skip first few days of post-listing chaos
    skip = min(IPO_BASE_SKIP_DAYS, total_days // 4)
    base_df = df.iloc[skip:]

    if len(base_df) < 10:
        return False, 0.0

    current_price = float(df["Close"].iloc[-1])
    base_high     = float(base_df["High"].max())
    base_low      = float(base_df["Low"].min())

    if base_high <= 0 or base_low <= 0:
        return False, 0.0

    # Base must be a genuine consolidation, not a wild swing
    base_range_pct = (base_high - base_low) / base_low
    if base_range_pct > IPO_BASE_RANGE_MAX:
        return False, 0.0

    # Price must be in the breakout zone around the base high
    pct_from_high = (current_price - base_high) / base_high
    if not (IPO_BREAKOUT_LOWER <= pct_from_high <= IPO_BREAKOUT_UPPER):
        return False, round(pct_from_high * 100, 2)

    # Volume: recent 5d avg should be ≥ the historical average (buying interest)
    if total_days >= VOL_LONG_PERIOD:
        recent_vol = float(df["Volume"].tail(VOL_SHORT_PERIOD).mean())
        hist_vol   = float(df["Volume"].tail(VOL_LONG_PERIOD).mean())
        if hist_vol > 0 and (recent_vol / hist_vol) < IPO_VOL_PICKUP_RATIO:
            return False, round(pct_from_high * 100, 2)

    return True, round(pct_from_high * 100, 2)


def ipo_days_listed(df: pd.DataFrame) -> int:
    """Number of trading days since listing (= number of rows in price history)."""
    return len(df)


def ema10_recent_touch(df: pd.DataFrame, lookback: int = 2) -> float:
    """
    Return 1.0 if any bar in the last `lookback` days had its High-Low range
    overlap with the EMA10 level (i.e. the bar touched or crossed the 10 EMA).
    Requires `add_indicators` to have been run (ema_10 column must exist).
    """
    if len(df) < lookback + 1 or "ema_10" not in df.columns:
        return 0.0
    for i in range(-lookback, 0):
        row = df.iloc[i]
        ema10 = float(row["ema_10"])
        if float(row["Low"]) <= ema10 <= float(row["High"]):
            return 1.0
    return 0.0


def calculate_rs_trend(
    stock_df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame],
) -> float:
    """
    Compute the RS line (stock Close / benchmark Close) and return:
      1.0  — EMA10 of RS line > EMA20 (uptrend / accelerating)
      0.0  — EMA10 ≤ EMA20 (downtrend / decelerating)
      0.5  — insufficient data (neutral)

    Uses the last 60 common trading days so the EMAs are stable.
    """
    if benchmark_df is None or len(benchmark_df) < 25 or len(stock_df) < 25:
        return 0.5

    stock_close = stock_df["Close"]
    bench_close = benchmark_df["Close"]

    if isinstance(bench_close, pd.DataFrame):
        bench_close = bench_close.iloc[:, 0]

    stock_close.index = pd.to_datetime(stock_close.index)
    bench_close.index = pd.to_datetime(bench_close.index)

    common = stock_close.index.intersection(bench_close.index)
    if len(common) < 25:
        return 0.5

    rs_line = stock_close.loc[common] / bench_close.loc[common]
    ema10_rs = rs_line.ewm(span=10, adjust=False).mean()
    ema20_rs = rs_line.ewm(span=20, adjust=False).mean()

    return 1.0 if float(ema10_rs.iloc[-1]) > float(ema20_rs.iloc[-1]) else 0.0


# ---------------------------------------------------------------------------
# Tight range + volume contraction score
# ---------------------------------------------------------------------------

def tight_range_vol_score(df: pd.DataFrame) -> float:
    """
    0–1 composite score: how tight is the recent price range and how much
    has volume contracted vs the 20-day baseline.

    range_score = 1 - (recent 5d avg daily range% / historical 20d avg daily range%)
    vol_score   = 1 - (recent 5d avg volume / historical 20d avg volume)
    Returns the average of the two, clamped to [0, 1].
    """
    if len(df) < VOL_LONG_PERIOD:
        return 0.0

    short = df.tail(VOL_SHORT_PERIOD)
    long_ = df.tail(VOL_LONG_PERIOD)

    # Daily range as % of close
    recent_range = ((short["High"] - short["Low"]) / short["Close"]).mean()
    hist_range   = ((long_["High"] - long_["Low"]) / long_["Close"]).mean()
    range_score  = max(0.0, 1.0 - recent_range / hist_range) if hist_range > 0 else 0.0

    # Volume contraction
    recent_vol  = short["Volume"].mean()
    hist_vol    = long_["Volume"].mean()
    vol_score   = max(0.0, 1.0 - recent_vol / hist_vol) if hist_vol > 0 else 0.0

    return min(1.0, (range_score + vol_score) / 2.0)


# ---------------------------------------------------------------------------
# Inside Day (Inside Bar)
# ---------------------------------------------------------------------------

def detect_inside_day(df: pd.DataFrame) -> bool:
    """True if today's High < yesterday's High AND today's Low > yesterday's Low."""
    if len(df) < 2:
        return False
    today     = df.iloc[-1]
    yesterday = df.iloc[-2]
    return (
        float(today["High"]) < float(yesterday["High"])
        and float(today["Low"]) > float(yesterday["Low"])
    )


# ---------------------------------------------------------------------------
# Relative Measured Volatility (RMV)
# ---------------------------------------------------------------------------

def calculate_rmv(df: pd.DataFrame, lookback: int = RMV_LOOKBACK) -> float:
    """
    Relative Measured Volatility — stochastic oscillator applied to bar range.

    RMV_t = clip[0,100]( 100 * (R_t - R_min) / (R_max - R_min) )

    Where R_t = today's High - Low, and R_min / R_max are the min and max
    bar ranges over the PREVIOUS `lookback` bars (current bar deliberately
    excluded so it cannot define its own normalization scale).

    RMV ≈ 0   → today's range is the smallest in recent memory (compression)
    RMV ≈ 100 → today's range is the largest in recent memory (expansion)
    RMV < 15  → unusually quiet bar, often a precursor to a breakout
    """
    if len(df) < lookback + 1:
        return np.nan

    current_range = float(df["High"].iloc[-1] - df["Low"].iloc[-1])

    prev          = df.iloc[-(lookback + 1):-1]   # previous N bars, exclude current
    prev_ranges   = prev["High"] - prev["Low"]
    r_min         = float(prev_ranges.min())
    r_max         = float(prev_ranges.max())

    denom = r_max - r_min if (r_max - r_min) != 0 else 1e-6
    return float(np.clip(100.0 * (current_range - r_min) / denom, 0.0, 100.0))


# ---------------------------------------------------------------------------
# Close to KMA (Key Moving Average) score
# ---------------------------------------------------------------------------

def kma_proximity_score(row: pd.Series) -> float:
    """
    0–1 score for how close the price is to the 10 EMA (Key Moving Average).

    Gradient: 1.0 when price == 10 EMA, 0.0 when price is ±KMA_BAND_PCT away
    or further. Works symmetrically above and below the EMA so stocks setting
    up just beneath the 10 EMA (still above 20 EMA via hard filter) also score.
    """
    ema10 = float(row["ema_10"])
    price = float(row["Close"])
    if ema10 <= 0:
        return 0.0
    pct = abs((price - ema10) / ema10)
    if pct >= KMA_BAND_PCT:
        return 0.0
    return 1.0 - pct / KMA_BAND_PCT


# ---------------------------------------------------------------------------
# IBD-style Relative Strength raw score
# ---------------------------------------------------------------------------

def calculate_rs_raw_score(
    stock_df: pd.DataFrame,
    index_df: Optional[pd.DataFrame] = None,
) -> float:
    """
    IBD-style raw relative strength score (unnormalized).

    Combines two methods:
      Method 2 — Absolute price performance over 3M/6M/9M/12M
      Method 3 — Stock performance relative to comparison index over same windows

    IBD weighting: 40% most-recent quarter + 20% each of the prior 3 quarters.

    Negative returns use the IBD inversion transform: -100 / |pct|, which
    penalises small sustained losses more harshly than a single large drop,
    capped at ±100.  Available windows are re-weighted proportionally so that
    stocks with < 12 months of history still receive a valid score.

    Returns a float (roughly −100 … +100). Call _normalize_rs_ratings() in
    run_screener() to map these raw scores to a 1-99 ordinal rating.
    """
    WINDOWS = [(63, 0.40), (126, 0.20), (189, 0.20), (252, 0.20)]

    def _transform(pct: float) -> float:
        """IBD return transform: positive unchanged, negative inverted."""
        if pct >= 0.0:
            return min(pct, 100.0)
        return max(-100.0 / max(abs(pct), 0.01), -100.0)

    close = stock_df["Close"].dropna()
    if len(close) < WINDOWS[0][0]:
        return 0.0

    idx_close: Optional[pd.Series] = None
    if index_df is not None and not index_df.empty and len(index_df) >= WINDOWS[0][0]:
        idx_close = index_df["Close"].dropna()

    m2_total, m2_w = 0.0, 0.0
    m3_total, m3_w = 0.0, 0.0

    for days, weight in WINDOWS:
        if len(close) <= days:
            continue
        stock_ret = (float(close.iloc[-1]) / float(close.iloc[-days]) - 1.0) * 100.0
        m2_total += weight * _transform(stock_ret)
        m2_w += weight

        if idx_close is not None and len(idx_close) > days:
            idx_ret = (float(idx_close.iloc[-1]) / float(idx_close.iloc[-days]) - 1.0) * 100.0
            m3_total += weight * _transform(stock_ret - idx_ret)
            m3_w += weight

    if m2_w == 0.0:
        return 0.0
    m2_score = m2_total / m2_w

    if m3_w > 0.0:
        return (m2_score + m3_total / m3_w) / 2.0
    return m2_score


# ---------------------------------------------------------------------------
# Symmetrical Triangle pattern
# ---------------------------------------------------------------------------

def detect_symmetrical_triangle(
    df: pd.DataFrame,
    lookback: int = SYMTRI_LOOKBACK,
    min_touches: int = SYMTRI_MIN_TOUCHES,
    swing_window: int = SYMTRI_SWING_WINDOW,
) -> dict:
    """
    Detect a Symmetrical Triangle: converging upper trendline (lower highs) and
    lower trendline (higher lows) that meet at a future apex.

    Parameters
    ----------
    df            : OHLCV DataFrame (requires High, Low, Close, Volume columns)
    lookback      : bars to scan (default SYMTRI_LOOKBACK = 90)
    min_touches   : minimum swing pivots per trendline (default SYMTRI_MIN_TOUCHES = 3)
    swing_window  : bars each side required to qualify a swing pivot (default 3)

    Notes
    -----
    Confirmation lag: the most recent `swing_window` bars cannot produce a pivot
    because the look-ahead window has not yet closed. Triangle signals therefore
    lag by `swing_window` bars — a deliberate tradeoff between recency and
    reliability that must be accounted for in entry timing.

    Returns
    -------
    dict with keys:
      detected          bool
      upper_slope       float | None   — negative slope of upper trendline
      upper_intercept   float | None
      lower_slope       float | None   — positive slope of lower trendline
      lower_intercept   float | None
      upper_pivots      list[(bar_idx, price)]  — swing highs used for fit
      lower_pivots      list[(bar_idx, price)]  — swing lows used for fit
      n_slice           int            — length of detection window (for chart mapping)
    """
    _empty = dict(
        detected=False,
        upper_slope=None, upper_intercept=None,
        lower_slope=None, lower_intercept=None,
        upper_pivots=[], lower_pivots=[],
        n_slice=0,
    )

    min_needed = 2 * swing_window + 2 * min_touches + 4
    if len(df) < min_needed:
        return _empty

    n_use = min(lookback, len(df))
    sl = df.iloc[-n_use:]
    n = len(sl)

    highs = sl["High"].to_numpy(dtype=float)
    lows = sl["Low"].to_numpy(dtype=float)

    swing_highs: list[tuple[int, float]] = []
    swing_lows:  list[tuple[int, float]] = []

    # Strict comparison on BOTH sides prevents double-counting pivots on flat-top
    # formations. The most recent swing_window bars are ineligible — see docstring.
    for i in range(swing_window, n - swing_window):
        left_h  = highs[i - swing_window: i].max()
        right_h = highs[i + 1: i + swing_window + 1].max()
        if highs[i] > left_h and highs[i] > right_h:       # strict both sides
            swing_highs.append((i, highs[i]))

        left_l  = lows[i - swing_window: i].min()
        right_l = lows[i + 1: i + swing_window + 1].min()
        if lows[i] < left_l and lows[i] < right_l:         # strict both sides
            swing_lows.append((i, lows[i]))

    if len(swing_highs) < min_touches or len(swing_lows) < min_touches:
        return _empty

    sh_x = np.array([p[0] for p in swing_highs], dtype=float)
    sh_y = np.array([p[1] for p in swing_highs], dtype=float)
    sl_x = np.array([p[0] for p in swing_lows],  dtype=float)
    sl_y = np.array([p[1] for p in swing_lows],  dtype=float)

    # Pattern must span enough bars so it's a real formation, not tick noise
    pivot_span = max(sh_x[-1], sl_x[-1]) - min(sh_x[0], sl_x[0])
    if pivot_span < SYMTRI_MIN_SPAN_BARS:
        return _empty

    u_slope, u_intercept = np.polyfit(sh_x, sh_y, 1)
    l_slope, l_intercept = np.polyfit(sl_x, sl_y, 1)

    # R² quality gate — polyfit on 3 points is near-deterministic; a good R²
    # filters noisy pivot sets that merely trend in the right direction by chance
    def _r2(x: np.ndarray, y: np.ndarray, slope: float, intercept: float) -> float:
        y_pred = slope * x + intercept
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return 1.0 - ss_res / ss_tot

    if _r2(sh_x, sh_y, u_slope, u_intercept) < SYMTRI_R2_THRESHOLD:
        return _empty
    if _r2(sl_x, sl_y, l_slope, l_intercept) < SYMTRI_R2_THRESHOLD:
        return _empty

    # Upper line must descend (lower highs), lower line must ascend (higher lows)
    if u_slope >= 0 or l_slope <= 0:
        return _empty

    # Slope symmetry: ensures we're detecting a genuinely symmetric triangle
    # and not a wedge or descending/ascending triangle mislabelled as symmetric
    slope_ratio = abs(u_slope) / abs(l_slope)
    if not (SYMTRI_SLOPE_SYMMETRY_MIN <= slope_ratio <= SYMTRI_SLOPE_SYMMETRY_MAX):
        return _empty

    # Slopes must be economically meaningful: each trendline must move by at
    # least SYMTRI_MIN_START_WIDTH_PCT / 2 of price across the pattern span
    current_close = float(sl["Close"].iloc[-1])
    if current_close <= 0:
        return _empty
    min_slope_mag = (current_close * SYMTRI_MIN_START_WIDTH_PCT / 2) / max(pivot_span, 1)
    if abs(u_slope) < min_slope_mag or abs(l_slope) < min_slope_mag:
        return _empty

    # Lines must intersect in the future (apex ahead of current bar)
    slope_diff = u_slope - l_slope
    if abs(slope_diff) < 1e-10:
        return _empty
    x_apex = (l_intercept - u_intercept) / slope_diff
    last_x = float(n - 1)
    if x_apex <= last_x:
        return _empty

    # Triangle must still be meaningfully open (apex is not imminent)
    if x_apex - last_x < SYMTRI_MIN_BARS_TO_APEX:
        return _empty

    # Current price must sit inside the triangle
    current_upper = u_slope * last_x + u_intercept
    current_lower = l_slope * last_x + l_intercept
    if not (current_lower <= current_close <= current_upper):
        return _empty

    # Starting width must be large enough to be a real pattern
    first_x = min(sh_x[0], sl_x[0])
    width_first = (u_slope * first_x + u_intercept) - (l_slope * first_x + l_intercept)
    if width_first < current_close * SYMTRI_MIN_START_WIDTH_PCT:
        return _empty

    # Width must genuinely contract from first pivot to current bar
    width_last = current_upper - current_lower
    if width_last >= width_first or width_first <= 0:
        return _empty

    # Pattern must not be essentially closed already (width > 0.5% of price)
    if width_last < current_close * 0.005:
        return _empty

    # Volume must contract into the apex (second-half avg < first-half avg)
    if SYMTRI_REQUIRE_VOL_CONTRACTION and "Volume" in sl.columns:
        mid = n // 2
        vol_first  = float(sl["Volume"].iloc[:mid].mean())
        vol_second = float(sl["Volume"].iloc[mid:].mean())
        if vol_first > 0 and vol_second >= vol_first:
            return _empty

    return dict(
        detected=True,
        upper_slope=float(u_slope),
        upper_intercept=float(u_intercept),
        lower_slope=float(l_slope),
        lower_intercept=float(l_intercept),
        upper_pivots=[(int(i), float(y)) for i, y in zip(sh_x, sh_y)],
        lower_pivots=[(int(i), float(y)) for i, y in zip(sl_x, sl_y)],
        n_slice=n,
    )


# ---------------------------------------------------------------------------
# Ascending Triangle pattern
# ---------------------------------------------------------------------------

def detect_ascending_triangle(
    df: pd.DataFrame,
    lookback: int = ASCTRI_LOOKBACK,
    min_touches_resist: int = ASCTRI_MIN_TOUCHES_RESIST,
    min_touches_support: int = ASCTRI_MIN_TOUCHES_SUPPORT,
    swing_window: int = ASCTRI_SWING_WINDOW,
) -> dict:
    """
    Detect an Ascending Triangle: flat-to-slightly-rising resistance (horizontal
    ceiling tested multiple times) and a rising support line (higher lows) that
    converges toward the resistance — a classic bullish continuation setup.

    Additional context filters (applied when indicator columns are present):
      - Price above 200-day EMA (ASCTRI_REQUIRE_ABOVE_200DMA)
      - RSI between ASCTRI_RSI_MIN and ASCTRI_RSI_MAX (not oversold / overbought)
      - Volume declining over the formation (linear slope < 0 + end/start ratio)
      - OBV net non-negative (accumulation, not distribution)

    Returns
    -------
    dict with keys:
      detected           bool
      resist_slope       float | None  — near-zero slope of resistance line
      resist_intercept   float | None
      support_slope      float | None  — positive slope of support line
      support_intercept  float | None
      resist_pivots      list[(bar_idx, price)]
      support_pivots     list[(bar_idx, price)]
      n_slice            int
    """
    _empty = dict(
        detected=False,
        resist_slope=None, resist_intercept=None,
        support_slope=None, support_intercept=None,
        resist_pivots=[], support_pivots=[],
        n_slice=0,
    )

    min_needed = 2 * swing_window + 2 * max(min_touches_resist, min_touches_support) + 4
    if len(df) < min_needed:
        return _empty

    # --- Trend context: above 200 EMA ---
    if ASCTRI_REQUIRE_ABOVE_200DMA and "ema_200" in df.columns:
        ema200 = float(df["ema_200"].iloc[-1])
        if ema200 > 0 and float(df["Close"].iloc[-1]) < ema200:
            return _empty

    n_use = min(lookback, len(df))
    if n_use < ASCTRI_MIN_SPAN_BARS + 2 * swing_window:
        return _empty
    sl = df.iloc[-n_use:]
    n = len(sl)

    highs = sl["High"].to_numpy(dtype=float)
    lows  = sl["Low"].to_numpy(dtype=float)

    resist_pivots: list[tuple[int, float]] = []
    support_pivots: list[tuple[int, float]] = []

    for i in range(swing_window, n - swing_window):
        lh = highs[i - swing_window: i].max()
        rh = highs[i + 1: i + swing_window + 1].max()
        if highs[i] > lh and highs[i] > rh:
            resist_pivots.append((i, highs[i]))

        ll = lows[i - swing_window: i].min()
        rl = lows[i + 1: i + swing_window + 1].min()
        if lows[i] < ll and lows[i] < rl:
            support_pivots.append((i, lows[i]))

    if len(resist_pivots) < min_touches_resist or len(support_pivots) < min_touches_support:
        return _empty

    rx = np.array([p[0] for p in resist_pivots],  dtype=float)
    ry = np.array([p[1] for p in resist_pivots],  dtype=float)
    sx = np.array([p[0] for p in support_pivots], dtype=float)
    sy = np.array([p[1] for p in support_pivots], dtype=float)

    # Pivot span must be in [MIN_SPAN, MAX_SPAN]
    pivot_span = max(rx[-1], sx[-1]) - min(rx[0], sx[0])
    if pivot_span < ASCTRI_MIN_SPAN_BARS or pivot_span > ASCTRI_MAX_SPAN_BARS:
        return _empty

    r_slope, r_intercept = np.polyfit(rx, ry, 1)
    s_slope, s_intercept = np.polyfit(sx, sy, 1)

    current_close = float(sl["Close"].iloc[-1])
    if current_close <= 0:
        return _empty

    # R² quality gate for support line
    def _r2(x: np.ndarray, y: np.ndarray, slope: float, intercept: float) -> float:
        y_pred = slope * x + intercept
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return 1.0 - ss_res / ss_tot

    if _r2(sx, sy, s_slope, s_intercept) < ASCTRI_SUPPORT_R2:
        return _empty

    # Resistance must be near-flat: |slope| < threshold
    if abs(r_slope) / current_close > ASCTRI_RESIST_MAX_SLOPE_PCT:
        return _empty

    # Support must be clearly ascending
    if s_slope / current_close < ASCTRI_SUPPORT_MIN_SLOPE_PCT:
        return _empty

    # Resistance must not be declining (flat or very slightly rising only)
    if r_slope < 0:
        return _empty

    # Lines must be converging: support rising toward flat resistance
    last_x = float(n - 1)
    first_x = min(rx[0], sx[0])
    resist_now   = r_slope * last_x + r_intercept
    support_now  = s_slope * last_x + s_intercept
    resist_start = r_slope * first_x + r_intercept
    support_start = s_slope * first_x + s_intercept

    width_first = resist_start - support_start
    width_last  = resist_now  - support_now
    if width_first <= 0 or width_last >= width_first:
        return _empty

    # Width must still be meaningful (not already at the apex)
    if width_last < current_close * 0.005:
        return _empty

    # Current price must be inside the triangle
    if not (support_now <= current_close <= resist_now):
        return _empty

    # RSI context filter (applied when rsi column is available)
    if "rsi" in sl.columns:
        current_rsi = float(sl["rsi"].iloc[-1])
        if not np.isnan(current_rsi) and not (ASCTRI_RSI_MIN <= current_rsi <= ASCTRI_RSI_MAX):
            return _empty

    # Volume declining over the formation
    vol_arr = sl["Volume"].to_numpy(dtype=float)
    vol_x   = np.arange(n, dtype=float)
    vol_slope, _ = np.polyfit(vol_x, vol_arr, 1)
    if vol_slope >= 0:
        return _empty

    vol_window  = max(5, min(10, n // 4))
    vol_start   = float(sl["Volume"].iloc[:vol_window].mean())
    vol_end     = float(sl["Volume"].iloc[-vol_window:].mean())
    if vol_start > 0 and vol_end / vol_start >= ASCTRI_VOL_END_RATIO:
        return _empty

    # OBV net non-negative (accumulation, not distribution)
    close_arr  = sl["Close"].to_numpy(dtype=float)
    close_diff = np.diff(close_arr)
    signed_vol = np.where(close_diff > 0, vol_arr[1:],
                 np.where(close_diff < 0, -vol_arr[1:], 0.0))
    obv        = np.concatenate([[0.0], np.cumsum(signed_vol)])
    obv_x      = np.arange(n, dtype=float)
    obv_slope, _ = np.polyfit(obv_x, obv, 1)
    avg_vol = float(np.mean(vol_arr)) + 1e-6
    # Reject if OBV is losing more than 30% of average daily volume per bar
    if obv_slope / avg_vol < -0.30:
        return _empty

    return dict(
        detected=True,
        resist_slope=float(r_slope),
        resist_intercept=float(r_intercept),
        support_slope=float(s_slope),
        support_intercept=float(s_intercept),
        resist_pivots=[(int(i), float(y)) for i, y in zip(rx, ry)],
        support_pivots=[(int(i), float(y)) for i, y in zip(sx, sy)],
        n_slice=n,
    )
