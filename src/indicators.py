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
    VOL_SHORT_PERIOD, VOL_LONG_PERIOD, VOL_CONTRACTION_RATIO,
    EMA10_UPPER_PCT, EMA10_LOWER_PCT, KMA_BAND_PCT,
    RMV_LOOKBACK, RMV_TIGHT_THRESHOLD,
    WEEKLY_DAYS, MONTHLY_DAYS,
    IPO_BASE_MAX_DAYS, IPO_BASE_MIN_DAYS, IPO_BASE_SKIP_DAYS,
    IPO_BASE_RANGE_MAX, IPO_BREAKOUT_LOWER, IPO_BREAKOUT_UPPER,
    IPO_VOL_PICKUP_RATIO,
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
    Returns True if the stock shows a Volatility Contraction Pattern:
      1. A prior run-up of ≥ VCP_RUNUP_MIN_PCT in the VCP_RUNUP_LOOKBACK days
         before the consolidation window.
      2. Last VCP_CONSOL_WEEKS weeks each show:
         - Smaller price range than the previous week (range = (High-Low)/Close_avg)
         - Lower average volume than the previous week
    """
    consol_days = VCP_CONSOL_WEEKS * 5
    min_needed = VCP_RUNUP_LOOKBACK + consol_days
    if len(df) < min_needed:
        return False

    runup_slice = df.iloc[-(min_needed):-consol_days]
    consol_slice = df.tail(consol_days)

    # --- Prior run-up check ---
    if len(runup_slice) < 5:
        return False
    runup_low = runup_slice["Low"].min()
    runup_high = runup_slice["High"].max()
    if runup_low <= 0:
        return False
    if (runup_high - runup_low) / runup_low < VCP_RUNUP_MIN_PCT:
        return False

    # --- Consolidation contraction check ---
    weeks = []
    for i in range(VCP_CONSOL_WEEKS):
        w = consol_slice.iloc[i * 5 : (i + 1) * 5]
        if len(w) == 0:
            continue
        close_avg = w["Close"].mean()
        price_range_pct = (w["High"].max() - w["Low"].min()) / close_avg if close_avg > 0 else 0
        avg_vol = w["Volume"].mean()
        weeks.append((price_range_pct, avg_vol))

    if len(weeks) < 2:
        return False

    range_contracting = all(weeks[i][0] > weeks[i + 1][0] for i in range(len(weeks) - 1))
    vol_contracting = all(weeks[i][1] > weeks[i + 1][1] for i in range(len(weeks) - 1))

    return range_contracting and vol_contracting


# ---------------------------------------------------------------------------
# Volume contraction
# ---------------------------------------------------------------------------

def detect_volume_contraction(df: pd.DataFrame) -> bool:
    """Recent 5-day average volume is below VOL_CONTRACTION_RATIO of 20-day average."""
    if len(df) < VOL_LONG_PERIOD:
        return False
    recent = df["Volume"].tail(VOL_SHORT_PERIOD).mean()
    historical = df["Volume"].tail(VOL_LONG_PERIOD).mean()
    if historical <= 0:
        return False
    return (recent / historical) < VOL_CONTRACTION_RATIO


# ---------------------------------------------------------------------------
# Flag pattern detection
# ---------------------------------------------------------------------------

# Tuneable constants (kept here for co-location with the algorithm)
_FLAG_DAYS = 15          # Max length of flag consolidation body
_POLE_DAYS = 10          # Max length of the flagpole
_BASELINE_DAYS = 20      # Days before the pole used for volume baseline
_MIN_POLE_GAIN = 0.15    # Pole must gain ≥ 15%
_MIN_VOL_RATIO = 2.0     # Pole avg volume ≥ 2× baseline avg volume
_MAX_FLAG_RANGE = 0.12   # Flag body range (High-Low)/CMP ≤ 12%
_MAX_RETRACE = 0.50      # Price must not give back more than 50% of pole move


def detect_flag(df: pd.DataFrame) -> bool:
    """
    Bull Flag Pattern:
      Pole  — price gained ≥ MIN_POLE_GAIN in POLE_DAYS with volume ≥ 2× baseline
      Flag  — last FLAG_DAYS days: tight range, declining volume, limited retracement

    Timeline (from oldest to newest):
      [...baseline_20d...][...pole_10d...][...flag_15d...] ← today
    """
    needed = _FLAG_DAYS + _POLE_DAYS + _BASELINE_DAYS
    if len(df) < needed:
        return False

    flag_slice = df.tail(_FLAG_DAYS)
    pole_slice = df.iloc[-(_FLAG_DAYS + _POLE_DAYS) : -_FLAG_DAYS]
    base_slice = df.iloc[-(_FLAG_DAYS + _POLE_DAYS + _BASELINE_DAYS) : -(_FLAG_DAYS + _POLE_DAYS)]

    if len(pole_slice) < 3 or len(base_slice) < 5:
        return False

    # --- Pole: price gain ---
    pole_start = float(pole_slice["Close"].iloc[0])
    pole_peak = float(pole_slice["High"].max())
    if pole_start <= 0:
        return False
    if (pole_peak - pole_start) / pole_start < _MIN_POLE_GAIN:
        return False

    # --- Pole: volume spike vs baseline ---
    pole_vol = float(pole_slice["Volume"].mean())
    base_vol = float(base_slice["Volume"].mean())
    if base_vol <= 0 or pole_vol / base_vol < _MIN_VOL_RATIO:
        return False

    # --- Flag: tight price range ---
    current_price = float(flag_slice["Close"].iloc[-1])
    flag_high = float(flag_slice["High"].max())
    flag_low = float(flag_slice["Low"].min())
    if current_price <= 0:
        return False
    if (flag_high - flag_low) / current_price > _MAX_FLAG_RANGE:
        return False

    # --- Flag: limited retracement from pole peak ---
    pole_move = pole_peak - pole_start
    max_allowable_low = pole_peak - pole_move * _MAX_RETRACE
    if flag_low < max_allowable_low:
        return False

    # --- Flag: volume declining vs pole ---
    flag_vol = float(flag_slice["Volume"].mean())
    if flag_vol >= pole_vol:
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
