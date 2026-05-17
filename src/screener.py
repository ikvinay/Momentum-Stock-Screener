"""
Main screening and ranking engine.

Applies hard filters to all Nifty 1000 stocks, computes a composite
ranking score for those that pass, and returns a sorted DataFrame.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import (
    MIN_MARKET_CAP_CR, PRICE_MIN_PCT_OF_52W_HIGH,
    MIN_HISTORY_DAYS, MONTHLY_DAYS, QUARTERLY_DAYS, WEEKLY_DAYS, INR_PER_CR,
    SCORE_WEIGHTS, EMA10_UPPER_PCT, EMA10_LOWER_PCT,
    IPO_BASE_MIN_DAYS, IPO_BASE_MAX_DAYS, COMBINED_RANK_WEIGHTS,
    RSI_MIN_SCORE, EMA10_MAX_EXTENSION_PCT,
)
from src.indicators import (
    add_indicators, ema_stack_ok, price_above_20ema,
    get_52w_high_low, get_performance,
    detect_vcp, detect_flag, detect_volume_contraction,
    near_10ema_flag, near_10ema_score,
    detect_ipo_base, ipo_days_listed, calculate_rsi,
    detect_inside_day, tight_range_vol_score,
    kma_proximity_score, calculate_rmv, calculate_rs_raw_score,
    ema10_recent_touch, calculate_rs_trend,
    detect_symmetrical_triangle,
    detect_ascending_triangle,
)
from src.sector_mapper import resolve_sector_index, index_return

logger = logging.getLogger(__name__)


def _normalize_rs_ratings(raw_scores: Dict[str, float]) -> Dict[str, int]:
    """
    Map raw RS scores to a 1-99 ordinal scale by percentile rank across all
    tickers that had enough price history to receive a score.
    """
    if not raw_scores:
        return {}
    s = pd.Series(raw_scores).dropna()
    if s.empty:
        return {}
    n = len(s)
    if n == 1:
        return {s.index[0]: 50}
    ranks = s.rank(method="average", ascending=True)
    ratings = (1.0 + 98.0 * (ranks - 1.0) / (n - 1.0)).round(0).clip(1, 99).astype(int)
    return ratings.to_dict()


def run_screener(
    price_data: Dict[str, pd.DataFrame],
    stock_info: Dict[str, dict],
    sector_indices: Dict[str, pd.DataFrame],
    benchmark: Optional[pd.DataFrame],
    freefloat: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Apply all filters and return a ranked DataFrame of passing stocks.

    Parameters
    ----------
    price_data : {ticker: OHLCV DataFrame}
    stock_info : {ticker: dict with sector, industry, pe, market_cap}
    sector_indices : {nse_index_ticker: OHLCV DataFrame}
    benchmark : OHLCV DataFrame for NiftyMidSmallCap400

    Returns
    -------
    DataFrame sorted by composite Score descending, grouped display ready.
    """
    benchmark_1m = index_return(benchmark, MONTHLY_DAYS)
    benchmark_3m = index_return(benchmark, QUARTERLY_DAYS)

    # Pre-compute RS raw scores across the FULL universe (before hard filters)
    # so that the 1-99 ranking reflects all stocks, not just those that pass.
    logger.info("Computing RS raw scores for full universe…")
    rs_raw: Dict[str, float] = {}
    for ticker, raw_df in price_data.items():
        try:
            rs_raw[ticker] = calculate_rs_raw_score(raw_df, benchmark)
        except Exception:
            pass
    rs_ratings = _normalize_rs_ratings(rs_raw)

    freefloat = freefloat or {}
    rows = []
    for ticker, raw_df in price_data.items():
        try:
            ff_pct = freefloat.get(ticker)
            result = _process_stock(ticker, raw_df, stock_info, sector_indices, benchmark_1m, benchmark_3m, benchmark, ff_pct)
            if result is not None:
                result["RS Rating"] = int(rs_ratings.get(ticker, 50))
                rows.append(result)
        except Exception as exc:
            logger.debug(f"Skipped {ticker}: {exc}")

    if not rows:
        logger.warning("No stocks passed the screener filters.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    w_s  = COMBINED_RANK_WEIGHTS["score"]     / 100.0
    w_rs = COMBINED_RANK_WEIGHTS["rs_rating"] / 100.0
    df["_rank_score"] = df["Score"] * w_s + df["RS Rating"] * w_rs
    df = df.sort_values("_rank_score", ascending=False).drop(columns=["_rank_score"]).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "Rank"

    logger.info(f"Screener complete: {len(df)} stocks passed")
    return df


def _process_stock(
    ticker: str,
    raw_df: pd.DataFrame,
    stock_info: Dict[str, dict],
    sector_indices: Dict[str, pd.DataFrame],
    benchmark_1m: float,
    benchmark_3m: float,
    benchmark_df: Optional[pd.DataFrame] = None,
    free_float_pct: Optional[float] = None,
) -> Optional[dict]:
    """Return a result dict if the stock passes all filters, else None."""
    info = stock_info.get(ticker, {})

    # --- Hard filter: minimum history ---
    if len(raw_df) < MIN_HISTORY_DAYS:
        return None

    # --- Hard filter: market cap > 2000 Cr ---
    mkt_cap_raw = info.get("market_cap") or 0
    mkt_cap_cr = mkt_cap_raw / INR_PER_CR
    if mkt_cap_cr < MIN_MARKET_CAP_CR:
        return None

    # --- Compute indicators ---
    df = add_indicators(raw_df)
    latest = df.iloc[-1]
    current_price = float(latest["Close"])

    # --- Hard filter: price > 52W high * 0.76 (within 24% of 52W high) ---
    high_52w, low_52w = get_52w_high_low(df)
    if current_price <= high_52w * PRICE_MIN_PCT_OF_52W_HIGH:
        return None

    # --- Hard filter: EMA stack 10 > 20 > 50 > 200 ---
    if not ema_stack_ok(latest):
        return None

    # --- Hard filter: price above 20 EMA ---
    if not price_above_20ema(latest):
        return None

    # --- Hard filter: price must not be more than 5% above 10 EMA ---
    if current_price > float(latest["ema_10"]) * (1 + EMA10_MAX_EXTENSION_PCT):
        return None

# --- Hard filter: RSI must be ≥ RSI_MIN_SCORE ---
    rsi_val = latest["rsi"]
    if np.isnan(rsi_val) or float(rsi_val) < RSI_MIN_SCORE:
        return None

    # --- Derived metrics ---
    rsi = round(float(rsi_val), 1)

    pe_raw = info.get("trailing_pe")
    pe = round(float(pe_raw), 1) if pe_raw and not np.isnan(float(pe_raw)) else None

    sector = info.get("sector") or "Unknown"
    industry = info.get("industry") or "Unknown"
    company_name = info.get("company_name") or ticker.replace(".NS", "")
    symbol = ticker.replace(".NS", "")

    weekly_pct, monthly_pct = get_performance(df)

    vcp = detect_vcp(df)
    flag = detect_flag(df)
    vol_contraction = detect_volume_contraction(df)
    near_ema_flag = near_10ema_flag(latest)
    near_ema_sc = near_10ema_score(latest)
    inside_day = detect_inside_day(df)
    sym_triangle = detect_symmetrical_triangle(df)["detected"]
    asc_triangle = detect_ascending_triangle(df)["detected"]
    trv_score  = tight_range_vol_score(df)
    kma_score  = kma_proximity_score(latest)
    rmv        = calculate_rmv(df)

    # --- Stock returns ---
    close = df["Close"]
    stock_1m_ret = (
        float((close.iloc[-1] / close.iloc[-MONTHLY_DAYS]  - 1) * 100)
        if len(close) >= MONTHLY_DAYS  else 0.0
    )
    stock_3m_ret = (
        float((close.iloc[-1] / close.iloc[-QUARTERLY_DAYS] - 1) * 100)
        if len(close) >= QUARTERLY_DAYS else 0.0
    )

    # --- Outperformance vs benchmark (avg of 1M and 3M) ---
    benchmark_outperf_1m = stock_1m_ret - benchmark_1m
    benchmark_outperf_3m = stock_3m_ret - benchmark_3m
    benchmark_outperf = round((benchmark_outperf_1m + benchmark_outperf_3m) / 2, 2)

    # --- Outperformance vs sector index (avg of 1M and 3M) ---
    sector_ticker = resolve_sector_index(sector)
    sector_df = sector_indices.get(sector_ticker)
    sector_1m = index_return(sector_df, MONTHLY_DAYS)
    sector_3m = index_return(sector_df, QUARTERLY_DAYS)
    sector_outperf_1m = stock_1m_ret - sector_1m
    sector_outperf_3m = stock_3m_ret - sector_3m
    sector_outperf = round((sector_outperf_1m + sector_outperf_3m) / 2, 2)

    # --- 52W high proximity ---
    pct_from_52h = round((high_52w - current_price) / high_52w * 100, 2) if high_52w > 0 else 0.0

    # --- New score components ---
    # RSI: normalise [55, 80] → [0, 1]; capped at 1 above 80
    rsi_sc = min(1.0, max(0.0, (float(rsi_val) - float(RSI_MIN_SCORE)) / (80.0 - float(RSI_MIN_SCORE))))

    # RMV: lower = more compressed = higher score; linear [0, 100] → [1, 0]
    rmv_sc = max(0.0, 1.0 - rmv / 100.0) if not np.isnan(rmv) else 0.5

    # RS trend: 1.0 if RS line (stock/benchmark) EMA10 > EMA20, else 0.0
    rs_trend_sc = calculate_rs_trend(df, benchmark_df)

    # EMA10 recent touch: 1.0 if bar range overlapped EMA10 in last 2 days
    ema10_touch_sc = ema10_recent_touch(df, lookback=2)

    # --- Composite score ---
    score = _compute_score(
        pct_from_52h=pct_from_52h,
        benchmark_outperf=benchmark_outperf,
        sector_outperf=sector_outperf,
        tight_range_vol=trv_score,
        close_to_kma=kma_score,
        rsi_sc=rsi_sc,
        rmv_sc=rmv_sc,
        rs_trend_sc=rs_trend_sc,
        ema10_touch_sc=ema10_touch_sc,
    )

    return {
        "Symbol":              symbol,
        "Company":             company_name,
        "Sector":              sector,
        "Industry":            industry,
        "CMP":                 round(current_price, 2),
        "PE":                  pe,
        "Free Float %":        free_float_pct,
        "RSI":                 rsi,
        "RMV":                 round(rmv, 1) if not np.isnan(rmv) else None,
        "Weekly %":            weekly_pct,
        "Monthly %":           monthly_pct,
        "Quarterly %":         round(stock_3m_ret, 2),
        "VCP":                 vcp,
        "Flag":                flag,
        "Vol Contraction":     vol_contraction,
        "Near 10 EMA":         near_ema_flag,
        "Inside Day":          inside_day,
        "Sym Triangle":        sym_triangle,
        "Asc Triangle":        asc_triangle,
        "52W High":            round(high_52w, 2),
        "52W Low":             round(low_52w, 2),
        "% from 52W High":     pct_from_52h,
        "Benchmark Outperf %": benchmark_outperf,
        "Sector Outperf %":    sector_outperf,
        "Market Cap (Cr)":     round(mkt_cap_cr, 0),
        "Sector Index":        sector_ticker,
        "EMA10":               round(float(latest["ema_10"]), 2),
        "EMA20":               round(float(latest["ema_20"]), 2),
        "EMA50":               round(float(latest["ema_50"]), 2),
        "EMA200":              round(float(latest["ema_200"]), 2),
        "Score":               round(score, 2),
    }


def _compute_score(
    pct_from_52h: float,
    benchmark_outperf: float,
    sector_outperf: float,
    tight_range_vol: float,
    close_to_kma: float,
    rsi_sc: float = 0.0,
    rmv_sc: float = 0.0,
    rs_trend_sc: float = 0.5,
    ema10_touch_sc: float = 0.0,
) -> float:
    """
    Weighted composite score 0–100.

    Weights from config.SCORE_WEIGHTS:
      52w_high_proximity : 20  — closer to 52W high
      benchmark_outperf  : 15  — avg 1M+3M vs NiftyMidSmallCap400
      sector_outperf     : 15  — avg 1M+3M vs sector index
      tight_range_vol    : 10  — tight price range + volume contraction (0–1)
      close_to_kma       :  0  — proximity to 10 EMA band (inactive)
      rsi_score          : 15  — RSI normalised in [55, 80] band (0–1)
      rmv_score          : 10  — lower RMV = higher score (0–1)
      rs_trend           :  5  — RS line EMA10 > EMA20 (0 or 1)
      ema10_touch        : 10  — bar touched 10 EMA in last 2 days (0 or 1)
    """
    w = SCORE_WEIGHTS

    # 1. 52W high proximity — 5% below → ~90 pts, 25% below → 50 pts
    s1 = max(0.0, (100.0 - pct_from_52h * 2.0) / 100.0) * w["52w_high_proximity"]

    # 2. Benchmark outperformance — band: -5% → 0, +10% → max
    s2 = min(1.0, max(0.0, (benchmark_outperf + 5.0) / 15.0)) * w["benchmark_outperf"]

    # 3. Sector outperformance — same normalisation
    s3 = min(1.0, max(0.0, (sector_outperf + 5.0) / 15.0)) * w["sector_outperf"]

    # 4. Tight price range + volume contraction (already 0–1)
    s4 = tight_range_vol * w["tight_range_vol"]

    # 5. Close to 10 EMA within ±1.5% band (already 0–1)
    s5 = close_to_kma * w["close_to_kma"]

    # 6. RSI score — normalised from [55, 80] to [0, 1]
    s6 = rsi_sc * w["rsi_score"]

    # 7. RMV score — lower RMV = more compressed bar = higher score (0–1)
    s7 = rmv_sc * w["rmv_score"]

    # 8. RS trend — 1.0 if RS line is in uptrend, 0.0 if not
    s8 = rs_trend_sc * w["rs_trend"]

    # 9. EMA10 recent touch — 1.0 if bar touched 10 EMA in last 2 days
    s9 = ema10_touch_sc * w["ema10_touch"]

    return s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9


# ---------------------------------------------------------------------------
# IPO Base screener (runs separately from the main screener)
# ---------------------------------------------------------------------------

def run_ipo_screener(
    price_data: Dict[str, pd.DataFrame],
    stock_info: Dict[str, dict],
    sector_indices: Optional[Dict[str, pd.DataFrame]] = None,
    nifty500: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Screen stocks with IPO_BASE_MIN_DAYS ≤ history < IPO_BASE_MAX_DAYS
    for the IPO Base breakout pattern.

    Filters applied:
      - Market Cap > MIN_MARKET_CAP_CR
      - EMA10 > EMA20  (short-term momentum confirmation)
      - detect_ipo_base() returns True

    Ranked by sector outperformance vs Nifty 500 (strongest sector first).
    """
    sector_indices = sector_indices or {}
    nifty500_ret   = index_return(nifty500, MONTHLY_DAYS) if nifty500 is not None else 0.0
    rows = []

    for ticker, raw_df in price_data.items():
        days = len(raw_df)
        if days < IPO_BASE_MIN_DAYS or days >= IPO_BASE_MAX_DAYS:
            continue

        try:
            info = stock_info.get(ticker, {})

            # Market cap filter
            mkt_cap_raw = info.get("market_cap") or 0
            mkt_cap_cr  = mkt_cap_raw / INR_PER_CR
            if mkt_cap_cr < MIN_MARKET_CAP_CR:
                continue

            # EMA10 > EMA20 filter
            close  = raw_df["Close"]
            ema10  = float(close.ewm(span=10, adjust=False).mean().iloc[-1])
            ema20  = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
            if ema10 <= ema20:
                continue

            # IPO base breakout filter
            is_base, pct_from_high = detect_ipo_base(raw_df)
            if not is_base:
                continue

            current_price = float(close.iloc[-1])
            days_listed   = ipo_days_listed(raw_df)

            rsi_series = calculate_rsi(close)
            rsi_val    = float(rsi_series.iloc[-1])
            rsi        = round(rsi_val, 2) if not np.isnan(rsi_val) else None

            pe_raw = info.get("trailing_pe")
            pe     = round(float(pe_raw), 2) if pe_raw and not np.isnan(float(pe_raw)) else None

            weekly_pct  = round(float((close.iloc[-1] / close.iloc[-WEEKLY_DAYS]  - 1) * 100), 2) if days > WEEKLY_DAYS  else 0.0
            monthly_pct = round(float((close.iloc[-1] / close.iloc[-MONTHLY_DAYS] - 1) * 100), 2) if days > MONTHLY_DAYS else 0.0

            skip      = min(5, days // 4)
            base_high = float(raw_df["High"].iloc[skip:].max())

            sector        = info.get("sector") or "Unknown"
            sector_ticker = resolve_sector_index(sector)
            sector_df     = sector_indices.get(sector_ticker)
            sector_ret    = index_return(sector_df, MONTHLY_DAYS)
            sector_outperf = round(sector_ret - nifty500_ret, 2)

            inside_day = detect_inside_day(raw_df)
            rmv        = calculate_rmv(raw_df)

            rows.append({
                "Symbol":             ticker.replace(".NS", ""),
                "Company":            info.get("company_name") or ticker.replace(".NS", ""),
                "Sector":             sector,
                "Industry":           info.get("industry") or "Unknown",
                "CMP":                round(current_price, 2),
                "Base High":          round(base_high, 2),
                "% from Base High":   pct_from_high,
                "Days Listed":        days_listed,
                "PE":                 pe,
                "RSI":                rsi,
                "RMV":                round(rmv, 1) if rmv is not None else None,
                "Inside Day":         inside_day,
                "Weekly %":           weekly_pct,
                "Monthly %":          monthly_pct,
                "Sector Outperf %":   sector_outperf,
                "Market Cap (Cr)":    round(mkt_cap_cr, 0),
            })

        except Exception as exc:
            logger.debug(f"IPO screener skipped {ticker}: {exc}")

    if not rows:
        logger.info("IPO screener: no stocks matched")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Primary sort: sector outperformance (strongest sector first)
    # Secondary sort: closest to base high (most actionable within the sector)
    df = df.sort_values(
        ["Sector Outperf %", "% from Base High"],
        ascending=[False, False],
    ).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "Rank"

    logger.info(f"IPO screener: {len(df)} stocks passed (EMA10>EMA20 + IPO base)")
    return df


def run_ipo_inside_bar_screener(
    price_data: Dict[str, pd.DataFrame],
    stock_info: Dict[str, dict],
    sector_indices: Optional[Dict[str, pd.DataFrame]] = None,
    nifty500: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Scan all IPO-age stocks for Inside Bar + Low RMV setups.

    Criteria:
      - IPO_BASE_MIN_DAYS <= trading days < IPO_BASE_MAX_DAYS
      - detect_inside_day() returns True
      - RMV <= RMV_TIGHT_THRESHOLD * 2

    Intentionally does NOT require detect_ipo_base() or a market-cap hard filter
    because many newly listed stocks lack yfinance cap data and haven't formed a
    full base yet, but already show compressed-volatility inside-bar setups.
    """
    from config import RMV_TIGHT_THRESHOLD as RMV_CAP_FACTOR
    rmv_cap        = RMV_CAP_FACTOR * 2
    sector_indices = sector_indices or {}
    nifty500_ret   = index_return(nifty500, MONTHLY_DAYS) if nifty500 is not None else 0.0
    rows = []

    for ticker, raw_df in price_data.items():
        days = len(raw_df)
        if days < IPO_BASE_MIN_DAYS or days >= IPO_BASE_MAX_DAYS:
            continue

        try:
            # Inside Bar required
            if not detect_inside_day(raw_df):
                continue

            # RMV must be low
            rmv = calculate_rmv(raw_df)
            if rmv is None or rmv > rmv_cap:
                continue

            info          = stock_info.get(ticker, {})
            close         = raw_df["Close"]
            current_price = float(close.iloc[-1])
            days_listed   = ipo_days_listed(raw_df)

            mkt_cap_raw = info.get("market_cap") or 0
            mkt_cap_cr  = mkt_cap_raw / INR_PER_CR

            ema10 = float(close.ewm(span=10, adjust=False).mean().iloc[-1])
            ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
            if ema10 <= ema20:
                continue

            rsi_series  = calculate_rsi(close)
            rsi_val     = float(rsi_series.iloc[-1])
            rsi         = round(rsi_val, 2) if not np.isnan(rsi_val) else None

            pe_raw = info.get("trailing_pe")
            pe     = round(float(pe_raw), 2) if pe_raw and not np.isnan(float(pe_raw)) else None

            weekly_pct  = round(float((close.iloc[-1] / close.iloc[-WEEKLY_DAYS]  - 1) * 100), 2) if days > WEEKLY_DAYS  else 0.0
            monthly_pct = round(float((close.iloc[-1] / close.iloc[-MONTHLY_DAYS] - 1) * 100), 2) if days > MONTHLY_DAYS else 0.0

            skip      = min(5, days // 4)
            base_high = float(raw_df["High"].iloc[skip:].max())
            pct_from_high = round((current_price / base_high - 1) * 100, 2)

            sector        = info.get("sector") or "Unknown"
            sector_ticker = resolve_sector_index(sector)
            sector_df     = sector_indices.get(sector_ticker)
            sector_ret    = index_return(sector_df, MONTHLY_DAYS)
            sector_outperf = round(sector_ret - nifty500_ret, 2)

            rows.append({
                "Symbol":           ticker.replace(".NS", ""),
                "Company":          info.get("company_name") or ticker.replace(".NS", ""),
                "Sector":           sector,
                "Industry":         info.get("industry") or "Unknown",
                "CMP":              round(current_price, 2),
                "Base High":        round(base_high, 2),
                "% from Base High": pct_from_high,
                "Days Listed":      days_listed,
                "EMA10 > EMA20":    ema10 > ema20,
                "RMV":              round(rmv, 1),
                "Inside Day":       True,
                "PE":               pe,
                "RSI":              rsi,
                "Weekly %":         weekly_pct,
                "Monthly %":        monthly_pct,
                "Sector Outperf %": sector_outperf,
                "Market Cap (Cr)":  round(mkt_cap_cr, 0) if mkt_cap_cr > 0 else None,
            })

        except Exception as exc:
            logger.debug(f"IPO inside-bar screener skipped {ticker}: {exc}")

    if not rows:
        logger.info("IPO inside-bar screener: no stocks matched")
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values(
        ["RMV", "% from Base High"], ascending=[True, False]
    ).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "Rank"

    logger.info(f"IPO inside-bar screener: {len(df)} stocks (Inside Day + RMV ≤ {rmv_cap})")
    return df
