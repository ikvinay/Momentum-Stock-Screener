"""
MCX Commodity screener — fetch OHLCV, score and rank tracked futures.

Data comes from Yahoo Finance international proxies (COMEX / NYMEX / LME).
RS is measured against:
  - Nifty 500  (^CRSLDX) — general benchmark
  - Nifty Metal (^CNXMETAL) — sectoral benchmark
Hard filter: EMA10 > EMA20 > EMA50.
"""

import logging
import os
import pickle
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    DATA_DIR,
    COMMODITY_OHLCV_FILE, COMMODITY_RESULTS_FILE,
    MCX_COMMODITY_TICKERS, MCX_COMMODITY_CATEGORY,
    COMMODITY_SCORE_WEIGHTS,
    COMMODITY_NIFTY500_TICKERS, COMMODITY_METAL_TICKERS,
    MONTHLY_DAYS, QUARTERLY_DAYS, WEEKLY_DAYS,
    KMA_BAND_PCT, RMV_LOOKBACK, RMV_TIGHT_THRESHOLD,
)
from src.indicators import calculate_rsi, calculate_rmv, detect_inside_day, detect_vcp, calculate_rs_raw_score
from src.sector_mapper import index_return

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OHLCV fetch
# ---------------------------------------------------------------------------

def fetch_commodity_ohlcv(period: str = "1y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
    """
    Download 1-year daily OHLCV for all MCX_COMMODITY_TICKERS.
    Returns {display_name: DataFrame} — unavailable tickers are silently skipped.
    """
    ticker_to_name = {v: k for k, v in MCX_COMMODITY_TICKERS.items()}
    all_tickers = list(MCX_COMMODITY_TICKERS.values())
    logger.info("Fetching OHLCV for %d commodity tickers…", len(all_tickers))

    result: Dict[str, pd.DataFrame] = {}
    try:
        raw = yf.download(
            all_tickers,
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as exc:
        logger.error("Commodity bulk download failed: %s", exc)
        return result

    for ticker, name in ticker_to_name.items():
        try:
            if len(all_tickers) == 1:
                df = raw.copy()
            else:
                df = raw[ticker].copy() if ticker in raw.columns.get_level_values(0) else pd.DataFrame()

            df = df.dropna(how="all")
            if df.empty or len(df) < 20:
                logger.debug("Skipping %s (%s) — insufficient data (%d rows)", name, ticker, len(df))
                continue

            df.index = pd.to_datetime(df.index)
            result[name] = df
            logger.debug("Loaded %s: %d rows", name, len(df))
        except Exception as exc:
            logger.debug("Skipping %s (%s): %s", name, ticker, exc)

    logger.info("Loaded %d/%d commodity tickers successfully", len(result), len(all_tickers))
    return result


def save_commodity_ohlcv(data: Dict[str, pd.DataFrame]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(COMMODITY_OHLCV_FILE, "wb") as f:
        pickle.dump(data, f)


def load_commodity_ohlcv() -> Dict[str, pd.DataFrame]:
    try:
        with open(COMMODITY_OHLCV_FILE, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}


def save_commodity_results(results: pd.DataFrame) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(COMMODITY_RESULTS_FILE, "wb") as f:
        pickle.dump(results, f)


def load_commodity_results() -> Optional[pd.DataFrame]:
    try:
        with open(COMMODITY_RESULTS_FILE, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Benchmark fetch helpers
# ---------------------------------------------------------------------------

def _fetch_series(tickers: list[str]) -> Optional[pd.DataFrame]:
    """Try each ticker in order; return the first successful 1y daily download."""
    for ticker in tickers:
        try:
            df = yf.download(ticker, period="1y", interval="1d", auto_adjust=True, progress=False)
            df = df.dropna(how="all")
            if not df.empty and len(df) >= 20:
                df.index = pd.to_datetime(df.index)
                return df
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# RS Rating normalisation (same logic as stock / index screeners)
# ---------------------------------------------------------------------------

def _normalize_rs_ratings(raw_scores: Dict[str, float]) -> Dict[str, int]:
    if not raw_scores:
        return {}
    s = pd.Series(raw_scores)
    n = len(s)
    if n == 1:
        return {s.index[0]: 50}
    ranks = s.rank(method="average", ascending=True)
    ratings = (1.0 + 98.0 * (ranks - 1.0) / (n - 1.0)).round(0).clip(1, 99).astype(int)
    return ratings.to_dict()


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def _score_52w_high(close: pd.Series, high_52w: float) -> float:
    if high_52w <= 0:
        return 0.0
    pct_below = (high_52w - float(close.iloc[-1])) / high_52w * 100
    return max(0.0, 100.0 - pct_below * 2.0) / 100.0


def _score_outperf(ret_1m: float, bench_ret_1m: float) -> float:
    """Generic outperformance score: diff normalised over −20% to +20% range."""
    diff = ret_1m - bench_ret_1m
    return min(max((diff + 0.20) / 0.40, 0.0), 1.0)


def _score_rsi(rsi_val: float) -> float:
    if rsi_val < 55:
        return 0.0
    if rsi_val >= 80:
        return 1.0
    return (rsi_val - 55.0) / 25.0


def _score_rmv(rmv_val: float) -> float:
    if rmv_val <= 0:
        return 1.0
    threshold = RMV_TIGHT_THRESHOLD * 3
    return max(0.0, 1.0 - rmv_val / threshold)


def _score_kma(close: float, ema10: float) -> float:
    if ema10 <= 0:
        return 0.0
    pct = (close - ema10) / ema10
    if abs(pct) > KMA_BAND_PCT:
        return 0.0
    return 1.0 - abs(pct) / KMA_BAND_PCT


def _score_rs_trend(close: pd.Series, benchmark: pd.Series) -> float:
    if benchmark is None or len(benchmark) < 20:
        return 0.0
    try:
        common = close.index.intersection(benchmark.index)
        if len(common) < 20:
            return 0.0
        rs = close.loc[common] / benchmark.loc[common]
        ema10 = float(rs.ewm(span=10, adjust=False).mean().iloc[-1])
        ema20 = float(rs.ewm(span=20, adjust=False).mean().iloc[-1])
        return 1.0 if ema10 > ema20 else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Main screener
# ---------------------------------------------------------------------------

def run_commodity_screener(
    commodity_ohlcv: Dict[str, pd.DataFrame],
    nifty500_df: Optional[pd.DataFrame] = None,
    metal_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Score and rank MCX commodities.

    Hard filter : EMA10 > EMA20 > EMA50
    Soft scoring : COMMODITY_SCORE_WEIGHTS
    """
    if not commodity_ohlcv:
        return pd.DataFrame()

    # Fetch benchmarks if not supplied
    if nifty500_df is None or nifty500_df.empty:
        logger.info("Fetching Nifty 500 for commodity RS…")
        nifty500_df = _fetch_series(COMMODITY_NIFTY500_TICKERS)

    if metal_df is None or metal_df.empty:
        logger.info("Fetching Nifty Metal Index for commodity RS…")
        metal_df = _fetch_series(COMMODITY_METAL_TICKERS)

    def _close_series(df: Optional[pd.DataFrame]) -> Optional[pd.Series]:
        if df is None or df.empty:
            return None
        s = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
        s.index = pd.to_datetime(s.index)
        return s

    n500_close = _close_series(nifty500_df)
    metal_close = _close_series(metal_df)
    n500_ret_1m  = index_return(nifty500_df, MONTHLY_DAYS) if nifty500_df is not None else 0.0
    metal_ret_1m = index_return(metal_df,    MONTHLY_DAYS) if metal_df    is not None else 0.0

    # Pre-compute RS raw scores across ALL commodities so 1-99 rating reflects
    # the full universe, not just those that pass the EMA stack filter.
    rs_raw: Dict[str, float] = {}
    for name, df in commodity_ohlcv.items():
        try:
            rs_raw[name] = calculate_rs_raw_score(df, nifty500_df)
        except Exception:
            pass
    rs_ratings = _normalize_rs_ratings(rs_raw)

    weights = COMMODITY_SCORE_WEIGHTS
    rows = []

    for name, df in commodity_ohlcv.items():
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
            close = df["Close"] if "Close" in df.columns else df.iloc[:, 0]

            if len(close) < 50:
                continue

            ema10 = float(close.ewm(span=10, adjust=False).mean().iloc[-1])
            ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
            ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])

            # Hard filter
            if not (ema10 > ema20 > ema50):
                continue

            cmp      = float(close.iloc[-1])
            high_52w = float(close.tail(252).max())
            low_52w  = float(close.tail(252).min())

            ret_1w = (cmp / float(close.iloc[-WEEKLY_DAYS])   - 1) * 100 if len(close) > WEEKLY_DAYS   else 0.0
            ret_1m = (cmp / float(close.iloc[-MONTHLY_DAYS])  - 1) * 100 if len(close) > MONTHLY_DAYS  else 0.0
            ret_3m = (cmp / float(close.iloc[-QUARTERLY_DAYS])- 1) * 100 if len(close) > QUARTERLY_DAYS else 0.0

            rsi_series = calculate_rsi(close)
            rsi_val    = float(rsi_series.iloc[-1]) if not rsi_series.dropna().empty else 50.0

            has_high = "High" in df.columns
            has_low  = "Low"  in df.columns
            rmv_val  = calculate_rmv(df) if (has_high and has_low) else float("nan")
            if isinstance(rmv_val, float) and np.isnan(rmv_val):
                rmv_val = 50.0

            inside_day = detect_inside_day(df) if (has_high and has_low) else False
            vcp        = detect_vcp(df) if (has_high and has_low and "Volume" in df.columns) else False

            rs_trend_score = _score_rs_trend(close, n500_close) if n500_close is not None else 0.0

            s_52w   = _score_52w_high(close, high_52w)
            s_n500  = _score_outperf(ret_1m, n500_ret_1m)
            s_metal = _score_outperf(ret_1m, metal_ret_1m)
            s_rsi   = _score_rsi(rsi_val)
            s_kma   = _score_kma(cmp, ema10)
            s_rmv   = _score_rmv(rmv_val)

            score = (
                s_52w   * weights["52w_high_proximity"]
                + s_n500  * weights["nifty500_outperf"]
                + s_metal * weights["metal_index_outperf"]
                + s_rsi   * weights["rsi_score"]
                + s_kma   * weights["close_to_kma"]
                + s_rmv   * weights["rmv_score"]
                + rs_trend_score * weights["rs_trend"]
            )

            rows.append({
                "Commodity":          name,
                "Category":           MCX_COMMODITY_CATEGORY.get(name, "Other"),
                "Ticker":             MCX_COMMODITY_TICKERS.get(name, ""),
                "CMP":                round(cmp, 2),
                "52W High":           round(high_52w, 2),
                "52W Low":            round(low_52w, 2),
                "% from High":        round((cmp / high_52w - 1) * 100, 2) if high_52w else 0.0,
                "Weekly %":           round(ret_1w, 2),
                "Monthly %":          round(ret_1m, 2),
                "Quarterly %":        round(ret_3m, 2),
                "vs Nifty500 1M":     round(ret_1m - n500_ret_1m,  2),
                "vs Metal Index 1M":  round(ret_1m - metal_ret_1m, 2),
                "RS Rating":          int(rs_ratings.get(name, 50)),
                "EMA10":              round(ema10, 2),
                "EMA20":              round(ema20, 2),
                "EMA50":              round(ema50, 2),
                "RSI":                round(rsi_val, 1),
                "RMV":                round(rmv_val, 1),
                "VCP":                vcp,
                "Inside Day":         inside_day,
                "RS Trend":           bool(rs_trend_score > 0),
                "Score":              round(score, 2),
            })

        except Exception as exc:
            logger.debug("Skipping commodity '%s': %s", name, exc)

    if not rows:
        return pd.DataFrame()

    results = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
    results.index = results.index + 1
    logger.info(
        "Commodity screener: %d/%d passed EMA stack filter",
        len(results), len(commodity_ohlcv),
    )
    return results
