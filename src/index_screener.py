"""
NSE Index screener — fetch OHLCV, score and rank all tracked NSE indices.

Scoring mirrors the stock screener but omits sector outperformance.
Relative Strength is measured against Nifty 500 (^CRSLDX / ^NSEI fallback).
"""

import logging
import pickle
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    DATA_DIR, INDEX_OHLCV_FILE, INDEX_RESULTS_FILE,
    NSE_INDEX_TICKERS, INDEX_SCORE_WEIGHTS,
    MONTHLY_DAYS, QUARTERLY_DAYS, WEEKLY_DAYS,
    KMA_BAND_PCT, RMV_LOOKBACK, RMV_TIGHT_THRESHOLD,
)
from src.indicators import (
    calculate_rsi, calculate_rmv,
    detect_vcp, detect_inside_day,
    calculate_rs_raw_score,
)
from src.sector_mapper import index_return

logger = logging.getLogger(__name__)

_NIFTY500_FALLBACK = ["^CRSLDX", "^NSEI"]


# ---------------------------------------------------------------------------
# OHLCV fetch
# ---------------------------------------------------------------------------

def fetch_index_ohlcv(period: str = "1y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
    """
    Download 1-year daily OHLCV for every ticker in NSE_INDEX_TICKERS.
    Returns {display_name: DataFrame} — indices with no data are skipped.
    """
    # Flatten all tickers to a reverse-lookup map: ticker → display_name
    ticker_to_name: Dict[str, str] = {}
    for _cat, entries in NSE_INDEX_TICKERS.items():
        for name, ticker in entries.items():
            ticker_to_name[ticker] = name

    all_tickers = list(ticker_to_name.keys())
    logger.info("Fetching OHLCV for %d NSE index tickers…", len(all_tickers))

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
        logger.error("yfinance bulk download failed: %s", exc)
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

    logger.info("Loaded %d/%d NSE indices successfully", len(result), len(all_tickers))
    return result


def save_index_ohlcv(data: Dict[str, pd.DataFrame]) -> None:
    import os
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(INDEX_OHLCV_FILE, "wb") as f:
        pickle.dump(data, f)


def load_index_ohlcv() -> Dict[str, pd.DataFrame]:
    try:
        with open(INDEX_OHLCV_FILE, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _fetch_nifty500() -> Optional[pd.DataFrame]:
    for ticker in _NIFTY500_FALLBACK:
        try:
            df = yf.download(ticker, period="1y", interval="1d", auto_adjust=True, progress=False)
            df = df.dropna(how="all")
            if not df.empty and len(df) >= 20:
                df.index = pd.to_datetime(df.index)
                return df
        except Exception:
            pass
    return None


def _score_52w_high(close: pd.Series, high_52w: float) -> float:
    if high_52w <= 0:
        return 0.0
    pct_below = (high_52w - float(close.iloc[-1])) / high_52w * 100
    return max(0.0, 100.0 - pct_below * 2.0) / 100.0


def _score_nifty500_outperf(ret_1m: float, n500_ret_1m: float) -> float:
    diff = ret_1m - n500_ret_1m
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
    band = KMA_BAND_PCT
    if abs(pct) > band:
        return 0.0
    return 1.0 - abs(pct) / band


def _score_rs_trend(close: pd.Series, nifty500: pd.Series) -> float:
    if nifty500 is None or len(nifty500) < 20:
        return 0.0
    try:
        common = close.index.intersection(nifty500.index)
        if len(common) < 20:
            return 0.0
        rs = close.loc[common] / nifty500.loc[common]
        ema10 = float(rs.ewm(span=10, adjust=False).mean().iloc[-1])
        ema20 = float(rs.ewm(span=20, adjust=False).mean().iloc[-1])
        return 1.0 if ema10 > ema20 else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Category lookup
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


def _index_category(name: str) -> str:
    for cat, entries in NSE_INDEX_TICKERS.items():
        if name in entries:
            return cat
    return "Other"


def _index_ticker(name: str) -> str:
    for _cat, entries in NSE_INDEX_TICKERS.items():
        if name in entries:
            return entries[name]
    return ""


# ---------------------------------------------------------------------------
# Main screener
# ---------------------------------------------------------------------------

def run_index_screener(
    index_ohlcv: Dict[str, pd.DataFrame],
    nifty500_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Score and rank all loaded NSE indices.

    Hard filters:
    - EMA10 > EMA20 > EMA50 (all must hold)

    Soft scoring (INDEX_SCORE_WEIGHTS):
    - 52w high proximity
    - Nifty 500 1M outperformance
    - KMA (close to 10 EMA)
    - RSI 55–80 band
    - RMV (lower = better)
    - RS trend (RS line EMA10 > EMA20)
    """
    if nifty500_df is None or nifty500_df.empty:
        logger.warning("Nifty 500 data not supplied — fetching…")
        nifty500_df = _fetch_nifty500()

    n500_close: Optional[pd.Series] = None
    n500_ret_1m = 0.0
    if nifty500_df is not None and not nifty500_df.empty:
        n500_close = nifty500_df["Close"] if "Close" in nifty500_df.columns else nifty500_df.iloc[:, 0]
        n500_close.index = pd.to_datetime(n500_close.index)
        n500_ret_1m = index_return(nifty500_df, MONTHLY_DAYS)

    # Pre-compute RS raw scores across ALL indices so the 1-99 rating reflects
    # the full universe, not just those that pass the EMA stack filter.
    rs_raw: Dict[str, float] = {}
    for name, df in index_ohlcv.items():
        try:
            rs_raw[name] = calculate_rs_raw_score(df, nifty500_df)
        except Exception:
            pass
    rs_ratings = _normalize_rs_ratings(rs_raw)

    weights = INDEX_SCORE_WEIGHTS
    rows = []

    for name, df in index_ohlcv.items():
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
            close = df["Close"] if "Close" in df.columns else df.iloc[:, 0]

            if len(close) < 50:
                continue

            # EMAs
            ema10 = float(close.ewm(span=10, adjust=False).mean().iloc[-1])
            ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
            ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])

            # Hard filter: EMA stack 10 > 20 > 50
            if not (ema10 > ema20 > ema50):
                continue

            cmp = float(close.iloc[-1])
            high_52w = float(close.tail(252).max())
            low_52w = float(close.tail(252).min())

            ret_1w = (cmp / float(close.iloc[-WEEKLY_DAYS]) - 1) * 100 if len(close) > WEEKLY_DAYS else 0.0
            ret_1m = (cmp / float(close.iloc[-MONTHLY_DAYS]) - 1) * 100 if len(close) > MONTHLY_DAYS else 0.0
            ret_3m = (cmp / float(close.iloc[-QUARTERLY_DAYS]) - 1) * 100 if len(close) > QUARTERLY_DAYS else 0.0

            rsi_series = calculate_rsi(close)
            rsi_val = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0
            has_high = "High" in df.columns
            has_low  = "Low" in df.columns
            rmv_val = calculate_rmv(df) if (has_high and has_low) else float("nan")
            if isinstance(rmv_val, float) and (rmv_val != rmv_val):  # nan
                rmv_val = 50.0

            # Inside Day detection
            inside_day = detect_inside_day(df) if (has_high and has_low) else False

            # VCP
            vcp = detect_vcp(df) if (has_high and has_low and "Volume" in df.columns) else False

            # RS trend vs Nifty 500
            rs_trend_score = _score_rs_trend(close, n500_close) if n500_close is not None else 0.0

            # Composite score
            s_52w  = _score_52w_high(close, high_52w)
            s_n500 = _score_nifty500_outperf(ret_1m, n500_ret_1m)
            s_kma  = _score_kma(cmp, ema10)
            s_rsi  = _score_rsi(rsi_val)
            s_rmv  = _score_rmv(rmv_val)

            score = (
                s_52w  * weights["52w_high_proximity"]
                + s_n500 * weights["nifty500_outperf"]
                + s_kma  * weights["close_to_kma"]
                + s_rsi  * weights["rsi_score"]
                + s_rmv  * weights["rmv_score"]
                + rs_trend_score * weights["rs_trend"]
            )

            rows.append({
                "Index":          name,
                "Category":       _index_category(name),
                "Ticker":         _index_ticker(name),
                "CMP":            round(cmp, 2),
                "52W High":       round(high_52w, 2),
                "52W Low":        round(low_52w, 2),
                "% from High":    round((cmp / high_52w - 1) * 100, 2) if high_52w else 0.0,
                "Weekly %":       round(ret_1w, 2),
                "Monthly %":      round(ret_1m, 2),
                "Quarterly %":    round(ret_3m, 2),
                "vs Nifty500 1M": round(ret_1m - n500_ret_1m, 2),
                "RS Rating":      int(rs_ratings.get(name, 50)),
                "EMA10":          round(ema10, 2),
                "EMA20":          round(ema20, 2),
                "EMA50":          round(ema50, 2),
                "RSI":            round(rsi_val, 1),
                "RMV":            round(rmv_val, 1),
                "VCP":            vcp,
                "Inside Day":     inside_day,
                "RS Trend":       bool(rs_trend_score > 0),
                "Score":          round(score, 2),
            })

        except Exception as exc:
            logger.debug("Skipping index '%s': %s", name, exc)

    if not rows:
        return pd.DataFrame()

    results = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
    results.index = results.index + 1
    logger.info("Index screener: %d/%d indices passed EMA stack filter", len(results), len(index_ohlcv))
    return results


def save_index_results(results: pd.DataFrame) -> None:
    import os
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(INDEX_RESULTS_FILE, "wb") as f:
        pickle.dump(results, f)


def load_index_results() -> Optional[pd.DataFrame]:
    try:
        with open(INDEX_RESULTS_FILE, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None
