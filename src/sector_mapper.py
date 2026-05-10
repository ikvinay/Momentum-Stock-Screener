"""
Maps yfinance sector names to NSE sector index tickers and fetches index price data.
"""

import logging
import os
import pickle
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from config import (
    SECTOR_INDEX_MAP, DEFAULT_SECTOR_INDEX,
    BENCHMARK_TICKERS, NIFTY500_TICKERS, INDEX_DATA_FILE, DATA_DIR,
    MONTHLY_DAYS, WEEKLY_DAYS,
)

logger = logging.getLogger(__name__)


def resolve_sector_index(sector: str) -> str:
    """
    Map a yfinance sector string to the best-matching NSE index ticker.
    Falls back to DEFAULT_SECTOR_INDEX if no match found.
    """
    sector_lower = sector.lower()
    for fragment, ticker in SECTOR_INDEX_MAP.items():
        if fragment in sector_lower:
            return ticker
    return DEFAULT_SECTOR_INDEX


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse MultiIndex columns returned by newer yfinance to a flat level."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _download_index(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
    try:
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
        df = _flatten_columns(df)
        if df is not None and len(df) > 20:
            return df
    except Exception as exc:
        logger.debug(f"Index download failed for {ticker}: {exc}")
    return None


def fetch_benchmark(period: str = "1y") -> Optional[pd.DataFrame]:
    """Try each benchmark ticker in order and return the first that works."""
    for ticker in BENCHMARK_TICKERS:
        df = _download_index(ticker, period)
        if df is not None:
            logger.info(f"Benchmark: using {ticker}")
            return df
    logger.warning("Could not fetch any benchmark index")
    return None


def fetch_nifty500(period: str = "1y") -> Optional[pd.DataFrame]:
    """Fetch Nifty 500 index data used for sector outperformance ordering."""
    for ticker in NIFTY500_TICKERS:
        df = _download_index(ticker, period)
        if df is not None:
            logger.info(f"Nifty 500: using {ticker}")
            return df
    logger.warning("Could not fetch Nifty 500 index")
    return None


def fetch_all_sector_indices(period: str = "1y") -> Dict[str, pd.DataFrame]:
    """
    Download price data for every unique NSE sector index used in the sector map,
    plus the default fallback index.
    """
    unique_tickers = set(SECTOR_INDEX_MAP.values()) | {DEFAULT_SECTOR_INDEX}
    index_data: Dict[str, pd.DataFrame] = {}

    for ticker in unique_tickers:
        df = _download_index(ticker, period)
        if df is not None:
            index_data[ticker] = df
            logger.info(f"Sector index fetched: {ticker} ({len(df)} rows)")
        else:
            logger.warning(f"Sector index unavailable: {ticker}")

    return index_data


def save_index_data(
    benchmark: Optional[pd.DataFrame],
    sector_indices: Dict[str, pd.DataFrame],
    nifty500: Optional[pd.DataFrame] = None,
) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    payload = {
        "benchmark":     benchmark,
        "sector_indices": sector_indices,
        "nifty500":      nifty500,
    }
    with open(INDEX_DATA_FILE, "wb") as f:
        pickle.dump(payload, f)


def load_index_data() -> tuple[Optional[pd.DataFrame], Dict[str, pd.DataFrame], Optional[pd.DataFrame]]:
    """Returns (benchmark, sector_indices, nifty500)."""
    if os.path.exists(INDEX_DATA_FILE):
        try:
            with open(INDEX_DATA_FILE, "rb") as f:
                payload = pickle.load(f)
            return (
                payload.get("benchmark"),
                payload.get("sector_indices", {}),
                payload.get("nifty500"),
            )
        except Exception as exc:
            logger.warning(f"index_data.pkl corrupted ({exc}) — please re-run Fetch Data")
            os.remove(INDEX_DATA_FILE)
    return None, {}, None


def is_index_data_fresh() -> bool:
    """Return True if index data was fetched today (within 20 hours)."""
    if not os.path.exists(INDEX_DATA_FILE):
        return False
    import time
    age_hours = (time.time() - os.path.getmtime(INDEX_DATA_FILE)) / 3600
    return age_hours < 20


def index_return(df: Optional[pd.DataFrame], days: int) -> float:
    """Return percentage return of an index over the last `days` trading days."""
    if df is None or len(df) < days:
        return 0.0
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    v_end = close.iloc[-1]
    v_start = close.iloc[-days]
    if isinstance(v_end, pd.Series):
        v_end = v_end.iloc[0]
    if isinstance(v_start, pd.Series):
        v_start = v_start.iloc[0]
    return float((v_end / v_start - 1) * 100)
