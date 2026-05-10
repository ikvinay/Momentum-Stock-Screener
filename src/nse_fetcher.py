"""
Fetches the list of NSE tradable equity stocks from:
  https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv

Filters for SERIES = 'EQ' (regular equities only — excludes BE, BZ trade-to-trade).
Result is cached locally and refreshed every EQUITY_LIST_CACHE_DAYS days.
"""

import io
import logging
import os
import time

import pandas as pd
import requests

from config import NIFTY1000_FILE, DATA_DIR, EQUITY_LIST_URL, EQUITY_LIST_CACHE_DAYS

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}


def _download_equity_list() -> pd.DataFrame | None:
    """Download EQUITY_L.csv from NSE archives and return filtered DataFrame."""
    try:
        resp = requests.get(EQUITY_LIST_URL, headers=_HEADERS, timeout=60)
        resp.raise_for_status()

        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = [c.strip() for c in df.columns]

        # Normalise values
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.strip()

        # Keep only regular equity series
        df = df[df["SERIES"] == "EQ"].copy()

        df = df.rename(columns={
            "SYMBOL":          "symbol",
            "NAME OF COMPANY": "company_name",
            "ISIN NUMBER":     "isin",
        })

        df["yf_ticker"] = df["symbol"] + ".NS"
        df = df[["symbol", "company_name", "isin", "yf_ticker"]].drop_duplicates("symbol").reset_index(drop=True)

        logger.info(f"Downloaded {len(df)} EQ-series stocks from NSE archives")
        return df

    except Exception as exc:
        logger.warning(f"Failed to download equity list: {exc}")
        return None


def fetch_nifty1000_stocks(force_refresh: bool = False) -> pd.DataFrame:
    """
    Return a DataFrame with columns: symbol, company_name, isin, yf_ticker.

    Source: NSE EQUITY_L.csv filtered to SERIES = 'EQ'.
    Cached locally; refreshed automatically after EQUITY_LIST_CACHE_DAYS days.
    Set force_refresh=True to bypass the cache.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if not force_refresh and os.path.exists(NIFTY1000_FILE):
        age_days = (time.time() - os.path.getmtime(NIFTY1000_FILE)) / 86400
        if age_days < EQUITY_LIST_CACHE_DAYS:
            df = pd.read_csv(NIFTY1000_FILE)
            logger.info(f"Loaded {len(df)} stocks from cache ({age_days:.1f}d old)")
            return df

    df = _download_equity_list()

    if df is not None and len(df) > 0:
        df.to_csv(NIFTY1000_FILE, index=False)
        logger.info(f"Saved {len(df)} stocks to {NIFTY1000_FILE}")
        return df

    # Fallback: use stale cache if live download failed
    if os.path.exists(NIFTY1000_FILE):
        logger.warning("Live download failed — using stale cache")
        return pd.read_csv(NIFTY1000_FILE)

    raise RuntimeError(
        f"Could not download NSE equity list from {EQUITY_LIST_URL}. "
        "Check your internet connection or place a valid 'data/nse_equity_list.csv' manually."
    )
