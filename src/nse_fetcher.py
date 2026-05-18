"""
Fetches the list of NSE tradable equity stocks from:
  https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv

Filters for SERIES = 'EQ' (regular equities only — excludes BE, BZ trade-to-trade).
Result is cached locally and refreshed every EQUITY_LIST_CACHE_DAYS days.
"""

import io
import logging
import os
import pickle
import time

import pandas as pd
import requests

from config import (
    NIFTY1000_FILE, DATA_DIR, EQUITY_LIST_URL, EQUITY_LIST_CACHE_DAYS,
    NSE_SECTOR_MAP_FILE, NSE_SECTOR_CACHE_DAYS, NSE_MACRO_TO_SECTOR,
)

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


# ---------------------------------------------------------------------------
# NSE Sector Map — fetches industry classification directly from NSE
# ---------------------------------------------------------------------------

_NSE_SESSION_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}


def _make_nse_session() -> requests.Session:
    """Create a requests.Session pre-warmed with NSE cookies."""
    session = requests.Session()
    session.headers.update(_NSE_SESSION_HEADERS)
    try:
        session.get("https://www.nseindia.com/", timeout=10)
    except Exception as exc:
        logger.debug(f"NSE warm-up request failed (non-fatal): {exc}")
    return session


def fetch_nse_sector_map(
    yf_tickers: list[str],
    force_refresh: bool = False,
    status_callback=None,
) -> dict[str, dict]:
    """
    Fetch NSE industry classification for the given yfinance tickers.

    Returns {yf_ticker: {"sector": str, "industry": str, "basic_industry": str}}.
    Results are cached for NSE_SECTOR_CACHE_DAYS days; only missing tickers
    trigger network calls.

    Parameters
    ----------
    yf_tickers:
        Tickers in yfinance format, e.g. ["RELIANCE.NS", "INFY.NS"].
    force_refresh:
        If True, bypass the cache and re-fetch every ticker.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # --- Load existing cache ---
    cache: dict[str, dict] = {}
    if not force_refresh and os.path.exists(NSE_SECTOR_MAP_FILE):
        age_days = (time.time() - os.path.getmtime(NSE_SECTOR_MAP_FILE)) / 86400
        if age_days < NSE_SECTOR_CACHE_DAYS:
            try:
                with open(NSE_SECTOR_MAP_FILE, "rb") as f:
                    cache = pickle.load(f)
                logger.info(
                    f"NSE sector map cache loaded: {len(cache)} entries ({age_days:.1f}d old)"
                )
            except Exception as exc:
                logger.warning(f"NSE sector map cache corrupted ({exc}) — re-fetching")
                cache = {}

    missing = [t for t in yf_tickers if t not in cache]
    if not missing:
        return cache

    logger.info(f"NSE sector map: fetching {len(missing)} missing tickers from NSE API")
    session = _make_nse_session()
    _SAVE_EVERY    = 50   # checkpoint frequency
    _SESSION_EVERY = 200  # re-warm session to prevent cookie expiry

    for i, yf_ticker in enumerate(missing):
        # Refresh session cookie periodically to avoid mid-run expiry
        if i > 0 and i % _SESSION_EVERY == 0:
            logger.info(f"NSE sector map: refreshing session at ticker {i}")
            session = _make_nse_session()

        symbol = yf_ticker.replace(".NS", "").upper()
        try:
            url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
            resp = session.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                info = data.get("industryInfo", {})
                macro = info.get("macro", "") or ""
                sector = NSE_MACRO_TO_SECTOR.get(macro, macro) or "Unknown"
                cache[yf_ticker] = {
                    "sector":         sector,
                    "industry":       info.get("industry", "") or "Unknown",
                    "basic_industry": info.get("basicIndustry", "") or "",
                }
            else:
                logger.debug(f"NSE API {resp.status_code} for {symbol}")
        except Exception as exc:
            logger.debug(f"NSE sector fetch failed for {symbol}: {exc}")

        done = i + 1
        if done % _SAVE_EVERY == 0:
            logger.info(f"NSE sector map: {done}/{len(missing)} fetched — saving checkpoint")
            try:
                with open(NSE_SECTOR_MAP_FILE, "wb") as f:
                    pickle.dump(cache, f)
            except Exception as exc:
                logger.warning(f"Checkpoint save failed: {exc}")
            if status_callback:
                status_callback(done, len(missing))

        time.sleep(0.8)

    # --- Final save ---
    try:
        with open(NSE_SECTOR_MAP_FILE, "wb") as f:
            pickle.dump(cache, f)
        logger.info(f"NSE sector map saved: {len(cache)} entries")
    except Exception as exc:
        logger.warning(f"Failed to save NSE sector map cache: {exc}")

    return cache
