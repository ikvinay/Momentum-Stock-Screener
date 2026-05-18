"""
Downloads historical price data and stock metadata from yfinance.

Price data: batched yf.download calls (50 tickers at a time)
Stock info: parallel ThreadPoolExecutor fetches (PE, sector, industry, market cap)
"""

import os
import time
import pickle
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from config import (
    PRICE_DATA_FILE, STOCK_INFO_FILE, DATA_DIR,
    BATCH_SIZE, BATCH_DELAY_SECONDS, INFO_MAX_WORKERS,
    STOCK_INFO_CACHE_DAYS, NSE_INDUSTRY_TO_SECTOR,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Price data
# ---------------------------------------------------------------------------

def fetch_price_data(
    tickers: List[str],
    period: str = "1y",
) -> Dict[str, pd.DataFrame]:
    """
    Download 1-year daily OHLCV for all tickers in batches.
    Returns {ticker: DataFrame} for tickers with sufficient history (≥50 rows).
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    all_data: Dict[str, pd.DataFrame] = {}
    batches = [tickers[i : i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]

    for idx, batch in enumerate(batches):
        logger.info(f"Price batch {idx + 1}/{len(batches)} — {len(batch)} tickers")
        try:
            raw = yf.download(
                batch,
                period=period,
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            for ticker in batch:
                try:
                    if len(batch) == 1:
                        df = raw.copy()
                    else:
                        df = raw[ticker].copy()
                    # Flatten MultiIndex columns produced by newer yfinance versions
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df = df.dropna(how="all")
                    if len(df) >= 50:
                        all_data[ticker] = df
                except Exception as exc:
                    logger.debug(f"No price data for {ticker}: {exc}")
        except Exception as exc:
            logger.error(f"Batch download error: {exc}")

        if idx < len(batches) - 1:
            time.sleep(BATCH_DELAY_SECONDS)

    logger.info(f"Price data fetched for {len(all_data)}/{len(tickers)} tickers")
    return all_data


def save_price_data(data: Dict[str, pd.DataFrame]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(PRICE_DATA_FILE, "wb") as f:
        pickle.dump(data, f)


def load_price_data() -> Optional[Dict[str, pd.DataFrame]]:
    if os.path.exists(PRICE_DATA_FILE):
        try:
            with open(PRICE_DATA_FILE, "rb") as f:
                return pickle.load(f)
        except Exception as exc:
            logger.warning(f"price_data.pkl corrupted ({exc}) — please re-run Fetch Data")
            os.remove(PRICE_DATA_FILE)
    return None


# ---------------------------------------------------------------------------
# Stock info (PE, sector, industry, market cap)
# ---------------------------------------------------------------------------

def _fetch_single_info(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        return {
            "symbol":          ticker,
            "company_name":    info.get("longName") or ticker.replace(".NS", ""),
            "sector":          info.get("sector") or "Unknown",
            "industry":        info.get("industry") or "Unknown",
            "trailing_pe":     info.get("trailingPE"),
            "market_cap":      info.get("marketCap"),           # in INR
            # Kept as fallback for the NSE freefloat module when NSE API returns []
            "heldPercentInsiders": info.get("heldPercentInsiders"),
        }
    except Exception as exc:
        logger.debug(f"Info fetch failed for {ticker}: {exc}")
        return {
            "symbol":              ticker,
            "company_name":        ticker.replace(".NS", ""),
            "sector":              "Unknown",
            "industry":            "Unknown",
            "trailing_pe":         None,
            "market_cap":          None,
            "heldPercentInsiders": None,
        }


def fetch_stock_info(tickers: List[str]) -> Dict[str, dict]:
    """
    Fetch PE ratio, sector, industry, market cap for all tickers in parallel.
    """
    results: Dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=INFO_MAX_WORKERS) as executor:
        futures = {executor.submit(_fetch_single_info, t): t for t in tickers}
        for i, future in enumerate(as_completed(futures)):
            info = future.result()
            results[info["symbol"]] = info
            if (i + 1) % 100 == 0:
                logger.info(f"Info fetched: {i + 1}/{len(tickers)}")
    logger.info(f"Stock info fetched for {len(results)} tickers")
    return results


def save_stock_info(data: Dict[str, dict]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(STOCK_INFO_FILE, "wb") as f:
        pickle.dump(data, f)


def load_stock_info() -> Optional[Dict[str, dict]]:
    if os.path.exists(STOCK_INFO_FILE):
        try:
            with open(STOCK_INFO_FILE, "rb") as f:
                return pickle.load(f)
        except Exception as exc:
            logger.warning(f"stock_info.pkl corrupted ({exc}) — please re-run Fetch Data")
            os.remove(STOCK_INFO_FILE)
    return None


def enrich_with_nse_sectors(
    stock_info: Dict[str, dict],
    nse_map: Dict[str, dict],
) -> Dict[str, dict]:
    """
    Override "Unknown" sector/industry in stock_info with data from the NSE sector map.

    Only replaces values that are missing or explicitly "Unknown"; confirmed
    yfinance values are kept as-is.

    Parameters
    ----------
    stock_info:
        {yf_ticker: info_dict} as returned by fetch_stock_info / load_stock_info.
    nse_map:
        {yf_ticker: {"sector", "industry", "basic_industry"}} from fetch_nse_sector_map.

    Returns
    -------
    The same stock_info dict (modified in place) with NSE data filled in.
    """
    _UNKNOWN = {"Unknown", None, ""}
    enriched_sector = 0
    enriched_industry = 0

    # Pass 1 — apply NSE API values for tickers present in nse_map
    for ticker, info in stock_info.items():
        nse = nse_map.get(ticker)
        if not nse:
            continue
        nse_industry = nse.get("industry") or ""
        if nse_industry not in _UNKNOWN:
            if info.get("industry") != nse_industry:
                enriched_industry += 1
            info["industry"] = nse_industry

    # Pass 2 — derive sector from industry for ALL stocks using NSE_INDUSTRY_TO_SECTOR.
    # This also catches stocks whose industry came from yfinance but matches the mapping.
    for ticker, info in stock_info.items():
        industry = info.get("industry") or ""
        if industry not in _UNKNOWN:
            mapped_sector = NSE_INDUSTRY_TO_SECTOR.get(industry)
            if mapped_sector and info.get("sector") != mapped_sector:
                info["sector"] = mapped_sector
                enriched_sector += 1

    logger.info(
        f"NSE sector enrichment: {enriched_sector} sectors, {enriched_industry} industries updated"
    )
    return stock_info


def is_info_cache_fresh() -> bool:
    """Return True if stock info cache exists and is within STOCK_INFO_CACHE_DAYS."""
    if not os.path.exists(STOCK_INFO_FILE):
        return False
    age_days = (time.time() - os.path.getmtime(STOCK_INFO_FILE)) / 86400
    return age_days < STOCK_INFO_CACHE_DAYS


def is_price_data_fresh() -> bool:
    """Return True if price data was fetched today (within 20 hours — post market-close window)."""
    if not os.path.exists(PRICE_DATA_FILE):
        return False
    age_hours = (time.time() - os.path.getmtime(PRICE_DATA_FILE)) / 3600
    return age_hours < 20


def price_data_last_fetched() -> str:
    """Human-readable age of the price data cache."""
    if not os.path.exists(PRICE_DATA_FILE):
        return "Never"
    age_mins = (time.time() - os.path.getmtime(PRICE_DATA_FILE)) / 60
    if age_mins < 60:
        return f"{int(age_mins)} min ago"
    age_hrs = age_mins / 60
    if age_hrs < 24:
        return f"{age_hrs:.1f} hr ago"
    return f"{age_hrs / 24:.1f} days ago"
