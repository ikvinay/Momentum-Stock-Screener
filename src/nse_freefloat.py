"""
NSE shareholding pattern API — fetches official free float % for NSE-listed stocks.

Endpoint:
  https://www.nseindia.com/api/corporate-share-holdings-master?index=equities&symbol={SYMBOL}

Returns a list of quarterly SEBI filings (newest first). Each entry contains:
  public_val    — public / free float %
  pr_and_prgrp  — promoter + promoter group %
  date          — quarter-end date

No authentication is required; a Referer header pointing to nseindia.com is sufficient.
The homepage is Akamai-protected (403 on datacenter IPs) but this API endpoint is not.

Rate limiting: 0.5 s between requests → 1 000 stocks ≈ 9 min.

Falls back to yfinance heldPercentInsiders for symbols that return [] from NSE (e.g.,
stocks with DVR history or very recent listings). Cache is refreshed quarterly (90 days)
since SEBI shareholding filings are submitted quarterly.
"""

import logging
import os
import pickle
import time
from typing import Dict, List, Optional

import requests

from config import FREEFLOAT_FILE, FREEFLOAT_CACHE_DAYS, DATA_DIR

logger = logging.getLogger(__name__)

NSE_SH_URL = (
    "https://www.nseindia.com/api/corporate-share-holdings-master"
    "?index=equities&symbol={symbol}"
)
REQUEST_DELAY      = 0.5   # seconds between NSE API calls
SLOW_REQUEST_SECS  = 10    # warn if a single request takes longer than this
CONSECUTIVE_WARN   = 10    # warn after this many consecutive NSE misses in a row


def _fmt_dur(seconds: float) -> str:
    """Format a duration as 'Xm Ys' for log readability."""
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

def _build_session() -> requests.Session:
    """Create a requests.Session with headers that satisfy NSE's basic checks."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Referer":        "https://www.nseindia.com/",
        "Accept":         "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
    })
    return s


# ---------------------------------------------------------------------------
# Single-ticker fetch
# ---------------------------------------------------------------------------

def fetch_freefloat(symbol: str, session: requests.Session) -> Optional[float]:
    """
    Fetch the most recent public (free float) shareholding % for one NSE symbol.

    Parameters
    ----------
    symbol : str
        NSE symbol WITHOUT the '.NS' suffix, e.g. 'RELIANCE'.
    session : requests.Session
        Pre-built session with NSE-compatible headers.

    Returns
    -------
    float or None
        Free float as a percentage (e.g. 49.7), or None on any failure / data gap.
    """
    url = NSE_SH_URL.format(symbol=symbol.upper())
    try:
        r = session.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or not data:
            return None
        # NSE returns filings newest-first; index 0 is the most recent quarter
        val = data[0].get("public_val")
        if val is None:
            return None
        return round(float(val), 1)
    except Exception as exc:
        logger.debug("NSE freefloat fetch failed for %s: %s", symbol, exc)
        return None


# ---------------------------------------------------------------------------
# Bulk fetch with fallback
# ---------------------------------------------------------------------------

def fetch_all_freefloat(
    yf_tickers: List[str],
    yf_info: Optional[Dict[str, dict]] = None,
) -> Dict[str, float]:
    """
    Fetch free float % for every ticker in the list.

    Primary source: NSE corporate shareholding API.
    Fallback:       yfinance heldPercentInsiders (already present in stock_info cache).

    Parameters
    ----------
    yf_tickers : list of str
        Tickers in yfinance format, e.g. ['RELIANCE.NS', 'INFY.NS'].
    yf_info : dict, optional
        stock_info dict keyed by yf ticker; used only as fallback when NSE returns [].

    Returns
    -------
    dict  {yf_ticker: free_float_pct}
        Only tickers where at least one source returned data are included.
    """
    yf_info = yf_info or {}
    session = _build_session()
    result:       Dict[str, float] = {}
    nse_count     = 0   # resolved via NSE API
    fallback_count = 0  # resolved via yfinance heldPercentInsiders
    no_data_count  = 0  # neither source had data
    consec_miss    = 0  # consecutive tickers with no NSE data

    total = len(yf_tickers)
    if total == 0:
        return result

    est_min = total * REQUEST_DELAY / 60
    logger.info(
        "Free float fetch START — %d tickers, est. %.0f min (0.5 s/ticker)",
        total, est_min,
    )
    t0 = time.time()

    for i, yf_ticker in enumerate(yf_tickers):
        nse_symbol = yf_ticker.replace(".NS", "")

        t_req  = time.time()
        ff_nse = fetch_freefloat(nse_symbol, session)
        req_secs = time.time() - t_req

        if req_secs > SLOW_REQUEST_SECS:
            logger.warning(
                "Slow NSE response — %s took %.1f s (possible throttling)",
                nse_symbol, req_secs,
            )

        if ff_nse is not None:
            result[yf_ticker] = ff_nse
            nse_count += 1
            consec_miss = 0
        else:
            consec_miss += 1
            if consec_miss == CONSECUTIVE_WARN:
                logger.warning(
                    "NSE API: %d consecutive tickers returned no data "
                    "(last: %s) — API may be throttling or down",
                    CONSECUTIVE_WARN, nse_symbol,
                )
            # Fallback: derive from yfinance insider-held %
            info = yf_info.get(yf_ticker, {})
            held = info.get("heldPercentInsiders")
            if held is not None:
                result[yf_ticker] = round((1.0 - float(held)) * 100, 1)
                fallback_count += 1
            else:
                no_data_count += 1

        # Progress log every 50 tickers
        done = i + 1
        if done % 50 == 0 or done == total:
            elapsed   = time.time() - t0
            rate      = done / elapsed  # tickers per second
            remaining = (total - done) / rate if rate > 0 else 0
            logger.info(
                "Free float [%d/%d] %.0f%% | elapsed %s | ETA %s | "
                "NSE=%d  fallback=%d  no-data=%d",
                done, total, done / total * 100,
                _fmt_dur(elapsed), _fmt_dur(remaining),
                nse_count, fallback_count, no_data_count,
            )

        # Sleep between requests but not after the last one
        if i < total - 1:
            time.sleep(REQUEST_DELAY)

    elapsed_total = time.time() - t0
    coverage = len(result) / total * 100 if total else 0
    logger.info(
        "Free float fetch COMPLETE in %s — "
        "%d/%d tickers covered (%.0f%%) | NSE=%d  fallback=%d  no-data=%d",
        _fmt_dur(elapsed_total),
        len(result), total, coverage,
        nse_count, fallback_count, no_data_count,
    )
    return result


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def is_freefloat_cache_fresh() -> bool:
    """True if the freefloat cache file exists and is less than FREEFLOAT_CACHE_DAYS old."""
    if not os.path.exists(FREEFLOAT_FILE):
        return False
    age_days = (time.time() - os.path.getmtime(FREEFLOAT_FILE)) / 86400
    return age_days < FREEFLOAT_CACHE_DAYS


def save_freefloat_cache(data: Dict[str, float]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(FREEFLOAT_FILE, "wb") as f:
        pickle.dump(data, f)


def load_freefloat_cache() -> Dict[str, float]:
    if os.path.exists(FREEFLOAT_FILE):
        try:
            with open(FREEFLOAT_FILE, "rb") as f:
                return pickle.load(f)
        except Exception as exc:
            logger.warning("freefloat_cache.pkl corrupted (%s) — will re-fetch", exc)
            os.remove(FREEFLOAT_FILE)
    return {}
