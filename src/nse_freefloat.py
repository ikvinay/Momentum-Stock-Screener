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

Rate limiting: FREEFLOAT_MAX_WORKERS parallel threads, each sleeping REQUEST_DELAY between
its own requests.  With the default of 3 workers × 0.5 s delay, 1 000 stocks take ~3 min
(vs ~9 min single-threaded).

Falls back to yfinance heldPercentInsiders for symbols that return [] from NSE (e.g.,
stocks with DVR history or very recent listings). Cache is refreshed weekly; SEBI
shareholding filings are submitted quarterly so weekly is more than adequate.
"""

import logging
import os
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import requests

from config import FREEFLOAT_FILE, FREEFLOAT_CACHE_DAYS, DATA_DIR, FREEFLOAT_MAX_WORKERS

logger = logging.getLogger(__name__)

NSE_SH_URL = (
    "https://www.nseindia.com/api/corporate-share-holdings-master"
    "?index=equities&symbol={symbol}"
)
REQUEST_DELAY     = 0.5   # seconds each worker sleeps between its own requests
SLOW_REQUEST_SECS = 10    # warn if a single request takes longer than this
HIGH_MISS_PCT     = 0.30  # warn when NSE miss rate exceeds this fraction of completed tickers


def _fmt_dur(seconds: float) -> str:
    """Format a duration as 'Xm Ys' for log readability."""
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


# ---------------------------------------------------------------------------
# Session — one per thread via threading.local
# ---------------------------------------------------------------------------

_thread_local = threading.local()


def _build_session() -> requests.Session:
    """Create a requests.Session with headers that satisfy NSE's basic checks."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Referer":         "https://www.nseindia.com/",
        "Accept":          "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
    })
    return s


def _get_thread_session() -> requests.Session:
    """Return this thread's requests.Session, creating it on first use."""
    if not hasattr(_thread_local, "session"):
        _thread_local.session = _build_session()
    return _thread_local.session


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
        r = session.get(url, timeout=5)
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
# Bulk fetch — multithreaded with per-thread rate limiting
# ---------------------------------------------------------------------------

def fetch_all_freefloat(
    yf_tickers: List[str],
    yf_info: Optional[Dict[str, dict]] = None,
    max_workers: int = FREEFLOAT_MAX_WORKERS,
) -> Dict[str, float]:
    """
    Fetch free float % for every ticker in the list using a thread pool.

    Each worker thread maintains its own requests.Session and sleeps REQUEST_DELAY
    seconds between its own requests, keeping each connection's rate conservative
    while achieving ~max_workers× overall throughput vs single-threaded.

    Primary source: NSE corporate shareholding API.
    Fallback:       yfinance heldPercentInsiders (from stock_info cache).

    Parameters
    ----------
    yf_tickers : list of str
        Tickers in yfinance format, e.g. ['RELIANCE.NS', 'INFY.NS'].
    yf_info : dict, optional
        stock_info dict keyed by yf ticker; used only as fallback when NSE returns [].
    max_workers : int
        Number of parallel fetch threads (default: FREEFLOAT_MAX_WORKERS from config).

    Returns
    -------
    dict  {yf_ticker: free_float_pct}
        Only tickers where at least one source returned data are included.
    """
    yf_info = yf_info or {}

    total = len(yf_tickers)
    if total == 0:
        return {}

    # With N workers each sleeping REQUEST_DELAY between their own requests the
    # effective wall-clock time is roughly total * REQUEST_DELAY / max_workers.
    est_min = total * REQUEST_DELAY / 60 / max_workers
    logger.info(
        "Free float fetch START — %d tickers, %d workers, est. %.0f min",
        total, max_workers, est_min,
    )
    t0 = time.time()

    result: Dict[str, float] = {}
    lock = threading.Lock()
    # Shared counters — always accessed under lock
    counters = {"nse": 0, "fallback": 0, "no_data": 0, "done": 0}

    def _fetch_one(yf_ticker: str) -> tuple[str, Optional[float], float]:
        """Worker: fetch one ticker, sleep, return (ticker, value_or_None, req_secs)."""
        nse_symbol = yf_ticker.replace(".NS", "")
        session    = _get_thread_session()
        t_req      = time.time()
        ff_nse     = fetch_freefloat(nse_symbol, session)
        req_secs   = time.time() - t_req
        time.sleep(REQUEST_DELAY)   # per-thread rate limiting
        return yf_ticker, ff_nse, req_secs

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ff_worker") as executor:
        futures = {executor.submit(_fetch_one, t): t for t in yf_tickers}

        for future in as_completed(futures):
            try:
                yf_ticker, ff_nse, req_secs = future.result()
            except Exception as exc:
                yf_ticker = futures[future]
                logger.debug("Worker exception for %s: %s", yf_ticker, exc)
                ff_nse, req_secs = None, 0.0

            if req_secs > SLOW_REQUEST_SECS:
                logger.warning(
                    "Slow NSE response — %s took %.1f s (possible throttling)",
                    yf_ticker.replace(".NS", ""), req_secs,
                )

            with lock:
                if ff_nse is not None:
                    result[yf_ticker] = ff_nse
                    counters["nse"] += 1
                else:
                    info = yf_info.get(yf_ticker, {})
                    held = info.get("heldPercentInsiders")
                    if held is not None:
                        result[yf_ticker] = round((1.0 - float(held)) * 100, 1)
                        counters["fallback"] += 1
                    else:
                        counters["no_data"] += 1

                counters["done"] += 1
                done = counters["done"]

                # Warn when NSE miss rate is high — likely throttling
                nse_misses = counters["fallback"] + counters["no_data"]
                if done >= 50 and nse_misses / done > HIGH_MISS_PCT and nse_misses % 50 == 0:
                    logger.warning(
                        "NSE API miss rate high: %d/%d (%.0f%%) — possible throttling",
                        nse_misses, done, nse_misses / done * 100,
                    )

                # Progress log every 50 completions
                if done % 50 == 0 or done == total:
                    elapsed   = time.time() - t0
                    rate      = done / elapsed if elapsed > 0 else 1
                    remaining = (total - done) / rate if rate > 0 else 0
                    logger.info(
                        "Free float [%d/%d] %.0f%% | elapsed %s | ETA %s | "
                        "NSE=%d  fallback=%d  no-data=%d",
                        done, total, done / total * 100,
                        _fmt_dur(elapsed), _fmt_dur(remaining),
                        counters["nse"], counters["fallback"], counters["no_data"],
                    )

    elapsed_total = time.time() - t0
    coverage = len(result) / total * 100 if total else 0
    logger.info(
        "Free float fetch COMPLETE in %s — "
        "%d/%d tickers covered (%.0f%%) | NSE=%d  fallback=%d  no-data=%d",
        _fmt_dur(elapsed_total),
        len(result), total, coverage,
        counters["nse"], counters["fallback"], counters["no_data"],
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
