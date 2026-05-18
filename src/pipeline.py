"""
Data pipeline — fetch, screener, sentiment orchestration.

All network and compute stages live here so they can be called from:
  - app.py          (scheduler + background threads)
  - pages/*.py      (sidebar buttons)
  - refresh.py      (CLI)

Status is persisted to data/refresh_status.json so the UI can poll it
across page navigations without shared in-process state.
"""

import json
import logging
import os
import pickle
from typing import Optional

import pandas as pd

from config import (
    DATA_DIR, SCREENER_RESULTS_FILE, LAST_UPDATED_FILE, REFRESH_STATUS_FILE,
    IPO_RESULTS_FILE, IPO_IB_RESULTS_FILE, INDEX_RESULTS_FILE,
    COMMODITY_RESULTS_FILE,
    MONTHLY_DAYS, QUARTERLY_DAYS,
    FREEFLOAT_FILE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status helpers — file-based so they survive page navigations
# ---------------------------------------------------------------------------

def write_status(stage: str, state: str, message: str) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    current = read_status()
    current[stage] = {"state": state, "message": message}
    with open(REFRESH_STATUS_FILE, "w") as f:
        json.dump(current, f)


def read_status() -> dict:
    if os.path.exists(REFRESH_STATUS_FILE):
        try:
            with open(REFRESH_STATUS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "fetch":     {"state": "idle", "message": ""},
        "screener":  {"state": "idle", "message": ""},
        "freefloat": {"state": "idle", "message": ""},
    }


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_results() -> Optional[pd.DataFrame]:
    if os.path.exists(SCREENER_RESULTS_FILE):
        try:
            with open(SCREENER_RESULTS_FILE, "rb") as f:
                return pickle.load(f)
        except Exception as exc:
            logger.warning("screener_results.pkl corrupted (%s) — re-run screener", exc)
            os.remove(SCREENER_RESULTS_FILE)
    return None


def load_ipo_results() -> Optional[pd.DataFrame]:
    if os.path.exists(IPO_RESULTS_FILE):
        try:
            with open(IPO_RESULTS_FILE, "rb") as f:
                return pickle.load(f)
        except Exception as exc:
            logger.warning("ipo_results.pkl corrupted (%s) — re-run screener", exc)
            os.remove(IPO_RESULTS_FILE)
    return None


def load_ipo_ib_results() -> Optional[pd.DataFrame]:
    if os.path.exists(IPO_IB_RESULTS_FILE):
        try:
            with open(IPO_IB_RESULTS_FILE, "rb") as f:
                return pickle.load(f)
        except Exception as exc:
            logger.warning("ipo_ib_results.pkl corrupted (%s) — re-run screener", exc)
            os.remove(IPO_IB_RESULTS_FILE)
    return None


def load_commodity_results() -> Optional[pd.DataFrame]:
    if os.path.exists(COMMODITY_RESULTS_FILE):
        try:
            with open(COMMODITY_RESULTS_FILE, "rb") as f:
                return pickle.load(f)
        except Exception as exc:
            logger.warning("commodity_results.pkl corrupted (%s) — re-run screener", exc)
            os.remove(COMMODITY_RESULTS_FILE)
    return None


def load_index_results() -> Optional[pd.DataFrame]:
    if os.path.exists(INDEX_RESULTS_FILE):
        try:
            with open(INDEX_RESULTS_FILE, "rb") as f:
                return pickle.load(f)
        except Exception as exc:
            logger.warning("index_results.pkl corrupted (%s) — re-run screener", exc)
            os.remove(INDEX_RESULTS_FILE)
    return None


def last_updated() -> str:
    if os.path.exists(LAST_UPDATED_FILE):
        with open(LAST_UPDATED_FILE) as f:
            return f.read().strip()
    return "Never"


# ---------------------------------------------------------------------------
# Computed helpers
# ---------------------------------------------------------------------------

def sector_outperformance_vs_nifty500() -> dict:
    """Return {sector: avg outperformance %} vs Nifty 500 across 1M + 3M."""
    from src.sector_mapper import load_index_data, resolve_sector_index, index_return
    try:
        _, sector_indices, nifty500 = load_index_data()
        nifty500_1m = index_return(nifty500, MONTHLY_DAYS)
        nifty500_3m = index_return(nifty500, QUARTERLY_DAYS)
        results = load_results()
        if results is None or results.empty:
            return {}
        outperf: dict = {}
        for sector in results["Sector"].unique():
            idx_ticker = resolve_sector_index(sector)
            sector_df  = sector_indices.get(idx_ticker)
            sector_1m  = index_return(sector_df, MONTHLY_DAYS)
            sector_3m  = index_return(sector_df, QUARTERLY_DAYS)
            outperf[sector] = round(((sector_1m - nifty500_1m) + (sector_3m - nifty500_3m)) / 2, 2)
        return outperf
    except Exception:
        return {}


def top_industries_vs_sector(results: pd.DataFrame, sector_indices: dict, top_n: int = 10) -> list:
    """Return list of dicts for top industries by outperformance vs their sector index."""
    from src.sector_mapper import resolve_sector_index, index_return
    rows = []
    for (sector, industry), grp in results.groupby(["Sector", "Industry"]):
        avg_1m = float(grp["Monthly %"].mean())
        avg_3m = float(grp["Quarterly %"].mean()) if "Quarterly %" in grp.columns else avg_1m
        idx_ticker = resolve_sector_index(sector)
        sector_df  = sector_indices.get(idx_ticker)
        sector_1m  = index_return(sector_df, MONTHLY_DAYS)
        sector_3m  = index_return(sector_df, QUARTERLY_DAYS)
        outperf    = round(((avg_1m - sector_1m) + (avg_3m - sector_3m)) / 2, 2)
        rows.append({"industry": industry, "sector": sector, "outperf": outperf, "n_stocks": len(grp)})
    rows.sort(key=lambda r: r["outperf"], reverse=True)
    return rows[:top_n]


# ---------------------------------------------------------------------------
# Pipeline stages — each is thread-safe (writes to files, no shared state)
# ---------------------------------------------------------------------------

def run_data_fetch(triggered_by: str = "manual") -> None:
    """Stage 1 — Network: fetch stock list, prices, info, index data."""
    write_status("fetch", "running", f"Data fetch started ({triggered_by})")
    try:
        from src.nse_fetcher import fetch_nifty1000_stocks
        from src.data_fetcher import (
            fetch_price_data, fetch_stock_info, save_price_data, save_stock_info,
            load_stock_info, is_info_cache_fresh,
        )
        from src.sector_mapper import fetch_benchmark, fetch_nifty500, fetch_all_sector_indices, save_index_data
        from src.index_screener import fetch_index_ohlcv, save_index_ohlcv

        write_status("fetch", "running", "Fetching NSE equity list…")
        stocks_df = fetch_nifty1000_stocks()
        tickers = stocks_df["yf_ticker"].tolist()
        logger.info("Stock list: %d tickers", len(tickers))

        write_status("fetch", "running", f"Downloading price data for {len(tickers)} stocks…")
        price_data = fetch_price_data(tickers)
        save_price_data(price_data)
        logger.info("Price data saved: %d tickers", len(price_data))

        if is_info_cache_fresh():
            write_status("fetch", "running", "Stock info cache is fresh — skipping download")
            stock_info = load_stock_info() or {}
        else:
            write_status("fetch", "running", "Fetching PE / sector / market cap info…")
            stock_info = fetch_stock_info(list(price_data.keys()))
            save_stock_info(stock_info)
            logger.info("Stock info saved: %d tickers", len(stock_info))

        # Enrich Unknown sector/industry via NSE Direct API
        from src.nse_fetcher import fetch_nse_sector_map
        from src.data_fetcher import enrich_with_nse_sectors
        unknown_count = sum(
            1 for info in stock_info.values()
            if info.get("sector") in ("Unknown", None, "")
        )
        write_status(
            "fetch", "running",
            f"Enriching sector/industry via NSE API "
            f"({unknown_count} stocks with Unknown sector)…",
        )
        def _nse_progress(done, total):
            write_status(
                "fetch", "running",
                f"NSE sector map: {done}/{total} tickers fetched…",
            )

        nse_map = fetch_nse_sector_map(list(price_data.keys()), status_callback=_nse_progress)
        stock_info = enrich_with_nse_sectors(stock_info, nse_map)
        save_stock_info(stock_info)
        logger.info("Stock info re-saved with NSE sector enrichment")

        from src.nse_freefloat import fetch_all_freefloat, save_freefloat_cache, load_freefloat_cache
        existing_ff  = load_freefloat_cache()
        all_tickers  = list(price_data.keys())
        missing_ff   = [t for t in all_tickers if t not in existing_ff]
        if missing_ff:
            n_missing = len(missing_ff)
            n_cached  = len(existing_ff)
            est_min   = round(n_missing * 0.5 / 60)
            write_status(
                "fetch", "running",
                f"Fetching free float % for {n_missing} new stocks "
                f"({n_cached} already cached; ~{est_min} min)…",
            )
            new_ff = fetch_all_freefloat(missing_ff, stock_info)
            existing_ff.update(new_ff)
            save_freefloat_cache(existing_ff)
            logger.info(
                "Free float cache updated: %d total (%d new, %d pre-existing)",
                len(existing_ff), len(new_ff), n_cached,
            )
        else:
            write_status("fetch", "running", "Free float cache complete — all stocks covered, skipping fetch")
            logger.info(
                "Free float cache complete — %d/%d tickers cached, no fetch needed",
                len(existing_ff), len(all_tickers),
            )

        write_status("fetch", "running", "Fetching benchmark, Nifty 500 and sector indices…")
        benchmark = fetch_benchmark()
        nifty500 = fetch_nifty500()
        sector_indices = fetch_all_sector_indices()
        save_index_data(benchmark, sector_indices, nifty500)
        logger.info("Sector index data saved: %d indices", len(sector_indices))

        write_status("fetch", "running", "Fetching NSE index OHLCV data…")
        index_ohlcv = fetch_index_ohlcv()
        save_index_ohlcv(index_ohlcv)
        logger.info("NSE index OHLCV saved: %d indices", len(index_ohlcv))

        write_status("fetch", "running", "Fetching MCX commodity OHLCV data…")
        from src.commodity_screener import fetch_commodity_ohlcv, save_commodity_ohlcv
        commodity_ohlcv = fetch_commodity_ohlcv()
        save_commodity_ohlcv(commodity_ohlcv)
        logger.info("Commodity OHLCV saved: %d instruments", len(commodity_ohlcv))

        from datetime import datetime
        import pytz
        from config import IST_TIMEZONE
        now_ist = datetime.now(pytz.timezone(IST_TIMEZONE)).strftime("%Y-%m-%d %H:%M:%S IST")
        write_status("fetch", "done", f"Data fetch complete — {now_ist}")
        logger.info("Data fetch complete")

    except Exception as exc:
        logger.exception("Data fetch failed")
        write_status("fetch", "error", str(exc))


def run_screener_only(triggered_by: str = "manual") -> None:
    """Stage 2 — Compute: load cached data, run all screeners, save results."""
    write_status("screener", "running", f"Screener started ({triggered_by})")
    try:
        from src.data_fetcher import load_price_data, load_stock_info
        from src.sector_mapper import load_index_data
        from src.screener import run_screener, run_ipo_screener, run_ipo_inside_bar_screener
        from src.index_screener import load_index_ohlcv, run_index_screener, save_index_results
        from src.tracker import save_snapshot
        from src.index_tracker import save_index_snapshot

        price_data = load_price_data()
        if not price_data:
            write_status("screener", "error", "No price data — run Fetch Data first")
            return

        stock_info = load_stock_info() or {}
        benchmark, sector_indices, nifty500 = load_index_data()

        from src.nse_freefloat import load_freefloat_cache
        freefloat = load_freefloat_cache()
        if freefloat:
            logger.info("Free float cache loaded: %d tickers", len(freefloat))
        else:
            logger.warning(
                "Free float cache is empty — data fetch may still be running. "
                "Re-run the screener after the fetch completes to get Free Float %% values."
            )

        write_status("screener", "running", f"Running main screener on {len(price_data)} stocks…")
        results = run_screener(price_data, stock_info, sector_indices, benchmark, freefloat)

        write_status("screener", "running", "Running IPO Base screener…")
        ipo_results = run_ipo_screener(price_data, stock_info, sector_indices, nifty500)

        write_status("screener", "running", "Running IPO Inside Bar screener…")
        ipo_ib_results = run_ipo_inside_bar_screener(price_data, stock_info, sector_indices, nifty500)

        os.makedirs(DATA_DIR, exist_ok=True)
        with open(SCREENER_RESULTS_FILE, "wb") as f:
            pickle.dump(results, f)
        with open(IPO_RESULTS_FILE, "wb") as f:
            pickle.dump(ipo_results, f)
        with open(IPO_IB_RESULTS_FILE, "wb") as f:
            pickle.dump(ipo_ib_results, f)

        from datetime import datetime
        import pytz
        from config import IST_TIMEZONE
        now_ist = datetime.now(pytz.timezone(IST_TIMEZONE)).strftime("%Y-%m-%d %H:%M:%S IST")
        with open(LAST_UPDATED_FILE, "w") as f:
            f.write(now_ist)

        write_status("screener", "running", "Saving daily stock snapshot…")
        n_snap = save_snapshot(results, force=True)
        logger.info("Stock snapshot: %d stocks", n_snap)

        write_status("screener", "running", "Running NSE Index screener…")
        try:
            index_ohlcv = load_index_ohlcv()
            if index_ohlcv:
                index_results = run_index_screener(index_ohlcv, nifty500)
                save_index_results(index_results)
                save_index_snapshot(index_results, force=True)
                logger.info("Index screener: %d indices passed", len(index_results))
            else:
                logger.warning("No index OHLCV data cached — run Fetch Data first to populate index tracker")
        except Exception as exc:
            logger.exception("Index screener failed (screener will continue): %s", exc)

        write_status("screener", "running", "Running Commodity screener…")
        try:
            from src.commodity_screener import load_commodity_ohlcv, run_commodity_screener, save_commodity_results
            from src.commodity_tracker import save_commodity_snapshot
            commodity_ohlcv = load_commodity_ohlcv()
            if commodity_ohlcv:
                commodity_results = run_commodity_screener(commodity_ohlcv, nifty500)
                save_commodity_results(commodity_results)
                save_commodity_snapshot(commodity_results, force=True)
                logger.info("Commodity screener: %d passed", len(commodity_results))
            else:
                logger.warning("No commodity OHLCV data cached — run Fetch Commodities to populate tracker")
        except Exception as exc:
            logger.exception("Commodity screener failed (screener will continue): %s", exc)

        write_status(
            "screener", "done",
            f"{len(results)} stocks | {len(ipo_results)} IPO bases — {now_ist}",
        )
        logger.info("Screener complete: %d stocks passed", len(results))

    except Exception as exc:
        logger.exception("Screener failed")
        write_status("screener", "error", str(exc))


def run_full_pipeline(triggered_by: str = "scheduler") -> None:
    """Chains fetch → screener. Called by the 4 PM APScheduler job."""
    run_data_fetch(triggered_by)
    run_screener_only(triggered_by)


def run_commodity_pipeline(triggered_by: str = "scheduler") -> None:
    """
    Standalone commodity refresh — fetch OHLCV then run screener.
    Called by the 23:45 IST APScheduler job and the sidebar button.
    """
    write_status("commodity", "running", f"Commodity fetch started ({triggered_by})")
    try:
        from src.commodity_screener import (
            fetch_commodity_ohlcv, save_commodity_ohlcv,
            run_commodity_screener, save_commodity_results,
        )
        from src.commodity_tracker import save_commodity_snapshot
        from src.sector_mapper import load_index_data

        write_status("commodity", "running", "Fetching MCX commodity OHLCV…")
        commodity_ohlcv = fetch_commodity_ohlcv()
        save_commodity_ohlcv(commodity_ohlcv)

        write_status("commodity", "running", "Running commodity screener…")
        _, _, nifty500 = load_index_data()
        commodity_results = run_commodity_screener(commodity_ohlcv, nifty500)
        save_commodity_results(commodity_results)
        save_commodity_snapshot(commodity_results, force=True)

        from datetime import datetime
        import pytz
        from config import IST_TIMEZONE
        now_ist = datetime.now(pytz.timezone(IST_TIMEZONE)).strftime("%Y-%m-%d %H:%M:%S IST")
        write_status("commodity", "done", f"{len(commodity_results)} commodities passing — {now_ist}")
        logger.info("Commodity pipeline complete: %d passed", len(commodity_results))

    except Exception as exc:
        logger.exception("Commodity pipeline failed")
        write_status("commodity", "error", str(exc))


def run_freefloat_refresh(triggered_by: str = "scheduler") -> None:
    """
    Full weekly refresh of free float data — re-fetches every ticker and overwrites
    the cache.  Called by the Saturday APScheduler job and the sidebar manual button.
    """
    write_status("freefloat", "running", f"Free float full refresh started ({triggered_by})")
    try:
        from src.data_fetcher import load_price_data, load_stock_info
        from src.nse_freefloat import fetch_all_freefloat, save_freefloat_cache

        price_data = load_price_data()
        if not price_data:
            write_status("freefloat", "error", "No price data — run Fetch Data first")
            return

        stock_info = load_stock_info() or {}
        tickers    = list(price_data.keys())
        n_tickers  = len(tickers)
        est_min    = round(n_tickers * 0.5 / 60)

        write_status(
            "freefloat", "running",
            f"Refreshing free float % for all {n_tickers} stocks (~{est_min} min)…",
        )
        freefloat = fetch_all_freefloat(tickers, stock_info)
        save_freefloat_cache(freefloat)

        from datetime import datetime
        import pytz
        from config import IST_TIMEZONE
        now_ist = datetime.now(pytz.timezone(IST_TIMEZONE)).strftime("%Y-%m-%d %H:%M:%S IST")
        write_status(
            "freefloat", "done",
            f"{len(freefloat)}/{n_tickers} tickers covered — {now_ist}",
        )
        logger.info("Free float full refresh complete: %d/%d tickers", len(freefloat), n_tickers)

    except Exception as exc:
        logger.exception("Free float refresh failed")
        write_status("freefloat", "error", str(exc))


def run_sentiment_fetch(triggered_by: str = "manual") -> None:
    """Background sentiment fetch for all screener-passing stocks."""
    try:
        results = load_results()
        if results is None or results.empty:
            logger.warning("Sentiment: no screener results — run screener first")
            return
        symbols   = results["Symbol"].tolist()
        companies = results["Company"].tolist() if "Company" in results.columns else symbols
        from src.sentiment import batch_sentiment
        logger.info("Sentiment fetch started for %d stocks (%s)", len(symbols), triggered_by)
        batch_sentiment(symbols, companies)
        logger.info("Sentiment fetch complete")
    except Exception as exc:
        logger.exception("Sentiment fetch failed: %s", exc)
