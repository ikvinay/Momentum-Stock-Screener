"""
Standalone data refresh script.

Usage:
    python refresh.py            # Normal refresh (uses cached stock info if fresh)
    python refresh.py --force    # Force refresh of everything including stock info

Runs the full pipeline:
  1. Fetch NSE equity list
  2. Download 1-year price data
  3. Fetch stock info (PE, sector, market cap)
  4. Fetch benchmark + sector indices + Nifty 500
  5. Run main screener + save snapshot
  6. Run IPO Base screener
  7. Run IPO Inside Bar screener
"""

import argparse
import logging
import os
import pickle
import sys
from datetime import datetime

import pytz

from config import (
    SCREENER_RESULTS_FILE, LAST_UPDATED_FILE, DATA_DIR, IST_TIMEZONE,
    IPO_RESULTS_FILE, IPO_IB_RESULTS_FILE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
IST = pytz.timezone(IST_TIMEZONE)


def run(force_info: bool = False) -> None:
    from src.nse_fetcher import fetch_nifty1000_stocks
    from src.data_fetcher import (
        fetch_price_data, fetch_stock_info, save_price_data, save_stock_info,
        load_stock_info, is_info_cache_fresh,
    )
    from src.sector_mapper import (
        fetch_benchmark, fetch_nifty500, fetch_all_sector_indices, save_index_data,
    )
    from src.screener import run_screener, run_ipo_screener, run_ipo_inside_bar_screener
    from src.tracker import save_snapshot

    os.makedirs(DATA_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Momentum Stock Screener — Full Refresh")
    logger.info("=" * 60)

    # 1. NSE equity list
    logger.info("[1/7] Fetching NSE equity list…")
    stocks_df = fetch_nifty1000_stocks(force_refresh=force_info)
    tickers   = stocks_df["yf_ticker"].tolist()
    logger.info("      %d tickers loaded", len(tickers))

    # 2. Price data
    logger.info("[2/7] Downloading 1-year daily price data…")
    price_data = fetch_price_data(tickers)
    save_price_data(price_data)
    logger.info("      Price data saved for %d tickers", len(price_data))

    # 3. Stock info
    if not force_info and is_info_cache_fresh():
        logger.info("[3/7] Stock info cache is fresh — skipping download")
        stock_info = load_stock_info() or {}
    else:
        logger.info("[3/7] Fetching stock info (PE, sector, market cap)…")
        stock_info = fetch_stock_info(list(price_data.keys()))
        save_stock_info(stock_info)
        logger.info("      Info saved for %d tickers", len(stock_info))

    # 4. Benchmark + sector indices + Nifty 500
    logger.info("[4/7] Fetching benchmark, sector indices and Nifty 500…")
    benchmark      = fetch_benchmark()
    nifty500       = fetch_nifty500()
    sector_indices = fetch_all_sector_indices()
    save_index_data(benchmark, sector_indices, nifty500)
    logger.info("      %d sector indices fetched", len(sector_indices))

    # 5. Main screener + snapshot
    logger.info("[5/7] Running main screener…")
    results = run_screener(price_data, stock_info, sector_indices, benchmark)
    logger.info("      %d stocks passed all filters", len(results))
    with open(SCREENER_RESULTS_FILE, "wb") as f:
        pickle.dump(results, f)

    n_snap = save_snapshot(results, force=True)
    logger.info("      Snapshot saved: %d stocks captured for today", n_snap)

    # 6. IPO Base screener
    logger.info("[6/7] Running IPO Base screener…")
    ipo_results = run_ipo_screener(price_data, stock_info, sector_indices, nifty500)
    logger.info("      %d IPO base stocks passed", len(ipo_results))
    with open(IPO_RESULTS_FILE, "wb") as f:
        pickle.dump(ipo_results, f)

    # 7. IPO Inside Bar screener
    logger.info("[7/7] Running IPO Inside Bar screener…")
    ipo_ib_results = run_ipo_inside_bar_screener(price_data, stock_info, sector_indices, nifty500)
    logger.info("      %d IPO inside-bar setups found", len(ipo_ib_results))
    with open(IPO_IB_RESULTS_FILE, "wb") as f:
        pickle.dump(ipo_ib_results, f)

    # Timestamp
    now_ist = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
    with open(LAST_UPDATED_FILE, "w") as f:
        f.write(now_ist)

    logger.info("=" * 60)
    logger.info("Refresh complete: %s", now_ist)
    logger.info("=" * 60)

    if not results.empty:
        logger.info("\nTop 10 by Combined Score:")
        top10 = results.head(10)[["Symbol", "Sector", "Score"]]
        logger.info("\n" + top10.to_string(index=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Momentum Stock Screener — Full Refresh")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download of stock info even if cache is fresh",
    )
    args = parser.parse_args()
    run(force_info=args.force)
