"""
Standalone script — fetch NSE sector/industry classification only.

Reads the existing price_data.pkl for the ticker list, then calls
fetch_nse_sector_map() which saves a checkpoint every 50 tickers and
resumes from the last checkpoint if interrupted.

Usage:
    python refresh_nse_sectors.py
"""

import logging
import pickle
import sys
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PRICE_DATA_FILE, NSE_SECTOR_MAP_FILE
from src.nse_fetcher import fetch_nse_sector_map
from src.data_fetcher import load_stock_info, save_stock_info, enrich_with_nse_sectors


def main():
    # Load ticker list from price data cache
    if not os.path.exists(PRICE_DATA_FILE):
        logger.error("price_data.pkl not found — run Fetch Data first to populate it.")
        sys.exit(1)

    with open(PRICE_DATA_FILE, "rb") as f:
        price_data = pickle.load(f)

    tickers = list(price_data.keys())
    logger.info(f"Loaded {len(tickers)} tickers from price_data.pkl")

    # Check existing cache
    cached_count = 0
    if os.path.exists(NSE_SECTOR_MAP_FILE):
        with open(NSE_SECTOR_MAP_FILE, "rb") as f:
            existing = pickle.load(f)
        cached_count = len(existing)
        missing = [t for t in tickers if t not in existing]
        logger.info(
            f"Existing cache: {cached_count} entries — {len(missing)} tickers still missing"
        )
    else:
        logger.info("No existing cache — fetching all tickers from scratch")

    def _progress(done, total):
        pct = done / total * 100
        logger.info(f"Progress: {done}/{total} ({pct:.0f}%)")

    nse_map = fetch_nse_sector_map(tickers, status_callback=_progress)
    logger.info(f"NSE sector map complete: {len(nse_map)} entries in cache")

    # Enrich stock_info if available
    stock_info = load_stock_info()
    if stock_info:
        unknown_before = sum(
            1 for info in stock_info.values()
            if info.get("sector") in ("Unknown", None, "")
        )
        stock_info = enrich_with_nse_sectors(stock_info, nse_map)
        unknown_after = sum(
            1 for info in stock_info.values()
            if info.get("sector") in ("Unknown", None, "")
        )
        save_stock_info(stock_info)
        logger.info(
            f"stock_info enriched: Unknown sectors {unknown_before} → {unknown_after} "
            f"({unknown_before - unknown_after} fixed)"
        )
    else:
        logger.warning("stock_info.pkl not found — skipping enrichment step")


if __name__ == "__main__":
    main()
