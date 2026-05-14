"""
MCX Commodity daily snapshot tracker.

Captures which commodities passed the screener each day, then computes
forward returns (1D / 1W / 2W / 1M) from cached OHLCV data.
Independent from the stock and index trackers.
"""

import logging
import os
import pickle
from datetime import date

import numpy as np
import pandas as pd

from config import DATA_DIR, COMMODITY_SNAPSHOTS_FILE, COMMODITY_OHLCV_FILE

logger = logging.getLogger(__name__)

_FORWARD_WINDOWS = [
    ("Ret 1D %",  1),
    ("Ret 1W %",  5),
    ("Ret 2W %", 10),
    ("Ret 1M %", 21),
]


# ---------------------------------------------------------------------------
# Snapshot save / load
# ---------------------------------------------------------------------------

def save_commodity_snapshot(
    results: pd.DataFrame,
    capture_date: str = None,
    force: bool = False,
) -> int:
    """
    Append today's commodity screener results to the persistent snapshot log.
    Returns number of new rows written (0 if skipped).
    """
    if results is None or results.empty:
        return 0

    capture_date = capture_date or date.today().isoformat()

    existing = load_commodity_snapshots()
    if not existing.empty and capture_date in existing["Capture Date"].values:
        if not force:
            logger.info("Commodity snapshot for %s already exists — skipped", capture_date)
            return 0
        existing = existing[existing["Capture Date"] != capture_date].reset_index(drop=True)
        logger.info("Replacing commodity snapshot for %s (force=True)", capture_date)

    snap = results.reset_index(drop=True).copy()
    snap.insert(0, "Capture Date", capture_date)

    combined = pd.concat([existing, snap], ignore_index=True, sort=False)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(COMMODITY_SNAPSHOTS_FILE, "wb") as f:
        pickle.dump(combined, f)

    logger.info("Commodity snapshot saved: %d commodities for %s", len(snap), capture_date)
    return len(snap)


def load_commodity_snapshots() -> pd.DataFrame:
    """Return accumulated commodity snapshot DataFrame; empty DataFrame if none."""
    if not os.path.exists(COMMODITY_SNAPSHOTS_FILE):
        return pd.DataFrame()
    try:
        with open(COMMODITY_SNAPSHOTS_FILE, "rb") as f:
            return pickle.load(f)
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Forward return computation
# ---------------------------------------------------------------------------

def compute_commodity_forward_returns(snapshots: pd.DataFrame) -> pd.DataFrame:
    """
    Augment commodity snapshots with forward-return columns using cached OHLCV.
    Entry price = closing price on or after the capture date.
    """
    if snapshots.empty:
        return snapshots

    commodity_ohlcv: dict = {}
    try:
        with open(COMMODITY_OHLCV_FILE, "rb") as f:
            commodity_ohlcv = pickle.load(f)
    except FileNotFoundError:
        logger.warning("commodity_ohlcv.pkl not found — forward returns unavailable")
        return snapshots

    df = snapshots.copy()
    for label, _ in _FORWARD_WINDOWS:
        df[label] = np.nan

    for idx, row in df.iterrows():
        name = row.get("Commodity")
        if not name or name not in commodity_ohlcv:
            continue

        px_df = commodity_ohlcv[name].copy()
        px_df.index = pd.to_datetime(px_df.index)
        future = px_df[px_df.index >= str(row["Capture Date"])]
        if future.empty:
            continue

        close_col = "Close" if "Close" in future.columns else future.columns[0]
        entry = float(future[close_col].iloc[0])
        if entry <= 0:
            continue

        close = future[close_col]
        for label, days in _FORWARD_WINDOWS:
            if len(close) > days:
                df.at[idx, label] = round((float(close.iloc[days]) / entry - 1) * 100, 2)

    return df


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def commodity_score_band_summary(df: pd.DataFrame, bands: int = 5) -> pd.DataFrame:
    """Bucket commodities by Score quintile and compute average forward returns."""
    if "Score" not in df.columns or df.empty:
        return pd.DataFrame()

    df = df.copy()
    try:
        df["Score Band"] = pd.qcut(
            df["Score"], q=bands, duplicates="drop",
            labels=[f"Q{i+1}" for i in range(bands)],
        )
    except Exception:
        return pd.DataFrame()

    rows = []
    for band, grp in df.groupby("Score Band", observed=True):
        w1 = grp["Ret 1W %"].dropna()
        m1 = grp["Ret 1M %"].dropna()
        score_range = f"{grp['Score'].min():.0f}–{grp['Score'].max():.0f}"
        rows.append({
            "Score Band": f"{band}  ({score_range})",
            "Captures":   len(grp),
            "Avg Ret 1W": f"{w1.mean():+.2f}%" if len(w1) else "—",
            "Avg Ret 1M": f"{m1.mean():+.2f}%" if len(m1) else "—",
        })
    return pd.DataFrame(rows)
