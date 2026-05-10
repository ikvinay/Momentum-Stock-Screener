"""
Daily screener snapshot tracker.

Captures which stocks passed the screener each day, then computes forward
returns (1D / 1W / 2W / 1M) from the existing price cache.  Because returns
are computed dynamically against price_data.pkl they automatically reflect
the latest downloaded prices every time the tracker tab is opened.

Extensibility contract
──────────────────────
Snapshots capture ALL columns from the screener results DataFrame.  Adding a
new pattern or metric to the screener automatically includes it in future
snapshots without any change here.  Old snapshots that predate the new column
will have NaN for it; boolean pattern columns are backfilled to False so that
pattern_summary() doesn't miscount them.  Column additions never break
existing data; column removals leave a NaN column in old rows (displayed as
"—"), which is harmless.
"""

import logging
import os
import pickle
from datetime import date

import numpy as np
import pandas as pd

from config import SNAPSHOTS_FILE, DATA_DIR, PATTERN_COLS, COMBINED_RANK_WEIGHTS

logger = logging.getLogger(__name__)

# (column label, trading-day offset from entry price)
_FORWARD_WINDOWS = [
    ("Ret 1D %",  1),
    ("Ret 1W %",  5),
    ("Ret 2W %", 10),
    ("Ret 1M %", 21),
]

# Columns that must exist in every snapshot (added if missing from results)
_REQUIRED_COLS = ["Symbol", "Company", "Sector", "Industry", "CMP", "Score"]


# ---------------------------------------------------------------------------
# Snapshot save / load
# ---------------------------------------------------------------------------

def save_snapshot(results: pd.DataFrame, capture_date: str = None, force: bool = False) -> int:
    """
    Append today's screener results to the persistent snapshot log.

    Captures ALL columns from *results* so that new patterns or metrics are
    included automatically.  Skips if a snapshot for this date already exists
    unless *force=True*, which replaces today's entry (used when re-running
    the screener manually to pick up config/weight changes).
    Returns the number of new rows written (0 if skipped).
    """
    if results is None or results.empty:
        return 0

    capture_date = capture_date or date.today().isoformat()

    existing = load_snapshots()
    if not existing.empty and capture_date in existing["Capture Date"].values:
        if not force:
            logger.info(f"Snapshot for {capture_date} already exists — skipped")
            return 0
        # Drop the existing entry for this date so we can replace it
        existing = existing[existing["Capture Date"] != capture_date].reset_index(drop=True)
        logger.info(f"Replacing snapshot for {capture_date} (force=True)")

    snap = results.reset_index(drop=True).copy()
    snap.insert(0, "Capture Date", capture_date)

    # Merge with existing: pd.concat fills missing columns with NaN across both sides
    combined = pd.concat([existing, snap], ignore_index=True, sort=False)

    # Back-fill NaN in pattern columns with False so they don't skew analysis
    _normalise_pattern_cols(combined)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SNAPSHOTS_FILE, "wb") as f:
        pickle.dump(combined, f)

    logger.info(f"Snapshot saved: {len(snap)} stocks for {capture_date}")
    return len(snap)


def load_snapshots() -> pd.DataFrame:
    """Return the accumulated snapshot DataFrame; empty DataFrame if none exist."""
    if not os.path.exists(SNAPSHOTS_FILE):
        return pd.DataFrame()
    try:
        with open(SNAPSHOTS_FILE, "rb") as f:
            df = pickle.load(f)
        _normalise_pattern_cols(df)
        return df
    except Exception:
        return pd.DataFrame()


def _normalise_pattern_cols(df: pd.DataFrame) -> None:
    """Fill NaN in any known pattern column with False (in-place)."""
    for col in PATTERN_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)


# ---------------------------------------------------------------------------
# Forward return computation
# ---------------------------------------------------------------------------

def compute_forward_returns(snapshots: pd.DataFrame, price_data: dict | None) -> pd.DataFrame:
    """
    Augment snapshots with forward-return columns.

    Entry price = closing price on or after the capture date (next trading
    day if the market was closed that day).  Returns are relative to that
    entry price.
    """
    if snapshots.empty or not price_data:
        return snapshots

    df = snapshots.copy()
    for label, _ in _FORWARD_WINDOWS:
        df[label] = np.nan

    for idx, row in df.iterrows():
        ticker = f"{row['Symbol']}.NS"
        px_df  = price_data.get(ticker)
        if px_df is None or px_df.empty:
            continue

        px = px_df.copy()
        px.index = pd.to_datetime(px.index)
        future = px[px.index >= row["Capture Date"]]
        if future.empty:
            continue

        entry = float(future["Close"].iloc[0])
        if entry <= 0:
            continue

        close = future["Close"]
        for label, days in _FORWARD_WINDOWS:
            if len(close) > days:
                df.at[idx, label] = round((float(close.iloc[days]) / entry - 1) * 100, 2)

    return df


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def pattern_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each pattern in PATTERN_COLS (plus any extra boolean columns present
    in *df*) return: N captures, Win Rate 1W, Avg Ret 1W, Win Rate 1M, Avg Ret 1M.
    Only rows with available forward data count toward rates/averages.
    """
    # Use configured patterns that exist in df, plus any extra bool columns
    present = [c for c in PATTERN_COLS if c in df.columns]
    extra   = [c for c in df.columns
               if c not in PATTERN_COLS
               and c not in ("Capture Date", "Symbol", "Company", "Sector", "Industry")
               and pd.api.types.is_bool_dtype(df[c])]
    cols = present + extra

    rows = []
    for pat in cols:
        sub = df[df[pat] == True]
        if sub.empty:
            continue
        w1 = sub["Ret 1W %"].dropna()
        m1 = sub["Ret 1M %"].dropna()
        rows.append({
            "Pattern":     pat,
            "Captures":    len(sub),
            "Win Rate 1W": f"{(w1 > 0).mean() * 100:.0f}%" if len(w1) else "—",
            "Avg Ret 1W":  f"{w1.mean():+.2f}%"            if len(w1) else "—",
            "Win Rate 1M": f"{(m1 > 0).mean() * 100:.0f}%" if len(m1) else "—",
            "Avg Ret 1M":  f"{m1.mean():+.2f}%"            if len(m1) else "—",
        })
    return pd.DataFrame(rows)


def score_band_summary(df: pd.DataFrame, bands: int = 5) -> pd.DataFrame:
    """
    Bucket stocks by combined rank score quintile and compute average forward
    returns per bucket.  Combined score = (Momentum Score + RS Rating) / 2
    when RS Rating is available, otherwise falls back to Score alone.
    """
    if "Score" not in df.columns or df.empty:
        return pd.DataFrame()

    df = df.copy()
    w_s  = COMBINED_RANK_WEIGHTS["score"]     / 100.0
    w_rs = COMBINED_RANK_WEIGHTS["rs_rating"] / 100.0
    if "RS Rating" in df.columns:
        df["_combined"] = df["Score"] * w_s + df["RS Rating"] * w_rs
        band_label = "Combined Score Band"
    else:
        df["_combined"] = df["Score"]
        band_label = "Score Band"

    df[band_label] = pd.qcut(df["_combined"], q=bands, duplicates="drop",
                             labels=[f"Q{i+1}" for i in range(bands)])
    rows = []
    for band, grp in df.groupby(band_label, observed=True):
        w1 = grp["Ret 1W %"].dropna()
        m1 = grp["Ret 1M %"].dropna()
        score_range = f"{grp['_combined'].min():.0f}–{grp['_combined'].max():.0f}"
        rows.append({
            band_label:   f"{band}  ({score_range})",
            "Captures":   len(grp),
            "Avg Ret 1W": f"{w1.mean():+.2f}%" if len(w1) else "—",
            "Avg Ret 1M": f"{m1.mean():+.2f}%" if len(m1) else "—",
        })
    return pd.DataFrame(rows)
