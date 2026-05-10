"""
Backtesting engine for the Momentum Stock Screener.

Entry rule  : Price crosses above previous day's High AND Volume > 20D Vol MA.
              Checked on each of the next 3 trading days after the signal date.
              Entry price = previous day's High (the breakout level).
Exit rule   : First daily close below EMA20.  Open trade if not yet triggered.
Universe    : Stocks from the latest screener output (scored + ranked).
Signal dates: Every trading day in the last `lookback_weeks` weeks.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import PATTERN_COLS, COMBINED_RANK_WEIGHTS

logger = logging.getLogger(__name__)

ENTRY_MAX_DAYS   = 3      # Max days after signal to look for a breakout entry
VOL_MA_PERIOD    = 20     # Volume MA period for entry confirmation
EXIT_EMA_PERIOD  = 20     # Exit when Close < this EMA
STOPLOSS_LOOKBACK = 2     # Number of prior days used to compute the stop-loss low
STOPLOSS_BUFFER   = 0.0025  # 0.25% below the min low of last N days


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _signal_dates(price_data: dict, lookback_weeks: int) -> list:
    """Return sorted list of Timestamps covering the last `lookback_weeks` of trading."""
    sample = next(iter(price_data.values()))
    all_ts = pd.to_datetime(sample.index).sort_values()
    cutoff = all_ts[-1] - pd.Timedelta(weeks=lookback_weeks)
    return all_ts[all_ts >= cutoff].tolist()


def _find_entry(
    df: pd.DataFrame,
    signal_ts: pd.Timestamp,
) -> tuple:
    """
    Scan up to ENTRY_MAX_DAYS after signal_ts for a valid breakout entry.

    Conditions
    ----------
    - today's High > prev day's High (price crossed above)
    - today's Volume > 20D Vol MA (volume confirms)
    - today's Low ≤ prev day's High (entry level was actually touched intraday)

    Returns
    -------
    (entry_date, entry_price)  or  (None, None)
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    hist = df[df.index <= signal_ts]
    if len(hist) < VOL_MA_PERIOD:
        return None, None

    vol_ma    = float(hist["Volume"].tail(VOL_MA_PERIOD).mean())
    prev_high = float(hist["High"].iloc[-1])   # Signal day's high

    future = df[df.index > signal_ts]
    if future.empty:
        return None, None

    for i in range(min(ENTRY_MAX_DAYS, len(future))):
        day        = future.iloc[i]
        today_high = float(day["High"])
        today_low  = float(day["Low"])
        today_vol  = float(day["Volume"])

        if today_high > prev_high and today_vol > vol_ma and today_low <= prev_high:
            return future.index[i], round(prev_high, 2)

        prev_high = today_high   # Update for next iteration

    return None, None


def _compute_stoploss(df: pd.DataFrame, entry_ts: pd.Timestamp) -> float:
    """
    Stop-loss = min Low of the last STOPLOSS_LOOKBACK days before (and including)
    the entry bar, minus STOPLOSS_BUFFER (0.25%).
    """
    hist = df[df.index <= entry_ts]
    if len(hist) < STOPLOSS_LOOKBACK:
        min_low = float(hist["Low"].min())
    else:
        min_low = float(hist["Low"].iloc[-STOPLOSS_LOOKBACK:].min())
    return round(min_low * (1 - STOPLOSS_BUFFER), 2)


def _find_exit(
    df: pd.DataFrame,
    entry_ts: pd.Timestamp,
    entry_price: float,
    stop_loss: float,
) -> tuple:
    """
    Scan forward from entry_ts. Exit on whichever triggers first:
      1. Intraday Low ≤ stop_loss  → exit at stop_loss price
      2. Daily Close < EMA20       → exit at that day's Close

    Returns
    -------
    (exit_date, exit_price, holding_days, status, exit_reason, max_gain_pct, max_dd_pct)
    status      : "closed" | "open"
    exit_reason : "Stop Loss" | "EMA20" | "Open"
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    ema20  = df["Close"].ewm(span=EXIT_EMA_PERIOD, adjust=False).mean()
    future = df[df.index >= entry_ts]

    if len(future) < 2:
        last = float(df["Close"].iloc[-1])
        return df.index[-1], last, 0, "open", "Open", 0.0, 0.0

    peak   = entry_price
    trough = entry_price

    for i in range(1, len(future)):
        row     = future.iloc[i]
        ts      = future.index[i]
        low     = float(row["Low"])
        close   = float(row["Close"])
        ema_val = float(ema20.loc[ts])

        peak   = max(peak,   float(row["High"]))
        trough = min(trough, low)

        max_gain = round((peak   / entry_price - 1) * 100, 2)
        max_dd   = round((trough / entry_price - 1) * 100, 2)

        # Stop-loss check first (intraday low breaches the level)
        if low <= stop_loss:
            return ts, stop_loss, i, "closed", "Stop Loss", max_gain, max_dd

        # EMA20 exit (end-of-day close below EMA)
        if close < ema_val:
            return ts, round(close, 2), i, "closed", "EMA20", max_gain, max_dd

    # No exit triggered — position still open
    last_close = float(future["Close"].iloc[-1])
    max_gain   = round((peak   / entry_price - 1) * 100, 2)
    max_dd     = round((trough / entry_price - 1) * 100, 2)
    return future.index[-1], round(last_close, 2), len(future) - 1, "open", "Open", max_gain, max_dd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_backtest(
    price_data: Dict[str, pd.DataFrame],
    screener_results: pd.DataFrame,
    lookback_weeks: int = 4,
) -> pd.DataFrame:
    """
    Simulate trades for every stock in screener_results over the last
    lookback_weeks of price history.

    Parameters
    ----------
    price_data       : {ticker: OHLCV DataFrame}  (full 1-year history)
    screener_results : Latest screener output DataFrame
    lookback_weeks   : Number of weeks to look back for signal dates

    Returns
    -------
    DataFrame with one row per trade (entry triggered), columns:
    Signal Date, Symbol, Sector, Score, RS Rating, Combined Score,
    Entry Date, Entry Price, Stop Loss, Exit Date, Exit Price, Return %,
    Holding Days, Status, Exit Reason, Max Gain %, Max DD %, <pattern flags>
    """
    if screener_results is None or screener_results.empty:
        logger.warning("No screener results available — run the screener first.")
        return pd.DataFrame()

    signal_dates = _signal_dates(price_data, lookback_weeks)
    if not signal_dates:
        logger.warning("No trading dates found in the lookback window.")
        return pd.DataFrame()

    logger.info(
        f"Backtesting {len(screener_results)} stocks over "
        f"{len(signal_dates)} signal dates ({lookback_weeks}W)…"
    )

    w_s  = COMBINED_RANK_WEIGHTS["score"]     / 100.0
    w_rs = COMBINED_RANK_WEIGHTS["rs_rating"] / 100.0
    trades = []

    for _, stock_row in screener_results.reset_index(drop=True).iterrows():
        symbol = stock_row["Symbol"]
        ticker = f"{symbol}.NS"
        df = price_data.get(ticker)
        if df is None or df.empty:
            continue

        df = df.copy()
        df.index = pd.to_datetime(df.index)

        rs = stock_row.get("RS Rating", 50)
        combined = round(stock_row["Score"] * w_s + float(rs) * w_rs, 1)

        for signal_ts in signal_dates:
            if signal_ts not in df.index:
                continue

            entry_date, entry_price = _find_entry(df, signal_ts)
            if entry_date is None:
                continue

            stop_loss = _compute_stoploss(df, entry_date)

            exit_date, exit_price, holding_days, status, exit_reason, max_gain, max_dd = \
                _find_exit(df, entry_date, entry_price, stop_loss)

            ret_pct = (
                round((exit_price / entry_price - 1) * 100, 2)
                if entry_price and exit_price else None
            )

            trade = {
                "Signal Date":    signal_ts.date(),
                "Symbol":         symbol,
                "Sector":         stock_row.get("Sector", ""),
                "Score":          round(float(stock_row["Score"]), 1),
                "RS Rating":      int(rs),
                "Combined Score": combined,
                "Entry Date":     entry_date.date(),
                "Entry Price":    entry_price,
                "Stop Loss":      stop_loss,
                "Exit Date":      exit_date.date() if exit_date is not None else None,
                "Exit Price":     exit_price,
                "Return %":       ret_pct,
                "Holding Days":   holding_days,
                "Status":         status,
                "Exit Reason":    exit_reason,
                "Max Gain %":     max_gain,
                "Max DD %":       max_dd,
            }
            for p in PATTERN_COLS:
                trade[p] = bool(stock_row.get(p, False))

            trades.append(trade)

    if not trades:
        logger.info("No entry triggers fired in the backtest window.")
        return pd.DataFrame()

    result = pd.DataFrame(trades)
    counts = result["Status"].value_counts().to_dict()
    logger.info(f"Backtest complete — {len(result)} trades: {counts}")
    return result
