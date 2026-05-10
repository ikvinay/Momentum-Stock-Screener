"""
Momentum Stock Screener — Streamlit UI

Run:  streamlit run app.py
Port: 8501 (default)

Auto-refreshes data at 4:00 PM IST daily via APScheduler background thread.
"""

import json
import logging
import os
import pickle
import sys
import threading
from datetime import datetime
from typing import Optional

import pandas as pd
import pytz
import streamlit as st
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from config import (
    SCREENER_RESULTS_FILE, LAST_UPDATED_FILE, REFRESH_STATUS_FILE,
    REFRESH_HOUR_IST, REFRESH_MINUTE_IST, IST_TIMEZONE, DATA_DIR,
    MONTHLY_DAYS, QUARTERLY_DAYS, IPO_RESULTS_FILE, SNAPSHOTS_FILE, PATTERN_COLS,
    COMBINED_RANK_WEIGHTS, RMV_TIGHT_THRESHOLD, IPO_BASE_MAX_DAYS,
    IPO_IB_RESULTS_FILE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
IST = pytz.timezone(IST_TIMEZONE)

# ---------------------------------------------------------------------------
# Scheduler (module-level — one instance for the whole process)
# ---------------------------------------------------------------------------
_scheduler: Optional[BackgroundScheduler] = None
_scheduler_lock = threading.Lock()


def _run_data_fetch(triggered_by: str = "manual") -> None:
    """
    Stage 1 — Network stage: fetch stock list, price data, stock info, index data.
    Writes to cache files only. Does NOT run the screener.
    """
    _write_status("fetch", "running", f"Data fetch started ({triggered_by})")
    try:
        from src.nse_fetcher import fetch_nifty1000_stocks
        from src.data_fetcher import (
            fetch_price_data, fetch_stock_info, save_price_data, save_stock_info,
            load_stock_info, is_info_cache_fresh,
        )
        from src.sector_mapper import (
            fetch_benchmark, fetch_nifty500, fetch_all_sector_indices, save_index_data,
        )

        # 1. Stock list
        _write_status("fetch", "running", "Fetching NSE equity list…")
        stocks_df = fetch_nifty1000_stocks()
        tickers = stocks_df["yf_ticker"].tolist()
        logger.info(f"Stock list: {len(tickers)} tickers")

        # 2. Price data
        _write_status("fetch", "running", f"Downloading price data for {len(tickers)} stocks (batches of 50)…")
        price_data = fetch_price_data(tickers)
        save_price_data(price_data)
        logger.info(f"Price data saved: {len(price_data)} tickers")

        # 3. Stock info (PE, sector, market cap) — cached 7 days
        if is_info_cache_fresh():
            _write_status("fetch", "running", "Stock info cache is fresh — skipping download")
            logger.info("Stock info cache fresh — skipped")
        else:
            _write_status("fetch", "running", "Fetching PE / sector / market cap info (~5 min)…")
            stock_info = fetch_stock_info(list(price_data.keys()))
            save_stock_info(stock_info)
            logger.info(f"Stock info saved: {len(stock_info)} tickers")

        # 4. Benchmark + sector indices + Nifty 500
        _write_status("fetch", "running", "Fetching benchmark, Nifty 500 and sector indices…")
        benchmark = fetch_benchmark()
        nifty500 = fetch_nifty500()
        sector_indices = fetch_all_sector_indices()
        save_index_data(benchmark, sector_indices, nifty500)
        logger.info(f"Index data saved: {len(sector_indices)} sector indices")

        now_ist = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
        _write_status("fetch", "done", f"Data fetch complete — {now_ist}")
        logger.info("Data fetch complete")

    except Exception as exc:
        logger.exception("Data fetch failed")
        _write_status("fetch", "error", str(exc))


def _run_screener_only(triggered_by: str = "manual") -> None:
    """
    Stage 2 — Compute stage: load cached data, apply filters + patterns, save results.
    No network calls. Runs in seconds once data is cached.
    """
    _write_status("screener", "running", f"Screener started ({triggered_by})")
    try:
        from src.data_fetcher import load_price_data, load_stock_info
        from src.sector_mapper import load_index_data
        from src.screener import run_screener, run_ipo_screener, run_ipo_inside_bar_screener

        price_data = load_price_data()
        if not price_data:
            _write_status("screener", "error", "No price data found — run Fetch Data first")
            return

        stock_info = load_stock_info() or {}
        benchmark, sector_indices, nifty500 = load_index_data()

        _write_status("screener", "running", f"Running main screener on {len(price_data)} stocks…")
        results = run_screener(price_data, stock_info, sector_indices, benchmark)

        _write_status("screener", "running", "Running IPO Base screener…")
        ipo_results = run_ipo_screener(price_data, stock_info, sector_indices, nifty500)

        _write_status("screener", "running", "Running IPO Inside Bar screener…")
        ipo_ib_results = run_ipo_inside_bar_screener(price_data, stock_info, sector_indices, nifty500)

        os.makedirs(DATA_DIR, exist_ok=True)
        with open(SCREENER_RESULTS_FILE, "wb") as f:
            pickle.dump(results, f)
        with open(IPO_RESULTS_FILE, "wb") as f:
            pickle.dump(ipo_results, f)
        with open(IPO_IB_RESULTS_FILE, "wb") as f:
            pickle.dump(ipo_ib_results, f)

        now_ist = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
        with open(LAST_UPDATED_FILE, "w") as f:
            f.write(now_ist)

        # Snapshot — one row per passing stock, keyed by today's date.
        # force=True so re-runs (config/weight changes) always update today's entry.
        _write_status("screener", "running", "Saving daily snapshot…")
        from src.tracker import save_snapshot
        n_snap = save_snapshot(results, force=True)
        if n_snap:
            logger.info(f"Snapshot: {n_snap} stocks captured")

        _write_status("screener", "done", f"{len(results)} stocks passed | {len(ipo_results)} IPO bases — {now_ist}")
        logger.info(f"Screener complete: {len(results)} stocks passed")

    except Exception as exc:
        logger.exception("Screener failed")
        _write_status("screener", "error", str(exc))


def _run_full_pipeline(triggered_by: str = "scheduler") -> None:
    """Chains data fetch → screener. Used by the 4 PM scheduler."""
    _run_data_fetch(triggered_by)
    _run_screener_only(triggered_by)


def _run_sentiment_fetch(triggered_by: str = "manual") -> None:
    """Fetch sentiment for all screener-passing stocks (background task)."""
    try:
        results = _load_results()
        if results is None or results.empty:
            logger.warning("Sentiment: no screener results — run screener first")
            return
        symbols   = results["Symbol"].tolist()
        companies = (
            results["Company"].tolist()
            if "Company" in results.columns
            else symbols
        )
        from src.sentiment import batch_sentiment
        logger.info("Sentiment fetch started for %d stocks (%s)", len(symbols), triggered_by)
        batch_sentiment(symbols, companies)
        logger.info("Sentiment fetch complete")
    except Exception as exc:
        logger.exception("Sentiment fetch failed: %s", exc)


def _write_status(stage: str, state: str, message: str) -> None:
    """Update one stage's status inside the shared JSON file."""
    os.makedirs(DATA_DIR, exist_ok=True)
    current = _read_status()
    current[stage] = {"state": state, "message": message}
    with open(REFRESH_STATUS_FILE, "w") as f:
        json.dump(current, f)


def _read_status() -> dict:
    if os.path.exists(REFRESH_STATUS_FILE):
        try:
            with open(REFRESH_STATUS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "fetch":    {"state": "idle", "message": ""},
        "screener": {"state": "idle", "message": ""},
    }


def _start_scheduler() -> None:
    global _scheduler
    with _scheduler_lock:
        if _scheduler is not None:
            return
        _scheduler = BackgroundScheduler(timezone=IST)
        _scheduler.add_job(
            func=lambda: _run_full_pipeline("scheduler"),
            trigger=CronTrigger(
                hour=REFRESH_HOUR_IST,
                minute=REFRESH_MINUTE_IST,
                timezone=IST,
            ),
            id="daily_refresh",
            name=f"Daily refresh at {REFRESH_HOUR_IST:02d}:{REFRESH_MINUTE_IST:02d} IST",
            replace_existing=True,
        )
        _scheduler.start()
        logger.info(
            f"Scheduler started — daily refresh at {REFRESH_HOUR_IST:02d}:{REFRESH_MINUTE_IST:02d} IST"
        )


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_results() -> Optional[pd.DataFrame]:
    if os.path.exists(SCREENER_RESULTS_FILE):
        try:
            with open(SCREENER_RESULTS_FILE, "rb") as f:
                return pickle.load(f)
        except Exception as exc:
            logger.warning(f"screener_results.pkl corrupted ({exc}) — re-run screener")
            os.remove(SCREENER_RESULTS_FILE)
    return None


def _load_ipo_results() -> Optional[pd.DataFrame]:
    if os.path.exists(IPO_RESULTS_FILE):
        try:
            with open(IPO_RESULTS_FILE, "rb") as f:
                return pickle.load(f)
        except Exception as exc:
            logger.warning(f"ipo_results.pkl corrupted ({exc}) — re-run screener")
            os.remove(IPO_RESULTS_FILE)
    return None


def _load_ipo_ib_results() -> Optional[pd.DataFrame]:
    if os.path.exists(IPO_IB_RESULTS_FILE):
        try:
            with open(IPO_IB_RESULTS_FILE, "rb") as f:
                return pickle.load(f)
        except Exception as exc:
            logger.warning(f"ipo_ib_results.pkl corrupted ({exc}) — re-run screener")
            os.remove(IPO_IB_RESULTS_FILE)
    return None


def _last_updated() -> str:
    if os.path.exists(LAST_UPDATED_FILE):
        with open(LAST_UPDATED_FILE) as f:
            return f.read().strip()
    return "Never"


def _sector_outperformance_vs_nifty500() -> dict[str, float]:
    """
    Return {sector_name: outperformance_%} averaged across 1M and 3M periods.
    Outperformance = avg(sector_1M - nifty500_1M, sector_3M - nifty500_3M).
    Falls back to 0.0 if data is unavailable.
    """
    from src.sector_mapper import load_index_data, resolve_sector_index, index_return
    try:
        _, sector_indices, nifty500 = load_index_data()
        nifty500_1m = index_return(nifty500, MONTHLY_DAYS)
        nifty500_3m = index_return(nifty500, QUARTERLY_DAYS)

        results = _load_results()
        if results is None or results.empty:
            return {}

        outperf: dict[str, float] = {}
        for sector in results["Sector"].unique():
            idx_ticker = resolve_sector_index(sector)
            sector_df  = sector_indices.get(idx_ticker)
            sector_1m  = index_return(sector_df, MONTHLY_DAYS)
            sector_3m  = index_return(sector_df, QUARTERLY_DAYS)
            outperf[sector] = round(((sector_1m - nifty500_1m) + (sector_3m - nifty500_3m)) / 2, 2)
        return outperf
    except Exception:
        return {}


def _top_industries_vs_sector(
    results: pd.DataFrame,
    sector_indices: dict,
    top_n: int = 3,
) -> list[dict]:
    """
    For every Industry in the screener results, compute outperformance averaged
    across 1M and 3M periods:
      avg( avg(stock_1M%) - sector_1M,  avg(stock_3M%) - sector_3M )

    Returns a list of dicts sorted by outperformance descending, capped at top_n.
    """
    from src.sector_mapper import resolve_sector_index, index_return

    rows = []
    for (sector, industry), grp in results.groupby(["Sector", "Industry"]):
        if len(grp) < 1:
            continue
        avg_1m     = float(grp["Monthly %"].mean())
        avg_3m     = float(grp["Quarterly %"].mean()) if "Quarterly %" in grp.columns else avg_1m
        idx_ticker = resolve_sector_index(sector)
        sector_df  = sector_indices.get(idx_ticker)
        sector_1m  = index_return(sector_df, MONTHLY_DAYS)
        sector_3m  = index_return(sector_df, QUARTERLY_DAYS)
        outperf    = round(((avg_1m - sector_1m) + (avg_3m - sector_3m)) / 2, 2)
        rows.append({
            "industry": industry,
            "sector":   sector,
            "outperf":  outperf,
            "n_stocks": len(grp),
        })

    rows.sort(key=lambda r: r["outperf"], reverse=True)
    return rows[:top_n]


def _render_highlights(
    results: Optional[pd.DataFrame],
    sector_indices: dict,
) -> None:
    """Render Top 5 Industries by outperformance vs their sector index."""
    if results is None or results.empty:
        return

    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]

    st.markdown("### Market Highlights")
    st.markdown("**Top Industries vs their Sector**")

    top_inds = _top_industries_vs_sector(results, sector_indices, top_n=5)
    if top_inds:
        cols = st.columns(len(top_inds))
        for i, row in enumerate(top_inds):
            op           = row["outperf"]
            sign         = "+" if op >= 0 else ""
            delta_colour = "normal" if op >= 0 else "inverse"
            cols[i].metric(
                label=f"{medals[i]} {row['industry']}",
                value=f"{sign}{op:.2f}%",
                delta=f"{row['sector']}  ·  {row['n_stocks']} stocks",
                delta_color=delta_colour,
            )
    else:
        st.info("Run the screener to see industry data.")

    st.divider()


def _render_top_stocks(results: Optional[pd.DataFrame], sentiment_cache: dict = None, top_n: int = 10) -> None:
    """Render the Top N ranked stocks as two rows of 5 cards each."""
    if results is None or results.empty:
        return

    st.markdown("### Top Ranked Stocks")

    # Rank by weighted blend of Momentum Score and RS Rating (from config)
    df_top = results.copy()
    w_s  = COMBINED_RANK_WEIGHTS["score"]     / 100.0
    w_rs = COMBINED_RANK_WEIGHTS["rs_rating"] / 100.0
    if "RS Rating" in df_top.columns:
        df_top["_composite"] = df_top["Score"] * w_s + df_top["RS Rating"] * w_rs
    else:
        df_top["_composite"] = df_top["Score"]
    df_top = df_top.sort_values("_composite", ascending=False).head(top_n).reset_index(drop=True)

    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
    pat_cols = [c for c in PATTERN_COLS if c in df_top.columns]

    row_size = 5
    for row_start in range(0, top_n, row_size):
        row_slice = df_top.iloc[row_start : row_start + row_size]
        cols = st.columns(row_size)
        for col_idx, (_, row) in enumerate(row_slice.iterrows()):
            i = row_start + col_idx
            active_patterns = [p for p in pat_cols if row.get(p) is True]
            pct_h = row.get("% from 52W High", None)
            pct_h_str = f"{pct_h:.1f}% from high" if pct_h is not None else ""
            rsi   = row.get("RSI", None)
            rsi_str = f"RSI {rsi:.0f}" if rsi is not None else ""

            rs = row.get("RS Rating", None)
            rs_str = f"RS {int(rs)}" if rs is not None and not pd.isna(rs) else ""
            rs_color = (
                "#22c55e" if rs is not None and rs >= 80
                else "#f59e0b" if rs is not None and rs >= 50
                else "#ef4444"
            ) if rs is not None else "#6b7280"

            composite = row.get("_composite", row["Score"])

            # Sentiment badge from cache
            sent = (sentiment_cache or {}).get(row["Symbol"], {})
            sent_label = sent.get("label", "")
            sent_badge = (
                "<span style='font-size:0.7rem;background:#14532d;color:#4ade80;"
                "border-radius:4px;padding:1px 6px;margin-left:4px'>🟢 Bullish</span>"
                if sent_label == "Bullish" else
                "<span style='font-size:0.7rem;background:#450a0a;color:#f87171;"
                "border-radius:4px;padding:1px 6px;margin-left:4px'>🔴 Bearish</span>"
                if sent_label == "Bearish" else
                ""
            )

            with cols[col_idx]:
                st.markdown(
                    f"<div style='"
                    f"background:rgba(255,255,255,0.04);"
                    f"border:1px solid rgba(255,255,255,0.09);"
                    f"border-radius:10px;"
                    f"padding:14px 16px;"
                    f"height:100%'>"
                    f"<div style='font-size:0.75rem;color:#6b7280;font-weight:700;letter-spacing:0.06em'>"
                    f"{medals[i]} RANK {i+1}</div>"
                    f"<div style='font-size:1.3rem;font-weight:800;margin:4px 0 2px'>{row['Symbol']}{sent_badge}</div>"
                    f"<div style='font-size:0.78rem;color:#9ca3af;margin-bottom:8px'>{row.get('Sector','')}</div>"
                    f"<div style='font-size:1rem;font-weight:600'>₹{row['CMP']:,.2f}</div>"
                    f"<div style='font-size:0.85rem;color:#e2e8f0;font-weight:700;margin-top:2px'>Combined {composite:.1f}</div>"
                    f"<div style='display:flex;gap:10px;align-items:center;margin-top:1px'>"
                    f"<span style='font-size:0.75rem;color:#a78bfa'>Score {row['Score']:.1f}</span>"
                    f"<span style='font-size:0.75rem;color:{rs_color}'>{rs_str}</span>"
                    f"</div>"
                    f"<div style='font-size:0.75rem;color:#6b7280;margin-top:6px'>{pct_h_str}"
                    f"{'  ·  ' if pct_h_str and rsi_str else ''}{rsi_str}</div>"
                    + (
                        f"<div style='margin-top:8px;font-size:0.7rem;color:#4ade80'>"
                        f"{'  '.join(active_patterns)}</div>"
                        if active_patterns else ""
                    )
                    + "</div>",
                    unsafe_allow_html=True,
                )
        st.write("")

    st.divider()


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _colour_pct(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return "color: #2ecc71" if val > 0 else ("color: #e74c3c" if val < 0 else "")


def _colour_rsi(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    if val >= 70:
        return "color: #e74c3c; font-weight: bold"
    if val <= 30:
        return "color: #2ecc71; font-weight: bold"
    return ""


def _colour_rs(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    if val >= 80:
        return "color: #22c55e; font-weight: bold"
    if val >= 50:
        return "color: #f59e0b"
    return "color: #ef4444"


def _colour_rmv(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    if val < 15:
        return "color: #22c55e; font-weight: bold"   # tight / compressed
    if val < 40:
        return "color: #86efac"                       # moderately quiet
    if val > 75:
        return "color: #ef4444"                       # expanding / volatile
    return ""


def _tick(val: bool) -> str:
    return "✓" if val else "—"


def _colour_sentiment(val) -> str:
    v = str(val)
    if "Bullish" in v:
        return "background-color: #14532d; color: #4ade80; font-weight: 600"
    if "Bearish" in v:
        return "background-color: #450a0a; color: #f87171; font-weight: 600"
    return "color: #6b7280"


# Pattern columns are inserted between these two halves at render time.
# To add a new pattern: add it to config.PATTERN_COLS — no change needed here.
_DISPLAY_COLS_HEAD = [
    "Symbol", "Company", "CMP", "Score", "RS Rating",
    "Sentiment", "News Vol", "Reddit",
    "RSI", "RMV", "% from 52W High",
    "Weekly %", "Monthly %", "Benchmark Outperf %", "Sector Outperf %",
]
_DISPLAY_COLS_TAIL = ["PE", "Market Cap (Cr)"]

_STYLE_COLS = {
    "Weekly %":            _colour_pct,
    "Monthly %":           _colour_pct,
    "Benchmark Outperf %": _colour_pct,
    "Sector Outperf %":    _colour_pct,
    "RSI":                 _colour_rsi,
    "RS Rating":           _colour_rs,
    "RMV":                 _colour_rmv,
    "Sentiment":           _colour_sentiment,
}


_IPO_DISPLAY_COLS = [
    "Symbol", "Company", "Sector", "Industry",
    "CMP", "Base High", "% from Base High", "Days Listed",
    "Inside Day", "RMV", "PE", "RSI", "Weekly %", "Monthly %",
    "Sector Outperf %", "Market Cap (Cr)",
]

_IPO_NUM_FORMAT = {
    "CMP":               "{:.2f}",
    "Base High":         "{:.2f}",
    "% from Base High":  "{:.2f}",
    "PE":                "{:.2f}",
    "RSI":               "{:.2f}",
    "RMV":               "{:.1f}",
    "Weekly %":          "{:.2f}",
    "Monthly %":         "{:.2f}",
    "Sector Outperf %":  "{:.2f}",
    "Market Cap (Cr)":   "{:.0f}",
}

_IPO_STYLE_COLS = {
    "Weekly %":         _colour_pct,
    "Monthly %":        _colour_pct,
    "Sector Outperf %": _colour_pct,
    "RSI":              _colour_rsi,
}


def _render_stock_chart(symbol: str, price_data: dict, lookback: int = 90) -> None:
    """
    Render an interactive candlestick chart with EMA overlays and volume for
    the given symbol.  Uses the last `lookback` trading days.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ticker = f"{symbol}.NS"
    df_raw = price_data.get(ticker)
    if df_raw is None or df_raw.empty:
        st.warning(f"No price data available for {symbol}.")
        return

    df_full = df_raw.copy()
    df_full.index = pd.to_datetime(df_full.index)

    # Compute EMAs on full history so EMA200 is accurate, then slice for display
    close_full = df_full["Close"]
    ema = {
        10:  close_full.ewm(span=10,  adjust=False).mean().tail(lookback),
        20:  close_full.ewm(span=20,  adjust=False).mean().tail(lookback),
        50:  close_full.ewm(span=50,  adjust=False).mean().tail(lookback),
        200: close_full.ewm(span=200, adjust=False).mean().tail(lookback),
    }
    df = df_full.tail(lookback)
    ema_colors = {10: "#22c55e", 20: "#3b82f6", 50: "#f59e0b", 200: "#ef4444"}

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
    )

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price",
        increasing_line_color="#22c55e",
        decreasing_line_color="#ef4444",
        increasing_fillcolor="#22c55e",
        decreasing_fillcolor="#ef4444",
        line_width=1,
    ), row=1, col=1)

    # EMA lines
    for period, series in ema.items():
        fig.add_trace(go.Scatter(
            x=df.index, y=series,
            mode="lines",
            name=f"EMA{period}",
            line=dict(color=ema_colors[period], width=1.5),
        ), row=1, col=1)

    # Volume bars
    vol_colors = [
        "#22c55e" if c >= o else "#ef4444"
        for c, o in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        name="Volume",
        marker_color=vol_colors,
        marker_line_width=0,
        opacity=0.7,
    ), row=2, col=1)

    fig.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=36, b=0),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(color="#94a3b8", size=11),
        title=dict(
            text=f"<b>{symbol}</b> — last {lookback} trading days",
            font=dict(color="#f1f5f9", size=14),
            x=0.01,
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ),
        xaxis_rangeslider_visible=False,
        xaxis2=dict(showgrid=False),
    )
    fig.update_xaxes(
        gridcolor="#334155", showgrid=True,
        zeroline=False, tickfont=dict(size=10),
    )
    fig.update_yaxes(
        gridcolor="#334155", showgrid=True,
        zeroline=False, tickfont=dict(size=10),
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_ipo_section(ipo_df: Optional[pd.DataFrame]) -> None:
    """Render the IPO Base breakout section at the top of the dashboard."""
    st.subheader("🚀 IPO Base — Breakout Zone  (< 200 days since listing)")

    if ipo_df is None or ipo_df.empty:
        st.info("No IPO base stocks detected in the current scan.")
        st.divider()
        return

    available = [c for c in _IPO_DISPLAY_COLS if c in ipo_df.columns]
    display   = ipo_df[available].copy()
    if "Inside Day" in display.columns:
        display["Inside Day"] = display["Inside Day"].apply(_tick)

    styled = display.style.format(_IPO_NUM_FORMAT, na_rep="—")
    for col, fn in _IPO_STYLE_COLS.items():
        if col in display.columns:
            styled = styled.map(fn, subset=[col])

    # Highlight rows where price is above the base high (confirmed breakout)
    def _highlight_breakout(row):
        colour = "background-color: #1a3a1a" if row["% from Base High"] >= 0 else ""
        return [colour] * len(row)

    styled = styled.apply(_highlight_breakout, axis=1)

    st.caption(
        f"{len(ipo_df)} stock{'s' if len(ipo_df) != 1 else ''} approaching or clearing IPO base.  "
        "🟩 = price already above base high (confirmed breakout)."
    )
    st.dataframe(styled, width="stretch", height=min(38 * (len(display) + 1), 500))
    st.divider()


_NUM_FORMAT = {
    "CMP":                 "{:.2f}",
    "PE":                  "{:.2f}",
    "RSI":                 "{:.2f}",
    "RMV":                 "{:.1f}",
    "RS Rating":           "{:.0f}",
    "Weekly %":            "{:.2f}",
    "Monthly %":           "{:.2f}",
    "% from 52W High":     "{:.2f}",
    "Benchmark Outperf %": "{:.2f}",
    "Sector Outperf %":    "{:.2f}",
    "Market Cap (Cr)":     "{:.0f}",
    "Score":               "{:.2f}",
    "News Vol":            "{:.0f}",
    "Reddit":              "{:.0f}",
}


def _render_table(sub: pd.DataFrame) -> None:
    pat_cols = [c for c in PATTERN_COLS if c in sub.columns]
    ordered  = _DISPLAY_COLS_HEAD + pat_cols + _DISPLAY_COLS_TAIL
    available = [c for c in ordered if c in sub.columns]
    display = sub[available].copy()
    for col in pat_cols:
        display[col] = display[col].apply(_tick)

    styled = display.style.format(_NUM_FORMAT, na_rep="—")
    for col, fn in _STYLE_COLS.items():
        if col in display.columns:
            styled = styled.map(fn, subset=[col])

    st.dataframe(
        styled,
        width="stretch",
        height=min(38 * (len(display) + 1), 600),
    )



# ---------------------------------------------------------------------------
# Performance Tracker tab
# ---------------------------------------------------------------------------

# Pattern columns are appended at render time from PATTERN_COLS.
_TRACKER_BASE_COLS = [
    "Capture Date", "Symbol", "Company", "Sector",
    "CMP", "Combined Score", "Score", "RS Rating",
    "Ret 1D %", "Ret 1W %", "Ret 2W %", "Ret 1M %",
    "RSI", "RMV", "% from 52W High",
]

_TRACKER_NUM_FORMAT = {
    "CMP":              "{:.2f}",
    "Combined Score":   "{:.1f}",
    "Score":            "{:.2f}",
    "RS Rating":        "{:.0f}",
    "RSI":              "{:.1f}",
    "RMV":              "{:.1f}",
    "% from 52W High":  "{:.2f}",
    "Ret 1D %":         "{:+.2f}",
    "Ret 1W %":         "{:+.2f}",
    "Ret 2W %":         "{:+.2f}",
    "Ret 1M %":         "{:+.2f}",
}


def _colour_ret(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "color: #6b7280"
    return "color: #22c55e" if val > 0 else ("color: #ef4444" if val < 0 else "")


def _render_tracker_tab() -> None:
    from src.tracker import load_snapshots, compute_forward_returns, pattern_summary, score_band_summary
    from src.data_fetcher import load_price_data

    snapshots = load_snapshots()

    if snapshots.empty:
        st.info(
            "No snapshots captured yet. "
            "Run the screener (🔍 Run Screener) to start recording daily results. "
            "Forward returns are computed automatically as new price data arrives."
        )
        return

    # Compute forward returns against the latest price cache
    with st.spinner("Computing forward returns…"):
        price_data = load_price_data() or {}
        df = compute_forward_returns(snapshots, price_data)

    # ── Summary metrics ───────────────────────────────────────────────────
    n_dates    = df["Capture Date"].nunique()
    n_captures = len(df)
    w1 = df["Ret 1W %"].dropna() if "Ret 1W %" in df.columns else pd.Series(dtype=float)
    m1 = df["Ret 1M %"].dropna() if "Ret 1M %" in df.columns else pd.Series(dtype=float)
    wr_1w = f"{(w1 > 0).mean() * 100:.0f}%" if len(w1) else "—"
    wr_1m = f"{(m1 > 0).mean() * 100:.0f}%" if len(m1) else "—"
    avg_1w = f"{w1.mean():+.2f}%" if len(w1) else "—"
    avg_1m = f"{m1.mean():+.2f}%" if len(m1) else "—"

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Capture Days",   n_dates)
    c2.metric("Total Captures", n_captures)
    c3.metric("Win Rate 1W",    wr_1w)
    c4.metric("Avg Return 1W",  avg_1w)
    c5.metric("Win Rate 1M",    wr_1m)
    c6.metric("Avg Return 1M",  avg_1m)

    st.divider()

    # ── Analysis panels ───────────────────────────────────────────────────
    col_pat, col_score = st.columns(2)

    with col_pat:
        st.markdown("##### Pattern Effectiveness")
        pat_df = pattern_summary(df)
        if pat_df.empty:
            st.caption("Not enough data yet.")
        else:
            st.dataframe(pat_df, width="stretch", hide_index=True)

    with col_score:
        st.markdown("##### Score Band Performance")
        sb_df = score_band_summary(df)
        if sb_df.empty:
            st.caption("Not enough data yet.")
        else:
            st.dataframe(sb_df, width="stretch", hide_index=True)

    st.divider()

    # ── Capture log ───────────────────────────────────────────────────────
    st.markdown("##### Capture Log")

    all_dates = sorted(df["Capture Date"].unique(), reverse=True)
    f1, f2, f3 = st.columns([2, 2, 3])
    with f1:
        sel_dates = st.multiselect("Date", all_dates, default=all_dates[:5],
                                   placeholder="All dates")
    with f2:
        sel_sector = st.multiselect("Sector", sorted(df["Sector"].unique()),
                                    placeholder="All sectors")
    with f3:
        sym_search = st.text_input("Symbol search", placeholder="e.g. INFY")

    view = df.copy()
    if sel_dates:
        view = view[view["Capture Date"].isin(sel_dates)]
    if sel_sector:
        view = view[view["Sector"].isin(sel_sector)]
    if sym_search:
        view = view[view["Symbol"].str.contains(sym_search.strip().upper(), na=False)]

    # Compute Combined Score using configured weights for display
    w_s  = COMBINED_RANK_WEIGHTS["score"]     / 100.0
    w_rs = COMBINED_RANK_WEIGHTS["rs_rating"] / 100.0
    if "Score" in view.columns and "RS Rating" in view.columns:
        view["Combined Score"] = (view["Score"] * w_s + view["RS Rating"] * w_rs).round(1)
    elif "Score" in view.columns:
        view["Combined Score"] = view["Score"].round(1)

    pat_cols  = [c for c in PATTERN_COLS if c in view.columns]
    ordered   = _TRACKER_BASE_COLS + pat_cols
    available = [c for c in ordered if c in view.columns]
    display   = view[available].copy()

    for col in pat_cols:
        display[col] = display[col].apply(_tick)

    styled = display.style.format(_TRACKER_NUM_FORMAT, na_rep="—")
    for ret_col in ("Ret 1D %", "Ret 1W %", "Ret 2W %", "Ret 1M %"):
        if ret_col in display.columns:
            styled = styled.map(_colour_ret, subset=[ret_col])
    if "RS Rating" in display.columns:
        styled = styled.map(_colour_rs, subset=["RS Rating"])
    if "RMV" in display.columns:
        styled = styled.map(_colour_rmv, subset=["RMV"])

    st.dataframe(styled, width="stretch",
                 height=min(38 * (len(display) + 1), 600))

    st.divider()
    col_dl, _ = st.columns([2, 5])
    with col_dl:
        st.download_button(
            label=f"⬇  Export tracker data  ({len(view)} rows)",
            data=view.to_csv(index=False),
            file_name="screener_tracker.csv",
            mime="text/csv",
            key="dl_tracker",
        )


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

def _inject_css() -> None:
    st.markdown("""<style>
/* ── Metric cards ────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 14px 16px !important;
}
[data-testid="stMetricLabel"] > div {
    font-size: 0.7rem !important;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #9ca3af !important;
}
[data-testid="stMetricValue"] > div {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}
/* ── Download button ─────────────────────────────────────── */
[data-testid="stDownloadButton"] > button {
    width: 100%;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.14);
    color: #e2e8f0;
    border-radius: 8px;
    font-size: 0.9rem;
    padding: 10px 0;
    margin-top: 4px;
    transition: background 0.2s, border-color 0.2s;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(255,255,255,0.09);
    border-color: rgba(255,255,255,0.28);
}
/* ── Tabs ────────────────────────────────────────────────── */
button[data-baseweb="tab"] {
    font-size: 0.97rem !important;
    font-weight: 500 !important;
    padding: 8px 22px !important;
}
/* ── Expander header ─────────────────────────────────────── */
details > summary p {
    font-size: 0.97rem !important;
    font-weight: 600 !important;
}
/* ── Sidebar section labels ──────────────────────────────── */
.sidebar-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6b7280;
    margin-bottom: 6px;
}

/* ── Responsive: tablet ≤ 768 px ────────────────────────── */
@media (max-width: 768px) {
    /* Wrap every st.columns() row into a two-column flex grid */
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        gap: 0.5rem !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        flex: 1 1 calc(50% - 0.5rem) !important;
        min-width: calc(50% - 0.5rem) !important;
        width: auto !important;
    }
    /* Shrink metric value font so cards don't overflow */
    [data-testid="stMetricValue"] > div {
        font-size: 1.1rem !important;
    }
    /* Plotly chart — reduce height on tablet */
    [data-testid="stPlotlyChart"] > div {
        height: 380px !important;
    }
    [data-testid="stPlotlyChart"] .js-plotly-plot,
    [data-testid="stPlotlyChart"] .plot-container {
        height: 380px !important;
    }
    /* Tab labels — smaller so they fit */
    button[data-baseweb="tab"] {
        font-size: 0.8rem !important;
        padding: 6px 12px !important;
    }
}

/* ── Responsive: phone ≤ 480 px ─────────────────────────── */
@media (max-width: 480px) {
    /* Collapse every column row to full-width stacked cards */
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }
    /* Stock cards in Top Ranked section — reduce padding */
    [data-testid="stHorizontalBlock"] div[style*="border-radius:10px"] {
        padding: 10px 12px !important;
    }
    /* Plotly chart — compact on phone */
    [data-testid="stPlotlyChart"] > div,
    [data-testid="stPlotlyChart"] .js-plotly-plot,
    [data-testid="stPlotlyChart"] .plot-container {
        height: 300px !important;
    }
    /* Readable tab labels on small screens */
    button[data-baseweb="tab"] {
        font-size: 0.72rem !important;
        padding: 5px 8px !important;
    }
    /* Chip panels — allow wrapping without overflow */
    div[style*="line-height:2.2"] {
        line-height: 2 !important;
    }
}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main Streamlit app
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Momentum Stock Screener",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _start_scheduler()
    _inject_css()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 📈 Momentum Screener")
        st.caption("EQ universe · Auto-refresh 16:00 IST")
        st.divider()

        from src.data_fetcher import is_price_data_fresh, price_data_last_fetched
        from src.sector_mapper import is_index_data_fresh

        # Display filters — generated from PATTERN_COLS; add a new pattern
        # to config.py and a checkbox appears here automatically
        st.markdown('<p class="sidebar-label">🔍 Display Filters</p>', unsafe_allow_html=True)
        pattern_filters = {pat: st.checkbox(pat) for pat in PATTERN_COLS}
        st.write("")
        min_score = st.slider("Min Score", 0, 100, 50, step=5)
        min_rs    = st.slider("Min RS Rating", 1, 99, 75)
        rsi_range = st.slider("RSI Range", 0, 100, (0, 100))
        max_rmv   = st.select_slider(
                        "Max RMV",
                        options=list(range(100, -1, -1)),
                        value=100,
                        help="RMV < 15 = compressed / tight bar. Drag right for tighter filter.",
                    )

        _CAP_OPTIONS = [
            2_000, 3_000, 5_000, 7_500, 10_000, 15_000, 20_000,
            30_000, 50_000, 75_000, 1_00_000, 1_50_000, 2_00_000,
            3_00_000, 5_00_000,
        ]
        def _fmt_cap(v):
            if v >= 1_00_000: return f"₹{v//1_00_000}L Cr"
            if v >= 1_000:    return f"₹{v//1_000}K Cr"
            return f"₹{v} Cr"

        cap_range = st.select_slider(
            "Market Cap Range",
            options=_CAP_OPTIONS,
            value=(2_000, 20_000),
            format_func=_fmt_cap,
            help="Small Cap < ₹10K Cr  ·  Mid Cap ₹10K–₹20K Cr  ·  Large Cap > ₹20K Cr",
        )
        cap_min, cap_max = cap_range

        st.divider()

        # Criteria reference
        with st.expander("📋 Screener Criteria", expanded=False):
            st.markdown("""
**Hard filters**
- Market Cap > ₹2,000 Cr
- Price > 52W High × 0.76
- EMA 10 > 20 > 50 > 200
- Price > 20 EMA

**Ranking**
- Tight range + vol contraction: 30
- Benchmark outperf (1M+3M): 25
- Sector outperf (1M+3M): 25
- 52W High proximity: 20
            """)

        # Data management — collapsible, at the bottom
        data_fresh = is_price_data_fresh() and is_index_data_fresh()
        age        = price_data_last_fetched()
        dot_color  = "#22c55e" if data_fresh else "#f59e0b"
        dm_label   = f"⚙️ Data Management  {'✓' if data_fresh else '⚠'}"
        with st.expander(dm_label, expanded=False):
            status    = _read_status()
            fetch_st  = status.get("fetch", {})
            screen_st = status.get("screener", {})
            if fetch_st.get("state") == "running":
                st.info(f"📥 {fetch_st['message']}")
            elif screen_st.get("state") == "running":
                st.info(f"🔍 {screen_st['message']}")
            elif fetch_st.get("state") == "error":
                st.error(f"❌ {fetch_st['message']}")
            elif screen_st.get("state") == "error":
                st.error(f"❌ {screen_st['message']}")

            fetch_label = "📥 Fetch Data  ✓" if data_fresh else "📥 Fetch Data"
            if st.button(fetch_label, width="stretch",
                         help="Downloads price data, stock info and index data. 10–30 min on first run."):
                st.toast("Data fetch started in background…")
                threading.Thread(target=_run_data_fetch, args=("manual",), daemon=True).start()

            st.markdown(
                f"<small style='color:{dot_color}'>● {age}</small>",
                unsafe_allow_html=True,
            )
            st.write("")

            has_data = os.path.exists(os.path.join(DATA_DIR, "price_data.pkl"))
            if st.button("🔍 Run Screener", type="primary", width="stretch",
                         disabled=not has_data,
                         help="Applies all filters and pattern detection on cached data. Runs in seconds."):
                st.toast("Screener started…")
                threading.Thread(target=_run_screener_only, args=("manual",), daemon=True).start()
            if not has_data:
                st.caption("⚠ Fetch data first.")

            has_results = os.path.exists(SCREENER_RESULTS_FILE)
            if st.button("🧠 Fetch Sentiment", width="stretch",
                         disabled=not has_results,
                         help="Pulls news from yfinance, Google News & Reddit for screener stocks. ~2 min."):
                st.toast("Sentiment fetch started in background…")
                threading.Thread(target=_run_sentiment_fetch, args=("manual",), daemon=True).start()
            if not has_results:
                st.caption("⚠ Run screener first.")

    # ── Header ───────────────────────────────────────────────────────────────
    col_title, col_meta = st.columns([5, 2])
    with col_title:
        st.markdown("# 📈 Momentum Stock Screener")
        st.caption("EMA Stack 10 › 20 › 50 › 200  ·  " + "  ·  ".join(PATTERN_COLS) + "  ·  Sector Outperformance")
    with col_meta:
        last = _last_updated()
        st.markdown(
            f"<div style='text-align:right; padding-top:14px'>"
            f"<span style='color:#6b7280; font-size:0.8rem'>LAST UPDATED</span><br>"
            f"<span style='color:#d1d5db; font-size:0.88rem; font-weight:600'>{last}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.divider()

    # ── Load data ─────────────────────────────────────────────────────────────
    results        = _load_results()
    ipo_results    = _load_ipo_results()
    ipo_ib_results = _load_ipo_ib_results()

    # Load sentiment cache (empty dict if not yet fetched)
    try:
        from src.sentiment import load_sentiment
        sentiment_cache = load_sentiment()
    except Exception:
        sentiment_cache = {}

    try:
        from src.sector_mapper import load_index_data
        _, sector_indices, _ = load_index_data()
    except Exception:
        sector_indices = {}

    sector_outperf = _sector_outperformance_vs_nifty500()

    # ── Market Highlights ─────────────────────────────────────────────────────
    _render_highlights(results, sector_indices)

    # ── Top Ranked Stocks ─────────────────────────────────────────────────────
    _render_top_stocks(results, sentiment_cache)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_screen, tab_ipo, tab_track = st.tabs([
        "  📊  Screener Results  ",
        "  🚀  IPO Setups  ",
        "  📈  Performance Tracker  ",
    ])

    # ── Tab 1 — Screener ──────────────────────────────────────────────────────
    with tab_screen:
        if results is None or results.empty:
            st.warning(
                "No screener results yet. "
                "Click **📥 Fetch Data** then **🔍 Run Screener** in the sidebar."
            )
        else:
            # Apply pattern filters (driven by PATTERN_COLS — no edits needed here
            # when a new pattern is added)
            df = results.copy()

            # Merge sentiment columns if data is available
            if sentiment_cache:
                df["Sentiment"] = df["Symbol"].map(
                    lambda s: (
                        ("🟢 Bullish" if sentiment_cache[s]["label"] == "Bullish" else
                         "🔴 Bearish" if sentiment_cache[s]["label"] == "Bearish" else
                         "⚪ Neutral")
                        if s in sentiment_cache else ""
                    )
                )
                df["News Vol"] = df["Symbol"].map(
                    lambda s: sentiment_cache[s]["article_count"] if s in sentiment_cache else None
                )
                df["Reddit"] = df["Symbol"].map(
                    lambda s: sentiment_cache[s]["reddit_mentions"] if s in sentiment_cache else None
                )

            for pat, active in pattern_filters.items():
                if active and pat in df.columns:
                    df = df[df[pat] == True]
            df = df[df["Score"] >= min_score]
            if "RS Rating" in df.columns and min_rs > 1:
                df = df[df["RS Rating"] >= min_rs]
            if "RMV" in df.columns and max_rmv < 100:
                df = df[df["RMV"].isna() | (df["RMV"] <= max_rmv)]
            if "Market Cap (Cr)" in df.columns:
                df = df[
                    (df["Market Cap (Cr)"] >= cap_min) &
                    (df["Market Cap (Cr)"] <= cap_max)
                ]
            if "RSI" in df.columns:
                df = df[
                    df["RSI"].isna()
                    | ((df["RSI"] >= rsi_range[0]) & (df["RSI"] <= rsi_range[1]))
                ]

            if df.empty:
                st.info("No stocks match the current filters.")
            else:
                # Summary metrics — 2 fixed + one per pattern
                n_stocks  = len(df)
                n_sectors = df["Sector"].nunique()
                metric_cols = st.columns(2 + len(PATTERN_COLS))
                metric_cols[0].metric("Stocks",  n_stocks)
                metric_cols[1].metric("Sectors", n_sectors)
                for i, pat in enumerate(PATTERN_COLS):
                    count = int(df[pat].sum()) if pat in df.columns else 0
                    metric_cols[i + 2].metric(pat, count)

                st.write("")

                # ── Top Industries vs their Sector panel ──────────────────
                sectors_sorted = sorted(
                    df["Sector"].unique(),
                    key=lambda s: sector_outperf.get(s, float("-inf")),
                    reverse=True,
                )
                top_inds = _top_industries_vs_sector(df, sector_indices, top_n=10)
                chips_html = ""
                for _row in top_inds:
                    _op   = _row["outperf"]
                    _clr  = "#22c55e" if _op >= 0 else "#ef4444"
                    _sign = "+" if _op >= 0 else ""
                    chips_html += (
                        f"<span style='display:inline-block;"
                        f"background:rgba(255,255,255,0.04);"
                        f"border:1px solid {_clr}44;border-radius:20px;"
                        f"padding:3px 12px;font-size:0.74rem;"
                        f"margin:2px 3px;white-space:nowrap'>"
                        f"<span style='color:#e2e8f0'>{_row['industry']}</span>"
                        f"&nbsp;<span style='color:{_clr};font-weight:700'>"
                        f"{_sign}{_op:.1f}%</span>"
                        f"&nbsp;<span style='color:#6b7280'>·&nbsp;{_row['sector']}"
                        f"&nbsp;·&nbsp;{_row['n_stocks']}</span>"
                        f"</span>"
                    )
                if chips_html:
                    st.markdown(
                        f"<div style='margin-bottom:10px;line-height:2.2'>"
                        f"<span style='font-size:0.72rem;color:#6b7280;font-weight:700;"
                        f"text-transform:uppercase;letter-spacing:0.06em;"
                        f"margin-right:8px'>Top Industries vs their Sector</span>"
                        f"{chips_html}</div>",
                        unsafe_allow_html=True,
                    )

                # ── Group-by control ──────────────────────────────────────────
                gb_col, _ = st.columns([3, 5])
                with gb_col:
                    group_by = st.radio(
                        "Group by",
                        ["None", "Sector", "Industry"],
                        index=1,
                        horizontal=True,
                        label_visibility="visible",
                    )

                # ── Build sorted display DataFrame ────────────────────────────
                _disp = df.copy().reset_index(drop=True)

                if group_by == "Sector":
                    _s_rank = {s: i for i, s in enumerate(sectors_sorted)}
                    _disp["_g"] = _disp["Sector"].map(_s_rank)
                    _disp = _disp.sort_values(["_g", "Score"], ascending=[True, False]) \
                                 .drop(columns=["_g"]).reset_index(drop=True)
                elif group_by == "Industry":
                    _s_rank = {s: i for i, s in enumerate(sectors_sorted)}
                    _disp["_g"] = _disp["Sector"].map(_s_rank)
                    _disp = _disp.sort_values(["_g", "Industry", "Score"],
                                              ascending=[True, True, False]) \
                                 .drop(columns=["_g"]).reset_index(drop=True)
                else:
                    _disp = _disp.sort_values("Score", ascending=False).reset_index(drop=True)

                # Group values for alternating row stripes
                if group_by == "Sector":
                    _gvals = _disp["Sector"].tolist()
                elif group_by == "Industry":
                    _gvals = _disp["Industry"].tolist()
                else:
                    _gvals = []
                _guniq    = list(dict.fromkeys(_gvals))
                _gidx_map = {g: i for i, g in enumerate(_guniq)}

                # Column ordering — prepend group key(s) so they're visible
                _pat_cols = [c for c in PATTERN_COLS if c in _disp.columns]
                if group_by == "Sector":
                    _head = ["Sector"] + _DISPLAY_COLS_HEAD
                elif group_by == "Industry":
                    _head = ["Sector", "Industry"] + _DISPLAY_COLS_HEAD
                else:
                    _head = list(_DISPLAY_COLS_HEAD)
                _ordered   = _head + _pat_cols + _DISPLAY_COLS_TAIL
                _available = [c for c in _ordered if c in _disp.columns]
                _display   = _disp[_available].copy()
                for _c in _pat_cols:
                    _display[_c] = _display[_c].apply(_tick)

                _styled = _display.style.format(_NUM_FORMAT, na_rep="—")
                for _col, _fn in _STYLE_COLS.items():
                    if _col in _display.columns:
                        _styled = _styled.map(_fn, subset=[_col])

                # Alternating background stripe per group
                if _gvals:
                    def _row_bg(row, gvals=_gvals, gmap=_gidx_map):
                        shade = gmap.get(gvals[row.name], 0) % 2
                        bg    = "rgba(255,255,255,0.05)" if shade else "rgba(0,0,0,0)"
                        return [f"background-color:{bg}"] * len(row)
                    _styled = _styled.apply(_row_bg, axis=1)

                st.dataframe(
                    _styled,
                    width="stretch",
                    height=min(38 * (len(_display) + 1), 820),
                )

                st.divider()
                col_dl, col_spacer = st.columns([2, 5])
                with col_dl:
                    st.download_button(
                        label=f"⬇  Export {len(df)} symbols  (.txt)",
                        data=",".join(df["Symbol"].tolist()),
                        file_name="symbols.txt",
                        mime="text/plain",
                    )

                # ── Chart View ────────────────────────────────────────────────
                st.divider()
                st.markdown("### 📈 Chart View")
                chart_col1, chart_col2 = st.columns([3, 1])
                with chart_col1:
                    ranked_symbols = (
                        df.sort_values("Score", ascending=False)["Symbol"].tolist()
                    )
                    chart_symbol = st.selectbox(
                        "Select stock",
                        ["— select a stock —"] + ranked_symbols,
                        label_visibility="collapsed",
                        key="chart_symbol_select",
                    )
                with chart_col2:
                    chart_lookback = st.select_slider(
                        "Period",
                        options=[30, 60, 90, 120, 180, 252],
                        value=90,
                        format_func=lambda x: f"{x}D",
                        key="chart_lookback_select",
                    )

                if chart_symbol != "— select a stock —":
                    from src.data_fetcher import load_price_data as _load_pd
                    with st.spinner(f"Loading chart for {chart_symbol}…"):
                        _pd = _load_pd()
                    if _pd:
                        _render_stock_chart(chart_symbol, _pd, lookback=chart_lookback)
                    else:
                        st.warning("Price data not available — run Fetch Data first.")

                    # ── News Headlines ────────────────────────────────────────
                    sent_data = sentiment_cache.get(chart_symbol, {})
                    headlines = sent_data.get("headlines", [])
                    if headlines:
                        score_val  = sent_data.get("score", 0.0)
                        label      = sent_data.get("label", "Neutral")
                        reddit_ct  = sent_data.get("reddit_mentions", 0)
                        label_html = (
                            "<span style='color:#4ade80;font-weight:700'>🟢 Bullish</span>"
                            if label == "Bullish" else
                            "<span style='color:#f87171;font-weight:700'>🔴 Bearish</span>"
                            if label == "Bearish" else
                            "<span style='color:#9ca3af'>⚪ Neutral</span>"
                        )
                        st.markdown(
                            f"**Sentiment** — {label_html} &nbsp;·&nbsp; "
                            f"<span style='color:#94a3b8'>VADER score: {score_val:+.3f}</span> &nbsp;·&nbsp; "
                            f"<span style='color:#94a3b8'>{len(headlines)} articles &nbsp; "
                            f"Reddit buzz: {reddit_ct}</span>",
                            unsafe_allow_html=True,
                        )
                        st.markdown("---")
                        from src.sentiment import _vader
                        for title, h_score in headlines:
                            dot = "🟢" if h_score >= 0.05 else ("🔴" if h_score <= -0.05 else "⚪")
                            st.markdown(
                                f"<div style='font-size:0.85rem;padding:3px 0;color:#cbd5e1'>"
                                f"{dot}&nbsp; {title}"
                                f"<span style='color:#475569;font-size:0.75rem;margin-left:8px'>"
                                f"[{h_score:+.2f}]</span></div>",
                                unsafe_allow_html=True,
                            )
                    elif chart_symbol:
                        st.caption("No sentiment data — click 🧠 Fetch Sentiment in the sidebar.")

    # ── Tab 2 — IPO Base ──────────────────────────────────────────────────────
    with tab_ipo:
        _render_ipo_section(ipo_results)

        # ── Inside Bar + Low RMV Setups (IPO-age stocks only) ────────────────
        st.subheader("🕯️ Inside Bar + Low RMV  —  IPO-age stocks")
        st.caption(
            f"Recently listed stocks (< {IPO_BASE_MAX_DAYS} trading days) with a daily "
            f"Inside Bar AND RMV ≤ {RMV_TIGHT_THRESHOLD * 2} — compressed volatility, "
            "potential breakout setup. No IPO-base or market-cap filter applied."
        )

        if ipo_ib_results is None or ipo_ib_results.empty:
            st.info("Run the screener first to populate this section.")
        else:
            _ib_cols = [
                "Symbol", "Company", "Sector", "Industry",
                "CMP", "% from Base High", "Days Listed",
                "EMA10 > EMA20", "RMV", "RSI",
                "Weekly %", "Monthly %", "Sector Outperf %", "Market Cap (Cr)",
            ]
            _ib_fmt = {
                "CMP":               "{:.2f}",
                "% from Base High":  "{:.2f}",
                "RMV":               "{:.1f}",
                "RSI":               "{:.1f}",
                "Weekly %":          "{:.2f}",
                "Monthly %":         "{:.2f}",
                "Sector Outperf %":  "{:.2f}",
                "Market Cap (Cr)":   "{:.0f}",
            }
            available = [c for c in _ib_cols if c in ipo_ib_results.columns]
            display   = ipo_ib_results[available].copy()
            if "EMA10 > EMA20" in display.columns:
                display["EMA10 > EMA20"] = display["EMA10 > EMA20"].apply(_tick)

            styled = display.style.format(_ib_fmt, na_rep="—")
            if "RMV" in display.columns:
                styled = styled.map(_colour_rmv, subset=["RMV"])
            for col in ("Weekly %", "Monthly %", "Sector Outperf %"):
                if col in display.columns:
                    styled = styled.map(_colour_pct, subset=[col])
            if "RSI" in display.columns:
                styled = styled.map(_colour_rsi, subset=["RSI"])

            st.caption(
                f"{len(display)} setup{'s' if len(display) != 1 else ''}  ·  "
                "sorted by RMV ascending (tightest first)"
            )
            st.dataframe(styled, width="stretch", height=min(38 * (len(display) + 1), 500))

    # ── Tab 3 — Performance Tracker ───────────────────────────────────────────
    with tab_track:
        _render_tracker_tab()

    # ── Footer ────────────────────────────────────────────────────────────────
    st.caption(
        "Data: Yahoo Finance (yfinance)  ·  Benchmark: NiftyMidSmallCap400  ·  "
        "EMA = Daily Exponential Moving Average  ·  VCP = Volatility Contraction Pattern (Minervini)"
    )


if __name__ == "__main__":
    main()
