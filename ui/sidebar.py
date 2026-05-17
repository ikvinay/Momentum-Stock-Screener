"""
Shared sidebar renderer.

render_data_management() — Admin page only. Shows data pipeline controls.
render_stock_filters()   — Stocks page sidebar filters.
render_index_filters()   — Themes page sidebar filters.
render_commodity_filters() — Commodities page sidebar filters.
"""

import os
import threading

import streamlit as st

from config import DATA_DIR, SCREENER_RESULTS_FILE, PATTERN_COLS
from src.pipeline import (
    run_data_fetch, run_screener_only, run_sentiment_fetch,
    run_commodity_pipeline, run_freefloat_refresh, read_status,
)


def _sidebar_section(label: str) -> None:
    """Render a small uppercase section label for the sidebar."""
    st.markdown(
        f'<div style="font-size:10px;font-weight:700;letter-spacing:.1em;'
        f'text-transform:uppercase;color:#334155;margin:14px 4px 6px;">{label}</div>',
        unsafe_allow_html=True,
    )


def _status_badge(state: str, message: str) -> None:
    """Compact inline status line for running / error states."""
    if state == "running":
        st.markdown(
            f'<div style="font-size:11.5px;color:#f59e0b;padding:4px 0 2px;">'
            f'<i class="ti ti-loader-2"></i> {message}</div>',
            unsafe_allow_html=True,
        )
    elif state == "error":
        st.markdown(
            f'<div style="font-size:11.5px;color:#ef4444;padding:4px 0 2px;">'
            f'<i class="ti ti-circle-x"></i> {message}</div>',
            unsafe_allow_html=True,
        )
    elif state == "done":
        st.markdown(
            f'<div style="font-size:11px;color:#22c55e;padding:2px 0;">'
            f'<i class="ti ti-circle-check"></i> {message}</div>',
            unsafe_allow_html=True,
        )


def render_data_management() -> None:
    """Render the sidebar brand header + data pipeline controls."""
    from src.data_fetcher import is_price_data_fresh, price_data_last_fetched
    from src.sector_mapper import is_index_data_fresh

    data_fresh = is_price_data_fresh() and is_index_data_fresh()
    age        = price_data_last_fetched()
    dot_color  = "#22c55e" if data_fresh else "#f59e0b"

    status       = read_status()
    fetch_st     = status.get("fetch",     {})
    screen_st    = status.get("screener",  {})
    commodity_st = status.get("commodity", {})
    freefloat_st = status.get("freefloat", {})

    # ── Data Controls (collapsible) ───────────────────────────────────────────
    with st.expander("Data Controls", expanded=False):

        # Stocks
        _sidebar_section("Stocks & Indices")
        _status_badge(fetch_st.get("state", ""), fetch_st.get("message", ""))
        fetch_label = "Refresh Stock Data ✓" if data_fresh else "Refresh Stock Data"
        if st.button(fetch_label, use_container_width=True,
                     help="Downloads price data, stock info and index data. 10–30 min on first run."):
            st.toast("Stock data fetch started in background…")
            threading.Thread(target=run_data_fetch, args=("manual",), daemon=True).start()
        st.markdown(
            f'<div style="font-size:11px;color:{dot_color};margin-top:3px;">● {age}</div>',
            unsafe_allow_html=True,
        )

        # Free Float
        st.markdown(
            '<hr style="border:none;border-top:1px solid rgba(255,255,255,0.06);margin:10px 0 4px;"/>',
            unsafe_allow_html=True,
        )
        _sidebar_section("Free Float %")
        _status_badge(freefloat_st.get("state", ""), freefloat_st.get("message", ""))
        ff_busy = freefloat_st.get("state") == "running"
        if st.button("Refresh Free Float Data", use_container_width=True, disabled=ff_busy,
                     help="Full re-fetch of NSE shareholding % (~8 min). Auto-runs every Saturday 10:00 IST."):
            st.toast("Free float full refresh started in background…")
            threading.Thread(target=run_freefloat_refresh, args=("manual",), daemon=True).start()

        # Commodities
        st.markdown(
            '<hr style="border:none;border-top:1px solid rgba(255,255,255,0.06);margin:10px 0 4px;"/>',
            unsafe_allow_html=True,
        )
        _sidebar_section("Commodities")
        _status_badge(commodity_st.get("state", ""), commodity_st.get("message", ""))
        if st.button("Refresh Commodities Data", use_container_width=True,
                     help="Fetches MCX commodity OHLCV and re-runs screener. Auto-runs at 23:45 IST."):
            st.toast("Commodity refresh started in background…")
            threading.Thread(target=run_commodity_pipeline, args=("manual",), daemon=True).start()

        # Run Screener
        st.markdown(
            '<hr style="border:none;border-top:1px solid rgba(255,255,255,0.06);margin:10px 0 4px;"/>',
            unsafe_allow_html=True,
        )
        _sidebar_section("Screener")
        _status_badge(screen_st.get("state", ""), screen_st.get("message", ""))
        has_data   = os.path.exists(os.path.join(DATA_DIR, "price_data.pkl"))
        fetch_busy = fetch_st.get("state") == "running"
        if st.button("Run Screener", use_container_width=True,
                     disabled=not (has_data and not fetch_busy),
                     help="Applies all filters and pattern detection on cached price data. Runs in seconds."):
            st.toast("Screener started…")
            threading.Thread(target=run_screener_only, args=("manual",), daemon=True).start()
        if not has_data:
            st.markdown('<div style="font-size:11px;color:#64748b;margin-top:3px;">Fetch stock data first.</div>',
                        unsafe_allow_html=True)
        elif fetch_busy:
            st.markdown('<div style="font-size:11px;color:#64748b;margin-top:3px;">Waiting for fetch to finish…</div>',
                        unsafe_allow_html=True)

        has_results = os.path.exists(SCREENER_RESULTS_FILE)
        if st.button("Fetch Sentiment", use_container_width=True, disabled=not has_results,
                     help="Pulls news + Reddit sentiment for screener stocks. ~2 min."):
            st.toast("Sentiment fetch started in background…")
            threading.Thread(target=run_sentiment_fetch, args=("manual",), daemon=True).start()
        if not has_results:
            st.markdown('<div style="font-size:11px;color:#64748b;margin-top:3px;">Run screener first.</div>',
                        unsafe_allow_html=True)


def render_stock_filters() -> dict:
    """
    Render stock display filters (pattern checkboxes + sliders).
    Returns a dict of filter values for use by the Stocks page.
    """
    _CAP_OPTIONS = [
        2_000, 3_000, 5_000, 7_500, 10_000, 15_000, 20_000,
        30_000, 50_000, 75_000, 1_00_000, 1_50_000, 2_00_000,
        3_00_000, 5_00_000,
    ]

    def _fmt_cap(v):
        if v >= 1_00_000: return f"₹{v//1_00_000}L Cr"
        if v >= 1_000:    return f"₹{v//1_000}K Cr"
        return f"₹{v} Cr"

    with st.expander("Filters", expanded=False):
        pattern_opts = st.multiselect(
            "Patterns",
            options=PATTERN_COLS,
            default=[],
            key="stk_patterns",
            placeholder="Any pattern",
        )

        min_score = st.slider("Min Score",     0,  100, 50, step=5)
        min_rs    = st.slider("Min RS Rating", 1,   99, 75)
        rsi_range = st.slider("RSI Range",     0,  100, (0, 100))
        max_rmv   = st.select_slider(
            "Max RMV",
            options=list(range(100, -1, -1)),
            value=100,
            help="RMV < 15 = compressed / tight bar.",
        )
        min_float = st.select_slider(
            "Min Free Float %",
            options=[0, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80],
            value=0,
            format_func=lambda v: "Any" if v == 0 else f"{v}%",
            help="Show only stocks with known Free Float ≥ this value. 'Any' includes stocks with no data.",
        )
        cap_range = st.select_slider(
            "Market Cap Range",
            options=_CAP_OPTIONS,
            value=(2_000, 20_000),
            format_func=_fmt_cap,
            help="Small Cap < ₹10K Cr  ·  Mid Cap ₹10K–₹20K Cr  ·  Large Cap > ₹20K Cr",
        )
        cap_min, cap_max = cap_range

    with st.expander("Screener Criteria", expanded=False):
        st.markdown("""
**Hard filters**
- Market Cap > ₹2,000 Cr
- Price > 52W High × 0.76
- EMA 10 > 20 > 50 > 200
- Price > 20 EMA

**Ranking**
- 52W High proximity: 20 pts
- Benchmark outperf (1M): 15 pts
- Sector outperf (1M): 15 pts
- RSI score: 15 pts
- RS trend: 25 pts
- KMA proximity: 5 pts
- RMV score: 5 pts
        """)

    return {
        "pattern_opts": pattern_opts,
        "min_score":    min_score,
        "min_rs":           min_rs,
        "rsi_range":        rsi_range,
        "max_rmv":          max_rmv,
        "min_float":        min_float,
        "cap_min":          cap_min,
        "cap_max":          cap_max,
    }


def render_index_filters(categories: list | None = None) -> dict:
    """
    Render index display filters in the sidebar.
    categories: list of unique Category values from the loaded results.
    Returns a dict of filter values for use by the Indices page.
    """
    cats = ["All"] + sorted(categories or [])

    with st.expander("Filters", expanded=False):
        cat_filter  = st.selectbox("Category", cats, key="idx_cat")
        min_score   = st.slider("Min Score",     0,  100,  0, step=5, key="idx_score")
        min_rs      = st.slider("Min RS Rating", 1,   99,  1,         key="idx_rs")
        min_rsi     = st.slider("Min RSI",      40,   80, 40,         key="idx_rsi")
        max_rmv     = st.slider("Max RMV",       5,  100, 100,        key="idx_rmv")
        pattern_opts = st.multiselect(
            "Pattern", ["VCP", "Inside Day", "RS Trend"], default=[], key="idx_patterns",
        )

    return {
        "cat_filter":   cat_filter,
        "min_score":    min_score,
        "min_rs":       min_rs,
        "min_rsi":      min_rsi,
        "max_rmv":      max_rmv,
        "pattern_opts": pattern_opts,
    }


def render_commodity_filters(categories: list | None = None) -> dict:
    """
    Render commodity display filters in the sidebar.
    categories: list of unique Category values from the loaded results.
    Returns a dict of filter values for use by the Commodities page.
    """
    cats = ["All"] + sorted(categories or [])

    with st.expander("Filters", expanded=False):
        cat_filter   = st.selectbox("Category", cats,          key="com_cat")
        min_score    = st.slider("Min Score",     0,  100,  0, step=5, key="com_score")
        min_rs       = st.slider("Min RS Rating", 1,   99,  1,         key="com_rs")
        rsi_range    = st.slider("RSI Range",     0,  100, (0, 100),   key="com_rsi")
        max_rmv      = st.slider("Max RMV",       0,  100, 100,        key="com_rmv")
        pattern_opts = st.multiselect(
            "Pattern", ["VCP", "Inside Day"], default=[], key="com_patterns",
        )

    return {
        "cat_filter":   cat_filter,
        "min_score":    min_score,
        "min_rs":       min_rs,
        "rsi_range":    rsi_range,
        "max_rmv":      max_rmv,
        "pattern_opts": pattern_opts,
    }
