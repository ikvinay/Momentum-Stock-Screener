"""
Shared sidebar renderer.

Call render_data_management() from every page to show the data pipeline
controls (Fetch / Run Screener / Fetch Sentiment buttons + status).

Call render_stock_filters() only from the Stocks page to get pattern
checkboxes, score/RS/RSI/RMV/cap sliders. Returns a dict of filter values.
"""

import os
import threading

import streamlit as st

from config import DATA_DIR, SCREENER_RESULTS_FILE, PATTERN_COLS
from src.pipeline import run_data_fetch, run_screener_only, run_sentiment_fetch, read_status


def render_data_management() -> None:
    """Render the ⚙️ Data Management expander in the sidebar."""
    from src.data_fetcher import is_price_data_fresh, price_data_last_fetched
    from src.sector_mapper import is_index_data_fresh

    data_fresh = is_price_data_fresh() and is_index_data_fresh()
    age        = price_data_last_fetched()
    dot_color  = "#22c55e" if data_fresh else "#f59e0b"
    dm_label   = f"⚙️ Data Management  {'✓' if data_fresh else '⚠'}"

    with st.expander(dm_label, expanded=False):
        status    = read_status()
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
        if st.button(fetch_label, use_container_width=True,
                     help="Downloads price data, stock info and index data. 10–30 min on first run."):
            st.toast("Data fetch started in background…")
            threading.Thread(target=run_data_fetch, args=("manual",), daemon=True).start()

        st.markdown(f"<small style='color:{dot_color}'>● {age}</small>", unsafe_allow_html=True)
        st.write("")

        has_data = os.path.exists(os.path.join(DATA_DIR, "price_data.pkl"))
        if st.button("🔍 Run Screener", type="primary", use_container_width=True,
                     disabled=not has_data,
                     help="Applies all filters and pattern detection on cached data. Runs in seconds."):
            st.toast("Screener started…")
            threading.Thread(target=run_screener_only, args=("manual",), daemon=True).start()
        if not has_data:
            st.caption("⚠ Fetch data first.")

        has_results = os.path.exists(SCREENER_RESULTS_FILE)
        if st.button("🧠 Fetch Sentiment", use_container_width=True,
                     disabled=not has_results,
                     help="Pulls news + Reddit sentiment for screener stocks. ~2 min."):
            st.toast("Sentiment fetch started in background…")
            threading.Thread(target=run_sentiment_fetch, args=("manual",), daemon=True).start()
        if not has_results:
            st.caption("⚠ Run screener first.")


def render_stock_filters() -> dict:
    """
    Render stock display filters (pattern checkboxes + sliders).
    Returns a dict of filter values for use by the Stocks page.
    """
    st.markdown('<p class="sidebar-label">🔍 Display Filters</p>', unsafe_allow_html=True)
    pattern_filters = {pat: st.checkbox(pat) for pat in PATTERN_COLS}
    st.write("")

    min_score = st.slider("Min Score",     0,  100, 50, step=5)
    min_rs    = st.slider("Min RS Rating", 1,   99, 75)
    rsi_range = st.slider("RSI Range",     0,  100, (0, 100))
    max_rmv   = st.select_slider(
        "Max RMV",
        options=list(range(100, -1, -1)),
        value=100,
        help="RMV < 15 = compressed / tight bar.",
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

    with st.expander("📋 Screener Criteria", expanded=False):
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
        "pattern_filters": pattern_filters,
        "min_score":        min_score,
        "min_rs":           min_rs,
        "rsi_range":        rsi_range,
        "max_rmv":          max_rmv,
        "cap_min":          cap_min,
        "cap_max":          cap_max,
    }
