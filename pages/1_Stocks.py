"""
Stocks page — NSE Momentum Screener, IPO Setups, Performance Tracker.
"""

import streamlit as st

from ui.components import (
    inject_css, tick, colour_pct, colour_rsi, colour_rmv, colour_rs, colour_ret, colour_float,
    render_stock_chart, render_top_stocks, render_ipo_section,
    DISPLAY_COLS_HEAD, DISPLAY_COLS_TAIL, STYLE_COLS, NUM_FORMAT,
    TRACKER_BASE_COLS, TRACKER_NUM_FORMAT,
)
from ui.sidebar import render_data_management, render_stock_filters
from src.pipeline import (
    load_results, load_ipo_results, load_ipo_ib_results,
    last_updated, sector_outperformance_vs_nifty500, top_industries_vs_sector,
)
from config import (
    PATTERN_COLS, COMBINED_RANK_WEIGHTS, MONTHLY_DAYS, QUARTERLY_DAYS,
    IPO_BASE_MAX_DAYS, RMV_TIGHT_THRESHOLD,
)

import pandas as pd

inject_css()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    render_data_management()
    st.divider()
    filters = render_stock_filters()

pattern_filters = filters["pattern_filters"]
min_score       = filters["min_score"]
min_rs          = filters["min_rs"]
rsi_range       = filters["rsi_range"]
max_rmv         = filters["max_rmv"]
min_float       = filters["min_float"]
cap_min         = filters["cap_min"]
cap_max         = filters["cap_max"]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
col_title, col_meta = st.columns([5, 2])
with col_title:
    st.markdown("# 📊 Momentum Stock Screener")
    st.caption("EMA Stack 10 › 20 › 50 › 200  ·  " + "  ·  ".join(PATTERN_COLS) + "  ·  Sector Outperformance")
with col_meta:
    last = last_updated()
    st.markdown(
        f"<div style='text-align:right;padding-top:14px'>"
        f"<span style='color:#6b7280;font-size:0.8rem'>LAST UPDATED</span><br>"
        f"<span style='color:#d1d5db;font-size:0.88rem;font-weight:600'>{last}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
st.divider()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
results        = load_results()
ipo_results    = load_ipo_results()
ipo_ib_results = load_ipo_ib_results()

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

sector_outperf = sector_outperformance_vs_nifty500()


# ---------------------------------------------------------------------------
# Market highlights (Top Industries vs their Sector)
# ---------------------------------------------------------------------------
def _render_highlights(results, sector_indices):
    if results is None or results.empty:
        return
    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
    st.markdown("### Market Highlights")
    st.markdown("**Top Industries vs their Sector**")
    top_inds = top_industries_vs_sector(results, sector_indices, top_n=5)
    if top_inds:
        cols = st.columns(len(top_inds))
        for i, row in enumerate(top_inds):
            op = row["outperf"]
            sign = "+" if op >= 0 else ""
            cols[i].metric(
                label=f"{medals[i]} {row['industry']}",
                value=f"{sign}{op:.2f}%",
                delta=f"{row['sector']}  ·  {row['n_stocks']} stocks",
                delta_color="normal" if op >= 0 else "inverse",
            )
    else:
        st.info("Run the screener to see industry data.")
    st.divider()


_render_highlights(results, sector_indices)
render_top_stocks(results, sentiment_cache)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_screen, tab_ipo, tab_track = st.tabs([
    "  📊  Screener Results  ",
    "  🚀  IPO Setups  ",
    "  📈  Performance Tracker  ",
])

# ── Tab 1 — Screener ─────────────────────────────────────────────────────────
with tab_screen:
    if results is None or results.empty:
        st.warning("No screener results yet. Click **📥 Fetch Data** then **🔍 Run Screener** in the sidebar.")
    else:
        df = results.copy()

        # Merge sentiment columns
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

        # Apply filters
        for pat, active in pattern_filters.items():
            if active and pat in df.columns:
                df = df[df[pat] == True]
        df = df[df["Score"] >= min_score]
        if "RS Rating" in df.columns and min_rs > 1:
            df = df[df["RS Rating"] >= min_rs]
        if "RMV" in df.columns and max_rmv < 100:
            df = df[df["RMV"].isna() | (df["RMV"] <= max_rmv)]
        if "Market Cap (Cr)" in df.columns:
            df = df[(df["Market Cap (Cr)"] >= cap_min) & (df["Market Cap (Cr)"] <= cap_max)]
        if "RSI" in df.columns:
            df = df[df["RSI"].isna() | ((df["RSI"] >= rsi_range[0]) & (df["RSI"] <= rsi_range[1]))]
        if "Free Float %" in df.columns and min_float > 0:
            df = df[df["Free Float %"].notna() & (df["Free Float %"] >= min_float)]

        if df.empty:
            st.info("No stocks match the current filters.")
        else:
            # Summary metrics
            metric_cols = st.columns(2 + len(PATTERN_COLS))
            metric_cols[0].metric("Stocks",  len(df))
            metric_cols[1].metric("Sectors", df["Sector"].nunique())
            for i, pat in enumerate(PATTERN_COLS):
                count = int(df[pat].sum()) if pat in df.columns else 0
                metric_cols[i + 2].metric(pat, count)
            st.write("")

            # Top Industries chips
            sectors_sorted = sorted(
                df["Sector"].unique(),
                key=lambda s: sector_outperf.get(s, float("-inf")),
                reverse=True,
            )
            top_inds = top_industries_vs_sector(df, sector_indices, top_n=10)
            chips_html = ""
            for _row in top_inds:
                _op   = _row["outperf"]
                _clr  = "#22c55e" if _op >= 0 else "#ef4444"
                _sign = "+" if _op >= 0 else ""
                chips_html += (
                    f"<span style='display:inline-block;background:rgba(255,255,255,0.04);"
                    f"border:1px solid {_clr}44;border-radius:20px;padding:3px 12px;"
                    f"font-size:0.74rem;margin:2px 3px;white-space:nowrap'>"
                    f"<span style='color:#e2e8f0'>{_row['industry']}</span>"
                    f"&nbsp;<span style='color:{_clr};font-weight:700'>{_sign}{_op:.1f}%</span>"
                    f"&nbsp;<span style='color:#6b7280'>·&nbsp;{_row['sector']}&nbsp;·&nbsp;{_row['n_stocks']}</span>"
                    f"</span>"
                )
            if chips_html:
                st.markdown(
                    f"<div style='margin-bottom:10px;line-height:2.2'>"
                    f"<span style='font-size:0.72rem;color:#6b7280;font-weight:700;"
                    f"text-transform:uppercase;letter-spacing:0.06em;margin-right:8px'>"
                    f"Top Industries vs their Sector</span>{chips_html}</div>",
                    unsafe_allow_html=True,
                )

            # Group-by control
            gb_col, _ = st.columns([3, 5])
            with gb_col:
                group_by = st.radio(
                    "Group by", ["None", "Sector", "Industry"],
                    index=1, horizontal=True,
                )

            # Build sorted display DataFrame
            _disp = df.copy().reset_index(drop=True)
            if group_by == "Sector":
                _s_rank = {s: i for i, s in enumerate(sectors_sorted)}
                _disp["_g"] = _disp["Sector"].map(_s_rank)
                _disp = _disp.sort_values(["_g", "Score"], ascending=[True, False]).drop(columns=["_g"]).reset_index(drop=True)
            elif group_by == "Industry":
                _s_rank = {s: i for i, s in enumerate(sectors_sorted)}
                _disp["_g"] = _disp["Sector"].map(_s_rank)
                _disp = _disp.sort_values(["_g", "Industry", "Score"], ascending=[True, True, False]).drop(columns=["_g"]).reset_index(drop=True)
            else:
                _disp = _disp.sort_values("Score", ascending=False).reset_index(drop=True)

            # Group stripe colours
            if group_by == "Sector":
                _gvals = _disp["Sector"].tolist()
            elif group_by == "Industry":
                _gvals = _disp["Industry"].tolist()
            else:
                _gvals = []
            _guniq    = list(dict.fromkeys(_gvals))
            _gidx_map = {g: i for i, g in enumerate(_guniq)}

            # Column ordering
            _pat_cols = [c for c in PATTERN_COLS if c in _disp.columns]
            if group_by == "Sector":
                _head = ["Sector"] + list(DISPLAY_COLS_HEAD)
            elif group_by == "Industry":
                _head = ["Sector", "Industry"] + list(DISPLAY_COLS_HEAD)
            else:
                _head = list(DISPLAY_COLS_HEAD)
            _ordered   = _head + _pat_cols + list(DISPLAY_COLS_TAIL)
            _available = [c for c in _ordered if c in _disp.columns]
            _display   = _disp[_available].copy()
            for _c in _pat_cols:
                _display[_c] = _display[_c].apply(tick)

            _styled = _display.style.format(NUM_FORMAT, na_rep="—")
            for _col, _fn in STYLE_COLS.items():
                if _col in _display.columns:
                    _styled = _styled.map(_fn, subset=[_col])
            if _gvals:
                def _row_bg(row, gvals=_gvals, gmap=_gidx_map):
                    shade = gmap.get(gvals[row.name], 0) % 2
                    bg    = "rgba(255,255,255,0.05)" if shade else "rgba(0,0,0,0)"
                    return [f"background-color:{bg}"] * len(row)
                _styled = _styled.apply(_row_bg, axis=1)

            st.dataframe(_styled, width="stretch", height=min(38 * (len(_display) + 1), 820))
            st.divider()

            col_dl, _ = st.columns([2, 5])
            with col_dl:
                st.download_button(
                    label=f"⬇  Export {len(df)} symbols  (.txt)",
                    data=",".join(df["Symbol"].tolist()),
                    file_name="symbols.txt",
                    mime="text/plain",
                )

            # Chart View
            st.divider()
            st.markdown("### 📈 Chart View")
            chart_col1, chart_col2 = st.columns([3, 1])
            with chart_col1:
                ranked_symbols = df.sort_values("Score", ascending=False)["Symbol"].tolist()
                chart_symbol = st.selectbox(
                    "Select stock",
                    ["— select a stock —"] + ranked_symbols,
                    label_visibility="collapsed",
                )
            with chart_col2:
                chart_lookback = st.select_slider(
                    "Period", options=[30, 60, 90, 120, 180, 252],
                    value=90, format_func=lambda x: f"{x}D",
                )

            if chart_symbol != "— select a stock —":
                from src.data_fetcher import load_price_data as _load_pd
                with st.spinner(f"Loading chart for {chart_symbol}…"):
                    _pd = _load_pd()
                if _pd:
                    render_stock_chart(chart_symbol, _pd, lookback=chart_lookback)
                else:
                    st.warning("Price data not available — run Fetch Data first.")

                # News Headlines
                sent_data = sentiment_cache.get(chart_symbol, {})
                headlines = sent_data.get("headlines", [])
                if headlines:
                    score_val = sent_data.get("score", 0.0)
                    label     = sent_data.get("label", "Neutral")
                    reddit_ct = sent_data.get("reddit_mentions", 0)
                    label_html = (
                        "<span style='color:#4ade80;font-weight:700'>🟢 Bullish</span>" if label == "Bullish" else
                        "<span style='color:#f87171;font-weight:700'>🔴 Bearish</span>" if label == "Bearish" else
                        "<span style='color:#9ca3af'>⚪ Neutral</span>"
                    )
                    st.markdown(
                        f"**Sentiment** — {label_html} &nbsp;·&nbsp; "
                        f"<span style='color:#94a3b8'>VADER score: {score_val:+.3f}</span> &nbsp;·&nbsp; "
                        f"<span style='color:#94a3b8'>{len(headlines)} articles &nbsp; Reddit buzz: {reddit_ct}</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("---")
                    for title, h_score in headlines:
                        dot = "🟢" if h_score >= 0.05 else ("🔴" if h_score <= -0.05 else "⚪")
                        st.markdown(
                            f"<div style='font-size:0.85rem;padding:3px 0;color:#cbd5e1'>"
                            f"{dot}&nbsp; {title}"
                            f"<span style='color:#475569;font-size:0.75rem;margin-left:8px'>[{h_score:+.2f}]</span></div>",
                            unsafe_allow_html=True,
                        )
                elif chart_symbol:
                    st.caption("No sentiment data — click 🧠 Fetch Sentiment in the sidebar.")

# ── Tab 2 — IPO Setups ───────────────────────────────────────────────────────
with tab_ipo:
    render_ipo_section(ipo_results)

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
            "CMP": "{:.2f}", "% from Base High": "{:.2f}", "RMV": "{:.1f}",
            "RSI": "{:.1f}", "Weekly %": "{:.2f}", "Monthly %": "{:.2f}",
            "Sector Outperf %": "{:.2f}", "Market Cap (Cr)": "{:.0f}",
        }
        available = [c for c in _ib_cols if c in ipo_ib_results.columns]
        display   = ipo_ib_results[available].copy()
        if "EMA10 > EMA20" in display.columns:
            display["EMA10 > EMA20"] = display["EMA10 > EMA20"].apply(tick)
        styled = display.style.format(_ib_fmt, na_rep="—")
        if "RMV" in display.columns:
            styled = styled.map(colour_rmv, subset=["RMV"])
        for col in ("Weekly %", "Monthly %", "Sector Outperf %"):
            if col in display.columns:
                styled = styled.map(colour_pct, subset=[col])
        if "RSI" in display.columns:
            styled = styled.map(colour_rsi, subset=["RSI"])
        st.caption(f"{len(display)} setup{'s' if len(display) != 1 else ''}  ·  sorted by RMV ascending (tightest first)")
        st.dataframe(styled, width="stretch", height=min(38 * (len(display) + 1), 500))

# ── Tab 3 — Performance Tracker ──────────────────────────────────────────────
with tab_track:
    from src.tracker import load_snapshots, compute_forward_returns, pattern_summary, score_band_summary
    from src.data_fetcher import load_price_data

    snapshots = load_snapshots()
    if snapshots.empty:
        st.info(
            "No snapshots captured yet. Run the screener to start recording daily results. "
            "Forward returns are computed automatically as new price data arrives."
        )
    else:
        with st.spinner("Computing forward returns…"):
            price_data = load_price_data() or {}
            df = compute_forward_returns(snapshots, price_data)

        n_dates    = df["Capture Date"].nunique()
        n_captures = len(df)
        w1 = df["Ret 1W %"].dropna() if "Ret 1W %" in df.columns else pd.Series(dtype=float)
        m1 = df["Ret 1M %"].dropna() if "Ret 1M %" in df.columns else pd.Series(dtype=float)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Capture Days",   n_dates)
        c2.metric("Total Captures", n_captures)
        c3.metric("Win Rate 1W",    f"{(w1 > 0).mean() * 100:.0f}%" if len(w1) else "—")
        c4.metric("Avg Return 1W",  f"{w1.mean():+.2f}%" if len(w1) else "—")
        c5.metric("Win Rate 1M",    f"{(m1 > 0).mean() * 100:.0f}%" if len(m1) else "—")
        c6.metric("Avg Return 1M",  f"{m1.mean():+.2f}%" if len(m1) else "—")
        st.divider()

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

        st.markdown("##### Capture Log")
        all_dates = sorted(df["Capture Date"].unique(), reverse=True)
        f1, f2, f3 = st.columns([2, 2, 3])
        with f1:
            sel_dates = st.multiselect("Date", all_dates, default=all_dates[:5], placeholder="All dates")
        with f2:
            sel_sector = st.multiselect("Sector", sorted(df["Sector"].unique()), placeholder="All sectors")
        with f3:
            sym_search = st.text_input("Symbol search", placeholder="e.g. INFY")

        view = df.copy()
        if sel_dates:
            view = view[view["Capture Date"].isin(sel_dates)]
        if sel_sector:
            view = view[view["Sector"].isin(sel_sector)]
        if sym_search:
            view = view[view["Symbol"].str.contains(sym_search.strip().upper(), na=False)]

        w_s  = COMBINED_RANK_WEIGHTS["score"]     / 100.0
        w_rs = COMBINED_RANK_WEIGHTS["rs_rating"] / 100.0
        if "Score" in view.columns and "RS Rating" in view.columns:
            view["Combined Score"] = (view["Score"] * w_s + view["RS Rating"] * w_rs).round(1)
        elif "Score" in view.columns:
            view["Combined Score"] = view["Score"].round(1)

        pat_cols  = [c for c in PATTERN_COLS if c in view.columns]
        ordered   = TRACKER_BASE_COLS + pat_cols
        available = [c for c in ordered if c in view.columns]
        display   = view[available].copy()
        for col in pat_cols:
            display[col] = display[col].apply(tick)

        styled = display.style.format(TRACKER_NUM_FORMAT, na_rep="—")
        for ret_col in ("Ret 1D %", "Ret 1W %", "Ret 2W %", "Ret 1M %"):
            if ret_col in display.columns:
                styled = styled.map(colour_ret, subset=[ret_col])
        if "RS Rating" in display.columns:
            styled = styled.map(colour_rs, subset=["RS Rating"])
        if "RMV" in display.columns:
            styled = styled.map(colour_rmv, subset=["RMV"])

        st.dataframe(styled, width="stretch", height=min(38 * (len(display) + 1), 600))
        st.divider()

        col_dl, _ = st.columns([2, 5])
        with col_dl:
            st.download_button(
                label=f"⬇  Export tracker data  ({len(view)} rows)",
                data=view.to_csv(index=False),
                file_name="screener_tracker.csv",
                mime="text/csv",
            )

st.caption(
    "Data: Yahoo Finance (yfinance)  ·  Benchmark: NiftyMidSmallCap400  ·  "
    "EMA = Daily Exponential Moving Average  ·  VCP = Volatility Contraction Pattern (Minervini)"
)
