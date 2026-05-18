"""
Stocks page — NSE Market Insights, IPO Setups, Performance Tracker.
"""

import streamlit as st

from ui.components import (
    tick, colour_pct, colour_rsi, colour_rmv, colour_rs, colour_ret, colour_float,
    render_stock_chart, render_top_stocks, render_ipo_section, render_rrg_chart,
    DISPLAY_COLS_HEAD, DISPLAY_COLS_TAIL, STYLE_COLS, NUM_FORMAT,
    TRACKER_BASE_COLS, TRACKER_NUM_FORMAT,
)
from ui.sidebar import render_stock_filters
from src.pipeline import (
    load_results, load_ipo_results, load_ipo_ib_results,
    last_updated, sector_outperformance_vs_nifty500, top_industries_vs_sector,
)
from src.data_fetcher import is_price_data_fresh, price_data_last_fetched
from config import (
    PATTERN_COLS, COMBINED_RANK_WEIGHTS, MONTHLY_DAYS, QUARTERLY_DAYS,
    IPO_BASE_MAX_DAYS, RMV_TIGHT_THRESHOLD,
)

import pandas as pd

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    filters = render_stock_filters()

pattern_opts = filters["pattern_opts"]
min_score    = filters["min_score"]
min_rs          = filters["min_rs"]
rsi_range       = filters["rsi_range"]
max_rmv         = filters["max_rmv"]
max_float       = filters["max_float"]
cap_min         = filters["cap_min"]
cap_max         = filters["cap_max"]

# ---------------------------------------------------------------------------
# Header — freshness-aware masthead
# ---------------------------------------------------------------------------
_fresh       = is_price_data_fresh()
_age         = price_data_last_fetched()
_fresh_rgb   = "34,197,94" if _fresh else "245,158,11"
_fresh_color = "#22c55e"   if _fresh else "#f59e0b"
_fresh_label = "LIVE"      if _fresh else "STALE"
_pat_list    = "  ·  ".join(PATTERN_COLS)

st.markdown(
    f'<div style="display:flex;justify-content:space-between;align-items:flex-start;padding-bottom:6px">'
    f'<div>'
    f'<div style="font-size:22px;font-weight:700;color:#f1f5f9;letter-spacing:-0.02em">'
    f'<i class="ti ti-chart-bar" style="color:#6366f1;margin-right:8px"></i>Stocks Screener</div>'
    f'<div style="font-size:12px;color:#64748b;margin-top:4px;line-height:1.6">'
    f'EMA Stack 10 › 20 › 50 › 200  ·  {_pat_list}  ·  Sector Outperformance</div>'
    f'</div>'
    f'<div style="text-align:right;flex-shrink:0;padding-left:16px">'
    f'<span style="display:inline-flex;align-items:center;gap:5px;'
    f'background:rgba({_fresh_rgb},0.1);border:1px solid rgba({_fresh_rgb},0.25);'
    f'color:{_fresh_color};border-radius:6px;padding:4px 10px;'
    f'font-size:0.7rem;font-weight:700;letter-spacing:0.07em">'
    f'<span style="width:6px;height:6px;border-radius:50%;background:{_fresh_color};'
    f'display:inline-block;flex-shrink:0"></span>{_fresh_label}</span>'
    f'<div style="font-size:0.72rem;color:#64748b;margin-top:5px">{_age}</div>'
    f'</div>'
    f'</div>',
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
    _benchmark_df, sector_indices, _nifty500_df = load_index_data()
except Exception:
    _benchmark_df, sector_indices, _nifty500_df = None, {}, None

sector_outperf = sector_outperformance_vs_nifty500()


# ---------------------------------------------------------------------------
# Market highlights (Top Industries vs their Sector)
# ---------------------------------------------------------------------------
def _render_highlights(results, sector_indices):
    if results is None or results.empty:
        return
    medals = ["#1", "#2", "#3", "#4", "#5"]
    st.markdown("### Market Highlights")

    # ── Sector Highlights ─────────────────────────────────────────────────────
    st.markdown("**Sector Highlights**")
    _sh_ok = all(c in results.columns for c in ["Sector", "Monthly %", "Quarterly %"])
    if _sh_ok:
        _sh = results[results["Sector"].notna() & (results["Sector"] != "Unknown")]
        _sector_stats = (
            _sh.groupby("Sector")
            .agg(
                avg_1m=("Monthly %", "mean"),
                avg_3m=("Quarterly %", "mean"),
                n_stocks=("Symbol", "count"),
            )
            .reset_index()
        )
        _sector_stats["blended"] = (_sector_stats["avg_1m"] + _sector_stats["avg_3m"]) / 2
        _sector_stats = _sector_stats.sort_values("blended", ascending=False).head(5).reset_index(drop=True)
        if not _sector_stats.empty:
            sh_cols = st.columns(len(_sector_stats))
            for i, srow in _sector_stats.iterrows():
                b  = srow["blended"]
                m  = srow["avg_1m"]
                q  = srow["avg_3m"]
                sh_cols[i].metric(
                    label=f"{medals[i]} {srow['Sector']}  ·  {int(srow['n_stocks'])} stocks",
                    value=f"{'+'if b>=0 else ''}{b:.1f}%  avg",
                    delta=f"1M {'+'if m>=0 else ''}{m:.1f}%  ·  3M {'+'if q>=0 else ''}{q:.1f}%",
                    delta_color="normal" if b >= 0 else "inverse",
                )
        else:
            st.info("No sector data available.")
    else:
        st.info("Run the screener to see sector highlights.")

    st.markdown("<div style='margin-bottom:4px;'></div>", unsafe_allow_html=True)

    # ── Top Industries vs their Sector ────────────────────────────────────────
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
tab_screen, tab_ipo, tab_track, tab_rrg = st.tabs([
    "  Screener Results  ",
    "  IPO Setups  ",
    "  Performance Tracker  ",
    "  Sector Rotation  ",
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
                    ("● Bullish" if sentiment_cache[s]["label"] == "Bullish" else
                     "● Bearish" if sentiment_cache[s]["label"] == "Bearish" else
                     "● Neutral")
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
        for pat in pattern_opts:
            if pat in df.columns:
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
        if "Free Float %" in df.columns and max_float < 100:
            df = df[df["Free Float %"].notna() & (df["Free Float %"] <= max_float)]

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
            st.markdown("### Chart View")
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
                        "<span style='color:#4ade80;font-weight:700'><span style='color:#22c55e'>●</span> Bullish</span>" if label == "Bullish" else
                        "<span style='color:#f87171;font-weight:700'><span style='color:#ef4444'>●</span> Bearish</span>" if label == "Bearish" else
                        "<span style='color:#9ca3af'><span style='color:#6b7280'>●</span> Neutral</span>"
                    )
                    st.markdown(
                        f"**Sentiment** — {label_html} &nbsp;·&nbsp; "
                        f"<span style='color:#94a3b8'>VADER score: {score_val:+.3f}</span> &nbsp;·&nbsp; "
                        f"<span style='color:#94a3b8'>{len(headlines)} articles &nbsp; Reddit buzz: {reddit_ct}</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("---")
                    for title, h_score in headlines:
                        dot_color = "#22c55e" if h_score >= 0.05 else ("#ef4444" if h_score <= -0.05 else "#6b7280")
                        dot = f"<span style='color:{dot_color}'>●</span>"
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

    st.subheader("Inside Bar + Low RMV  —  IPO-age stocks")
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

# ── Tab 4 — Sector Rotation (RRG) ────────────────────────────────────────────
with tab_rrg:
    from src.rrg import compute_rrg
    from src.index_screener import load_index_ohlcv
    from src.data_fetcher import load_price_data as _lpd
    from config import NSE_INDEX_TICKERS, SECTOR_INDEX_MAP

    # ── Build index display-name → sector-name mapping ────────────────────────
    _ticker_to_sector = {v: k for k, v in SECTOR_INDEX_MAP.items()}
    _index_to_sector: dict[str, str] = {}
    for _cat, _entries in NSE_INDEX_TICKERS.items():
        for _iname, _iticker in _entries.items():
            _frag = _ticker_to_sector.get(_iticker)
            if _frag:
                _index_to_sector[_iname] = _frag

    def _resolve_sector_for_index(index_display_name: str) -> str | None:
        frag = _index_to_sector.get(index_display_name, "").lower()
        if not frag or results is None or results.empty:
            return None
        for s in results["Sector"].dropna().unique():
            if frag in s.lower() or s.lower() in frag:
                return s
        return None

    index_ohlcv   = load_index_ohlcv()
    rrg_benchmark = (
        _nifty500_df if (_nifty500_df is not None and not _nifty500_df.empty)
        else _benchmark_df
    )

    if not index_ohlcv or rrg_benchmark is None:
        st.info(
            "No index data available. Run **📥 Refresh Stock Data** "
            "then **🔍 Run Screener** in the sidebar."
        )
    else:
        _Q_ORDER = {"Leading": 0, "Improving": 1, "Weakening": 2, "Lagging": 3}
        _Q_STYLE = {
            "Leading":   ("#22c55e", "rgba(34,197,94,0.08)"),
            "Improving": ("#60a5fa", "rgba(96,165,250,0.08)"),
            "Weakening": ("#f59e0b", "rgba(245,158,11,0.08)"),
            "Lagging":   ("#ef4444", "rgba(239,68,68,0.08)"),
        }

        def _color_quadrant(val):
            c = {"Leading": "#22c55e", "Improving": "#60a5fa",
                 "Weakening": "#f59e0b", "Lagging": "#ef4444"}.get(val, "")
            return f"color: {c}" if c else ""

        def _synthetic_df(symbols: list[str], all_pd: dict) -> pd.DataFrame | None:
            closes = [
                all_pd[f"{s}.NS"]["Close"].rename(s)
                for s in symbols
                if f"{s}.NS" in all_pd and "Close" in all_pd[f"{s}.NS"].columns
            ]
            if not closes:
                return None
            df = pd.concat(closes, axis=1).dropna(how="all")
            return pd.DataFrame({"Close": df.mean(axis=1)}) if not df.empty else None

        def _quadrant_badges(rrg_df: pd.DataFrame) -> None:
            qb1, qb2, qb3, qb4 = st.columns(4)
            for col, qname in [(qb1, "Leading"), (qb2, "Improving"),
                               (qb3, "Weakening"), (qb4, "Lagging")]:
                color, bg = _Q_STYLE[qname]
                in_q  = rrg_df[rrg_df["quadrant"] == qname]["name"].tolist()
                names_str = "  ·  ".join(in_q) if in_q else "—"
                col.markdown(
                    f"<div style='padding:10px 12px;background:{bg};border-radius:8px;"
                    f"border-top:2px solid {color};min-height:76px'>"
                    f"<div style='display:flex;align-items:baseline;gap:8px;margin-bottom:4px'>"
                    f"<span style='font-size:1.4rem;font-weight:800;color:{color}'>{len(in_q)}</span>"
                    f"<span style='font-size:0.72rem;font-weight:600;color:{color};"
                    f"letter-spacing:0.06em;text-transform:uppercase'>{qname}</span></div>"
                    f"<div style='font-size:0.72rem;color:#94a3b8;line-height:1.5'>{names_str}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        def _detail_table(rrg_df: pd.DataFrame, col_label: str) -> None:
            tbl = rrg_df[["name", "rs_ratio", "rs_momentum", "quadrant"]].copy()
            tbl["_q"] = tbl["quadrant"].map(_Q_ORDER)
            tbl = (tbl.sort_values(["_q", "rs_ratio"], ascending=[True, False])
                   .drop(columns=["_q"]).reset_index(drop=True))
            tbl.columns = [col_label, "RS-Ratio", "RS-Momentum", "Quadrant"]
            st.dataframe(
                tbl.style
                   .format({"RS-Ratio": "{:.2f}", "RS-Momentum": "{:.2f}"})
                   .map(_color_quadrant, subset=["Quadrant"]),
                use_container_width=True, hide_index=True,
                height=min(38 * (len(tbl) + 1), 400),
            )

        def _render_rrg_view(weekly: bool, rrg_m: int, rrg_k: int, rrg_tail: int, freq_key: str) -> None:
            _freq_label = "Weekly" if weekly else "Daily"
            _path_key   = f"rrg_path_{freq_key}"
            if _path_key not in st.session_state:
                st.session_state[_path_key] = [
                    {"label": "Indices", "level": 0, "sector": None, "industry": None}
                ]

            path  = st.session_state[_path_key]
            level = path[-1]["level"]

            # ── Breadcrumb ────────────────────────────────────────────────────
            bc_left, bc_right = st.columns([9, 1])
            with bc_left:
                _level_hint = {0: "click a dot to drill → Sectors",
                               1: "click a dot to drill → Industries",
                               2: "click a dot to drill → Stocks",
                               3: ""}
                crumb_parts = []
                for i, crumb in enumerate(path):
                    is_last = (i == len(path) - 1)
                    if is_last:
                        hint = _level_hint.get(level, "")
                        crumb_parts.append(
                            f'<span style="color:#f1f5f9;font-weight:600">{crumb["label"]}</span>'
                            + (f'<span style="font-size:0.72rem;color:#64748b;margin-left:10px">'
                               f'{hint}</span>' if hint else "")
                        )
                    else:
                        crumb_parts.append(
                            f'<span style="color:#60a5fa">{crumb["label"]}</span>'
                        )
                st.markdown(
                    '<div style="font-size:0.9rem;padding:6px 0">'
                    + ' <span style="color:#475569">›</span> '.join(crumb_parts)
                    + "</div>",
                    unsafe_allow_html=True,
                )
            with bc_right:
                if len(path) > 1 and st.button("↑ Back", key=f"rrg_back_{freq_key}",
                                                use_container_width=True):
                    st.session_state[_path_key] = path[:-1]
                    st.rerun()

            if len(path) > 2:
                anc_cols = st.columns(len(path) - 1)
                for i, crumb in enumerate(path[:-1]):
                    if anc_cols[i].button(f"← {crumb['label']}",
                                          key=f"rrg_crumb_{freq_key}_{i}",
                                          use_container_width=True):
                        st.session_state[_path_key] = path[:i + 1]
                        st.rerun()

            # ── Search / jump box ─────────────────────────────────────────────
            if results is not None and not results.empty:
                _all_sectors_j    = sorted(results["Sector"].dropna().unique().tolist())
                _all_industries_j = sorted(results["Industry"].dropna().unique().tolist())
                _jump_opts = ([""] +
                    [f"📊 {s}" for s in _all_sectors_j] +
                    [f"🏭 {i}" for i in _all_industries_j])
                _jump = st.selectbox(
                    "jump", _jump_opts, key=f"rrg_jump_{freq_key}",
                    label_visibility="collapsed",
                    placeholder="🔍 Jump to any sector or industry…",
                )
                if _jump:
                    if _jump.startswith("📊 "):
                        _sec = _jump[2:].strip()
                        st.session_state[_path_key] = [
                            {"label": "Indices", "level": 0, "sector": None, "industry": None},
                            {"label": "Sectors", "level": 1, "sector": _sec, "industry": None},
                            {"label": _sec,      "level": 2, "sector": _sec, "industry": None},
                        ]
                    elif _jump.startswith("🏭 "):
                        _ind = _jump[2:].strip()
                        _ind_sec = results[results["Industry"] == _ind]["Sector"].mode()
                        _ind_sec = _ind_sec.iloc[0] if not _ind_sec.empty else "Unknown"
                        st.session_state[_path_key] = [
                            {"label": "Indices", "level": 0, "sector": None,     "industry": None},
                            {"label": "Sectors", "level": 1, "sector": _ind_sec, "industry": None},
                            {"label": _ind_sec,  "level": 2, "sector": _ind_sec, "industry": None},
                            {"label": _ind,      "level": 3, "sector": _ind_sec, "industry": _ind},
                        ]
                    st.rerun()

            st.write("")

            # ── Build price_dict for the current level ────────────────────────
            if level == 0:
                price_dict  = index_ohlcv
                can_drill   = True
                chart_title = f"NSE Sector Indices vs Nifty 500 — {_freq_label}"
                col_label   = "Index"

            elif level == 1:
                price_dict  = {}
                can_drill   = True
                chart_title = f"Sectors (synthetic) vs Nifty 500 — {_freq_label}"
                col_label   = "Sector"
                if results is not None and not results.empty:
                    with st.spinner("Building sector series…"):
                        _all_pd = _lpd() or {}
                    for _sec, _grp in results.groupby("Sector"):
                        if not _sec or _sec == "Unknown":
                            continue
                        _sdf = _synthetic_df(_grp["Symbol"].tolist(), _all_pd)
                        if _sdf is not None and len(_sdf) >= 50:
                            price_dict[_sec] = _sdf

            elif level == 2:
                _sec = path[-1]["sector"]
                price_dict  = {}
                can_drill   = True
                chart_title = f"{_sec} — Industries vs Nifty 500 · {_freq_label}"
                col_label   = "Industry"
                if results is not None and not results.empty and _sec:
                    with st.spinner("Building industry series…"):
                        _all_pd = _lpd() or {}
                    for _ind, _grp in results[results["Sector"] == _sec].groupby("Industry"):
                        if not _ind or _ind == "Unknown":
                            continue
                        _sdf = _synthetic_df(_grp["Symbol"].tolist(), _all_pd)
                        if _sdf is not None and len(_sdf) >= 50:
                            price_dict[_ind] = _sdf

            elif level == 3:
                _sec = path[-1]["sector"]
                _ind = path[-1]["industry"]
                price_dict  = {}
                can_drill   = False
                chart_title = f"{_ind} — Stocks vs Nifty 500 · {_freq_label}"
                col_label   = "Stock"
                if results is not None and not results.empty:
                    with st.spinner("Loading price data…"):
                        _all_pd = _lpd() or {}
                    _mask = results["Industry"] == _ind
                    if _sec:
                        _mask &= results["Sector"] == _sec
                    for _, _row in results[_mask].iterrows():
                        _t = f"{_row['Symbol']}.NS"
                        if _t in _all_pd:
                            price_dict[_row.get("Company", _row["Symbol"])] = _all_pd[_t]
            else:
                price_dict  = {}
                can_drill   = False
                chart_title = "RRG"
                col_label   = "Name"

            # ── Compute & render RRG ──────────────────────────────────────────
            if not price_dict:
                st.info("No data available at this level. Navigate back or run the screener first.")
            else:
                with st.spinner(f"Computing {_freq_label} RRG — {len(price_dict)} items…"):
                    rrg_df = compute_rrg(
                        price_dict, rrg_benchmark,
                        m=rrg_m, k=rrg_k, tail=rrg_tail, weekly=weekly,
                    )

                if rrg_df.empty:
                    st.info(
                        f"Not enough history to render the {_freq_label} RRG at this level. "
                        "Try Daily mode or navigate back."
                    )
                else:
                    if level == 1 and path[-1].get("sector"):
                        _hint_sec = path[-1]["sector"]
                        if _hint_sec in set(rrg_df["name"]):
                            st.caption(
                                f"💡 You came from an index in **{_hint_sec}** — "
                                "click that sector to drill into its industries."
                            )

                    _quadrant_badges(rrg_df)
                    if can_drill:
                        st.caption("Click any dot on the chart to drill down.")
                    st.write("")

                    _chart_key = f"rrg_{freq_key}_" + "_".join(str(p["level"]) for p in path)
                    _event = render_rrg_chart(
                        rrg_df,
                        title=chart_title,
                        key=_chart_key,
                        on_select="rerun" if can_drill else "ignore",
                    )

                    if can_drill and _event is not None:
                        _pts = getattr(getattr(_event, "selection", None), "points", None)
                        if _pts:
                            _clicked = _pts[0].get("customdata", [None])[0]
                            if _clicked:
                                if level == 0:
                                    _matched_sec = _resolve_sector_for_index(_clicked)
                                    st.session_state[_path_key] = path + [{
                                        "label":    "Sectors",
                                        "level":    1,
                                        "sector":   _matched_sec,
                                        "industry": None,
                                    }]
                                elif level == 1:
                                    st.session_state[_path_key] = path + [{
                                        "label":    _clicked,
                                        "level":    2,
                                        "sector":   _clicked,
                                        "industry": None,
                                    }]
                                elif level == 2:
                                    st.session_state[_path_key] = path + [{
                                        "label":    _clicked,
                                        "level":    3,
                                        "sector":   path[-1]["sector"],
                                        "industry": _clicked,
                                    }]
                                st.rerun()

                    with st.expander("Detail table"):
                        _detail_table(rrg_df, col_label)

                    with st.expander("How to read the RRG"):
                        st.markdown(
                            "**RS-Ratio** (x-axis) — outperforming Nifty 500 (>100) or underperforming (<100)  \n"
                            "**RS-Momentum** (y-axis) — that outperformance accelerating (>100) or fading (<100)  \n\n"
                            "Normal rotation is **clockwise**: "
                            ":blue[Improving] → :green[Leading] → :orange[Weakening] → :red[Lagging]  \n"
                            "**Actionable**: Leading (hold/add) · Improving (early entry, tail curling up-right)"
                        )

        # ── Timeframe tabs ────────────────────────────────────────────────────
        rrg_tab_daily, rrg_tab_weekly = st.tabs(["  Daily  ", "  Weekly  "])

        with rrg_tab_daily:
            _def_m, _def_k, _def_tail = 14, 10, 4
            with st.expander("Advanced settings"):
                _ac1, _ac2, _ac3 = st.columns(3)
                rrg_m = _ac1.number_input(
                    "EMA Smoothing", 5, 30, _def_m, 1, key="rrg_m_daily",
                    help="Smoothing window for RS-Ratio / RS-Momentum. Daily default: 14.",
                )
                rrg_k = _ac2.number_input(
                    "RS Period", 3, 30, _def_k, 1, key="rrg_k_daily",
                    help="Look-back for relative-strength vs benchmark. Daily default: 10.",
                )
                rrg_tail = _ac3.number_input(
                    "Trail Length", 3, 16, _def_tail, 1, key="rrg_tail_daily",
                    help="Number of past periods shown as a tail. Daily default: 4.",
                )
            _render_rrg_view(
                weekly=False,
                rrg_m=int(rrg_m), rrg_k=int(rrg_k), rrg_tail=int(rrg_tail),
                freq_key="daily",
            )

        with rrg_tab_weekly:
            _def_m, _def_k, _def_tail = 14, 10, 4
            with st.expander("Advanced settings"):
                _ac1, _ac2, _ac3 = st.columns(3)
                rrg_m = _ac1.number_input(
                    "EMA Smoothing", 5, 30, _def_m, 1, key="rrg_m_weekly",
                    help="Smoothing window for RS-Ratio / RS-Momentum. Weekly default: 14.",
                )
                rrg_k = _ac2.number_input(
                    "RS Period", 3, 30, _def_k, 1, key="rrg_k_weekly",
                    help="Look-back for relative-strength vs benchmark. Weekly default: 10.",
                )
                rrg_tail = _ac3.number_input(
                    "Trail Length", 3, 16, _def_tail, 1, key="rrg_tail_weekly",
                    help="Number of past periods shown as a tail. Weekly default: 4.",
                )
            _render_rrg_view(
                weekly=True,
                rrg_m=int(rrg_m), rrg_k=int(rrg_k), rrg_tail=int(rrg_tail),
                freq_key="weekly",
            )


st.caption(
    "Data: Yahoo Finance (yfinance)  ·  Benchmark: NiftyMidSmallCap400  ·  "
    "EMA = Daily Exponential Moving Average  ·  VCP = Volatility Contraction Pattern (Minervini)"
)
