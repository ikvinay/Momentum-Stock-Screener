"""
NSE Indices page — ranked by momentum, independent performance tracker.
"""

import streamlit as st
import pandas as pd
from ui.components import inject_css, tick, colour_pct, colour_rsi, colour_rmv, colour_rs, colour_score
from ui.sidebar import render_data_management, render_index_filters
from src.pipeline import load_index_results, last_updated

inject_css()

# ---------------------------------------------------------------------------
# Load data first so categories are available for the sidebar filter
# ---------------------------------------------------------------------------
index_results = load_index_results()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    render_data_management()
    if index_results is not None and not index_results.empty:
        st.divider()
        _cats = index_results["Category"].unique().tolist()
        _idx_filters = render_index_filters(categories=_cats)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
col_title, col_meta = st.columns([5, 2])
with col_title:
    st.markdown("# 🗂️ NSE Indices — Momentum Ranking")
    st.caption("Hard filter: EMA10 > EMA20 > EMA50  ·  RS measured vs Nifty 500  ·  Same scoring as stocks")
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
# Guard — no data
# ---------------------------------------------------------------------------
if index_results is None or index_results.empty:
    st.info("No index results yet. Click **📥 Refresh Stock Data** then **🔍 Run Screener** in the sidebar.")
    st.stop()

# ---------------------------------------------------------------------------
# Highlights — top 5 by Score (unfiltered, always from full results)
# ---------------------------------------------------------------------------
st.markdown("### 🏆 Highlights — Top 5 Sector / Theme Indices")
top5 = index_results.head(5)
if not top5.empty:
    h_cols = st.columns(min(5, len(top5)))
    for i, (_, row) in enumerate(top5.iterrows()):
        with h_cols[i]:
            score_color = "#4ade80" if row.get("Score", 0) >= 70 else "#fbbf24"
            trend_arrow = "▲" if row.get("RS Trend") else "▼"
            rs_rating   = row.get("RS Rating", "—")
            st.markdown(
                f"<div style='background:rgba(255,255,255,0.05);border-radius:8px;"
                f"padding:12px;text-align:center;border:1px solid rgba(255,255,255,0.1)'>"
                f"<div style='font-size:0.72rem;color:#94a3b8'>{row.get('Category','')}</div>"
                f"<div style='font-weight:700;font-size:0.9rem;margin:4px 0'>{row.get('Index','')}</div>"
                f"<div style='font-size:1.4rem;font-weight:800;color:{score_color}'>{row.get('Score',0):.0f}</div>"
                f"<div style='font-size:0.78rem;color:#94a3b8'>Score</div>"
                f"<div style='font-size:0.82rem;margin-top:4px'>"
                f"RS {rs_rating} &nbsp;·&nbsp; RSI {row.get('RSI',0):.0f} &nbsp;·&nbsp; {trend_arrow} RS Trend"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
st.markdown("<div style='margin-bottom:24px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Apply sidebar filters
# ---------------------------------------------------------------------------
cat_filter   = _idx_filters["cat_filter"]
min_score    = _idx_filters["min_score"]
min_rs       = _idx_filters["min_rs"]
min_rsi      = _idx_filters["min_rsi"]
max_rmv      = _idx_filters["max_rmv"]
pattern_opts = _idx_filters["pattern_opts"]

df = index_results.copy()
if cat_filter != "All":
    df = df[df["Category"] == cat_filter]
df = df[
    (df["Score"] >= min_score)
    & (df.get("RS Rating", pd.Series(99, index=df.index)) >= min_rs)
    & (df["RSI"] >= min_rsi)
    & (df["RMV"] <= max_rmv)
]
for pat in pattern_opts:
    if pat in df.columns:
        df = df[df[pat] == True]

st.caption(f"{len(df)} indices match filters")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_rank, tab_track = st.tabs(["  🗂️  Rankings  ", "  📈  Performance Tracker  "])

# ── Tab 1 — Rankings ─────────────────────────────────────────────────────────
with tab_rank:
    if df.empty:
        st.info("No indices match the current filters.")
    else:
        _DISP = [
            "Index", "Category", "Ticker", "CMP",
            "% from High", "Weekly %", "Monthly %", "Quarterly %",
            "vs Nifty500 1M", "RS Rating", "RSI", "RMV", "VCP", "Inside Day", "RS Trend", "Score",
        ]
        available = [c for c in _DISP if c in df.columns]
        disp = df[available].copy()
        for col in ("VCP", "Inside Day", "RS Trend"):
            if col in disp.columns:
                disp[col] = disp[col].apply(tick)

        fmt = {
            "CMP": "{:.2f}", "% from High": "{:.2f}", "Weekly %": "{:.2f}",
            "Monthly %": "{:.2f}", "Quarterly %": "{:.2f}", "vs Nifty500 1M": "{:.2f}",
            "RS Rating": "{:.0f}", "RSI": "{:.1f}", "RMV": "{:.1f}", "Score": "{:.1f}",
        }
        styled = disp.style.format(fmt, na_rep="—")
        for col in ("Weekly %", "Monthly %", "Quarterly %", "vs Nifty500 1M"):
            if col in disp.columns:
                styled = styled.map(colour_pct, subset=[col])
        if "RSI" in disp.columns:
            styled = styled.map(colour_rsi, subset=["RSI"])
        if "RMV" in disp.columns:
            styled = styled.map(colour_rmv, subset=["RMV"])
        if "RS Rating" in disp.columns:
            styled = styled.map(colour_rs, subset=["RS Rating"])
        if "Score" in disp.columns:
            styled = styled.map(colour_score, subset=["Score"])

        st.dataframe(styled, use_container_width=True, height=min(42 * (len(disp) + 1), 600))

        col_dl, _ = st.columns([2, 5])
        with col_dl:
            st.download_button(
                label=f"⬇  Export {len(df)} indices  (.csv)",
                data=df.to_csv(index=False),
                file_name="index_results.csv",
                mime="text/csv",
            )

# ── Tab 2 — Performance Tracker ──────────────────────────────────────────────
with tab_track:
    from src.index_tracker import load_index_snapshots, compute_index_forward_returns, index_score_band_summary

    snapshots = load_index_snapshots()
    if snapshots.empty:
        st.info("No index snapshots yet — run the screener to start tracking.")
    else:
        snapshots_with_ret = compute_index_forward_returns(snapshots)

        sub_snap, sub_band = st.tabs(["Daily Snapshots", "Score Band Analysis"])

        with sub_snap:
            dates = sorted(snapshots_with_ret["Capture Date"].unique(), reverse=True)
            sel_date = st.selectbox("Snapshot date", dates, key="idx_snap_date")
            snap_df = snapshots_with_ret[snapshots_with_ret["Capture Date"] == sel_date].copy()

            _SNAP_COLS = [
                "Index", "Category", "Score", "RS Rating",
                "Monthly %", "vs Nifty500 1M", "RSI", "RMV",
                "Ret 1D %", "Ret 1W %", "Ret 2W %", "Ret 1M %",
            ]
            snap_avail = [c for c in _SNAP_COLS if c in snap_df.columns]
            snap_disp  = snap_df[snap_avail].copy()
            snap_styled = snap_disp.style.format(
                {c: "{:.2f}" for c in ("Score", "Monthly %", "vs Nifty500 1M", "RSI", "RMV",
                                       "Ret 1D %", "Ret 1W %", "Ret 2W %", "Ret 1M %")}
                | ({"RS Rating": "{:.0f}"} if "RS Rating" in snap_disp.columns else {}),
                na_rep="—",
            )
            for col in ("Ret 1D %", "Ret 1W %", "Ret 2W %", "Ret 1M %", "Monthly %", "vs Nifty500 1M"):
                if col in snap_disp.columns:
                    snap_styled = snap_styled.map(colour_pct, subset=[col])
            if "RS Rating" in snap_disp.columns:
                snap_styled = snap_styled.map(colour_rs, subset=["RS Rating"])

            st.dataframe(snap_styled, use_container_width=True, height=min(42 * (len(snap_disp) + 1), 500))
            st.caption(f"{len(snap_disp)} indices captured on {sel_date}")

            col_dl, _ = st.columns([2, 5])
            with col_dl:
                st.download_button(
                    label=f"⬇  Export snapshot  ({len(snap_disp)} rows)",
                    data=snap_disp.to_csv(index=False),
                    file_name=f"index_snapshot_{sel_date}.csv",
                    mime="text/csv",
                )

        with sub_band:
            perf_df = index_score_band_summary(snapshots_with_ret)
            if perf_df.empty:
                st.info("Not enough data for score band analysis yet.")
            else:
                st.dataframe(perf_df, use_container_width=True)
                st.caption("Forward returns by score quintile across all index snapshots")

st.caption("Data: Yahoo Finance (yfinance)  ·  RS benchmark: Nifty 500  ·  Hard filter: EMA10 > EMA20 > EMA50")
