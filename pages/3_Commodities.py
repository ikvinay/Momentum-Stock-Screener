"""
Commodities page — MCX futures momentum ranking and performance tracker.
"""

import streamlit as st
import pandas as pd

from ui.components import tick, colour_pct, colour_rsi, colour_rmv, colour_rs, colour_score
from ui.sidebar import render_commodity_filters
from src.pipeline import load_commodity_results, last_updated

# ---------------------------------------------------------------------------
# Load data first so categories are available for the sidebar filter
# ---------------------------------------------------------------------------
commodity_results = load_commodity_results()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    if commodity_results is not None and not commodity_results.empty:
        _cats = commodity_results["Category"].unique().tolist()
        _com_filters = render_commodity_filters(categories=_cats)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
_last = last_updated()
st.markdown(
    f'<div style="display:flex;justify-content:space-between;align-items:flex-start;padding-bottom:6px">'
    f'<div>'
    f'<div style="font-size:22px;font-weight:700;color:#f1f5f9;letter-spacing:-0.02em">'
    f'<i class="ti ti-gem" style="color:#6366f1;margin-right:8px"></i>MCX Commodities — Momentum Ranking</div>'
    f'<div style="font-size:12px;color:#64748b;margin-top:4px;line-height:1.6">'
    f'Hard filter: EMA10 &rsaquo; EMA20 &rsaquo; EMA50  ·  RS vs Nifty 500 + Metal Index  ·  Auto-refresh 23:45 IST</div>'
    f'</div>'
    f'<div style="text-align:right;flex-shrink:0;padding-left:16px">'
    f'<div style="font-size:0.68rem;color:#64748b;font-weight:700;letter-spacing:0.07em;text-transform:uppercase">Last Updated</div>'
    f'<div style="font-size:0.88rem;color:#d1d5db;font-weight:600;margin-top:3px">{_last}</div>'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)
st.divider()

if commodity_results is None or commodity_results.empty:
    st.warning(
        "No commodity data yet.  \n"
        "Click **🏗️ Refresh Commodities** in the sidebar to fetch and screen MCX futures.  \n"
        "The screener runs automatically at **23:45 IST** daily after MCX closes."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_rank, tab_track = st.tabs(["  Rankings  ", "  Performance Tracker  "])

# ── Tab 1 — Rankings ─────────────────────────────────────────────────────────
with tab_rank:
    cat_filter   = _com_filters["cat_filter"]
    min_score    = _com_filters["min_score"]
    min_rs       = _com_filters["min_rs"]
    rsi_range    = _com_filters["rsi_range"]
    max_rmv      = _com_filters["max_rmv"]
    patterns     = _com_filters["pattern_opts"]

    df = commodity_results.copy()
    if cat_filter != "All":
        df = df[df["Category"] == cat_filter]
    df = df[df["Score"] >= min_score]
    if "RS Rating" in df.columns and min_rs > 1:
        df = df[df["RS Rating"] >= min_rs]
    if "RSI" in df.columns:
        df = df[df["RSI"].isna() | ((df["RSI"] >= rsi_range[0]) & (df["RSI"] <= rsi_range[1]))]
    if max_rmv < 100 and "RMV" in df.columns:
        df = df[df["RMV"].isna() | (df["RMV"] <= max_rmv)]
    for pat in patterns:
        if pat in df.columns:
            df = df[df[pat] == True]

    if df.empty:
        st.info("No commodities match the current filters.")
    else:
        # Summary metrics
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Passing",       len(df))
        mc2.metric("Categories",    df["Category"].nunique())
        mc3.metric("With VCP",      int(df["VCP"].sum()) if "VCP" in df.columns else 0)
        mc4.metric("With Inside Day", int(df["Inside Day"].sum()) if "Inside Day" in df.columns else 0)
        st.write("")

        # Top 3 score cards
        top3 = df.head(3)
        if len(top3):
            card_cols = st.columns(min(3, len(top3)))
            for i, (_, row) in enumerate(top3.iterrows()):
                with card_cols[i]:
                    delta_color = "normal" if row.get("Monthly %", 0) >= 0 else "inverse"
                    st.metric(
                        label=f"#{i+1}  {row['Commodity']}",
                        value=f"Score: {row['Score']:.0f}",
                        delta=f"{row.get('Monthly %', 0):+.2f}% (1M)",
                        delta_color=delta_color,
                    )
            st.write("")

        # Display columns
        _DISPLAY_COLS = [
            "Commodity", "Category", "CMP", "Score",
            "% from High", "Weekly %", "Monthly %", "Quarterly %",
            "vs Nifty500 1M", "vs Metal Index 1M", "RS Rating",
            "RSI", "RMV", "EMA10", "EMA20", "EMA50",
            "VCP", "Inside Day", "RS Trend",
        ]
        _NUM_FMT = {
            "CMP": "{:.2f}", "Score": "{:.1f}",
            "% from High": "{:.2f}", "Weekly %": "{:.2f}",
            "Monthly %": "{:.2f}", "Quarterly %": "{:.2f}",
            "vs Nifty500 1M": "{:.2f}", "vs Metal Index 1M": "{:.2f}",
            "RS Rating": "{:.0f}", "RSI": "{:.1f}", "RMV": "{:.1f}",
            "EMA10": "{:.2f}", "EMA20": "{:.2f}", "EMA50": "{:.2f}",
        }

        available = [c for c in _DISPLAY_COLS if c in df.columns]
        display = df[available].copy()
        for col in ("VCP", "Inside Day", "RS Trend"):
            if col in display.columns:
                display[col] = display[col].apply(tick)

        styled = display.style.format(_NUM_FMT, na_rep="—")
        for col in ("Weekly %", "Monthly %", "Quarterly %", "vs Nifty500 1M", "vs Metal Index 1M"):
            if col in display.columns:
                styled = styled.map(colour_pct, subset=[col])
        if "RS Rating" in display.columns:
            styled = styled.map(colour_rs, subset=["RS Rating"])
        if "RSI" in display.columns:
            styled = styled.map(colour_rsi, subset=["RSI"])
        if "RMV" in display.columns:
            styled = styled.map(colour_rmv, subset=["RMV"])
        if "Score" in display.columns:
            styled = styled.map(colour_score, subset=["Score"])

        st.dataframe(styled, use_container_width=True, height=min(38 * (len(display) + 1), 600))
        st.divider()

        col_dl, _ = st.columns([2, 5])
        with col_dl:
            st.download_button(
                label=f"⬇  Export {len(df)} commodities  (.csv)",
                data=df.to_csv(index=False),
                file_name="commodity_results.csv",
                mime="text/csv",
            )

# ── Tab 2 — Performance Tracker ──────────────────────────────────────────────
with tab_track:
    from src.commodity_tracker import load_commodity_snapshots, compute_commodity_forward_returns, commodity_score_band_summary

    snapshots = load_commodity_snapshots()
    if snapshots.empty:
        st.info(
            "No snapshots yet. Run the commodity screener to start recording daily results. "
            "Forward returns are computed automatically as new price data arrives."
        )
    else:
        with st.spinner("Computing commodity forward returns…"):
            df_track = compute_commodity_forward_returns(snapshots)

        n_dates    = df_track["Capture Date"].nunique()
        n_captures = len(df_track)
        w1 = df_track["Ret 1W %"].dropna() if "Ret 1W %" in df_track.columns else pd.Series(dtype=float)
        m1 = df_track["Ret 1M %"].dropna() if "Ret 1M %" in df_track.columns else pd.Series(dtype=float)

        tc1, tc2, tc3, tc4, tc5, tc6 = st.columns(6)
        tc1.metric("Capture Days",   n_dates)
        tc2.metric("Total Captures", n_captures)
        tc3.metric("Win Rate 1W",    f"{(w1 > 0).mean() * 100:.0f}%" if len(w1) else "—")
        tc4.metric("Avg Return 1W",  f"{w1.mean():+.2f}%" if len(w1) else "—")
        tc5.metric("Win Rate 1M",    f"{(m1 > 0).mean() * 100:.0f}%" if len(m1) else "—")
        tc6.metric("Avg Return 1M",  f"{m1.mean():+.2f}%" if len(m1) else "—")
        st.divider()

        sub_snap, sub_band = st.tabs(["Daily Snapshots", "Score Band Analysis"])

        with sub_snap:
            all_dates  = sorted(df_track["Capture Date"].unique(), reverse=True)
            sf1, sf2   = st.columns([2, 3])
            with sf1:
                sel_dates = st.multiselect("Date", all_dates, default=all_dates[:5], placeholder="All dates")
            with sf2:
                sym_search = st.text_input("Search commodity", placeholder="e.g. Gold")

            view = df_track.copy()
            if sel_dates:
                view = view[view["Capture Date"].isin(sel_dates)]
            if sym_search:
                view = view[view["Commodity"].str.contains(sym_search.strip(), case=False, na=False)]

            _TRACK_COLS = [
                "Capture Date", "Commodity", "Category", "Score", "RS Rating",
                "CMP", "RSI", "RMV",
                "Ret 1D %", "Ret 1W %", "Ret 2W %", "Ret 1M %",
            ]
            _TRACK_FMT = {
                "Score": "{:.1f}", "RS Rating": "{:.0f}", "CMP": "{:.2f}",
                "RSI": "{:.1f}", "RMV": "{:.1f}",
                "Ret 1D %": "{:.2f}", "Ret 1W %": "{:.2f}",
                "Ret 2W %": "{:.2f}", "Ret 1M %": "{:.2f}",
            }
            avail   = [c for c in _TRACK_COLS if c in view.columns]
            display = view[avail].copy()
            styled  = display.style.format(_TRACK_FMT, na_rep="—")
            for ret_col in ("Ret 1D %", "Ret 1W %", "Ret 2W %", "Ret 1M %"):
                if ret_col in display.columns:
                    styled = styled.map(colour_pct, subset=[ret_col])
            if "RS Rating" in display.columns:
                styled = styled.map(colour_rs, subset=["RS Rating"])

            st.dataframe(styled, use_container_width=True, height=min(38 * (len(display) + 1), 600))
            st.divider()
            col_dl2, _ = st.columns([2, 5])
            with col_dl2:
                st.download_button(
                    label=f"⬇  Export tracker data  ({len(view)} rows)",
                    data=view.to_csv(index=False),
                    file_name="commodity_tracker.csv",
                    mime="text/csv",
                )

        with sub_band:
            st.markdown("##### Score Band Performance")
            sb_df = commodity_score_band_summary(df_track)
            if sb_df.empty:
                st.caption("Not enough data yet — needs at least 5 snapshot days.")
            else:
                st.dataframe(sb_df, use_container_width=True, hide_index=True)

st.caption(
    "Data: Yahoo Finance international proxies (COMEX / NYMEX / LME)  ·  "
    "RS vs Nifty 500 (^CRSLDX) and Nifty Metal Index (^CNXMETAL)  ·  "
    "MCX closes 23:30 IST — screener runs at 23:45 IST"
)
