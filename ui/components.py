"""
Shared Streamlit display components.

Import and call from any page — no Streamlit state is stored here, so
every function is safe to call multiple times per render cycle.
"""

from typing import Optional
import pandas as pd
import streamlit as st

from config import PATTERN_COLS, COMBINED_RANK_WEIGHTS


# ---------------------------------------------------------------------------
# Colour / style helpers (used with pandas Styler.map())
# ---------------------------------------------------------------------------

def colour_pct(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return "color: #2ecc71" if val > 0 else ("color: #e74c3c" if val < 0 else "")


def colour_rsi(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    if val >= 70:
        return "color: #e74c3c; font-weight: bold"
    if val <= 30:
        return "color: #2ecc71; font-weight: bold"
    return ""


def colour_rs(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    if val >= 80:
        return "color: #22c55e; font-weight: bold"
    if val >= 50:
        return "color: #f59e0b"
    return "color: #ef4444"


def colour_rmv(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    if val < 15:
        return "color: #22c55e; font-weight: bold"
    if val < 40:
        return "color: #86efac"
    if val > 75:
        return "color: #ef4444"
    return ""


def colour_ret(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "color: #6b7280"
    return "color: #22c55e" if val > 0 else ("color: #ef4444" if val < 0 else "")


def colour_sentiment(val) -> str:
    v = str(val)
    if "Bullish" in v:
        return "background-color: #14532d; color: #4ade80; font-weight: 600"
    if "Bearish" in v:
        return "background-color: #450a0a; color: #f87171; font-weight: 600"
    return "color: #6b7280"


def colour_score(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    if val >= 70:
        return "color:#4ade80;font-weight:700"
    if val >= 50:
        return "color:#fbbf24"
    return "color:#f87171"


def tick(val: bool) -> str:
    return "✓" if val else "—"


# ---------------------------------------------------------------------------
# Display column constants
# ---------------------------------------------------------------------------

DISPLAY_COLS_HEAD = [
    "Symbol", "Company", "CMP", "Score", "RS Rating",
    "Sentiment", "News Vol", "Reddit",
    "RSI", "RMV", "% from 52W High",
    "Weekly %", "Monthly %", "Benchmark Outperf %", "Sector Outperf %",
]
DISPLAY_COLS_TAIL = ["PE", "Market Cap (Cr)"]

STYLE_COLS = {
    "Weekly %":            colour_pct,
    "Monthly %":           colour_pct,
    "Benchmark Outperf %": colour_pct,
    "Sector Outperf %":    colour_pct,
    "RSI":                 colour_rsi,
    "RS Rating":           colour_rs,
    "RMV":                 colour_rmv,
    "Sentiment":           colour_sentiment,
}

NUM_FORMAT = {
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

IPO_DISPLAY_COLS = [
    "Symbol", "Company", "Sector", "Industry",
    "CMP", "Base High", "% from Base High", "Days Listed",
    "Inside Day", "RMV", "PE", "RSI", "Weekly %", "Monthly %",
    "Sector Outperf %", "Market Cap (Cr)",
]

IPO_NUM_FORMAT = {
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

IPO_STYLE_COLS = {
    "Weekly %":         colour_pct,
    "Monthly %":        colour_pct,
    "Sector Outperf %": colour_pct,
    "RSI":              colour_rsi,
}

TRACKER_BASE_COLS = [
    "Capture Date", "Symbol", "Company", "Sector",
    "CMP", "Combined Score", "Score", "RS Rating",
    "Ret 1D %", "Ret 1W %", "Ret 2W %", "Ret 1M %",
    "RSI", "RMV", "% from 52W High",
]

TRACKER_NUM_FORMAT = {
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


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

def inject_css() -> None:
    st.markdown("""<style>
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
button[data-baseweb="tab"] {
    font-size: 0.97rem !important;
    font-weight: 500 !important;
    padding: 8px 22px !important;
}
details > summary p {
    font-size: 0.97rem !important;
    font-weight: 600 !important;
}
.sidebar-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6b7280;
    margin-bottom: 6px;
}
@media (max-width: 768px) {
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        gap: 0.5rem !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        flex: 1 1 calc(50% - 0.5rem) !important;
        min-width: calc(50% - 0.5rem) !important;
        width: auto !important;
    }
    [data-testid="stMetricValue"] > div { font-size: 1.1rem !important; }
    [data-testid="stPlotlyChart"] > div { height: 380px !important; }
    [data-testid="stPlotlyChart"] .js-plotly-plot,
    [data-testid="stPlotlyChart"] .plot-container { height: 380px !important; }
    button[data-baseweb="tab"] { font-size: 0.8rem !important; padding: 6px 12px !important; }
}
@media (max-width: 480px) {
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }
    [data-testid="stHorizontalBlock"] div[style*="border-radius:10px"] {
        padding: 10px 12px !important;
    }
    [data-testid="stPlotlyChart"] > div,
    [data-testid="stPlotlyChart"] .js-plotly-plot,
    [data-testid="stPlotlyChart"] .plot-container { height: 300px !important; }
    button[data-baseweb="tab"] { font-size: 0.72rem !important; padding: 5px 8px !important; }
    div[style*="line-height:2.2"] { line-height: 2 !important; }
}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def render_stock_chart(symbol: str, price_data: dict, lookback: int = 90) -> None:
    """Render interactive candlestick + EMA + volume chart for a stock."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ticker = f"{symbol}.NS"
    df_raw = price_data.get(ticker)
    if df_raw is None or df_raw.empty:
        st.warning(f"No price data available for {symbol}.")
        return

    df_full = df_raw.copy()
    df_full.index = pd.to_datetime(df_full.index)
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
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.03,
    )
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price",
        increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
        increasing_fillcolor="#22c55e", decreasing_fillcolor="#ef4444",
        line_width=1,
    ), row=1, col=1)
    for period, series in ema.items():
        fig.add_trace(go.Scatter(
            x=df.index, y=series, mode="lines",
            name=f"EMA{period}",
            line=dict(color=ema_colors[period], width=1.5),
        ), row=1, col=1)
    vol_colors = ["#22c55e" if c >= o else "#ef4444" for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="Volume",
        marker_color=vol_colors, marker_line_width=0, opacity=0.7,
    ), row=2, col=1)
    fig.update_layout(
        height=520, margin=dict(l=0, r=0, t=36, b=0),
        paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
        font=dict(color="#94a3b8", size=11),
        title=dict(text=f"<b>{symbol}</b> — last {lookback} trading days",
                   font=dict(color="#f1f5f9", size=14), x=0.01),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="left", x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        xaxis_rangeslider_visible=False,
        xaxis2=dict(showgrid=False),
    )
    fig.update_xaxes(gridcolor="#334155", showgrid=True, zeroline=False, tickfont=dict(size=10))
    fig.update_yaxes(gridcolor="#334155", showgrid=True, zeroline=False, tickfont=dict(size=10))
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Table renderers
# ---------------------------------------------------------------------------

def render_table(sub: pd.DataFrame) -> None:
    """Render the main screener results table with formatting and colour."""
    pat_cols = [c for c in PATTERN_COLS if c in sub.columns]
    ordered  = DISPLAY_COLS_HEAD + pat_cols + DISPLAY_COLS_TAIL
    available = [c for c in ordered if c in sub.columns]
    display = sub[available].copy()
    for col in pat_cols:
        display[col] = display[col].apply(tick)

    styled = display.style.format(NUM_FORMAT, na_rep="—")
    for col, fn in STYLE_COLS.items():
        if col in display.columns:
            styled = styled.map(fn, subset=[col])

    st.dataframe(styled, width="stretch", height=min(38 * (len(display) + 1), 600))


def render_ipo_section(ipo_df: Optional[pd.DataFrame]) -> None:
    """Render the IPO Base breakout section."""
    st.subheader("🚀 IPO Base — Breakout Zone  (< 200 days since listing)")
    if ipo_df is None or ipo_df.empty:
        st.info("No IPO base stocks detected in the current scan.")
        st.divider()
        return

    available = [c for c in IPO_DISPLAY_COLS if c in ipo_df.columns]
    display   = ipo_df[available].copy()
    if "Inside Day" in display.columns:
        display["Inside Day"] = display["Inside Day"].apply(tick)

    styled = display.style.format(IPO_NUM_FORMAT, na_rep="—")
    for col, fn in IPO_STYLE_COLS.items():
        if col in display.columns:
            styled = styled.map(fn, subset=[col])

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


# ---------------------------------------------------------------------------
# Top-ranked stock cards
# ---------------------------------------------------------------------------

def render_top_stocks(results: Optional[pd.DataFrame], sentiment_cache: dict = None, top_n: int = 10) -> None:
    """Render Top N ranked stocks as two rows of 5 score cards."""
    if results is None or results.empty:
        return

    st.markdown("### Top Ranked Stocks")
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

    for row_start in range(0, top_n, 5):
        row_slice = df_top.iloc[row_start:row_start + 5]
        cols = st.columns(5)
        for col_idx, (_, row) in enumerate(row_slice.iterrows()):
            i = row_start + col_idx
            active_patterns = [p for p in pat_cols if row.get(p) is True]
            pct_h = row.get("% from 52W High", None)
            pct_h_str = f"{pct_h:.1f}% from high" if pct_h is not None else ""
            rsi = row.get("RSI", None)
            rsi_str = f"RSI {rsi:.0f}" if rsi is not None else ""
            rs = row.get("RS Rating", None)
            rs_str = f"RS {int(rs)}" if rs is not None and not pd.isna(rs) else ""
            rs_color = (
                "#22c55e" if rs is not None and rs >= 80 else
                "#f59e0b" if rs is not None and rs >= 50 else
                "#ef4444"
            ) if rs is not None else "#6b7280"
            composite = row.get("_composite", row["Score"])
            sent = (sentiment_cache or {}).get(row["Symbol"], {})
            sent_label = sent.get("label", "")
            sent_badge = (
                "<span style='font-size:0.7rem;background:#14532d;color:#4ade80;"
                "border-radius:4px;padding:1px 6px;margin-left:4px'>🟢 Bullish</span>"
                if sent_label == "Bullish" else
                "<span style='font-size:0.7rem;background:#450a0a;color:#f87171;"
                "border-radius:4px;padding:1px 6px;margin-left:4px'>🔴 Bearish</span>"
                if sent_label == "Bearish" else ""
            )
            with cols[col_idx]:
                st.markdown(
                    f"<div style='background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.09);"
                    f"border-radius:10px;padding:14px 16px;height:100%'>"
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
                    + (f"<div style='margin-top:8px;font-size:0.7rem;color:#4ade80'>"
                       f"{'  '.join(active_patterns)}</div>" if active_patterns else "")
                    + "</div>",
                    unsafe_allow_html=True,
                )
        st.write("")
    st.divider()
