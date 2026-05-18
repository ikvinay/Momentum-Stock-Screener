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
    if val > 85:
        return "color: #22c55e; font-weight: bold"
    if val > 70:
        return "color: #f59e0b; font-weight: bold"
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


def colour_float(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    if val < 25:
        return "color: #22c55e; font-weight: bold"
    if val < 50:
        return "color: #86efac"
    if val > 75:
        return "color: #ef4444"
    return ""


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
    "Symbol", "Company", "Score", "RS Rating",
    "Sentiment",
    "% from 52W High",
    "Weekly %", "Monthly %", "Benchmark Outperf %", "Sector Outperf %",
]
DISPLAY_COLS_TAIL = ["Free Float %"]

STYLE_COLS = {
    "Weekly %":            colour_pct,
    "Monthly %":           colour_pct,
    "Benchmark Outperf %": colour_pct,
    "Sector Outperf %":    colour_pct,
    "RSI":                 colour_rsi,
    "RS Rating":           colour_rs,
    "RMV":                 colour_rmv,
    "Free Float %":        colour_float,
    "Sentiment":           colour_sentiment,
}

NUM_FORMAT = {
    "CMP":                 "{:.2f}",
    "PE":                  "{:.2f}",
    "Free Float %":        "{:.1f}",
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

/* ── Tabler Icons webfont ─────────────────────────────────────────── */
@import url('https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3.31.0/dist/tabler-icons.min.css');

.ti { font-size: 16px; vertical-align: -2px; line-height: 1; }
.ti-sm  { font-size: 14px; }
.ti-md  { font-size: 18px; }
.ti-lg  { font-size: 22px; }
.ti-xl  { font-size: 26px; }
.ti-accent { color: #6366f1; }
.ti-muted  { color: #475569; }
.ti-success { color: #22c55e; }
.ti-danger  { color: #ef4444; }
.ti-warning { color: #f59e0b; }

/* =====================================================================
   SIDEBAR — Redesigned nav (heuristic-driven)
   ===================================================================== */

/* ── Shell ──────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background-color: #080c12 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
    min-width: 272px !important;
    max-width: 272px !important;
}
section[data-testid="stSidebar"] { scrollbar-width: thin; scrollbar-color: rgba(255,255,255,0.08) transparent; }
section[data-testid="stSidebar"]::-webkit-scrollbar { width: 3px; }
section[data-testid="stSidebar"]::-webkit-scrollbar-track { background: transparent; }
section[data-testid="stSidebar"]::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 2px; }

/* ── App header ─────────────────────────────────────────────────────── */
[data-testid="stSidebarHeader"] {
    height: auto !important;
    min-height: 64px !important;
    padding: 18px 10px 16px 18px !important;
    border-bottom: 1px solid rgba(255,255,255,0.06) !important;
    margin-bottom: 0 !important;
    background: #080c12 !important;
}
[data-testid="stSidebarLogo"] {
    flex: 1 !important;
    display: flex !important;
    align-items: center !important;
}
[data-testid="stSidebarLogo"] img.stLogo {
    height: 50px !important;
    width: auto !important;
    max-width: 240px !important;
}

/* ── Nav container ──────────────────────────────────────────────────── */
[data-testid="stSidebarNav"] {
    padding: 10px 0 10px 0 !important;
    border-bottom: 1px solid rgba(255,255,255,0.06) !important;
    margin-bottom: 0 !important;
}

/* Strip Streamlit's default wrapper chrome */
[data-testid="stSidebarNavLink"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
    border-radius: 0 !important;
}

/* ── Nav item link ──────────────────────────────────────────────────── */
[data-testid="stSidebarNavLink"] a {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    padding: 9px 14px 9px 16px !important;
    margin: 2px 8px !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #64748b !important;
    text-decoration: none !important;
    border-left: 3px solid transparent !important;
    transition: background 0.15s, color 0.15s, border-color 0.15s !important;
    letter-spacing: 0.01em !important;
}

/* Hover — reveal but don't commit */
[data-testid="stSidebarNavLink"] a:hover {
    background: rgba(255,255,255,0.04) !important;
    color: #94a3b8 !important;
}

/* ── Active page — high-contrast solid fill + left rail ────────────── */
[data-testid="stSidebarNavLink"] a[aria-current="page"] {
    background: rgba(99,102,241,0.14) !important;
    color: #c7d2fe !important;
    border-left-color: #6366f1 !important;
    font-weight: 600 !important;
}
[data-testid="stSidebarNavLink"] a[aria-current="page"]:hover {
    background: rgba(99,102,241,0.20) !important;
}

/* ── Icons — full opacity on active, dimmed on inactive ─────────────── */
[data-testid="stSidebarNavLink"] a span:first-child {
    font-size: 16px !important;
    opacity: 0.45 !important;
    transition: opacity 0.15s !important;
}
[data-testid="stSidebarNavLink"] a:hover span:first-child {
    opacity: 0.7 !important;
}
[data-testid="stSidebarNavLink"] a[aria-current="page"] span:first-child {
    opacity: 1 !important;
}

/* Hide Admin from nav (accessible via top-right gear or direct URL) */
[data-testid="stSidebarNavLink"]:has(a[href*="admin" i]) {
    display: none !important;
}

/* ── Controls area (below nav) ──────────────────────────────────────── */
[data-testid="stSidebarUserContent"] {
    padding: 16px 10px 24px !important;
}

/* Section label injected above user-content area */
[data-testid="stSidebarUserContent"]::before {
    content: "CONTROLS";
    display: block;
    font-size: 9.5px;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: #334155;
    padding: 0 8px 10px 8px;
    text-transform: uppercase;
}

/* ── Expanders ──────────────────────────────────────────────────────── */
[data-testid="stSidebar"] details {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    margin-bottom: 6px !important;
}
[data-testid="stSidebar"] details > summary {
    padding: 9px 14px !important;
    cursor: pointer !important;
    list-style: none !important;
}
[data-testid="stSidebar"] details > summary::-webkit-details-marker { display: none; }
[data-testid="stSidebar"] details > summary p {
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #64748b !important;
}
[data-testid="stSidebar"] details[open] > summary {
    border-bottom: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] details > div {
    padding: 10px 12px 12px !important;
}

/* ── Bordered containers ────────────────────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 8px !important;
    padding: 10px 12px !important;
    margin-bottom: 6px !important;
}

/* ── Buttons ────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: #64748b !important;
    border-radius: 7px !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
    padding: 7px 12px !important;
    transition: background 0.15s, border-color 0.15s, color 0.15s !important;
    letter-spacing: 0.01em !important;
}
[data-testid="stSidebar"] .stButton > button:hover:not(:disabled) {
    background: rgba(255,255,255,0.07) !important;
    border-color: rgba(255,255,255,0.15) !important;
    color: #cbd5e1 !important;
}
[data-testid="stSidebar"] .stButton > button:disabled {
    opacity: 0.25 !important;
    cursor: not-allowed !important;
}

/* ── Widget labels inside sidebar ───────────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] label p {
    font-size: 11px !important;
    font-weight: 600 !important;
    color: #475569 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}

/* Slider track and thumb */
[data-testid="stSidebar"] [data-testid="stSliderThumb"] {
    background: #6366f1 !important;
    border-color: #6366f1 !important;
}
[data-testid="stSidebar"] [data-testid="stSliderTrackFill"] {
    background: #6366f1 !important;
}

/* Selectbox / multiselect */
[data-testid="stSidebar"] [data-baseweb="select"] {
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(255,255,255,0.1) !important;
    border-radius: 7px !important;
}

/* Checkbox */
[data-testid="stSidebar"] [data-baseweb="checkbox"] span {
    border-color: rgba(255,255,255,0.18) !important;
}
[data-testid="stSidebar"] [data-baseweb="checkbox"] [aria-checked="true"] span {
    background: #6366f1 !important;
    border-color: #6366f1 !important;
}

/* Divider inside sidebar */
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.07) !important;
    margin: 10px 0 !important;
}

/* Info / error alerts in sidebar */
[data-testid="stSidebar"] [data-testid="stAlert"] {
    border-radius: 8px !important;
    font-size: 12px !important;
    padding: 8px 10px !important;
}

/* =====================================================================
   MAIN AREA — enterprise analytics skin
   ===================================================================== */

/* ── Metric cards: accent top border + hover lift ────────────────── */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-top: 2px solid #6366f1 !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
    transition: background 0.15s !important;
}
[data-testid="metric-container"]:hover {
    background: rgba(255,255,255,0.055) !important;
}
[data-testid="stMetricLabel"] > div {
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: #64748b !important;
    margin-bottom: 3px !important;
}
[data-testid="stMetricValue"] > div {
    font-size: 1.45rem !important;
    font-weight: 700 !important;
    color: #f1f5f9 !important;
    line-height: 1.15 !important;
}
[data-testid="stMetricDelta"] { font-size: 0.76rem !important; }

/* ── Tabs: pill container + crisp active state ───────────────────── */
[data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.025) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
    padding: 3px !important;
    gap: 2px !important;
}
button[data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-radius: 7px !important;
    font-size: 0.86rem !important;
    font-weight: 500 !important;
    padding: 6px 16px !important;
    color: #64748b !important;
    transition: background 0.15s, color 0.15s !important;
    white-space: nowrap !important;
    letter-spacing: 0.01em !important;
}
button[data-baseweb="tab"]:hover:not([aria-selected="true"]) {
    background: rgba(255,255,255,0.05) !important;
    color: #94a3b8 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: rgba(99,102,241,0.15) !important;
    color: #a5b4fc !important;
    font-weight: 600 !important;
}
[data-baseweb="tab-highlight"],
[data-baseweb="tab-border"] { display: none !important; }

/* ── Download button: indigo-tinted ──────────────────────────────── */
[data-testid="stDownloadButton"] > button {
    background: rgba(99,102,241,0.08) !important;
    border: 1px solid rgba(99,102,241,0.28) !important;
    color: #a5b4fc !important;
    border-radius: 8px !important;
    font-size: 0.87rem !important;
    font-weight: 500 !important;
    padding: 8px 20px !important;
    width: auto !important;
    margin-top: 4px !important;
    transition: background 0.15s, border-color 0.15s, color 0.15s !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(99,102,241,0.16) !important;
    border-color: rgba(99,102,241,0.48) !important;
    color: #c4b5fd !important;
}

/* ── Section headings ────────────────────────────────────────────── */
[data-testid="stMarkdown"] h3 {
    font-size: 1rem !important;
    font-weight: 700 !important;
    color: #e2e8f0 !important;
    letter-spacing: -0.01em !important;
    margin-bottom: 2px !important;
}
[data-testid="stMarkdown"] h4,
[data-testid="stMarkdown"] h5 {
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    color: #cbd5e1 !important;
}

/* ── Captions ─────────────────────────────────────────────────────── */
[data-testid="stCaptionContainer"] p {
    font-size: 0.77rem !important;
    color: #64748b !important;
    line-height: 1.5 !important;
}

/* ── Expander summary (main area only) ───────────────────────────── */
:not([data-testid="stSidebar"]) details > summary p {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}

/* ── Focus visible — WCAG 2.1 AA ────────────────────────────────── */
button:focus-visible,
a:focus-visible,
[data-baseweb="select"]:focus-visible,
input:focus-visible {
    outline: 2px solid #6366f1 !important;
    outline-offset: 2px !important;
}

/* ── Radio labels ────────────────────────────────────────────────── */
[data-baseweb="radio"] [data-testid="stMarkdownContainer"] p {
    font-size: 0.86rem !important;
}

/* ── Utility ─────────────────────────────────────────────────────── */
.sidebar-label {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748b;
    margin-bottom: 6px;
}

/* ── Mobile 768px ────────────────────────────────────────────────── */
@media (max-width: 768px) {
    [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; gap: 0.5rem !important; }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        flex: 1 1 calc(50% - 0.5rem) !important;
        min-width: calc(50% - 0.5rem) !important;
        width: auto !important;
    }
    [data-testid="stMetricValue"] > div { font-size: 1.1rem !important; }
    button[data-baseweb="tab"] { font-size: 0.78rem !important; padding: 5px 10px !important; }
    [data-testid="stPlotlyChart"] > div { height: 380px !important; }
    [data-testid="stPlotlyChart"] .js-plotly-plot,
    [data-testid="stPlotlyChart"] .plot-container { height: 380px !important; }
    section[data-testid="stSidebar"] { min-width: 220px !important; max-width: 220px !important; }
}

/* ── Mobile 480px ────────────────────────────────────────────────── */
@media (max-width: 480px) {
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        flex: 1 1 100% !important; min-width: 100% !important;
    }
    [data-testid="stMetricValue"] > div { font-size: 1rem !important; }
    button[data-baseweb="tab"] { font-size: 0.72rem !important; padding: 4px 8px !important; }
    [data-testid="stPlotlyChart"] > div,
    [data-testid="stPlotlyChart"] .js-plotly-plot,
    [data-testid="stPlotlyChart"] .plot-container { height: 300px !important; }
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

    # Ascending Triangle overlay
    try:
        import numpy as _np2
        from src.indicators import detect_ascending_triangle
        asc = detect_ascending_triangle(df_full)
        if asc["detected"]:
            n_sl = asc["n_slice"]
            chart_len = len(df)
            if n_sl <= chart_len:
                sl_dates = df.index[-n_sl:]
            else:
                sl_dates = df.index
                n_sl = chart_len
            xs = _np2.arange(n_sl, dtype=float)
            resist_y  = asc["resist_slope"] * xs + asc["resist_intercept"]
            support_y = asc["support_slope"] * xs + asc["support_intercept"]
            fig.add_trace(go.Scatter(
                x=sl_dates, y=resist_y, mode="lines", name="Asc△ Resist",
                line=dict(color="#f97316", width=1.5, dash="dash"),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=sl_dates, y=support_y, mode="lines", name="_asc_support",
                line=dict(color="#22c55e", width=1.5, dash="dash"),
                showlegend=False,
            ), row=1, col=1)
            r_px = [(sl_dates[i], y) for i, y in asc["resist_pivots"]  if i < len(sl_dates)]
            s_px = [(sl_dates[i], y) for i, y in asc["support_pivots"] if i < len(sl_dates)]
            if r_px:
                fig.add_trace(go.Scatter(
                    x=[p[0] for p in r_px], y=[p[1] for p in r_px],
                    mode="markers", name="",
                    marker=dict(color="#f97316", size=7, symbol="triangle-down"),
                    showlegend=False,
                ), row=1, col=1)
            if s_px:
                fig.add_trace(go.Scatter(
                    x=[p[0] for p in s_px], y=[p[1] for p in s_px],
                    mode="markers", name="",
                    marker=dict(color="#22c55e", size=7, symbol="triangle-up"),
                    showlegend=False,
                ), row=1, col=1)
    except Exception:
        pass

    # Symmetrical Triangle overlay
    try:
        import numpy as _np
        from src.indicators import detect_symmetrical_triangle
        sym = detect_symmetrical_triangle(df_full)
        if sym["detected"]:
            n_sl = sym["n_slice"]
            # Detection ran on df_full.tail(n_sl); map to the chart's date axis
            chart_len = len(df)
            if n_sl <= chart_len:
                sl_dates = df.index[-n_sl:]
            else:
                sl_dates = df.index
                n_sl = chart_len
            xs = _np.arange(n_sl, dtype=float)
            upper_y = sym["upper_slope"] * xs + sym["upper_intercept"]
            lower_y = sym["lower_slope"] * xs + sym["lower_intercept"]
            fig.add_trace(go.Scatter(
                x=sl_dates, y=upper_y, mode="lines", name="Sym△",
                line=dict(color="#a855f7", width=1.5, dash="dash"),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=sl_dates, y=lower_y, mode="lines", name="_sym_lower",
                line=dict(color="#a855f7", width=1.5, dash="dash"),
                showlegend=False,
            ), row=1, col=1)
            # Pivot markers
            u_px = [(sl_dates[i], y) for i, y in sym["upper_pivots"] if i < len(sl_dates)]
            l_px = [(sl_dates[i], y) for i, y in sym["lower_pivots"] if i < len(sl_dates)]
            if u_px:
                fig.add_trace(go.Scatter(
                    x=[p[0] for p in u_px], y=[p[1] for p in u_px],
                    mode="markers", name="",
                    marker=dict(color="#a855f7", size=7, symbol="triangle-down"),
                    showlegend=False,
                ), row=1, col=1)
            if l_px:
                fig.add_trace(go.Scatter(
                    x=[p[0] for p in l_px], y=[p[1] for p in l_px],
                    mode="markers", name="",
                    marker=dict(color="#a855f7", size=7, symbol="triangle-up"),
                    showlegend=False,
                ), row=1, col=1)
    except Exception:
        pass

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Relative Rotation Graph chart
# ---------------------------------------------------------------------------

def render_rrg_chart(
    rrg_df: pd.DataFrame,
    title: str = "Relative Rotation Graph",
    key: str = "rrg_chart",
    on_select: str = "ignore",
) -> object:
    """Render a Relative Rotation Graph (RRG) scatter chart with quadrant shading and trails.

    Returns the plotly_chart event object (selection data available when on_select='rerun').
    """
    import plotly.graph_objects as go
    from src.rrg import QUADRANT_FILL, QUADRANT_COLORS

    if rrg_df is None or rrg_df.empty:
        st.info("Not enough price history to render the RRG chart.")
        return

    all_x = [v for tx in rrg_df["tail_x"] for v in tx]
    all_y = [v for ty in rrg_df["tail_y"] for v in ty]
    pad  = 1.5
    x_lo = min(min(all_x) - pad, 97.5)
    x_hi = max(max(all_x) + pad, 102.5)
    y_lo = min(min(all_y) - pad, 97.5)
    y_hi = max(max(all_y) + pad, 102.5)

    fig = go.Figure()

    # Quadrant background shading
    _QLABEL = {"LEADING": "Leading", "WEAKENING": "Weakening",
               "LAGGING": "Lagging", "IMPROVING": "Improving"}
    for (x0, x1, y0, y1), qname in [
        ((100, x_hi, 100, y_hi), "Leading"),
        ((100, x_hi, y_lo, 100), "Weakening"),
        ((x_lo, 100, y_lo, 100), "Lagging"),
        ((x_lo, 100, 100, y_hi), "Improving"),
    ]:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                      fillcolor=QUADRANT_FILL[qname], line_width=0, layer="below")

    # Quadrant corner labels
    for label, lx, ly, xa, ya in [
        ("LEADING",   x_hi, y_hi, "right", "top"),
        ("WEAKENING", x_hi, y_lo, "right", "bottom"),
        ("LAGGING",   x_lo, y_lo, "left",  "bottom"),
        ("IMPROVING", x_lo, y_hi, "left",  "top"),
    ]:
        fig.add_annotation(
            x=lx, y=ly, text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=10, color=QUADRANT_COLORS[_QLABEL[label]]),
            xanchor=xa, yanchor=ya, opacity=0.65,
        )

    # Crosshair at centre (100, 100)
    fig.add_hline(y=100, line=dict(color="rgba(255,255,255,0.18)", width=1, dash="dot"))
    fig.add_vline(x=100, line=dict(color="rgba(255,255,255,0.18)", width=1, dash="dot"))

    # Trails, arrows and current-position markers
    for _, row in rrg_df.iterrows():
        color = row["color"]
        tx, ty = row["tail_x"], row["tail_y"]
        name   = row["name"]

        # Faded trail line (all but last point)
        if len(tx) > 1:
            fig.add_trace(go.Scatter(
                x=tx[:-1], y=ty[:-1],
                mode="lines",
                line=dict(color=color, width=1.2),
                opacity=0.30,
                showlegend=False,
                hoverinfo="skip",
            ))

        # Arrow from second-to-last → last point
        if len(tx) >= 2:
            fig.add_annotation(
                x=tx[-1], y=ty[-1],
                ax=tx[-2], ay=ty[-2],
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=3, arrowsize=1.0,
                arrowcolor=color, arrowwidth=2.0,
            )

        # Current position marker + label
        fig.add_trace(go.Scatter(
            x=[tx[-1]], y=[ty[-1]],
            mode="markers+text",
            marker=dict(size=10, color=color, line=dict(color="white", width=1.5)),
            text=[name],
            textposition="top center",
            textfont=dict(size=8, color="#e2e8f0"),
            name=name,
            customdata=[[name, row["rs_ratio"], row["rs_momentum"], row["quadrant"]]],
            hovertemplate=(
                f"<b>{name}</b><br>"
                "RS-Ratio: %{customdata[1]:.2f}<br>"
                "RS-Momentum: %{customdata[2]:.2f}<br>"
                "Quadrant: %{customdata[3]}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#e2e8f0"), x=0.0),
        xaxis=dict(
            title="← Underperforming  |  RS-Ratio  |  Outperforming →",
            range=[x_lo, x_hi],
            gridcolor="rgba(255,255,255,0.07)",
            zeroline=False, tickfont=dict(size=9), color="#94a3b8",
        ),
        yaxis=dict(
            title="↓ Decelerating  |  RS-Momentum  |  Accelerating ↑",
            range=[y_lo, y_hi],
            gridcolor="rgba(255,255,255,0.07)",
            zeroline=False, tickfont=dict(size=9), color="#94a3b8",
        ),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=8),
            itemsizing="constant",
        ),
        height=580,
        margin=dict(l=65, r=20, t=45, b=65),
    )
    return st.plotly_chart(
        fig,
        use_container_width=True,
        key=key,
        on_select=on_select,
        selection_mode="points",
    )


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
    st.subheader("IPO Base — Breakout Zone  (< 200 days since listing)")
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
        "Green highlight = price already above base high (confirmed breakout)."
    )
    st.dataframe(styled, width="stretch", height=min(38 * (len(display) + 1), 500))
    st.divider()


# ---------------------------------------------------------------------------
# Top-ranked stock cards
# ---------------------------------------------------------------------------

def render_top_stocks(results: Optional[pd.DataFrame], sentiment_cache: dict = None, top_n: int = 5) -> None:
    """Render Top N ranked stocks as a single row of compact score cards."""
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

    pat_cols = [c for c in PATTERN_COLS if c in df_top.columns]
    cols     = st.columns(min(top_n, 5))

    _RANK_C = {0: "#f59e0b", 1: "#94a3b8", 2: "#cd7c2e"}  # gold / silver / bronze

    for i, (_, row) in enumerate(df_top.iterrows()):
        rank_bg    = _RANK_C.get(i, "#334155")
        rank_txt   = "#111" if i < 3 else "#e2e8f0"

        rs = row.get("RS Rating", None)
        rs_val = int(rs) if (rs is not None and not pd.isna(rs)) else None
        if rs_val and rs_val >= 80:
            rs_color, rs_bg = "#22c55e", "34,197,94"
        elif rs_val and rs_val >= 50:
            rs_color, rs_bg = "#f59e0b", "245,158,11"
        elif rs_val:
            rs_color, rs_bg = "#ef4444", "239,68,68"
        else:
            rs_color, rs_bg = "#6b7280", "107,114,128"

        score       = row["Score"]
        score_color = "#4ade80" if score >= 70 else ("#fbbf24" if score >= 50 else "#f87171")

        monthly   = row.get("Monthly %", None)
        m_str     = f"{monthly:+.1f}%" if (monthly is not None and not pd.isna(monthly)) else ""
        m_color   = "#22c55e" if (monthly or 0) >= 0 else "#ef4444"

        pct_h     = row.get("% from 52W High", None)
        pct_h_str = f"{pct_h:.1f}% from high" if (pct_h is not None and not pd.isna(pct_h)) else ""

        active_pats = [p for p in pat_cols if row.get(p) is True]
        pat_tags = "".join(
            f"<span style='background:rgba(99,102,241,0.15);color:#a5b4fc;"
            f"border-radius:4px;padding:1px 6px;font-size:0.62rem;"
            f"font-weight:700;letter-spacing:0.04em'>{p}</span>"
            for p in active_pats
        )
        pat_html = (
            f"<div style='display:flex;gap:3px;flex-wrap:wrap;margin-top:7px'>{pat_tags}</div>"
            if pat_tags else ""
        )

        sent       = (sentiment_cache or {}).get(row["Symbol"], {})
        sent_label = sent.get("label", "")
        if sent_label == "Bullish":
            sent_dot = "<span style='color:#22c55e;font-size:0.6rem;margin-left:4px'>● B</span>"
        elif sent_label == "Bearish":
            sent_dot = "<span style='color:#ef4444;font-size:0.6rem;margin-left:4px'>● Ba</span>"
        else:
            sent_dot = ""

        sector = str(row.get("Sector", ""))[:22]
        cmp    = row["CMP"]

        with cols[i % 5]:
            st.markdown(
                f"<div style='background:rgba(255,255,255,0.04);"
                f"border:1px solid rgba(255,255,255,0.09);border-radius:10px;"
                f"padding:14px;position:relative;overflow:hidden;height:100%'>"

                # Corner rank badge (Fitts's Law — small but anchored to corner)
                f"<div style='position:absolute;top:0;left:0;background:{rank_bg};"
                f"border-radius:10px 0 8px 0;padding:2px 9px;"
                f"font-size:0.62rem;font-weight:800;color:{rank_txt}'>#{i+1}</div>"

                # Symbol — primary identifier
                f"<div style='margin-top:14px;font-size:1.05rem;font-weight:800;"
                f"color:#f1f5f9;letter-spacing:-0.02em;line-height:1.15'>"
                f"{row['Symbol']}{sent_dot}</div>"

                # Sector — supporting context
                f"<div style='font-size:0.68rem;color:#64748b;margin-top:2px;"
                f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis'>{sector}</div>"

                # CMP left, 1M return right — immediate financial signal
                f"<div style='display:flex;justify-content:space-between;"
                f"align-items:baseline;margin-top:8px'>"
                f"<span style='font-size:0.95rem;font-weight:700;color:#e2e8f0'>₹{cmp:,.0f}</span>"
                f"<span style='font-size:0.78rem;font-weight:600;color:{m_color}'>{m_str}</span>"
                f"</div>"

                # Score + RS as mini-stat boxes — quality signals
                f"<div style='display:flex;gap:5px;margin-top:7px'>"
                f"<div style='flex:1;background:rgba(99,102,241,0.1);"
                f"border-radius:6px;padding:4px 6px;text-align:center'>"
                f"<div style='font-size:0.55rem;color:#6b7280;font-weight:700;"
                f"letter-spacing:0.06em;text-transform:uppercase'>Score</div>"
                f"<div style='font-size:0.92rem;font-weight:700;color:{score_color};"
                f"line-height:1.2'>{score:.0f}</div></div>"
                f"<div style='flex:1;background:rgba({rs_bg},0.1);"
                f"border-radius:6px;padding:4px 6px;text-align:center'>"
                f"<div style='font-size:0.55rem;color:#6b7280;font-weight:700;"
                f"letter-spacing:0.06em;text-transform:uppercase'>RS</div>"
                f"<div style='font-size:0.92rem;font-weight:700;color:{rs_color};"
                f"line-height:1.2'>{rs_val if rs_val else '—'}</div></div>"
                f"</div>"

                # Tertiary: distance from high
                + (f"<div style='font-size:0.68rem;color:#64748b;margin-top:5px'>"
                   f"{pct_h_str}</div>" if pct_h_str else "")

                # Pattern tags — action signals
                + pat_html
                + "</div>",
                unsafe_allow_html=True,
            )

    st.write("")
    st.divider()
