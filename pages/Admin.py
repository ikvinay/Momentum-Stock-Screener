"""
Admin page — screener configuration.
Not shown in the sidebar nav; access directly at /Admin.
"""

import re
import streamlit as st
from pathlib import Path

from ui.components import inject_css
from ui.sidebar import render_data_management
import config as _cfg

inject_css()

with st.sidebar:
    render_data_management()

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    '<div style="font-size:22px;font-weight:700;color:#f1f5f9;margin-bottom:4px;">'
    '<i class="ti ti-settings-2" style="color:#6366f1;margin-right:8px;"></i>'
    'Configuration</div>'
    '<div style="font-size:13px;color:#64748b;margin-bottom:4px;">'
    'Edit screener parameters below. '
    '<b style="color:#94a3b8">Save to config.py</b> writes changes to disk — '
    'restart the app to apply.</div>',
    unsafe_allow_html=True,
)
st.divider()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sec(label: str) -> None:
    st.markdown(
        f'<div style="font-size:10.5px;font-weight:700;color:#475569;'
        f'letter-spacing:.1em;text-transform:uppercase;margin:4px 0 10px;">'
        f'{label}</div>',
        unsafe_allow_html=True,
    )


def _weight_badge(keys: list, prefix: str) -> None:
    total = sum(st.session_state.get(f"{prefix}{k}", 0) for k in keys)
    ok = total == 100
    color = "#22c55e" if ok else "#ef4444"
    st.markdown(
        f'<div style="font-size:11px;font-weight:600;color:{color};margin-top:2px;">'
        f'Sum = {total} {"✓" if ok else " — must equal 100"}</div>',
        unsafe_allow_html=True,
    )


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_filt, tab_rank, tab_pat, tab_sys, tab_docs = st.tabs([
    "  Screener Filters  ",
    "  Ranking Weights  ",
    "  Pattern Detection  ",
    "  System & Scheduler  ",
    "  Pattern Reference  ",
])

# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Screener Filters
# ─────────────────────────────────────────────────────────────────────────────
with tab_filt:
    with st.container(border=True):
        _sec("Hard Filters — Stocks")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.number_input(
                "Min Market Cap (₹ Cr)",
                min_value=100, max_value=500_000, step=100,
                value=_cfg.MIN_MARKET_CAP_CR,
                key="MIN_MARKET_CAP_CR",
                help="Stocks below this market cap are excluded from screening.",
            )
        with c2:
            st.number_input(
                "Min Price vs 52W High (%)",
                min_value=50, max_value=100,
                value=int(_cfg.PRICE_MIN_PCT_OF_52W_HIGH * 100),
                key="PRICE_PCT_52W",
                help="Price must be ≥ this % of its 52-week high (e.g. 76 = within 24% of high).",
            )
        with c3:
            st.number_input(
                "Min History (trading days)",
                min_value=50, max_value=500, step=10,
                value=_cfg.MIN_HISTORY_DAYS,
                key="MIN_HISTORY_DAYS",
            )
        c4, c5, _ = st.columns(3)
        with c4:
            st.number_input(
                "RSI Hard Cut-off",
                min_value=20, max_value=70,
                value=_cfg.RSI_MIN_SCORE,
                key="RSI_MIN_SCORE",
                help="Stocks with RSI below this value are excluded before scoring.",
            )
        with c5:
            st.number_input(
                "EMA10 Max Extension (%)",
                min_value=1, max_value=25,
                value=int(_cfg.EMA10_MAX_EXTENSION_PCT * 100),
                key="EMA10_MAX_EXT_PCT",
                help="Stocks more than this % above EMA10 are excluded (overextended).",
            )

    with st.container(border=True):
        _sec("RSI Settings")
        r1, r2, _ = st.columns(3)
        with r1:
            st.number_input(
                "RSI Period",
                min_value=5, max_value=30,
                value=_cfg.RSI_PERIOD,
                key="RSI_PERIOD",
            )

# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Ranking Weights
# ─────────────────────────────────────────────────────────────────────────────
with tab_rank:
    # ── Stock scoring weights ──────────────────────────────────────────────────
    with st.container(border=True):
        _sec("Stock Scoring Weights  (must sum to 100)")
        SW = _cfg.SCORE_WEIGHTS
        SW_LABELS = {
            "52w_high_proximity": "52W High Proximity",
            "benchmark_outperf":  "Benchmark Outperf (1M)",
            "sector_outperf":     "Sector Outperf (1M)",
            "tight_range_vol":    "Tight Range / Volume",
            "close_to_kma":       "KMA Proximity",
            "rsi_score":          "RSI Score",
            "rmv_score":          "RMV Score",
            "rs_trend":           "RS Trend",
            "ema10_touch":        "EMA10 Touch",
        }
        sw_keys = list(SW.keys())
        sw_cols = st.columns(3)
        for i, k in enumerate(sw_keys):
            with sw_cols[i % 3]:
                st.number_input(
                    SW_LABELS.get(k, k),
                    min_value=0, max_value=100, step=5,
                    value=SW[k], key=f"sw_{k}",
                )
        _weight_badge(sw_keys, "sw_")

    # ── Combined rank weights ──────────────────────────────────────────────────
    with st.container(border=True):
        _sec("Combined Rank Weights  (must sum to 100)")
        cr1, cr2, _ = st.columns(3)
        with cr1:
            st.number_input(
                "Momentum Score",
                min_value=0, max_value=100, step=5,
                value=_cfg.COMBINED_RANK_WEIGHTS["score"],
                key="crw_score",
            )
        with cr2:
            st.number_input(
                "RS Rating",
                min_value=0, max_value=100, step=5,
                value=_cfg.COMBINED_RANK_WEIGHTS["rs_rating"],
                key="crw_rs_rating",
            )
        _weight_badge(["score", "rs_rating"], "crw_")

    # ── Index scoring weights ──────────────────────────────────────────────────
    with st.container(border=True):
        _sec("Index Scoring Weights  (must sum to 100)")
        IW = _cfg.INDEX_SCORE_WEIGHTS
        IW_LABELS = {
            "52w_high_proximity": "52W High Proximity",
            "nifty500_outperf":   "Nifty500 Outperf (1M)",
            "close_to_kma":       "KMA Proximity",
            "rsi_score":          "RSI Score",
            "rmv_score":          "RMV Score",
            "rs_trend":           "RS Trend",
        }
        iw_keys = list(IW.keys())
        iw_cols = st.columns(3)
        for i, k in enumerate(iw_keys):
            with iw_cols[i % 3]:
                st.number_input(
                    IW_LABELS.get(k, k),
                    min_value=0, max_value=100, step=5,
                    value=IW[k], key=f"iw_{k}",
                )
        _weight_badge(iw_keys, "iw_")

    # ── Commodity scoring weights ──────────────────────────────────────────────
    with st.container(border=True):
        _sec("Commodity Scoring Weights  (must sum to 100)")
        CW = _cfg.COMMODITY_SCORE_WEIGHTS
        CW_LABELS = {
            "52w_high_proximity":  "52W High Proximity",
            "nifty500_outperf":    "Nifty500 Outperf (1M)",
            "metal_index_outperf": "Metal Index Outperf (1M)",
            "rsi_score":           "RSI Score",
            "close_to_kma":        "KMA Proximity",
            "rmv_score":           "RMV Score",
            "rs_trend":            "RS Trend",
        }
        cw_keys = list(CW.keys())
        cw_cols = st.columns(3)
        for i, k in enumerate(cw_keys):
            with cw_cols[i % 3]:
                st.number_input(
                    CW_LABELS.get(k, k),
                    min_value=0, max_value=100, step=5,
                    value=CW[k], key=f"cw_{k}",
                )
        _weight_badge(cw_keys, "cw_")

# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Pattern Detection
# ─────────────────────────────────────────────────────────────────────────────
with tab_pat:
    pc1, pc2 = st.columns(2)

    with pc1:
        with st.container(border=True):
            _sec("VCP Pattern")
            st.number_input(
                "Run-up Lookback (days)", min_value=10, max_value=120, step=5,
                value=_cfg.VCP_RUNUP_LOOKBACK, key="VCP_RUNUP_LOOKBACK",
            )
            st.number_input(
                "Min Prior Run-up (%)", min_value=5, max_value=60, step=5,
                value=int(_cfg.VCP_RUNUP_MIN_PCT * 100), key="VCP_RUNUP_MIN_PCT",
                help="Minimum directional gain (low → subsequent high) in the run-up window.",
            )
            st.number_input(
                "Consolidation Weeks", min_value=1, max_value=12,
                value=_cfg.VCP_CONSOL_WEEKS, key="VCP_CONSOL_WEEKS",
                help="Number of 5-day weeks that must contract. 4+ recommended for Indian markets.",
            )
            st.number_input(
                "Contraction Tolerance (%)", min_value=5, max_value=40, step=5,
                value=int(_cfg.VCP_CONTRACTION_TOLERANCE * 100), key="VCP_CONTRACTION_TOLERANCE",
                help="Each week must be at least this % tighter than the prior week.",
            )
            st.number_input(
                "Max Base Depth (%)", min_value=3, max_value=30, step=1,
                value=int(_cfg.VCP_MAX_BASE_DEPTH_PCT * 100), key="VCP_MAX_BASE_DEPTH_PCT",
                help="Final (tightest) week's price range must not exceed this % of close.",
            )
            st.number_input(
                "Vol Dry-up Ratio", min_value=0.30, max_value=0.90, step=0.05, format="%.2f",
                value=float(_cfg.VCP_VOL_DRY_UP_RATIO), key="VCP_VOL_DRY_UP_RATIO",
                help="Consolidation avg volume must be below this fraction of run-up avg volume.",
            )

    with pc2:
        with st.container(border=True):
            _sec("Volume Contraction")
            st.number_input(
                "Recent Vol Window (days)", min_value=1, max_value=20,
                value=_cfg.VOL_SHORT_PERIOD, key="VOL_SHORT_PERIOD",
            )
            st.number_input(
                "Historical Vol Window (days)", min_value=5, max_value=60,
                value=_cfg.VOL_LONG_PERIOD, key="VOL_LONG_PERIOD",
                help="Tier 1 baseline: 5d avg vs this window.",
            )
            st.number_input(
                "Extended Vol Window (days)", min_value=20, max_value=120, step=5,
                value=_cfg.VOL_EXTENDED_PERIOD, key="VOL_EXTENDED_PERIOD",
                help="Tier 2 baseline: 5d avg vs this longer window (stronger signal).",
            )
            st.number_input(
                "Contraction Ratio", min_value=0.40, max_value=0.99, step=0.05, format="%.2f",
                value=float(_cfg.VOL_CONTRACTION_RATIO), key="VOL_CONTRACTION_RATIO",
                help="Recent vol must be < this fraction of both baselines.",
            )
            st.number_input(
                "Price Stability Max Drop (%)", min_value=1, max_value=10,
                value=int(_cfg.VOL_PRICE_STABILITY_PCT * 100), key="VOL_PRICE_STABILITY_PCT",
                help="If price fell more than this % during the recent window, flag as distribution.",
            )
            st.number_input(
                "Distribution Day Threshold (%)", min_value=1, max_value=5,
                value=int(_cfg.VOL_DISTRIB_DAY_PCT * 100), key="VOL_DISTRIB_DAY_PCT",
                help="Any bar closing >this % below its open on above-avg volume is a distribution day.",
            )

    pc3, pc4 = st.columns(2)

    with pc3:
        with st.container(border=True):
            _sec("EMA & KMA Band")
            st.number_input(
                "EMA10 Upper Band (%)", min_value=0.0, max_value=10.0, step=0.5, format="%.1f",
                value=float(_cfg.EMA10_UPPER_PCT * 100), key="EMA10_UPPER_PCT",
                help="Max % above EMA10 to qualify as 'Near 10 EMA'.",
            )
            st.number_input(
                "EMA10 Lower Band (%)", min_value=-10.0, max_value=0.0, step=0.5, format="%.1f",
                value=float(_cfg.EMA10_LOWER_PCT * 100), key="EMA10_LOWER_PCT",
                help="Max % below EMA10 to qualify.",
            )
            st.number_input(
                "KMA Band (%)", min_value=0.0, max_value=5.0, step=0.5, format="%.1f",
                value=float(_cfg.KMA_BAND_PCT * 100), key="KMA_BAND_PCT",
                help="±% around EMA10 used for KMA proximity scoring.",
            )

    with pc4:
        with st.container(border=True):
            _sec("RMV — Relative Measured Volatility")
            st.number_input(
                "Lookback Bars", min_value=5, max_value=60,
                value=_cfg.RMV_LOOKBACK, key="RMV_LOOKBACK",
            )
            st.number_input(
                "Tight Bar Threshold", min_value=1, max_value=50,
                value=_cfg.RMV_TIGHT_THRESHOLD, key="RMV_TIGHT_THRESHOLD",
                help="RMV < this is flagged as a compressed / tight bar.",
            )

    with st.container(border=True):
        _sec("IPO Base Pattern")
        ip1, ip2, ip3, ip4 = st.columns(4)
        with ip1:
            st.number_input("Max Days Listed", min_value=20, max_value=500,
                            value=_cfg.IPO_BASE_MAX_DAYS, key="IPO_BASE_MAX_DAYS")
        with ip2:
            st.number_input("Min Days Listed", min_value=1, max_value=50,
                            value=_cfg.IPO_BASE_MIN_DAYS, key="IPO_BASE_MIN_DAYS")
        with ip3:
            st.number_input("Skip First N Days", min_value=0, max_value=30,
                            value=_cfg.IPO_BASE_SKIP_DAYS, key="IPO_BASE_SKIP_DAYS")
        with ip4:
            st.number_input("Max Base Range (%)", min_value=10, max_value=100,
                            value=int(_cfg.IPO_BASE_RANGE_MAX * 100),
                            key="IPO_BASE_RANGE_MAX")
        ip5, ip6, ip7, _ = st.columns(4)
        with ip5:
            st.number_input("Breakout Lower (%)", min_value=-20, max_value=0,
                            value=int(_cfg.IPO_BREAKOUT_LOWER * 100),
                            key="IPO_BREAKOUT_LOWER")
        with ip6:
            st.number_input("Breakout Upper (%)", min_value=0, max_value=50,
                            value=int(_cfg.IPO_BREAKOUT_UPPER * 100),
                            key="IPO_BREAKOUT_UPPER")
        with ip7:
            st.number_input("Vol Pickup Ratio", min_value=0.5, max_value=3.0,
                            step=0.1, format="%.1f",
                            value=float(_cfg.IPO_VOL_PICKUP_RATIO),
                            key="IPO_VOL_PICKUP_RATIO")

    st_c1, st_c2 = st.columns(2)
    with st_c1:
        with st.container(border=True):
            _sec("Symmetrical Triangle")
            st.number_input(
                "Lookback (bars)", min_value=20, max_value=200, step=5,
                value=_cfg.SYMTRI_LOOKBACK, key="SYMTRI_LOOKBACK",
                help="Number of recent bars scanned. Real institutional triangles often need 80–120 bars.",
            )
            st.number_input(
                "Min Pivot Touches", min_value=2, max_value=6,
                value=_cfg.SYMTRI_MIN_TOUCHES, key="SYMTRI_MIN_TOUCHES",
                help="Minimum swing pivot touchpoints required per trendline. Higher = stricter.",
            )
            st.number_input(
                "Trendline R² Threshold", min_value=0.50, max_value=0.99, step=0.05, format="%.2f",
                value=float(_cfg.SYMTRI_R2_THRESHOLD), key="SYMTRI_R2_THRESHOLD",
                help="Minimum R² for the pivot line fit. Rejects noisy pivot sets (< 0.85 = poor fit).",
            )
    with st_c2:
        with st.container(border=True):
            _sec("Symmetrical Triangle (cont.)")
            st.number_input(
                "Swing Window (bars)", min_value=2, max_value=8,
                value=_cfg.SYMTRI_SWING_WINDOW, key="SYMTRI_SWING_WINDOW",
                help="Bars on each side a pivot must dominate. Note: last N bars cannot be pivots (lag).",
            )
            st.number_input(
                "Min Bars to Apex", min_value=2, max_value=30,
                value=_cfg.SYMTRI_MIN_BARS_TO_APEX, key="SYMTRI_MIN_BARS_TO_APEX",
                help="Triangle must not be at the apex yet — ensures pattern is still active.",
            )
            st.checkbox(
                "Require Volume Contraction", value=bool(_cfg.SYMTRI_REQUIRE_VOL_CONTRACTION),
                key="SYMTRI_REQUIRE_VOL_CONTRACTION",
                help="Second half of lookback window must have lower average volume than first half.",
            )

    with st.container(border=True):
        _sec("Symmetrical Triangle — Slope Symmetry")
        ss1, ss2, _ = st.columns(3)
        with ss1:
            st.number_input(
                "Slope Ratio Min", min_value=0.10, max_value=1.00, step=0.05, format="%.2f",
                value=float(_cfg.SYMTRI_SLOPE_SYMMETRY_MIN), key="SYMTRI_SLOPE_SYMMETRY_MIN",
                help="|upper slope| / |lower slope| must be ≥ this. Prevents wedge mislabelling.",
            )
        with ss2:
            st.number_input(
                "Slope Ratio Max", min_value=1.00, max_value=5.00, step=0.10, format="%.2f",
                value=float(_cfg.SYMTRI_SLOPE_SYMMETRY_MAX), key="SYMTRI_SLOPE_SYMMETRY_MAX",
                help="|upper slope| / |lower slope| must be ≤ this. Prevents wedge mislabelling.",
            )

    at_c1, at_c2 = st.columns(2)
    with at_c1:
        with st.container(border=True):
            _sec("Ascending Triangle")
            st.number_input(
                "Lookback (bars)", min_value=30, max_value=200, step=5,
                value=_cfg.ASCTRI_LOOKBACK, key="ASCTRI_LOOKBACK",
                help="Bars scanned for the formation. Ideal patterns span 40–90 bars.",
            )
            st.number_input(
                "Min Resistance Touches", min_value=2, max_value=6,
                value=_cfg.ASCTRI_MIN_TOUCHES_RESIST, key="ASCTRI_MIN_TOUCHES_RESIST",
                help="Minimum swing highs at the flat resistance level.",
            )
            st.number_input(
                "Min Support Touches", min_value=2, max_value=6,
                value=_cfg.ASCTRI_MIN_TOUCHES_SUPPORT, key="ASCTRI_MIN_TOUCHES_SUPPORT",
                help="Minimum rising swing lows forming the ascending support.",
            )
            st.number_input(
                "Swing Window (bars)", min_value=2, max_value=8,
                value=_cfg.ASCTRI_SWING_WINDOW, key="ASCTRI_SWING_WINDOW",
            )
            st.number_input(
                "Min Span (bars)", min_value=10, max_value=60,
                value=_cfg.ASCTRI_MIN_SPAN_BARS, key="ASCTRI_MIN_SPAN_BARS",
                help="Pattern must span at least this many bars. < 25 = noise.",
            )
    with at_c2:
        with st.container(border=True):
            _sec("Ascending Triangle (cont.)")
            st.number_input(
                "RSI Min", min_value=20, max_value=60,
                value=_cfg.ASCTRI_RSI_MIN, key="ASCTRI_RSI_MIN",
                help="RSI must be ≥ this during formation (not deeply oversold).",
            )
            st.number_input(
                "RSI Max", min_value=50, max_value=90,
                value=_cfg.ASCTRI_RSI_MAX, key="ASCTRI_RSI_MAX",
                help="RSI must be ≤ this during formation (not overbought).",
            )
            st.number_input(
                "Vol End Ratio", min_value=0.30, max_value=0.90, step=0.05, format="%.2f",
                value=float(_cfg.ASCTRI_VOL_END_RATIO), key="ASCTRI_VOL_END_RATIO",
                help="Volume in the final bars must be < this fraction of the first bars.",
            )
            st.number_input(
                "Support Min Slope (× 10⁻⁴)", min_value=1, max_value=20,
                value=int(_cfg.ASCTRI_SUPPORT_MIN_SLOPE_PCT * 10_000), key="ASCTRI_SUPPORT_MIN_SLOPE_PCT",
                help="Minimum rising support slope as a fraction of price per bar (× 10⁻⁴).",
            )
            st.number_input(
                "Support R² Threshold", min_value=0.40, max_value=0.95, step=0.05, format="%.2f",
                value=float(_cfg.ASCTRI_SUPPORT_R2), key="ASCTRI_SUPPORT_R2",
                help="Minimum R² for the rising support line fit.",
            )
            st.checkbox(
                "Require Above 200 EMA", value=bool(_cfg.ASCTRI_REQUIRE_ABOVE_200DMA),
                key="ASCTRI_REQUIRE_ABOVE_200DMA",
                help="Pattern must form in an established uptrend (price above 200-day EMA).",
            )

    with st.container(border=True):
        _sec("Performance Lookbacks")
        pl1, pl2, pl3, _ = st.columns(4)
        with pl1:
            st.number_input("Weekly (trading days)", min_value=1, max_value=10,
                            value=_cfg.WEEKLY_DAYS, key="WEEKLY_DAYS")
        with pl2:
            st.number_input("Monthly (trading days)", min_value=10, max_value=35,
                            value=_cfg.MONTHLY_DAYS, key="MONTHLY_DAYS")
        with pl3:
            st.number_input("Quarterly (trading days)", min_value=40, max_value=90, step=5,
                            value=_cfg.QUARTERLY_DAYS, key="QUARTERLY_DAYS")

# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — System & Scheduler
# ─────────────────────────────────────────────────────────────────────────────
with tab_sys:
    sy1, sy2 = st.columns(2)

    with sy1:
        with st.container(border=True):
            _sec("Stock Data Refresh Schedule (IST)")
            st.number_input("Hour (0–23)", min_value=0, max_value=23,
                            value=_cfg.REFRESH_HOUR_IST, key="REFRESH_HOUR_IST")
            st.number_input("Minute (0–59)", min_value=0, max_value=59,
                            value=_cfg.REFRESH_MINUTE_IST, key="REFRESH_MINUTE_IST")

    with sy2:
        with st.container(border=True):
            _sec("Commodity Refresh Schedule (IST)")
            st.number_input("Hour (0–23)", min_value=0, max_value=23,
                            value=_cfg.COMMODITY_REFRESH_HOUR_IST,
                            key="COMMODITY_REFRESH_HOUR_IST")
            st.number_input("Minute (0–59)", min_value=0, max_value=59,
                            value=_cfg.COMMODITY_REFRESH_MINUTE_IST,
                            key="COMMODITY_REFRESH_MINUTE_IST")

    ff1, ff2 = st.columns(2)

    with ff1:
        with st.container(border=True):
            _sec("Free Float Weekly Refresh")
            _DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
            st.selectbox("Day of Week", _DAYS,
                         index=_DAYS.index(_cfg.FREEFLOAT_REFRESH_DAY),
                         key="FREEFLOAT_REFRESH_DAY")
            st.number_input("Hour (IST)", min_value=0, max_value=23,
                            value=_cfg.FREEFLOAT_REFRESH_HOUR_IST,
                            key="FREEFLOAT_REFRESH_HOUR_IST")
            st.number_input("Minute", min_value=0, max_value=59,
                            value=_cfg.FREEFLOAT_REFRESH_MINUTE_IST,
                            key="FREEFLOAT_REFRESH_MINUTE_IST")
            st.number_input("Parallel Workers", min_value=1, max_value=10,
                            value=_cfg.FREEFLOAT_MAX_WORKERS,
                            key="FREEFLOAT_MAX_WORKERS")

    with ff2:
        with st.container(border=True):
            _sec("Data Fetch Settings")
            st.number_input("Batch Size (stocks / request)", min_value=10, max_value=200, step=10,
                            value=_cfg.BATCH_SIZE, key="BATCH_SIZE")
            st.number_input("Batch Delay (seconds)", min_value=0, max_value=30,
                            value=_cfg.BATCH_DELAY_SECONDS, key="BATCH_DELAY_SECONDS")
            st.number_input("Info Fetch Workers", min_value=1, max_value=30,
                            value=_cfg.INFO_MAX_WORKERS, key="INFO_MAX_WORKERS")
            st.number_input("Stock Info Cache (days)", min_value=1, max_value=30,
                            value=_cfg.STOCK_INFO_CACHE_DAYS,
                            key="STOCK_INFO_CACHE_DAYS")
            st.number_input("Equity List Cache (days)", min_value=1, max_value=30,
                            value=_cfg.EQUITY_LIST_CACHE_DAYS,
                            key="EQUITY_LIST_CACHE_DAYS")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 5 — Pattern Reference
# ─────────────────────────────────────────────────────────────────────────────
with tab_docs:
    _doc_path = Path("docs/pattern_detection.md")
    if _doc_path.exists():
        st.markdown(_doc_path.read_text(encoding="utf-8"))
    else:
        st.warning("Documentation file not found at `docs/pattern_detection.md`.")


# ─────────────────────────────────────────────────────────────────────────────
# Save logic
# ─────────────────────────────────────────────────────────────────────────────

def _build_config_patch() -> str:
    """
    Read the current config.py, apply all widget edits via regex substitution,
    and return the patched source as a string.
    """
    ss = st.session_state
    src = Path("config.py").read_text(encoding="utf-8")

    def _sub_scalar(name: str, new_val) -> None:
        nonlocal src
        src = re.sub(
            rf'^({re.escape(name)}\s*=\s*)[^\n]+',
            lambda m: m.group(1) + repr(new_val),
            src, flags=re.MULTILINE,
        )

    def _sub_dict(name: str, new_dict: dict) -> None:
        nonlocal src
        # Build new dict literal aligned like the original style
        lines = [f"{name} = {{"]
        items = list(new_dict.items())
        maxk  = max(len(f'    "{k}"') for k, _ in items)
        for k, v in items:
            key_part = f'    "{k}"'
            lines.append(f"{key_part:{maxk}}: {v},")
        lines.append("}")
        new_block = "\n".join(lines)
        # Replace from `NAME = {` to the closing `}`
        src = re.sub(
            rf'^{re.escape(name)}\s*=\s*\{{[^}}]*\}}',
            new_block,
            src, flags=re.MULTILINE | re.DOTALL,
        )

    # Screener filters
    _sub_scalar("MIN_MARKET_CAP_CR",          ss.get("MIN_MARKET_CAP_CR",  _cfg.MIN_MARKET_CAP_CR))
    _sub_scalar("PRICE_MIN_PCT_OF_52W_HIGH",  ss.get("PRICE_PCT_52W", int(_cfg.PRICE_MIN_PCT_OF_52W_HIGH * 100)) / 100)
    _sub_scalar("MIN_HISTORY_DAYS",           ss.get("MIN_HISTORY_DAYS",   _cfg.MIN_HISTORY_DAYS))
    _sub_scalar("RSI_MIN_SCORE",              ss.get("RSI_MIN_SCORE",      _cfg.RSI_MIN_SCORE))
    _sub_scalar("EMA10_MAX_EXTENSION_PCT",    ss.get("EMA10_MAX_EXT_PCT",  int(_cfg.EMA10_MAX_EXTENSION_PCT * 100)) / 100)
    _sub_scalar("RSI_PERIOD",                 ss.get("RSI_PERIOD",         _cfg.RSI_PERIOD))

    # Performance lookbacks
    _sub_scalar("WEEKLY_DAYS",   ss.get("WEEKLY_DAYS",   _cfg.WEEKLY_DAYS))
    _sub_scalar("MONTHLY_DAYS",  ss.get("MONTHLY_DAYS",  _cfg.MONTHLY_DAYS))
    _sub_scalar("QUARTERLY_DAYS",ss.get("QUARTERLY_DAYS",_cfg.QUARTERLY_DAYS))

    # VCP
    _sub_scalar("VCP_RUNUP_LOOKBACK",       ss.get("VCP_RUNUP_LOOKBACK",       _cfg.VCP_RUNUP_LOOKBACK))
    _sub_scalar("VCP_RUNUP_MIN_PCT",        ss.get("VCP_RUNUP_MIN_PCT",        int(_cfg.VCP_RUNUP_MIN_PCT * 100)) / 100)
    _sub_scalar("VCP_CONSOL_WEEKS",         ss.get("VCP_CONSOL_WEEKS",         _cfg.VCP_CONSOL_WEEKS))
    _sub_scalar("VCP_CONTRACTION_TOLERANCE",ss.get("VCP_CONTRACTION_TOLERANCE",int(_cfg.VCP_CONTRACTION_TOLERANCE * 100)) / 100)
    _sub_scalar("VCP_MAX_BASE_DEPTH_PCT",   ss.get("VCP_MAX_BASE_DEPTH_PCT",   int(_cfg.VCP_MAX_BASE_DEPTH_PCT * 100)) / 100)
    _sub_scalar("VCP_VOL_DRY_UP_RATIO",    ss.get("VCP_VOL_DRY_UP_RATIO",     _cfg.VCP_VOL_DRY_UP_RATIO))

    # Volume contraction
    _sub_scalar("VOL_SHORT_PERIOD",         ss.get("VOL_SHORT_PERIOD",         _cfg.VOL_SHORT_PERIOD))
    _sub_scalar("VOL_LONG_PERIOD",          ss.get("VOL_LONG_PERIOD",          _cfg.VOL_LONG_PERIOD))
    _sub_scalar("VOL_EXTENDED_PERIOD",      ss.get("VOL_EXTENDED_PERIOD",      _cfg.VOL_EXTENDED_PERIOD))
    _sub_scalar("VOL_CONTRACTION_RATIO",    ss.get("VOL_CONTRACTION_RATIO",    _cfg.VOL_CONTRACTION_RATIO))
    _sub_scalar("VOL_PRICE_STABILITY_PCT",  ss.get("VOL_PRICE_STABILITY_PCT",  int(_cfg.VOL_PRICE_STABILITY_PCT * 100)) / 100)
    _sub_scalar("VOL_DISTRIB_DAY_PCT",      ss.get("VOL_DISTRIB_DAY_PCT",      int(_cfg.VOL_DISTRIB_DAY_PCT * 100)) / 100)

    # EMA & KMA
    _sub_scalar("EMA10_UPPER_PCT", ss.get("EMA10_UPPER_PCT", _cfg.EMA10_UPPER_PCT * 100) / 100)
    _sub_scalar("EMA10_LOWER_PCT", ss.get("EMA10_LOWER_PCT", _cfg.EMA10_LOWER_PCT * 100) / 100)
    _sub_scalar("KMA_BAND_PCT",    ss.get("KMA_BAND_PCT",    _cfg.KMA_BAND_PCT    * 100) / 100)

    # RMV
    _sub_scalar("RMV_LOOKBACK",        ss.get("RMV_LOOKBACK",        _cfg.RMV_LOOKBACK))
    _sub_scalar("RMV_TIGHT_THRESHOLD", ss.get("RMV_TIGHT_THRESHOLD", _cfg.RMV_TIGHT_THRESHOLD))

    # Symmetrical Triangle
    _sub_scalar("SYMTRI_LOOKBACK",               ss.get("SYMTRI_LOOKBACK",               _cfg.SYMTRI_LOOKBACK))
    _sub_scalar("SYMTRI_MIN_TOUCHES",            ss.get("SYMTRI_MIN_TOUCHES",            _cfg.SYMTRI_MIN_TOUCHES))
    _sub_scalar("SYMTRI_SWING_WINDOW",           ss.get("SYMTRI_SWING_WINDOW",           _cfg.SYMTRI_SWING_WINDOW))
    _sub_scalar("SYMTRI_MIN_BARS_TO_APEX",       ss.get("SYMTRI_MIN_BARS_TO_APEX",       _cfg.SYMTRI_MIN_BARS_TO_APEX))
    _sub_scalar("SYMTRI_R2_THRESHOLD",           ss.get("SYMTRI_R2_THRESHOLD",           _cfg.SYMTRI_R2_THRESHOLD))
    _sub_scalar("SYMTRI_SLOPE_SYMMETRY_MIN",     ss.get("SYMTRI_SLOPE_SYMMETRY_MIN",     _cfg.SYMTRI_SLOPE_SYMMETRY_MIN))
    _sub_scalar("SYMTRI_SLOPE_SYMMETRY_MAX",     ss.get("SYMTRI_SLOPE_SYMMETRY_MAX",     _cfg.SYMTRI_SLOPE_SYMMETRY_MAX))
    _sub_scalar("SYMTRI_REQUIRE_VOL_CONTRACTION",ss.get("SYMTRI_REQUIRE_VOL_CONTRACTION",_cfg.SYMTRI_REQUIRE_VOL_CONTRACTION))

    # Ascending Triangle
    _sub_scalar("ASCTRI_LOOKBACK",               ss.get("ASCTRI_LOOKBACK",               _cfg.ASCTRI_LOOKBACK))
    _sub_scalar("ASCTRI_MIN_TOUCHES_RESIST",     ss.get("ASCTRI_MIN_TOUCHES_RESIST",     _cfg.ASCTRI_MIN_TOUCHES_RESIST))
    _sub_scalar("ASCTRI_MIN_TOUCHES_SUPPORT",    ss.get("ASCTRI_MIN_TOUCHES_SUPPORT",    _cfg.ASCTRI_MIN_TOUCHES_SUPPORT))
    _sub_scalar("ASCTRI_SWING_WINDOW",           ss.get("ASCTRI_SWING_WINDOW",           _cfg.ASCTRI_SWING_WINDOW))
    _sub_scalar("ASCTRI_MIN_SPAN_BARS",          ss.get("ASCTRI_MIN_SPAN_BARS",          _cfg.ASCTRI_MIN_SPAN_BARS))
    _sub_scalar("ASCTRI_RSI_MIN",                ss.get("ASCTRI_RSI_MIN",                _cfg.ASCTRI_RSI_MIN))
    _sub_scalar("ASCTRI_RSI_MAX",                ss.get("ASCTRI_RSI_MAX",                _cfg.ASCTRI_RSI_MAX))
    _sub_scalar("ASCTRI_VOL_END_RATIO",          ss.get("ASCTRI_VOL_END_RATIO",          _cfg.ASCTRI_VOL_END_RATIO))
    _sub_scalar("ASCTRI_SUPPORT_MIN_SLOPE_PCT",  ss.get("ASCTRI_SUPPORT_MIN_SLOPE_PCT",  int(_cfg.ASCTRI_SUPPORT_MIN_SLOPE_PCT * 10_000) / 10_000))
    _sub_scalar("ASCTRI_SUPPORT_R2",             ss.get("ASCTRI_SUPPORT_R2",             _cfg.ASCTRI_SUPPORT_R2))
    _sub_scalar("ASCTRI_REQUIRE_ABOVE_200DMA",   ss.get("ASCTRI_REQUIRE_ABOVE_200DMA",   _cfg.ASCTRI_REQUIRE_ABOVE_200DMA))

    # IPO
    _sub_scalar("IPO_BASE_MAX_DAYS",    ss.get("IPO_BASE_MAX_DAYS", _cfg.IPO_BASE_MAX_DAYS))
    _sub_scalar("IPO_BASE_MIN_DAYS",    ss.get("IPO_BASE_MIN_DAYS", _cfg.IPO_BASE_MIN_DAYS))
    _sub_scalar("IPO_BASE_SKIP_DAYS",   ss.get("IPO_BASE_SKIP_DAYS",_cfg.IPO_BASE_SKIP_DAYS))
    _sub_scalar("IPO_BASE_RANGE_MAX",   ss.get("IPO_BASE_RANGE_MAX", int(_cfg.IPO_BASE_RANGE_MAX * 100)) / 100)
    _sub_scalar("IPO_BREAKOUT_LOWER",   ss.get("IPO_BREAKOUT_LOWER", int(_cfg.IPO_BREAKOUT_LOWER * 100)) / 100)
    _sub_scalar("IPO_BREAKOUT_UPPER",   ss.get("IPO_BREAKOUT_UPPER", int(_cfg.IPO_BREAKOUT_UPPER * 100)) / 100)
    _sub_scalar("IPO_VOL_PICKUP_RATIO", ss.get("IPO_VOL_PICKUP_RATIO", _cfg.IPO_VOL_PICKUP_RATIO))

    # Scheduler
    _sub_scalar("REFRESH_HOUR_IST",               ss.get("REFRESH_HOUR_IST",               _cfg.REFRESH_HOUR_IST))
    _sub_scalar("REFRESH_MINUTE_IST",             ss.get("REFRESH_MINUTE_IST",             _cfg.REFRESH_MINUTE_IST))
    _sub_scalar("COMMODITY_REFRESH_HOUR_IST",     ss.get("COMMODITY_REFRESH_HOUR_IST",     _cfg.COMMODITY_REFRESH_HOUR_IST))
    _sub_scalar("COMMODITY_REFRESH_MINUTE_IST",   ss.get("COMMODITY_REFRESH_MINUTE_IST",   _cfg.COMMODITY_REFRESH_MINUTE_IST))

    # Free float
    _sub_scalar("FREEFLOAT_REFRESH_DAY",         ss.get("FREEFLOAT_REFRESH_DAY",         _cfg.FREEFLOAT_REFRESH_DAY))
    _sub_scalar("FREEFLOAT_REFRESH_HOUR_IST",    ss.get("FREEFLOAT_REFRESH_HOUR_IST",    _cfg.FREEFLOAT_REFRESH_HOUR_IST))
    _sub_scalar("FREEFLOAT_REFRESH_MINUTE_IST",  ss.get("FREEFLOAT_REFRESH_MINUTE_IST",  _cfg.FREEFLOAT_REFRESH_MINUTE_IST))
    _sub_scalar("FREEFLOAT_MAX_WORKERS",         ss.get("FREEFLOAT_MAX_WORKERS",         _cfg.FREEFLOAT_MAX_WORKERS))

    # Data fetch
    _sub_scalar("BATCH_SIZE",            ss.get("BATCH_SIZE",            _cfg.BATCH_SIZE))
    _sub_scalar("BATCH_DELAY_SECONDS",   ss.get("BATCH_DELAY_SECONDS",   _cfg.BATCH_DELAY_SECONDS))
    _sub_scalar("INFO_MAX_WORKERS",      ss.get("INFO_MAX_WORKERS",      _cfg.INFO_MAX_WORKERS))
    _sub_scalar("STOCK_INFO_CACHE_DAYS", ss.get("STOCK_INFO_CACHE_DAYS", _cfg.STOCK_INFO_CACHE_DAYS))
    _sub_scalar("EQUITY_LIST_CACHE_DAYS",ss.get("EQUITY_LIST_CACHE_DAYS",_cfg.EQUITY_LIST_CACHE_DAYS))

    # Scoring weight dicts
    _sub_dict("SCORE_WEIGHTS", {
        k: ss.get(f"sw_{k}", _cfg.SCORE_WEIGHTS[k])
        for k in _cfg.SCORE_WEIGHTS
    })
    _sub_dict("COMBINED_RANK_WEIGHTS", {
        "score":     ss.get("crw_score",      _cfg.COMBINED_RANK_WEIGHTS["score"]),
        "rs_rating": ss.get("crw_rs_rating",  _cfg.COMBINED_RANK_WEIGHTS["rs_rating"]),
    })
    _sub_dict("INDEX_SCORE_WEIGHTS", {
        k: ss.get(f"iw_{k}", _cfg.INDEX_SCORE_WEIGHTS[k])
        for k in _cfg.INDEX_SCORE_WEIGHTS
    })
    _sub_dict("COMMODITY_SCORE_WEIGHTS", {
        k: ss.get(f"cw_{k}", _cfg.COMMODITY_SCORE_WEIGHTS[k])
        for k in _cfg.COMMODITY_SCORE_WEIGHTS
    })

    return src


# ── Action buttons ────────────────────────────────────────────────────────────
st.divider()
btn_save, btn_dl, _ = st.columns([2, 2, 4])

with btn_save:
    if st.button("Save to config.py", use_container_width=True, type="primary"):
        try:
            new_src = _build_config_patch()
            Path("config.py").write_text(new_src, encoding="utf-8")
            st.success("config.py saved. Restart the app for changes to take effect.")
        except Exception as exc:
            st.error(f"Save failed: {exc}")

with btn_dl:
    try:
        dl_src = _build_config_patch()
        st.download_button(
            label="Download config.py",
            data=dl_src,
            file_name="config.py",
            mime="text/plain",
            use_container_width=True,
        )
    except Exception:
        pass
