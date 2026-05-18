"""Home page — Market Insights dashboard."""

import pandas as pd
import streamlit as st

from src.pipeline import (
    load_results, load_index_results, load_commodity_results, last_updated,
    top_industries_vs_sector,
)

try:
    from src.sector_mapper import load_index_data as _load_index_data
    _, _sector_indices, _ = _load_index_data()
except Exception:
    _sector_indices = {}


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _pct_color(val: float) -> str:
    return "#22c55e" if val >= 0 else "#ef4444"


def _kpi_card(label: str, name: str, sub: str, sub_color: str = "#22c55e") -> str:
    return (
        f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.09);'
        f'border-top:2px solid #6366f1;border-radius:10px;padding:14px 16px;">'
        f'<div style="font-size:0.67rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.1em;color:#64748b;margin-bottom:6px;">{label}</div>'
        f'<div style="font-size:16px;font-weight:600;color:#f1f5f9;white-space:nowrap;'
        f'overflow:hidden;text-overflow:ellipsis;" title="{name}">{name}</div>'
        f'<div style="font-size:13px;font-weight:600;color:{sub_color};margin-top:4px;">{sub}</div>'
        f'</div>'
    )


def _bar_list(items: list[dict], ranked: bool = False) -> str:
    """Render ranked rows: name left, +% right, colored bar below.
    Each item: {n, v, u (optional unit string)}.
    """
    if not items:
        return '<p style="font-size:13px;color:#64748b;">No data.</p>'
    max_abs = max((abs(it["v"]) for it in items), default=1) or 1
    rows = []
    for i, it in enumerate(items):
        bar_w  = round(abs(it["v"]) / max_abs * 100)
        mb     = "0" if i == len(items) - 1 else "13px"
        color  = _pct_color(it["v"])
        prefix = f'{i + 1}.&nbsp;' if ranked else ""
        unit   = (f'&thinsp;<span style="font-weight:400;color:#64748b;">{it["u"]}</span>'
                  if it.get("u") else "")
        rows.append(
            f'<div style="margin-bottom:{mb};">'
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:baseline;margin-bottom:5px;">'
            f'<span style="font-size:13px;color:#e2e8f0;">{prefix}{it["n"]}</span>'
            f'<span style="font-size:13px;font-weight:500;color:{color};">'
            f'{it["v"]:+.1f}%{unit}</span>'
            f'</div>'
            f'<div style="height:6px;background:rgba(255,255,255,0.07);'
            f'border-radius:99px;overflow:hidden;">'
            f'<div style="height:100%;width:{bar_w}%;background:{color};'
            f'border-radius:99px;"></div></div></div>'
        )
    return "\n".join(rows)


def _stock_grid(stocks: list[dict]) -> str:
    """4-column card grid. Each item: {ticker, name, cmp, ret}."""
    cards = []
    for s in stocks:
        color = _pct_color(s["ret"])
        arrow = "↑" if s["ret"] >= 0 else "↓"
        cards.append(
            f'<div style="background:rgba(255,255,255,0.05);border-radius:10px;padding:12px 14px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;">'
            f'<span style="font-size:14px;font-weight:500;color:#f1f5f9;">{s["ticker"]}</span>'
            f'<span style="font-size:13px;font-weight:500;color:{color};">'
            f'{arrow}&nbsp;{s["ret"]:+.1f}%</span>'
            f'</div>'
            f'<div style="font-size:12px;color:#94a3b8;margin:3px 0 6px;'
            f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{s["name"]}</div>'
            f'<div style="font-size:13px;color:#f1f5f9;">{s["cmp"]}</div>'
            f'</div>'
        )
    n = len(cards)
    return (
        f'<div style="display:grid;grid-template-columns:repeat({n},1fr);gap:12px;">'
        + "".join(cards)
        + "</div>"
    )


def _section_title(text: str, badge: str = "", badge_color: str = "#94a3b8",
                   badge_bg: str = "rgba(255,255,255,0.07)") -> str:
    pill = (
        f'<span style="font-size:12px;font-weight:400;color:{badge_color};'
        f'background:{badge_bg};padding:3px 10px;border-radius:6px;">{badge}</span>'
        if badge else ""
    )
    return (
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'margin-bottom:14px;">'
        f'<span style="font-size:15px;font-weight:500;color:#f1f5f9;">{text}</span>'
        f'{pill}</div>'
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    '<div style="font-size:20px;font-weight:600;color:#f1f5f9;margin-bottom:3px;">'
    'Market highlights</div>'
    '<div style="font-size:13px;color:#94a3b8;">'
    'Top performers across sectors, stocks &amp; commodities</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
results           = load_results()
index_results     = load_index_results()
commodity_results = load_commodity_results()


# ---------------------------------------------------------------------------
# Timeframe tabs
# ---------------------------------------------------------------------------
tab_weekly, tab_monthly = st.tabs(["  Weekly  ", "  Monthly  "])

for _tab, timeframe in [(tab_weekly, "Weekly"), (tab_monthly, "Monthly")]:
    with _tab:
        ret_col = "Weekly %" if timeframe == "Weekly" else "Monthly %"

        # -------------------------------------------------------------------
        # Derived aggregations
        # -------------------------------------------------------------------
        sector_perf  = pd.Series(dtype=float)
        ind_perf     = pd.Series(dtype=float)
        top_sector   = "—"
        top_sector_v: float | None = None
        advancers    = 0
        decliners    = 0

        if results is not None and not results.empty and ret_col in results.columns:
            sector_perf = (
                results[results["Sector"].notna() & (results["Sector"] != "Unknown")]
                .groupby("Sector")[ret_col].mean()
                .sort_values(ascending=False)
            )
            if not sector_perf.empty:
                top_sector   = str(sector_perf.index[0])
                top_sector_v = float(sector_perf.iloc[0])
            advancers = int((results[ret_col] > 0).sum())
            decliners = int((results[ret_col] <= 0).sum())

        top_idx_name, top_idx_val = "—", None
        if index_results is not None and not index_results.empty and ret_col in index_results.columns:
            row = index_results.dropna(subset=[ret_col]).sort_values(ret_col, ascending=False).iloc[0]
            top_idx_name = str(row.get("Index", "—"))
            top_idx_val  = float(row[ret_col])

        top_comm_name, top_comm_val = "—", None
        if commodity_results is not None and not commodity_results.empty and ret_col in commodity_results.columns:
            row = commodity_results.dropna(subset=[ret_col]).sort_values(ret_col, ascending=False).iloc[0]
            top_comm_name = str(row.get("Commodity", "—"))
            top_comm_val  = float(row[ret_col])

        # -------------------------------------------------------------------
        # ① KPI strip — 4 cards
        # -------------------------------------------------------------------
        k1, k2, k3, k4 = st.columns(4)

        with k1:
            st.markdown(
                _kpi_card("TOP SECTOR", top_sector,
                          f"{top_sector_v:+.1f}%" if top_sector_v is not None else "—",
                          _pct_color(top_sector_v or 0)),
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                _kpi_card("TOP INDEX", top_idx_name,
                          f"{top_idx_val:+.1f}%" if top_idx_val is not None else "—",
                          _pct_color(top_idx_val or 0)),
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                _kpi_card("TOP COMMODITY", top_comm_name,
                          f"{top_comm_val:+.1f}%" if top_comm_val is not None else "—",
                          _pct_color(top_comm_val or 0)),
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown(
                _kpi_card("MARKET BREADTH", f"{advancers} / {decliners}",
                          "advancers vs decliners", "#94a3b8"),
                unsafe_allow_html=True,
            )

        st.markdown("<div style='margin-bottom:6px;'></div>", unsafe_allow_html=True)

        # -------------------------------------------------------------------
        # ② Top sectors + industries
        # -------------------------------------------------------------------
        sc1, sc2 = st.columns(2, gap="large")

        with sc1:
            with st.container(border=True):
                st.markdown(
                    _section_title("Top performing sectors", badge=timeframe),
                    unsafe_allow_html=True,
                )
                items = [{"n": k, "v": float(v)} for k, v in sector_perf.head(5).items()]
                st.markdown(_bar_list(items, ranked=True), unsafe_allow_html=True)

        with sc2:
            with st.container(border=True):
                st.markdown(
                    _section_title("Industries outperforming their Sector", badge=timeframe),
                    unsafe_allow_html=True,
                )
                if results is not None and not results.empty:
                    top_inds = top_industries_vs_sector(results, _sector_indices, top_n=5)
                    if top_inds:
                        items = [
                            {
                                "n": f"{r['industry']}  ·  {r['sector']}",
                                "v": float(r["outperf"]),
                                "u": f"{r['n_stocks']} stocks",
                            }
                            for r in top_inds
                        ]
                        st.markdown(_bar_list(items, ranked=True), unsafe_allow_html=True)
                    else:
                        st.caption("No data — run the screener first.")
                else:
                    st.caption("No data.")

        # -------------------------------------------------------------------
        # ③ Top stocks from leading sector
        # -------------------------------------------------------------------
        if results is not None and not results.empty and top_sector != "—":
            sector_stocks = (
                results[results["Sector"] == top_sector]
                .sort_values("Score", ascending=False)
                .head(4)
            )
            with st.container(border=True):
                st.markdown(
                    _section_title(
                        "Top stocks from leading sector",
                        badge=top_sector,
                        badge_color="#60a5fa",
                        badge_bg="rgba(96,165,250,0.12)",
                    ),
                    unsafe_allow_html=True,
                )
                if not sector_stocks.empty:
                    stocks = []
                    for _, row in sector_stocks.iterrows():
                        rv  = row.get(ret_col)
                        ret = float(rv) if rv is not None and pd.notna(rv) else 0.0
                        cmp = row.get("CMP")
                        stocks.append({
                            "ticker": str(row["Symbol"]),
                            "name":   (str(row.get("Company", "") or ""))[:30],
                            "cmp":    f"₹{cmp:,.1f}" if cmp is not None and pd.notna(cmp) else "—",
                            "ret":    ret,
                        })
                    st.markdown(_stock_grid(stocks), unsafe_allow_html=True)

        # -------------------------------------------------------------------
        # ④ Themes & Indices + Commodities
        # -------------------------------------------------------------------
        ic1, ic2 = st.columns(2, gap="large")

        with ic1:
            with st.container(border=True):
                st.markdown(_section_title("Top themes &amp; indices"), unsafe_allow_html=True)
                if index_results is not None and not index_results.empty and ret_col in index_results.columns:
                    top_idx = (
                        index_results.dropna(subset=[ret_col])
                        .sort_values(ret_col, ascending=False)
                        .head(5)
                    )
                    items = [{"n": r["Index"], "v": float(r[ret_col])} for _, r in top_idx.iterrows()]
                    st.markdown(_bar_list(items), unsafe_allow_html=True)
                else:
                    st.caption("No index data.")

        with ic2:
            with st.container(border=True):
                st.markdown(_section_title("Top performing commodities"), unsafe_allow_html=True)
                if commodity_results is not None and not commodity_results.empty and ret_col in commodity_results.columns:
                    top_comm_df = (
                        commodity_results.dropna(subset=[ret_col])
                        .sort_values(ret_col, ascending=False)
                        .head(5)
                    )
                    items = [{"n": r["Commodity"], "v": float(r[ret_col])} for _, r in top_comm_df.iterrows()]
                    st.markdown(_bar_list(items), unsafe_allow_html=True)
                else:
                    st.caption("No commodity data.")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    f'<div style="margin-top:1rem;font-size:12px;color:#64748b;">'
    f'Last updated: {last_updated()}&thinsp;·&thinsp;'
    f'NSE Nifty 1000&thinsp;·&thinsp;Powered by yfinance</div>',
    unsafe_allow_html=True,
)
