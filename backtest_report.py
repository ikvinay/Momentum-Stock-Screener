"""
Standalone backtest report generator.

Usage:
    python backtest_report.py [--weeks N]

Generates data/backtest_report.html and opens it in the default browser.
Requires price_data.pkl and screener_results.pkl to exist (run Fetch Data +
Run Screener from the Streamlit app first).
"""

import argparse
import base64
import io
import logging
import os
import pickle
import sys
import webbrowser
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz

from config import DATA_DIR, PATTERN_COLS, COMBINED_RANK_WEIGHTS, IST_TIMEZONE
from src.backtester import run_backtest
from src.data_fetcher import load_price_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPORT_FILE = os.path.join(DATA_DIR, "backtest_report.html")

DARK_BG      = "#0f172a"
CARD_BG      = "#1e293b"
BORDER       = "#334155"
TEXT_PRIMARY = "#f1f5f9"
TEXT_MUTED   = "#94a3b8"
GREEN        = "#22c55e"
RED          = "#ef4444"
AMBER        = "#f59e0b"
BLUE         = "#3b82f6"
PURPLE       = "#a78bfa"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_screener_results() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "screener_results.pkl")
    if not os.path.exists(path):
        logger.error("screener_results.pkl not found — run the screener first.")
        sys.exit(1)
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        logger.error(f"Could not load screener_results.pkl: {exc}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _plt_defaults():
    plt.rcParams.update({
        "figure.facecolor":  DARK_BG,
        "axes.facecolor":    CARD_BG,
        "axes.edgecolor":    BORDER,
        "axes.labelcolor":   TEXT_MUTED,
        "axes.titlecolor":   TEXT_PRIMARY,
        "xtick.color":       TEXT_MUTED,
        "ytick.color":       TEXT_MUTED,
        "grid.color":        BORDER,
        "grid.linewidth":    0.5,
        "text.color":        TEXT_PRIMARY,
        "legend.facecolor":  CARD_BG,
        "legend.edgecolor":  BORDER,
        "legend.labelcolor": TEXT_PRIMARY,
    })


def _fig_to_tag(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f'<img src="data:image/png;base64,{b64}" class="chart-img">'


def _chart_return_distribution(closed: pd.DataFrame) -> str:
    ret = closed["Return %"].dropna()
    if ret.empty:
        return ""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(ret, bins=25, color=BLUE, edgecolor=DARK_BG, alpha=0.85, linewidth=0.4)
    ax.axvline(0,          color=AMBER, linewidth=1.5, linestyle="--", label="Break-even")
    ax.axvline(ret.mean(), color=GREEN, linewidth=1.5, linestyle="-",
               label=f"Mean {ret.mean():+.2f}%")
    ax.set_xlabel("Return %")
    ax.set_ylabel("Frequency")
    ax.set_title("Return Distribution — Closed Trades")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    return _fig_to_tag(fig)


def _chart_score_bands(closed: pd.DataFrame) -> str:
    df = closed.dropna(subset=["Return %", "Combined Score"]).copy()
    if len(df) < 5:
        return ""
    try:
        df["Band"] = pd.qcut(
            df["Combined Score"], q=5,
            labels=["Q1\n(Lowest)", "Q2", "Q3", "Q4", "Q5\n(Highest)"],
            duplicates="drop",
        )
    except Exception:
        return ""
    summary = (
        df.groupby("Band", observed=True)["Return %"]
        .agg(Avg="mean", WinRate=lambda x: (x > 0).mean() * 100, Count="count")
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = [GREEN if v >= 0 else RED for v in summary["Avg"]]
    bars = ax.bar(summary["Band"].astype(str), summary["Avg"], color=colors,
                  edgecolor=DARK_BG, linewidth=0.4)
    for bar, wr in zip(bars, summary["WinRate"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"WR {wr:.0f}%", ha="center", va="bottom",
                color=TEXT_MUTED, fontsize=8)
    ax.axhline(0, color=BORDER, linewidth=0.8)
    ax.set_xlabel("Combined Score Quintile")
    ax.set_ylabel("Avg Return %")
    ax.set_title("Avg Return by Score Quintile — Closed Trades")
    ax.grid(axis="y", alpha=0.4)
    return _fig_to_tag(fig)


def _chart_pattern_winrate(pat_df: pd.DataFrame) -> str:
    if pat_df.empty:
        return ""
    df = pat_df.sort_values("Win Rate %", ascending=True)
    colors = [GREEN if v >= 50 else RED for v in df["Win Rate %"]]
    fig, ax = plt.subplots(figsize=(9, max(3, len(df) * 0.8)))
    bars = ax.barh(df["Pattern"], df["Win Rate %"], color=colors,
                   edgecolor=DARK_BG, linewidth=0.4)
    for bar, avg in zip(bars, df["Avg Ret %"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"avg {avg:+.1f}%", va="center", color=TEXT_MUTED, fontsize=8)
    ax.axvline(50, color=AMBER, linewidth=1.5, linestyle="--", label="50% (break-even)")
    ax.set_xlabel("Win Rate %")
    ax.set_title("Win Rate by Pattern — Closed Trades")
    ax.set_xlim(0, max(df["Win Rate %"].max() + 15, 65))
    ax.legend()
    ax.grid(axis="x", alpha=0.4)
    return _fig_to_tag(fig)


def _chart_rs_bands(closed: pd.DataFrame) -> str:
    if "RS Rating" not in closed.columns:
        return ""
    bands = [(1, 50, "<50"), (50, 75, "50–74"), (75, 90, "75–89"), (90, 100, "90–99")]
    labels, avgs, wrs = [], [], []
    for lo, hi, label in bands:
        sub = closed[(closed["RS Rating"] >= lo) & (closed["RS Rating"] < hi)]["Return %"].dropna()
        if sub.empty:
            continue
        labels.append(label)
        avgs.append(sub.mean())
        wrs.append((sub > 0).mean() * 100)

    if not labels:
        return ""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    colors = [GREEN if v >= 0 else RED for v in avgs]
    ax1.bar(labels, avgs, color=colors, edgecolor=DARK_BG, linewidth=0.4)
    ax1.axhline(0, color=BORDER)
    ax1.set_title("Avg Return by RS Rating Band")
    ax1.set_ylabel("Avg Return %")
    ax1.grid(axis="y", alpha=0.4)

    ax2.bar(labels, wrs, color=PURPLE, edgecolor=DARK_BG, linewidth=0.4)
    ax2.axhline(50, color=AMBER, linestyle="--", linewidth=1.2)
    ax2.set_title("Win Rate by RS Rating Band")
    ax2.set_ylabel("Win Rate %")
    ax2.set_ylim(0, 100)
    ax2.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    return _fig_to_tag(fig)


def _chart_holding_days(closed: pd.DataFrame) -> str:
    hd = closed["Holding Days"].dropna()
    if hd.empty:
        return ""
    wins   = closed[closed["Return %"] > 0]["Holding Days"].dropna()
    losses = closed[closed["Return %"] <= 0]["Holding Days"].dropna()
    fig, ax = plt.subplots(figsize=(9, 4))
    if not wins.empty:
        ax.hist(wins,   bins=15, color=GREEN, alpha=0.7, label=f"Winners (n={len(wins)})")
    if not losses.empty:
        ax.hist(losses, bins=15, color=RED,   alpha=0.7, label=f"Losers  (n={len(losses)})")
    ax.set_xlabel("Holding Days")
    ax.set_ylabel("Frequency")
    ax.set_title("Holding Period Distribution — Closed Trades")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    return _fig_to_tag(fig)


# ---------------------------------------------------------------------------
# HTML table helpers
# ---------------------------------------------------------------------------

def _ret_span(val) -> str:
    try:
        v = float(val)
        color = GREEN if v > 0 else (RED if v < 0 else TEXT_MUTED)
        return f"<span style='color:{color};font-weight:600'>{v:+.2f}%</span>"
    except Exception:
        return f"<span style='color:{TEXT_MUTED}'>—</span>"


def _df_to_html(df: pd.DataFrame, ret_cols: tuple = (), pct_cols: tuple = ()) -> str:
    if df.empty:
        return f"<p class='muted'>No data available.</p>"
    headers = "".join(f"<th>{c}</th>" for c in df.columns)
    rows_html = ""
    for _, r in df.iterrows():
        cells = ""
        for col in df.columns:
            val = r[col]
            if col in ret_cols:
                cells += f"<td>{_ret_span(val)}</td>"
            elif col in pct_cols and val is not None:
                try:
                    cells += f"<td>{float(val):.0f}%</td>"
                except Exception:
                    cells += f"<td>{val}</td>"
            elif isinstance(val, float) and not pd.isna(val):
                cells += f"<td>{val:.2f}</td>"
            elif val is None or (isinstance(val, float) and pd.isna(val)):
                cells += f"<td><span class='muted'>—</span></td>"
            else:
                cells += f"<td>{val}</td>"
        rows_html += f"<tr>{cells}</tr>"
    return f"""
    <div class="table-wrap">
      <table>
        <thead><tr>{headers}</tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"""


def _kpi_card(label: str, value: str, color: str = TEXT_PRIMARY) -> str:
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value" style="color:{color}">{value}</div>
    </div>"""


# ---------------------------------------------------------------------------
# Analysis builders
# ---------------------------------------------------------------------------

def _pattern_table(closed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pat in PATTERN_COLS:
        if pat not in closed.columns:
            continue
        sub = closed[closed[pat] == True]["Return %"].dropna()
        no_sub = closed[closed[pat] == False]["Return %"].dropna()
        if sub.empty:
            continue
        rows.append({
            "Pattern":        pat,
            "Trades":         len(sub),
            "Win Rate %":     round((sub > 0).mean() * 100, 0),
            "Avg Ret %":      round(sub.mean(), 2),
            "Median Ret %":   round(sub.median(), 2),
            "Best %":         round(sub.max(), 2),
            "Worst %":        round(sub.min(), 2),
            "vs No-Pattern":  round(sub.mean() - (no_sub.mean() if not no_sub.empty else 0), 2),
        })
    return pd.DataFrame(rows).sort_values("Avg Ret %", ascending=False) if rows else pd.DataFrame()


def _score_band_table(closed: pd.DataFrame) -> pd.DataFrame:
    df = closed.dropna(subset=["Return %", "Combined Score"]).copy()
    if len(df) < 5:
        return pd.DataFrame()
    try:
        df["Band"] = pd.qcut(
            df["Combined Score"], q=5,
            labels=["Q1 (Lowest)", "Q2", "Q3", "Q4", "Q5 (Highest)"],
            duplicates="drop",
        )
    except Exception:
        return pd.DataFrame()
    rows = []
    for band, grp in df.groupby("Band", observed=True):
        r = grp["Return %"].dropna()
        rows.append({
            "Score Band":   str(band),
            "Score Range":  f"{grp['Combined Score'].min():.0f}–{grp['Combined Score'].max():.0f}",
            "Trades":       len(grp),
            "Win Rate %":   round((r > 0).mean() * 100, 0),
            "Avg Ret %":    round(r.mean(), 2),
            "Median Ret %": round(r.median(), 2),
            "Avg Hold Days": round(grp["Holding Days"].mean(), 0),
        })
    return pd.DataFrame(rows)


def _rs_band_table(closed: pd.DataFrame) -> pd.DataFrame:
    if "RS Rating" not in closed.columns:
        return pd.DataFrame()
    bands = [(1, 50, "<50"), (50, 75, "50–74"), (75, 90, "75–89"), (90, 100, "90–99")]
    rows = []
    for lo, hi, label in bands:
        sub = closed[(closed["RS Rating"] >= lo) & (closed["RS Rating"] < hi)]["Return %"].dropna()
        if sub.empty:
            continue
        rows.append({
            "RS Rating Band": label,
            "Trades":         len(sub),
            "Win Rate %":     round((sub > 0).mean() * 100, 0),
            "Avg Ret %":      round(sub.mean(), 2),
            "Median Ret %":   round(sub.median(), 2),
        })
    return pd.DataFrame(rows)


def _component_correlation(closed: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation of score components vs actual Return %."""
    components = {
        "Score (Overall)":   "Score",
        "RS Rating":         "RS Rating",
        "Combined Score":    "Combined Score",
    }
    for p in PATTERN_COLS:
        components[p] = p

    rows = []
    ret = closed["Return %"].dropna()
    for label, col in components.items():
        if col not in closed.columns:
            continue
        vals = closed.loc[ret.index, col].apply(
            lambda x: 1 if x is True else (0 if x is False else x)
        )
        paired = pd.concat([vals, ret], axis=1).dropna()
        if len(paired) < 5:
            continue
        corr = paired.iloc[:, 0].corr(paired.iloc[:, 1])
        rows.append({
            "Component":  label,
            "Correlation": round(corr, 3),
            "Strength":    (
                "Strong +"  if corr >= 0.3 else
                "Moderate +" if corr >= 0.1 else
                "Weak"       if corr >= -0.1 else
                "Moderate -" if corr >= -0.3 else
                "Strong -"
            ),
        })
    return pd.DataFrame(rows).sort_values("Correlation", ascending=False)


def _trade_log_html(trades: pd.DataFrame) -> str:
    cols = [
        "Signal Date", "Symbol", "Sector", "Combined Score",
        "Entry Date", "Entry Price", "Stop Loss",
        "Exit Date", "Exit Price", "Return %",
        "Holding Days", "Status", "Exit Reason", "Max Gain %", "Max DD %",
    ] + [p for p in PATTERN_COLS if p in trades.columns]

    available = [c for c in cols if c in trades.columns]
    df = trades[available].copy()

    # Add status colour
    def _status(val):
        color = GREEN if val == "closed" else AMBER
        return f"<span style='color:{color}'>{val}</span>"

    # Build manually with colour
    headers = "".join(f"<th>{c}</th>" for c in df.columns)
    rows_html = ""
    for _, r in df.sort_values("Return %", ascending=False, na_position="last").iterrows():
        cells = ""
        for col in df.columns:
            val = r[col]
            if col == "Return %" or col == "Max Gain %" or col == "Max DD %":
                cells += f"<td>{_ret_span(val)}</td>"
            elif col == "Status":
                cells += f"<td>{_status(val)}</td>"
            elif col == "Exit Reason":
                color = RED if val == "Stop Loss" else (BLUE if val == "EMA20" else AMBER)
                cells += f"<td><span style='color:{color}'>{val}</span></td>"
            elif col in PATTERN_COLS:
                icon = "✓" if val else "·"
                color = GREEN if val else TEXT_MUTED
                cells += f"<td style='color:{color}'>{icon}</td>"
            elif isinstance(val, float) and not pd.isna(val):
                cells += f"<td>{val:.2f}</td>"
            elif val is None or (isinstance(val, float) and pd.isna(val)):
                cells += f"<td><span class='muted'>—</span></td>"
            else:
                cells += f"<td>{val}</td>"
        rows_html += f"<tr>{cells}</tr>"
    return f"""
    <div class="table-wrap">
      <table>
        <thead><tr>{headers}</tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"""


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

CSS = f"""
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: {DARK_BG}; color: {TEXT_PRIMARY};
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 14px; line-height: 1.6; padding: 32px;
  }}
  h1 {{ font-size: 1.8rem; font-weight: 800; margin-bottom: 4px; }}
  h2 {{ font-size: 1.15rem; font-weight: 700; margin: 32px 0 14px;
        color: {TEXT_PRIMARY}; border-left: 3px solid {BLUE};
        padding-left: 10px; }}
  h3 {{ font-size: 0.95rem; font-weight: 600; color: {TEXT_MUTED}; margin-bottom: 10px; }}
  .subtitle {{ color: {TEXT_MUTED}; font-size: 0.85rem; margin-bottom: 28px; }}
  .kpi-row {{ display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 6px; }}
  .kpi-card {{
    background: {CARD_BG}; border: 1px solid {BORDER};
    border-radius: 10px; padding: 16px 22px; min-width: 130px; flex: 1;
  }}
  .kpi-label {{ font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
                letter-spacing: 0.08em; color: {TEXT_MUTED}; }}
  .kpi-value {{ font-size: 1.5rem; font-weight: 800; margin-top: 4px; }}
  .section {{ background: {CARD_BG}; border: 1px solid {BORDER};
              border-radius: 12px; padding: 22px 24px; margin-bottom: 22px; }}
  .chart-img {{ width: 100%; border-radius: 8px; margin-top: 10px; }}
  .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .table-wrap {{ overflow-x: auto; margin-top: 6px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.83rem; }}
  thead tr {{ background: rgba(255,255,255,0.04); }}
  th {{ padding: 9px 12px; text-align: left; font-weight: 700;
        color: {TEXT_MUTED}; font-size: 0.75rem; text-transform: uppercase;
        letter-spacing: 0.05em; border-bottom: 1px solid {BORDER}; white-space: nowrap; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid rgba(255,255,255,0.04);
        white-space: nowrap; }}
  tr:hover td {{ background: rgba(255,255,255,0.025); }}
  .muted {{ color: {TEXT_MUTED}; }}
  .divider {{ border: none; border-top: 1px solid {BORDER}; margin: 24px 0; }}
  @media(max-width:900px) {{ .chart-grid {{ grid-template-columns: 1fr; }} }}
</style>
"""


def _build_html(trades: pd.DataFrame, weeks: int, generated_at: str) -> str:
    closed = trades[trades["Status"] == "closed"].copy()
    open_  = trades[trades["Status"] == "open"].copy()

    n_signals  = len(trades)
    n_closed   = len(closed)
    n_open     = len(open_)
    ret        = closed["Return %"].dropna()

    win_rate   = f"{(ret > 0).mean() * 100:.0f}%" if len(ret) else "—"
    avg_ret    = f"{ret.mean():+.2f}%"             if len(ret) else "—"
    med_ret    = f"{ret.median():+.2f}%"           if len(ret) else "—"
    avg_hold   = f"{closed['Holding Days'].mean():.0f}d" if n_closed else "—"
    best       = f"{ret.max():+.2f}%"              if len(ret) else "—"
    worst      = f"{ret.min():+.2f}%"              if len(ret) else "—"
    profit_fac = (
        f"{ret[ret > 0].sum() / abs(ret[ret < 0].sum()):.2f}"
        if (ret < 0).any() and (ret > 0).any() else "—"
    )

    wr_color = GREEN if len(ret) and (ret > 0).mean() >= 0.5 else RED
    ar_color = GREEN if len(ret) and ret.mean() >= 0 else RED

    kpi_row = f"""
    <div class="kpi-row">
      {_kpi_card("Total Signals",   str(n_signals))}
      {_kpi_card("Entries Triggered", str(n_closed + n_open))}
      {_kpi_card("Closed Trades",   str(n_closed))}
      {_kpi_card("Open Trades",     str(n_open),  AMBER)}
      {_kpi_card("Win Rate",        win_rate,     wr_color)}
      {_kpi_card("Avg Return",      avg_ret,      ar_color)}
      {_kpi_card("Median Return",   med_ret,      ar_color)}
      {_kpi_card("Avg Hold",        avg_hold)}
      {_kpi_card("Best Trade",      best,         GREEN)}
      {_kpi_card("Worst Trade",     worst,        RED)}
      {_kpi_card("Profit Factor",   profit_fac)}
    </div>"""

    # Charts
    _plt_defaults()
    c_dist   = _chart_return_distribution(closed)
    c_hold   = _chart_holding_days(closed)
    c_score  = _chart_score_bands(closed)
    c_rs     = _chart_rs_bands(closed)

    pat_df    = _pattern_table(closed)
    c_pat     = _chart_pattern_winrate(pat_df)

    score_df  = _score_band_table(closed)
    rs_df     = _rs_band_table(closed)
    corr_df   = _component_correlation(closed)

    trade_log = _trade_log_html(trades)

    def _section(title, content):
        return f'<div class="section"><h2>{title}</h2>{content}</div>'

    sl_exits   = len(closed[closed["Exit Reason"] == "Stop Loss"]) if "Exit Reason" in closed.columns else 0
    ema_exits  = len(closed[closed["Exit Reason"] == "EMA20"])      if "Exit Reason" in closed.columns else 0

    entry_rules = f"""
    <ul style="color:{TEXT_MUTED}; font-size:0.85rem; margin-top:8px; padding-left:18px; line-height:2">
      <li>Universe: latest screener output ({len(trades['Symbol'].unique())} stocks)</li>
      <li>Lookback: last {weeks} weeks of trading days</li>
      <li>Entry: price crosses above prev day High + Volume &gt; 20D Vol MA (max 3-day wait)</li>
      <li>Entry price: previous day's High (breakout level)</li>
      <li>Stop Loss: 0.25% below the minimum Low of the last 2 days at entry</li>
      <li>Exit 1 — Stop Loss hit: intraday Low ≤ stop-loss level → exit at stop-loss price
          ({sl_exits} trades)</li>
      <li>Exit 2 — EMA20 breach: daily Close &lt; 20 EMA → exit at Close
          ({ema_exits} trades)</li>
      <li>Open trades: neither exit triggered yet — valued at latest Close</li>
    </ul>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Backtest Report — {generated_at}</title>
  {CSS}
</head>
<body>
  <h1>📊 Backtest Report</h1>
  <p class="subtitle">
    Generated: {generated_at} &nbsp;·&nbsp;
    Lookback: {weeks} weeks &nbsp;·&nbsp;
    Entry: breakout above prev High + volume &nbsp;·&nbsp;
    Exit: Close &lt; EMA20
  </p>

  {_section("Methodology", entry_rules)}
  {_section("Executive Summary", kpi_row)}

  <div class="section">
    <h2>Return Distribution &amp; Holding Period</h2>
    <div class="chart-grid">
      <div>{c_dist}</div>
      <div>{c_hold}</div>
    </div>
  </div>

  <div class="section">
    <h2>Score Band Performance</h2>
    {_df_to_html(score_df, ret_cols=("Avg Ret %", "Median Ret %"), pct_cols=("Win Rate %",))}
    {c_score}
  </div>

  <div class="section">
    <h2>Pattern Effectiveness</h2>
    {_df_to_html(pat_df, ret_cols=("Avg Ret %", "Median Ret %", "Best %", "Worst %", "vs No-Pattern"),
                 pct_cols=("Win Rate %",))}
    {c_pat}
  </div>

  <div class="section">
    <h2>RS Rating Band Performance</h2>
    {_df_to_html(rs_df, ret_cols=("Avg Ret %", "Median Ret %"), pct_cols=("Win Rate %",))}
    {c_rs}
  </div>

  <div class="section">
    <h2>Score Component Correlation with Returns</h2>
    <p style="color:{TEXT_MUTED};font-size:0.82rem;margin-bottom:10px">
      Pearson correlation between each component and actual Return %.
      Positive = higher score → better return. Values &gt;0.3 are practically significant.
    </p>
    {_df_to_html(corr_df)}
  </div>

  <div class="section">
    <h2>Full Trade Log ({len(trades)} trades, sorted by Return %)</h2>
    {trade_log}
  </div>

</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate backtest report")
    parser.add_argument("--weeks", type=int, default=4,
                        help="Number of weeks to look back (default: 4)")
    args = parser.parse_args()

    logger.info("Loading data…")
    price_data = load_price_data()
    if not price_data:
        logger.error("price_data.pkl not found or empty — run Fetch Data first.")
        sys.exit(1)

    screener_results = _load_screener_results()
    logger.info(f"Screener results: {len(screener_results)} stocks loaded.")

    logger.info(f"Running backtest (last {args.weeks} weeks)…")
    trades = run_backtest(price_data, screener_results, lookback_weeks=args.weeks)

    if trades.empty:
        logger.warning("No trades generated — try a longer lookback with --weeks N.")
        sys.exit(0)

    ist = pytz.timezone(IST_TIMEZONE)
    generated_at = datetime.now(ist).strftime("%Y-%m-%d %H:%M IST")

    logger.info("Building HTML report…")
    html = _build_html(trades, args.weeks, generated_at)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"Report saved → {REPORT_FILE}")
    webbrowser.open(f"file:///{REPORT_FILE.replace(os.sep, '/')}")


if __name__ == "__main__":
    main()
