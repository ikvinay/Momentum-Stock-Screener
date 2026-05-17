"""
Relative Rotation Graph (RRG) — Enhanced EMA-based computation.

Uses the enhanced variant of Julius de Kempenaer's methodology: EMA ratio
normalisation instead of z-score, giving 2-3 periods faster signals and
direct percentage interpretation.

Formulas
--------
RS         = Close / Benchmark_Close
EMA_RS     = EMA(RS, span=m)
RS_Ratio   = 100 × EMA_RS / RollingMean(EMA_RS, m)

ROC        = pct_change(RS_Ratio, k)
EMA_ROC    = EMA(ROC, span=m)
RS_Momentum = 100 + 100 × EMA_ROC

Both axes are centred at 100. Quadrants (clockwise rotation is normal cycle):
  Improving  (x≤100, y>100)  — underperforming but recovering    → early entry
  Leading    (x>100, y>100)  — outperforming and accelerating     → hold/add
  Weakening  (x>100, y≤100)  — outperforming but decelerating     → take profits
  Lagging    (x≤100, y≤100)  — underperforming and decelerating   → avoid/exit
"""

import logging
from typing import Dict, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

QUADRANT_COLORS = {
    "Leading":   "#22c55e",
    "Weakening": "#f59e0b",
    "Lagging":   "#ef4444",
    "Improving": "#60a5fa",
}

QUADRANT_FILL = {
    "Leading":   "rgba(34,197,94,0.09)",
    "Weakening": "rgba(245,158,11,0.09)",
    "Lagging":   "rgba(239,68,68,0.09)",
    "Improving": "rgba(96,165,250,0.09)",
}

_EMPTY_DF = pd.DataFrame(
    columns=["name", "rs_ratio", "rs_momentum", "quadrant", "color", "tail_x", "tail_y"]
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample a daily OHLCV DataFrame to weekly (Friday close)."""
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    agg: dict = {"Close": "last"}
    for col, fn in [("Open", "first"), ("High", "max"), ("Low", "min"), ("Volume", "sum")]:
        if col in df.columns:
            agg[col] = fn
    return df.resample("W-FRI").agg(agg).dropna(subset=["Close"])


def _quadrant(rs_ratio: float, rs_momentum: float) -> str:
    if rs_ratio >= 100 and rs_momentum >= 100:
        return "Leading"
    if rs_ratio >= 100:
        return "Weakening"
    if rs_momentum >= 100:
        return "Improving"
    return "Lagging"


def _compute_series(
    close: pd.Series,
    bench: pd.Series,
    m: int,
    k: int,
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute RS-Ratio and RS-Momentum series for one security vs a benchmark.

    Returns (rs_ratio, rs_momentum) aligned on a common date index.
    Returns two empty Series if there is insufficient history.
    """
    close.index = pd.to_datetime(close.index)
    bench.index = pd.to_datetime(bench.index)

    common = close.index.intersection(bench.index)
    if len(common) < m + k + 5:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    c = close.reindex(common).ffill().dropna()
    b = bench.reindex(common).ffill().dropna()
    idx = c.index.intersection(b.index)
    c, b = c.loc[idx], b.loc[idx]

    rs        = c / b
    ema_rs    = rs.ewm(span=m, adjust=False).mean()
    roll_mean = ema_rs.rolling(m).mean()
    rs_ratio  = (100.0 * ema_rs / roll_mean).dropna()

    roc      = rs_ratio.pct_change(k)
    ema_roc  = roc.ewm(span=m, adjust=False).mean()
    rs_mom   = (100.0 + 100.0 * ema_roc).dropna()

    shared = rs_ratio.index.intersection(rs_mom.index)
    return rs_ratio.loc[shared], rs_mom.loc[shared]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_rrg(
    price_dict: Dict[str, pd.DataFrame],
    benchmark_df: pd.DataFrame,
    m: int = 14,
    k: int = 10,
    tail: int = 8,
    weekly: bool = False,
) -> pd.DataFrame:
    """
    Compute RRG positions for all securities against a benchmark.

    Parameters
    ----------
    price_dict   : {display_name: OHLCV DataFrame}
    benchmark_df : DataFrame with a 'Close' column (daily or weekly)
    m            : EMA window — default 14
    k            : ROC lookback period — default 10
    tail         : number of historical trail points incl. current — default 8
    weekly       : if True, resample all daily data to weekly before computing

    Returns
    -------
    DataFrame with columns:
        name, rs_ratio, rs_momentum, quadrant, color, tail_x (list), tail_y (list)
    Only securities with enough history are included. Empty DataFrame if no data.
    """
    if benchmark_df is None or benchmark_df.empty:
        return _EMPTY_DF.copy()

    bench = _to_weekly(benchmark_df) if weekly else benchmark_df.copy()
    bench.index = pd.to_datetime(bench.index)
    bench_close = bench["Close"].dropna()

    rows = []
    for name, df in price_dict.items():
        try:
            sec = _to_weekly(df) if weekly else df.copy()
            sec.index = pd.to_datetime(sec.index)
            close = sec["Close"].dropna()

            rsr, rsm = _compute_series(close, bench_close, m, k)
            if rsr.empty or len(rsr) < tail:
                continue

            tx = rsr.iloc[-tail:].tolist()
            ty = rsm.iloc[-tail:].tolist()
            cur_x, cur_y = tx[-1], ty[-1]
            q = _quadrant(cur_x, cur_y)

            rows.append({
                "name":        name,
                "rs_ratio":    round(cur_x, 3),
                "rs_momentum": round(cur_y, 3),
                "quadrant":    q,
                "color":       QUADRANT_COLORS[q],
                "tail_x":      [round(v, 3) for v in tx],
                "tail_y":      [round(v, 3) for v in ty],
            })
        except Exception as exc:
            logger.debug("RRG: skipping %s — %s", name, exc)

    return pd.DataFrame(rows) if rows else _EMPTY_DF.copy()
