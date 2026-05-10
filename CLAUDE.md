# NSE Stock Screener — CLAUDE.md

## Project Overview

A daily stock screener for Nifty 1000 NSE-listed stocks. Fetches historical price data via `yfinance`, applies technical filters, ranks stocks using a composite scoring model, and displays results grouped by Sector > Industry in a Streamlit dashboard.

Auto-refreshes once daily at **4:00 PM IST** via APScheduler background thread.

**Run with:** `streamlit run app.py` → http://localhost:8501

---

## Architecture

```
STOCK SCREENER/
├── CLAUDE.md
├── requirements.txt
├── config.py                # All constants, weights, tickers, paths
├── app.py                   # Streamlit UI + APScheduler startup
├── refresh.py               # Standalone CLI refresh script
├── src/
│   ├── __init__.py
│   ├── nse_fetcher.py       # Fetch Nifty 1000 constituent list from NSE
│   ├── data_fetcher.py      # yfinance batch download for prices + info
│   ├── indicators.py        # EMA, RSI, VCP, volume contraction, performance
│   ├── sector_mapper.py     # Map sector names → NSE index tickers; fetch index data
│   └── screener.py          # Apply filters, compute scores, rank and return DataFrame
└── data/                    # Runtime cache (auto-created)
    ├── nifty1000_stocks.csv # Nifty 1000 constituent list (refreshed weekly)
    ├── price_data.pkl       # 1-year daily OHLCV for all stocks
    ├── stock_info.pkl       # PE, sector, industry, market cap per stock
    ├── index_data.pkl       # Benchmark + sector index price data
    ├── screener_results.pkl # Final screened + ranked DataFrame
    └── last_updated.txt     # ISO timestamp of last successful refresh
```

**Data flow:**
1. `nse_fetcher` → Nifty 1000 symbol list
2. `data_fetcher` → download 1y daily price data + stock info (PE, sector, mkt cap)
3. `sector_mapper` → fetch NiftyMidSmallCap400 benchmark + all sector indices
4. `indicators` → compute EMAs, RSI, VCP, volume contraction per stock
5. `screener` → apply hard filters, compute composite score, sort
6. `app.py` → read pkl cache, render Streamlit UI

---

## Screener Filters (Hard — must ALL pass)

| Filter | Condition |
|--------|-----------|
| Universe | Nifty 1000 (.NS yfinance tickers) |
| Market Cap | ≥ ₹2,000 Crores |
| Price vs 52W High | Current Price > 52W High × 0.76 (within 24% of 52W high) |
| EMA Stack | EMA10 > EMA20 > EMA50 > EMA200 (daily) |
| Price vs EMA20 | Current Price > EMA20 |
| Min data | ≥ 200 trading days of history |

---

## Ranking Score (0–100 composite)

Stocks that pass filters are ranked by a weighted composite score:

| Criterion | Weight | Logic |
|-----------|--------|-------|
| Closer to 52W High | 25 pts | `max(0, 100 - pct_below_52h * 2)` |
| NiftyMidSmallCap400 Outperf (1M) | 20 pts | Stock 1M return − benchmark 1M return |
| Sector Index Outperf (1M) | 15 pts | Stock 1M return − sector index 1M return |
| VCP Pattern detected | 20 pts | Binary: 20 or 0 |
| Volume Contraction | 10 pts | 5d avg vol < 80% of 20d avg vol |
| Near 10 EMA (−2.5% to +1.5%) | 10 pts | Normalized distance within band |

---

## VCP Pattern Detection

Volatility Contraction Pattern (simplified Minervini):
1. **Prior run-up**: Stock gained ≥ 15% in the 40 trading days before the consolidation window
2. **Consolidation** (last 3 weeks / 15 trading days): Each successive week shows:
   - Smaller price range (High − Low as % of close)
   - Lower average volume

---

## Benchmark & Sector Indices (yfinance tickers)

- **Benchmark**: `^NSMIDCP` (NiftyMidSmallCap400) → fallback `^CNXMID` → `^NSEI`
- **Sector mapping** (yfinance sector name → NSE index):
  - Financial Services / Banks → `^NSEBANK`
  - Technology / IT → `^CNXIT`
  - Healthcare / Pharma → `^CNXPHARMA`
  - Consumer Defensive / FMCG → `^CNXFMCG`
  - Basic Materials / Metal → `^CNXMETAL`
  - Energy / Oil & Gas → `^CNXENERGY`
  - Real Estate → `^CNXREALTY`
  - Industrials / Infra → `^CNXINFRA`
  - Communication / Media → `^CNXMEDIA`
  - Consumer Cyclical / Auto → `^CNXAUTO`
  - Default fallback → `^NSEI`

---

## Scheduler

- APScheduler `BackgroundScheduler` starts once when Streamlit app launches
- Cron trigger: **16:00 IST daily** (`Asia/Kolkata` timezone)
- Refresh writes results to `data/screener_results.pkl` + `data/last_updated.txt`
- UI reads from cache; "Refresh Now" button triggers an immediate synchronous refresh

---

## Performance Notes

- First run: ~15–30 min (fetching info for 1000 stocks + 1y price data)
- Subsequent refreshes: ~5–10 min (price data only; info cached for 7 days)
- yfinance batch size: 50 stocks per request with 3s delay between batches
- Stock info fetch: parallel via `ThreadPoolExecutor(max_workers=10)`

---

## Key Config (config.py)

All tunable parameters live in `config.py`:
- `MIN_MARKET_CAP_CR` — minimum market cap filter (default 2000)
- `PRICE_MIN_PCT_OF_52W_HIGH` — price vs 52W high threshold (default 0.76, i.e. within 24% of high)
- `EMA10_UPPER_PCT` / `EMA10_LOWER_PCT` — near-10-EMA band (+1.5% / −2.5%)
- `VCP_RUNUP_MIN_PCT` — minimum prior run-up for VCP (default 0.15 = 15%)
- `REFRESH_HOUR_IST` — auto-refresh hour in IST (default 16)
- `SCORE_WEIGHTS` — dict of criterion weights for composite score

---

## Dependencies

See `requirements.txt`. Key packages:
- `yfinance` — market data
- `streamlit` — UI
- `pandas`, `numpy` — data processing
- `apscheduler` — daily scheduler
- `pytz` — IST timezone
- `requests` — NSE stock list fetch
