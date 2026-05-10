import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# NSE equity list source
EQUITY_LIST_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
EQUITY_LIST_CACHE_DAYS = 7   # Re-fetch after 7 days

# Cache file paths
NIFTY1000_FILE = os.path.join(DATA_DIR, "nse_equity_list.csv")
PRICE_DATA_FILE = os.path.join(DATA_DIR, "price_data.pkl")
STOCK_INFO_FILE = os.path.join(DATA_DIR, "stock_info.pkl")
INDEX_DATA_FILE = os.path.join(DATA_DIR, "index_data.pkl")
SCREENER_RESULTS_FILE = os.path.join(DATA_DIR, "screener_results.pkl")
LAST_UPDATED_FILE = os.path.join(DATA_DIR, "last_updated.txt")
REFRESH_STATUS_FILE = os.path.join(DATA_DIR, "refresh_status.json")
SNAPSHOTS_FILE = os.path.join(DATA_DIR, "snapshots.pkl")

# Benchmark index (NiftyMidSmallCap400)
BENCHMARK_TICKERS = ["^NSMIDCP", "^CNXMID", "^NSEI"]

# Nifty 500 — used to rank sectors by outperformance
NIFTY500_TICKERS = ["^NSEI"]

# EMA periods (daily)
EMA_PERIODS = [10, 20, 50, 200]

# Hard filters
MIN_MARKET_CAP_CR = 2000
PRICE_MIN_PCT_OF_52W_HIGH = 0.76   # Price must be > 52W High × 0.76 (within 24% of high)
MIN_HISTORY_DAYS = 200

# RSI
RSI_PERIOD = 14

# Performance lookback
WEEKLY_DAYS = 5
MONTHLY_DAYS = 21
QUARTERLY_DAYS = 63   # ~3 months (21 × 3)

# VCP detection
VCP_RUNUP_LOOKBACK = 40       # Days before consolidation to check for run-up
VCP_RUNUP_MIN_PCT = 0.15      # Minimum 15% prior run-up
VCP_CONSOL_WEEKS = 3          # Number of consolidation weeks to check

# Volume contraction
VOL_SHORT_PERIOD = 5          # Recent volume window (days)
VOL_LONG_PERIOD = 20          # Historical volume window (days)
VOL_CONTRACTION_RATIO = 0.80  # Recent vol must be < 80% of historical vol

# Near 10 EMA band
EMA10_UPPER_PCT = 0.015       # +1.5%
EMA10_LOWER_PCT = -0.025      # -2.5%

# Scheduler: 4:00 PM IST daily
REFRESH_HOUR_IST = 16
REFRESH_MINUTE_IST = 0
IST_TIMEZONE = "Asia/Kolkata"

# yfinance fetch settings
BATCH_SIZE = 50
BATCH_DELAY_SECONDS = 3
INFO_MAX_WORKERS = 10
STOCK_INFO_CACHE_DAYS = 7     # Re-fetch stock info after 7 days
NIFTY1000_CACHE_DAYS = EQUITY_LIST_CACHE_DAYS

# RSI hard filter — stocks below this threshold are excluded before scoring
RSI_MIN_SCORE = 50

# EMA10 extension hard filter — stocks more than this % above 10 EMA are excluded
EMA10_MAX_EXTENSION_PCT = 0.06   # 6% above EMA10


# Composite score weights (must sum to 100)
SCORE_WEIGHTS = {
    "52w_high_proximity": 20,
    "benchmark_outperf":  15,
    "sector_outperf":     15,
    "tight_range_vol":     0,
    "close_to_kma":        5,   # Proximity gradient within ±1.5% of 10 EMA
    "rsi_score":          15,   # RSI normalised from 55→80 band
    "rmv_score":           5,   # Lower RMV = higher score
    "rs_trend":           25,   # RS line EMA10 > EMA20 → uptrend
    "ema10_touch":         0,   # Superseded by EMA10_MAX_EXTENSION_PCT hard filter
}

# Close to KMA (Key Moving Average = 10 EMA) scoring band
KMA_BAND_PCT = 0.015   # ±1.5% around 10 EMA; score = 1.0 at EMA, 0.0 at boundary

# Relative Measured Volatility (RMV)
RMV_LOOKBACK         = 15   # Number of prior bars used for min/max normalization
RMV_TIGHT_THRESHOLD  = 15   # RMV < this → flagged as unusually compressed / tight bar

# Combined rank weights — how Momentum Score and RS Rating are blended for final rank
# Must sum to 100.
COMBINED_RANK_WEIGHTS = {
    "score":     40,   # Momentum Score (0–100)
    "rs_rating": 60,   # RS Rating (1–99)
}

# NSE sector index tickers (yfinance) keyed by lowercase fragment of sector name
SECTOR_INDEX_MAP = {
    "financial services": "^NSEBANK",
    "bank":               "^NSEBANK",
    "financial":          "^NSEBANK",
    "technology":         "^CNXIT",
    "software":           "^CNXIT",
    "information technology": "^CNXIT",
    "healthcare":         "^CNXPHARMA",
    "pharmaceutical":     "^CNXPHARMA",
    "pharma":             "^CNXPHARMA",
    "drug":               "^CNXPHARMA",
    "consumer defensive": "^CNXFMCG",
    "consumer staples":   "^CNXFMCG",
    "fmcg":               "^CNXFMCG",
    "basic materials":    "^CNXMETAL",
    "metal":              "^CNXMETAL",
    "mining":             "^CNXMETAL",
    "energy":             "^CNXENERGY",
    "oil":                "^CNXENERGY",
    "gas":                "^CNXENERGY",
    "power":              "^CNXENERGY",
    "real estate":        "^CNXREALTY",
    "realty":             "^CNXREALTY",
    "industrials":        "^CNXINFRA",
    "infrastructure":     "^CNXINFRA",
    "capital goods":      "^CNXINFRA",
    "communication":      "^CNXMEDIA",
    "media":              "^CNXMEDIA",
    "telecom":            "^CNXMEDIA",
    "consumer cyclical":  "^CNXAUTO",
    "automobile":         "^CNXAUTO",
    "auto":               "^CNXAUTO",
    "utility":            "^CNXENERGY",
    "utilities":          "^CNXENERGY",
}

DEFAULT_SECTOR_INDEX = "^NSEI"  # Nifty 50 fallback

INR_PER_CR = 1_00_00_000  # 1 Crore = 10 million

# Pattern columns produced by the screener (boolean).
# Adding a new pattern: implement detect_X() in indicators.py, call it in
# screener.py/_process_stock(), add the key here — everything else updates
# automatically (sidebar filters, metrics bar, tables, tracker snapshots).
PATTERN_COLS = [
    "Flag",
    "VCP",
    "Vol Contraction",
    "Near 10 EMA",
    "Inside Day",
]

# IPO Base pattern
IPO_BASE_MAX_DAYS = 199     # Max trading days since listing to qualify
IPO_BASE_MIN_DAYS = 20      # Min days needed to have formed a base
IPO_BASE_SKIP_DAYS = 5      # Skip first N days (post-listing volatility)
IPO_BASE_RANGE_MAX = 0.60   # Base High-Low range / Low must be ≤ this (60%)
IPO_BREAKOUT_LOWER = -0.05  # Price can be up to 5% below base high
IPO_BREAKOUT_UPPER = 0.10   # Price can be up to 10% above base high (still in play)
IPO_VOL_PICKUP_RATIO = 1.0  # Recent vol ≥ historical vol (accumulation signal)

# Cache file for IPO screener results
IPO_RESULTS_FILE    = os.path.join(DATA_DIR, "ipo_results.pkl")
IPO_IB_RESULTS_FILE = os.path.join(DATA_DIR, "ipo_ib_results.pkl")
