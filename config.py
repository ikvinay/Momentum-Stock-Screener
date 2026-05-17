import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# NSE equity list source
EQUITY_LIST_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
EQUITY_LIST_CACHE_DAYS = 7

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
PRICE_MIN_PCT_OF_52W_HIGH = 0.76
MIN_HISTORY_DAYS = 200

# RSI
RSI_PERIOD = 14

# Performance lookback
WEEKLY_DAYS = 5
MONTHLY_DAYS = 21
QUARTERLY_DAYS = 63

# VCP detection
VCP_RUNUP_LOOKBACK = 40
VCP_RUNUP_MIN_PCT = 0.3
VCP_CONSOL_WEEKS = 4
VCP_CONTRACTION_TOLERANCE = 0.15
VCP_VOL_DRY_UP_RATIO = 0.65
VCP_MAX_BASE_DEPTH_PCT = 0.12

# Volume contraction
VOL_SHORT_PERIOD = 5
VOL_LONG_PERIOD = 20
VOL_EXTENDED_PERIOD = 50
VOL_CONTRACTION_RATIO = 0.8
VOL_PRICE_STABILITY_PCT = 0.03
VOL_DISTRIB_DAY_PCT = 0.01

# Near 10 EMA band
EMA10_UPPER_PCT = 0.015
EMA10_LOWER_PCT = -0.025

# Scheduler: 4:00 PM IST daily
REFRESH_HOUR_IST = 16
REFRESH_MINUTE_IST = 0
IST_TIMEZONE = "Asia/Kolkata"

# yfinance fetch settings
BATCH_SIZE = 50
BATCH_DELAY_SECONDS = 3
INFO_MAX_WORKERS = 10
STOCK_INFO_CACHE_DAYS = 7
NIFTY1000_CACHE_DAYS = EQUITY_LIST_CACHE_DAYS

# RSI hard filter — stocks below this threshold are excluded before scoring
RSI_MIN_SCORE = 50

# EMA10 extension hard filter — stocks more than this % above 10 EMA are excluded
EMA10_MAX_EXTENSION_PCT = 0.06


# Composite score weights (must sum to 100)
SCORE_WEIGHTS = {
    "52w_high_proximity": 0,
    "benchmark_outperf" : 25,
    "sector_outperf"    : 25,
    "tight_range_vol"   : 10,
    "close_to_kma"      : 10,
    "rsi_score"         : 5,
    "rmv_score"         : 0,
    "rs_trend"          : 20,
    "ema10_touch"       : 5,
}

# Close to KMA (Key Moving Average = 10 EMA) scoring band
KMA_BAND_PCT = 0.015

# Relative Measured Volatility (RMV)
RMV_LOOKBACK         = 15
RMV_TIGHT_THRESHOLD  = 15

# Combined rank weights — how Momentum Score and RS Rating are blended for final rank
# Must sum to 100.
COMBINED_RANK_WEIGHTS = {
    "score"    : 40,
    "rs_rating": 60,
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

# ---------------------------------------------------------------------------
# NSE Indices — tracked separately in the Indices tab
# ---------------------------------------------------------------------------

# yfinance tickers for all NSE indices, grouped by category.
# Indices with no confirmed yfinance ticker are included with best-guess
# symbols; fetch failures are silently skipped.
NSE_INDEX_TICKERS: dict[str, dict[str, str]] = {
    "Broad Market": {
        "Nifty 50":              "^NSEI",
        "Nifty Next 50":         "^NFNXT50",
        "Nifty 100":             "^CNX100",
        "Nifty 200":             "^CNX200",
        "Nifty 500":             "^CRSLDX",
        "Nifty Midcap 50":       "^NIFMID50",
        "Nifty Midcap 100":      "^CNXMC",
        "Nifty Midcap 150":      "^NIFMDCP150",
        "Nifty Smallcap 50":     "^NSMCAP50",
        "Nifty Smallcap 100":    "^CNXSC",
        "Nifty Smallcap 250":    "^NSMCAP250",
        "Nifty MidSmallcap 400": "^NSMIDCP",
        "Nifty LargeMidcap 250": "^NIFTYLARGMID250",
    },
    "Sectoral": {
        "Nifty Bank":               "^NSEBANK",
        "Nifty Auto":               "^CNXAUTO",
        "Nifty Financial Services": "^CNXFINANCE",
        "Nifty FMCG":               "^CNXFMCG",
        "Nifty IT":                 "^CNXIT",
        "Nifty Media":              "^CNXMEDIA",
        "Nifty Metal":              "^CNXMETAL",
        "Nifty Oil & Gas":          "^CNXOIL",
        "Nifty Pharma":             "^CNXPHARMA",
        "Nifty PSU Bank":           "^CNXPSUBANK",
        "Nifty Private Bank":       "^NIFPVTBANK",
        "Nifty Realty":             "^CNXREALTY",
        "Nifty Energy":             "^CNXENERGY",
    },
    "Thematic": {
        "Nifty Capital Markets":             "^NIFCAPMKT",
        "Nifty Commodities":                 "^CNXCMDT",
        "Nifty Core Housing":                "^NIFCOREHS",
        "Nifty CPSE":                        "^CNXCPSE",
        "Nifty EV & New Age Automotive":     "^NIFEV",
        "Nifty Housing":                     "^NIFHOUSING",
        "Nifty India Consumption":           "^CNXCONSUM",
        "Nifty India Defence":               "^NIFINDDEF",
        "Nifty India Digital":               "^NIFINDDIG",
        "Nifty India Infrastructure & Logistics": "^NIFINFLOGI",
        "Nifty India Internet":              "^NIFINDINT",
        "Nifty India Manufacturing":         "^NIFINDMFG",
        "Nifty India New Age Consumption":   "^NIFINNAC",
        "Nifty India Railways PSU":          "^NIFRAILPSU",
        "Nifty India Tourism":               "^NIFINDTOU",
        "Nifty Infrastructure":              "^CNXINFRA",
        "Nifty IPO":                         "^NIFIPO",
        "Nifty MNC":                         "^CNXMNC",
        "Nifty Mobility":                    "^NIFMOBIL",
        "Nifty PSE":                         "^CNXPSE",
        "Nifty REITs & InvITs":              "^NIFREIT",
        "Nifty Rural":                       "^NIFRURAL",
        "Nifty Non-Cyclical Consumer":       "^NIFNCYCCON",
        "Nifty Services Sector":             "^CNXSERVICE",
        "Nifty Transportation & Logistics":  "^NIFTRL",
        "Nifty SME Emerge":                  "^NIFSMEEMERGE",
        "Nifty Healthcare Index":            "^CNXPHARMA",
    },
}

# Thematic indices fetched via nselib (NSE India) — not reliably available on yfinance.
# These supplement NSE_INDEX_TICKERS and are fetched separately in fetch_thematic_index_ohlcv().
NSE_THEMATIC_INDICES: list[str] = [
    "Nifty Capital Markets",
    "Nifty Commodities",
    "Nifty Core Housing",
    "Nifty CPSE",
    "Nifty Energy",
    "Nifty EV & New Age Automotive",
    "Nifty Housing",
    "Nifty India Consumption",
    "Nifty India Defence",
    "Nifty India Digital",
    "Nifty India Infrastructure & Logistics",
    "Nifty India Internet",
    "Nifty India Manufacturing",
    "Nifty India New Age Consumption",
    "Nifty India Railways PSU",
    "Nifty India Tourism",
    "Nifty Infrastructure",
    "Nifty IPO",
    "Nifty MNC",
    "Nifty Mobility",
    "Nifty PSE",
    "Nifty REITs & InvITs",
    "Nifty Services Sector",
    "Nifty Transportation & Logistics",
    "Nifty SME Emerge",
    "Nifty Healthcare Index",
]

# Cache files for index screener
INDEX_OHLCV_FILE      = os.path.join(DATA_DIR, "index_ohlcv.pkl")
INDEX_RESULTS_FILE    = os.path.join(DATA_DIR, "index_results.pkl")
INDEX_SNAPSHOTS_FILE  = os.path.join(DATA_DIR, "index_snapshots.pkl")

# EMA stack hard filter for indices (10 > 20 > 50 must all be true)
INDEX_EMA_STACK_FILTER = True

# Composite score weights for indices (must sum to 100).
# No sector outperformance — those 15 pts are added to nifty500_outperf.
INDEX_SCORE_WEIGHTS = {
    "52w_high_proximity": 10,
    "nifty500_outperf"  : 30,
    "close_to_kma"      : 5,
    "rsi_score"         : 15,
    "rmv_score"         : 5,
    "rs_trend"          : 35,
}

# Symmetrical Triangle pattern
SYMTRI_LOOKBACK = 90
SYMTRI_MIN_TOUCHES = 3
SYMTRI_SWING_WINDOW = 3
SYMTRI_MIN_BARS_TO_APEX = 5
SYMTRI_MIN_SPAN_BARS = 15          # Minimum bars between first and last pivot (pattern must develop)
SYMTRI_MIN_START_WIDTH_PCT = 0.03  # Minimum starting width as % of price (filters noise in tight ranges)
SYMTRI_R2_THRESHOLD = 0.85
SYMTRI_SLOPE_SYMMETRY_MIN = 0.5
SYMTRI_SLOPE_SYMMETRY_MAX = 2.0
SYMTRI_REQUIRE_VOL_CONTRACTION = True

# Ascending Triangle pattern
ASCTRI_LOOKBACK = 90               # Lookback window in bars (covers ideal 40–90-bar formation)
ASCTRI_MIN_TOUCHES_RESIST = 2      # Minimum flat-top resistance touches
ASCTRI_MIN_TOUCHES_SUPPORT = 2     # Minimum rising-bottom support touches
ASCTRI_SWING_WINDOW = 3            # Bars each side a pivot must dominate
ASCTRI_MIN_SPAN_BARS = 25          # Minimum pivot span (< 25 bars = noise)
ASCTRI_MAX_SPAN_BARS = 150         # Maximum pivot span
ASCTRI_RESIST_MAX_SLOPE_PCT = 0.001   # Resistance |slope| < 0.1% of price per bar (near-flat)
ASCTRI_SUPPORT_MIN_SLOPE_PCT = 0.0003 # Rising support slope > 0.03% of price per bar
ASCTRI_SUPPORT_R2 = 0.65           # Minimum R² for support trendline fit
ASCTRI_VOL_END_RATIO = 0.70        # Volume in final 10 bars < 70% of first 10 bars
ASCTRI_RSI_MIN = 45                # RSI lower bound during formation
ASCTRI_RSI_MAX = 70                # RSI upper bound during formation
ASCTRI_REQUIRE_ABOVE_200DMA = True # Pattern must form above the 200-day EMA

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
    "Sym Triangle",
    "Asc Triangle",
]

# IPO Base pattern
IPO_BASE_MAX_DAYS = 199
IPO_BASE_MIN_DAYS = 5
IPO_BASE_SKIP_DAYS = 5
IPO_BASE_RANGE_MAX = 0.6
IPO_BREAKOUT_LOWER = -0.05
IPO_BREAKOUT_UPPER = 0.1
IPO_VOL_PICKUP_RATIO = 1.0

# Cache file for IPO screener results
IPO_RESULTS_FILE    = os.path.join(DATA_DIR, "ipo_results.pkl")
IPO_IB_RESULTS_FILE = os.path.join(DATA_DIR, "ipo_ib_results.pkl")

# ---------------------------------------------------------------------------
# MCX Commodities — tracked separately in the Commodities tab
# ---------------------------------------------------------------------------

# Yahoo Finance proxy tickers for MCX futures contracts.
# Prices are international (COMEX / NYMEX / LME) but highly correlated with
# MCX; LME tickers (^LMNI*, ^LMPB*, ^LMZS*) are silently skipped if unavailable.
MCX_COMMODITY_TICKERS: dict[str, str] = {
    "Gold Mini":         "GC=F",          # COMEX Gold
    "Silver Mini":       "SI=F",          # COMEX Silver
    "Aluminium Mini":    "ALI=F",         # COMEX Aluminium
    "Copper":            "HG=F",          # COMEX Copper
    "Nickel Mini":       "^LMNIBS03",     # LME Nickel 3-month
    "Lead Mini":         "^LMPBBS03",     # LME Lead 3-month
    "Zinc Mini":         "^LMZSBS03",     # LME Zinc 3-month
    "Crude Mini":        "CL=F",          # NYMEX WTI Crude Oil
    "Natural Gas Mini":  "NG=F",          # NYMEX Natural Gas
}

# Category grouping for display
MCX_COMMODITY_CATEGORY: dict[str, str] = {
    "Gold Mini":        "Precious Metals",
    "Silver Mini":      "Precious Metals",
    "Aluminium Mini":   "Base Metals",
    "Copper":           "Base Metals",
    "Nickel Mini":      "Base Metals",
    "Lead Mini":        "Base Metals",
    "Zinc Mini":        "Base Metals",
    "Crude Mini":       "Energy",
    "Natural Gas Mini": "Energy",
}

# RS benchmarks for commodities
COMMODITY_NIFTY500_TICKERS  = ["^CRSLDX", "^NSEI"]   # General RS benchmark
COMMODITY_METAL_TICKERS     = ["^CNXMETAL"]           # Sectoral RS benchmark

# Composite score weights for commodities (must sum to 100)
COMMODITY_SCORE_WEIGHTS: dict[str, int] = {
    "52w_high_proximity":  20,   # Proximity to 52-week high
    "nifty500_outperf":    20,   # 1M outperf vs Nifty 500
    "metal_index_outperf": 20,   # 1M outperf vs Nifty Metal Index
    "rsi_score":           15,   # RSI normalised 55→80 band
    "close_to_kma":         5,   # Proximity within ±1.5% of 10 EMA
    "rmv_score":            5,   # Lower RMV = higher score
    "rs_trend":            15,   # RS line EMA10 > EMA20 → uptrend
}

# Cache files for commodity screener
COMMODITY_OHLCV_FILE      = os.path.join(DATA_DIR, "commodity_ohlcv.pkl")
COMMODITY_RESULTS_FILE    = os.path.join(DATA_DIR, "commodity_results.pkl")
COMMODITY_SNAPSHOTS_FILE  = os.path.join(DATA_DIR, "commodity_snapshots.pkl")

# Scheduler: commodity refresh at 23:45 IST (after MCX closes at 23:30)
COMMODITY_REFRESH_HOUR_IST   = 23
COMMODITY_REFRESH_MINUTE_IST = 45

# ---------------------------------------------------------------------------
# NSE Free Float cache
# ---------------------------------------------------------------------------

# Free float % sourced from NSE's official shareholding pattern API.
# Daily stock fetch: only missing tickers are fetched (incremental).
# Full refresh: runs automatically once a week on the configured day.
FREEFLOAT_FILE       = os.path.join(DATA_DIR, "freefloat_cache.pkl")
FREEFLOAT_CACHE_DAYS = 7   # kept for reference; weekly job ignores this and always refreshes

# Weekly full-refresh schedule (APScheduler CronTrigger values, IST timezone)
# "sat" = Saturday; change to "sun" for Sunday.
FREEFLOAT_REFRESH_DAY        = 'sat'
FREEFLOAT_REFRESH_HOUR_IST   = 10
FREEFLOAT_REFRESH_MINUTE_IST = 0

# Number of parallel fetch threads.  3 is conservative and safe for the NSE API;
# raise to 5 if you want a further speedup and don't observe throttling.
FREEFLOAT_MAX_WORKERS = 3
