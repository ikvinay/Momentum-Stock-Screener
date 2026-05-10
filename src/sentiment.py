"""
Sentiment analysis for screener-passing stocks.

Sources (all free, no API keys required):
  1. yfinance .news      — recent headlines per ticker
  2. Google News RSS     — broader news coverage for Indian stocks
  3. Reddit JSON API     — r/IndianStockMarket mention count + post titles
                          (uses unauthenticated public endpoint)

Scoring: VADER compound score per headline, averaged across all sources.
  >= +0.05  → Bullish
  <= -0.05  → Bearish
  otherwise → Neutral

Cache: data/sentiment.pkl — refreshed every SENTIMENT_CACHE_HOURS hours.
"""

import logging
import os
import pickle
import time
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import requests
import yfinance as yf

from config import DATA_DIR

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
SENTIMENT_FILE        = os.path.join(DATA_DIR, "sentiment.pkl")
SENTIMENT_CACHE_HOURS = 6

YF_MAX_ARTICLES    = 10
GOOGLE_MAX_ARTICLES = 8
REDDIT_MAX_POSTS   = 25
REQUEST_DELAY      = 0.4     # seconds between outbound calls
REQUEST_TIMEOUT    = 8

BULLISH_THRESHOLD =  0.05
BEARISH_THRESHOLD = -0.05

_HEADERS = {
    "User-Agent": "Mozilla/5.0 NSEMomentumScreener/1.0 (research)"
}

# ── VADER (lazy-loaded) ───────────────────────────────────────────────────────
_analyzer = None


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def _vader(text: str) -> float:
    """Return VADER compound score for a headline string."""
    if not text:
        return 0.0
    return _get_analyzer().polarity_scores(text)["compound"]


# ── Fetchers ──────────────────────────────────────────────────────────────────

def _fetch_yf_news(ticker: str) -> List[str]:
    """Pull recent news headlines from yfinance for the given .NS ticker."""
    try:
        news = yf.Ticker(ticker).news or []
        titles = []
        for item in news[:YF_MAX_ARTICLES]:
            if not isinstance(item, dict):
                continue
            # yfinance >= 0.2.40 nests title inside 'content'
            title = (
                item.get("title")
                or (item.get("content") or {}).get("title", "")
            )
            if title:
                titles.append(title)
        return titles
    except Exception as exc:
        logger.debug("yf news failed %s: %s", ticker, exc)
        return []


def _fetch_google_news(company: str, symbol: str) -> List[str]:
    """Fetch headlines from Google News RSS for an NSE stock."""
    try:
        import feedparser
    except ImportError:
        logger.debug("feedparser not installed — skipping Google News")
        return []

    query = quote_plus(f"{company} {symbol} NSE stock")
    url   = (
        f"https://news.google.com/rss/search"
        f"?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    )
    try:
        feed = feedparser.parse(url)
        return [
            e.title
            for e in feed.entries[:GOOGLE_MAX_ARTICLES]
            if hasattr(e, "title") and e.title
        ]
    except Exception as exc:
        logger.debug("Google News failed %s: %s", symbol, exc)
        return []


def _fetch_reddit(symbol: str) -> Tuple[List[str], int]:
    """
    Search r/IndianStockMarket for recent posts mentioning the symbol.
    Returns (post_titles, mention_count).  No API key required.
    """
    url = (
        f"https://www.reddit.com/r/IndianStockMarket/search.json"
        f"?q={symbol}&sort=new&restrict_sr=1&limit={REDDIT_MAX_POSTS}&t=week"
    )
    try:
        resp  = requests.get(url, headers=_HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        posts = resp.json().get("data", {}).get("children", [])
        titles = [
            p["data"].get("title", "")
            for p in posts
            if p.get("data") and p["data"].get("title")
        ]
        return titles, len(titles)
    except Exception as exc:
        logger.debug("Reddit failed %s: %s", symbol, exc)
        return [], 0


# ── Core computation ──────────────────────────────────────────────────────────

def compute_sentiment(symbol: str, company: str) -> Dict:
    """
    Aggregate sentiment for one stock from all free sources.

    Returns
    -------
    dict with keys:
      score           float   VADER compound avg (-1 to +1)
      label           str     "Bullish" | "Neutral" | "Bearish"
      article_count   int     total unique headlines analysed
      reddit_mentions int     number of Reddit posts this week
      headlines       list    up to 10 headline strings
      fetched_at      datetime
    """
    ticker = f"{symbol}.NS"

    yf_titles      = _fetch_yf_news(ticker)
    time.sleep(REQUEST_DELAY)
    google_titles  = _fetch_google_news(company, symbol)
    time.sleep(REQUEST_DELAY)
    reddit_titles, reddit_mentions = _fetch_reddit(symbol)

    # Deduplicate while preserving order
    seen, all_titles = set(), []
    for t in yf_titles + google_titles + reddit_titles:
        t = t.strip()
        if t and t not in seen:
            seen.add(t)
            all_titles.append(t)

    if not all_titles:
        return {
            "score":           0.0,
            "label":           "Neutral",
            "article_count":   0,
            "reddit_mentions": reddit_mentions,
            "headlines":       [],
            "fetched_at":      datetime.utcnow(),
        }

    scores    = [_vader(t) for t in all_titles]
    avg_score = sum(scores) / len(scores)

    label = (
        "Bullish" if avg_score >= BULLISH_THRESHOLD else
        "Bearish" if avg_score <= BEARISH_THRESHOLD else
        "Neutral"
    )

    return {
        "score":           round(avg_score, 3),
        "label":           label,
        "article_count":   len(all_titles),
        "reddit_mentions": reddit_mentions,
        "headlines":       list(zip(all_titles[:10], [round(s, 2) for s in scores[:10]])),
        "fetched_at":      datetime.utcnow(),
    }


# ── Cache helpers ─────────────────────────────────────────────────────────────

def load_sentiment() -> Dict[str, Dict]:
    if os.path.exists(SENTIMENT_FILE):
        try:
            with open(SENTIMENT_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            os.remove(SENTIMENT_FILE)
    return {}


def save_sentiment(data: Dict[str, Dict]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SENTIMENT_FILE, "wb") as f:
        pickle.dump(data, f)


def _is_fresh(entry: Optional[Dict]) -> bool:
    if not entry:
        return False
    age = datetime.utcnow() - entry.get("fetched_at", datetime.min)
    return age < timedelta(hours=SENTIMENT_CACHE_HOURS)


# ── Batch processing ──────────────────────────────────────────────────────────

def batch_sentiment(
    symbols: List[str],
    companies: List[str],
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Dict]:
    """
    Compute and cache sentiment for a list of stocks.
    Uses cached results for any stock whose entry is still fresh.
    Saves after each fetch so partial progress is never lost.

    progress_cb(i, total, symbol) — optional callback after each stock.
    """
    cache = load_sentiment()
    total = len(symbols)

    for i, (symbol, company) in enumerate(zip(symbols, companies)):
        if _is_fresh(cache.get(symbol)):
            logger.debug("Sentiment cache hit: %s", symbol)
        else:
            logger.info("Sentiment [%d/%d]: %s", i + 1, total, symbol)
            try:
                cache[symbol] = compute_sentiment(symbol, company)
                save_sentiment(cache)
            except Exception as exc:
                logger.warning("Sentiment failed %s: %s", symbol, exc)

        if progress_cb:
            progress_cb(i + 1, total, symbol)

    return cache
