"""
Integration tests — live HTTP calls to the NSE shareholding pattern API.

Requires a working internet connection and access to nseindia.com from the
machine running the tests (local machines pass; datacenter IPs may be blocked
by Akamai on the homepage, but the API endpoint itself is not protected).

Tests are skipped automatically when the API is unreachable so that CI
environments that lack NSE access don't produce false failures.

Run manually with:
    pytest tests/integration/test_nse_freefloat_integration.py -v
"""

import pytest

from src.nse_freefloat import _build_session, fetch_freefloat, fetch_all_freefloat


# ---------------------------------------------------------------------------
# Reachability guard — skip the whole module if NSE API is unreachable
# ---------------------------------------------------------------------------

def _nse_api_reachable() -> bool:
    try:
        s = _build_session()
        r = s.get(
            "https://www.nseindia.com/api/corporate-share-holdings-master"
            "?index=equities&symbol=INFY",
            timeout=10,
        )
        return r.status_code == 200 and isinstance(r.json(), list)
    except Exception:
        return False


nse_available = pytest.mark.skipif(
    not _nse_api_reachable(),
    reason="NSE shareholding API not reachable from this environment",
)


# ---------------------------------------------------------------------------
# fetch_freefloat — live API, known promoter benchmarks
# ---------------------------------------------------------------------------

class TestFetchFreefloatLive:

    @nse_available
    def test_reliance_float_near_50_pct(self):
        """Mukesh Ambani group holds ~50 % → public float ~50 %."""
        ff = fetch_freefloat("RELIANCE", _build_session())
        assert ff is not None
        assert 40.0 <= ff <= 60.0, f"RELIANCE: expected ~50 %, got {ff}"

    @nse_available
    def test_wipro_float_near_27_pct(self):
        """Azim Premji group holds ~73 % → public float ~27 %."""
        ff = fetch_freefloat("WIPRO", _build_session())
        assert ff is not None
        assert 20.0 <= ff <= 40.0, f"WIPRO: expected 20-40 %, got {ff}"

    @nse_available
    def test_irfc_float_low_due_to_govt_holding(self):
        """Indian Railways Finance Corp — Govt of India holds ~86 % → float ~14 %."""
        ff = fetch_freefloat("IRFC", _build_session())
        assert ff is not None
        assert 8.0 <= ff <= 25.0, f"IRFC: expected 8-25 %, got {ff}"

    @nse_available
    def test_icicibank_float_near_100_pct(self):
        """ICICI Bank has no promoter → public float approaches 100 %."""
        ff = fetch_freefloat("ICICIBANK", _build_session())
        assert ff is not None
        assert ff >= 85.0, f"ICICIBANK: expected ≥85 %, got {ff}"

    @nse_available
    def test_invalid_symbol_returns_none_not_exception(self):
        result = fetch_freefloat("THISDOESNOTEXIST99999", _build_session())
        assert result is None

    @nse_available
    def test_return_value_is_valid_percentage(self):
        for symbol in ("HDFCBANK", "BAJFINANCE", "POLYCAB"):
            ff = fetch_freefloat(symbol, _build_session())
            if ff is not None:
                assert 0.0 < ff <= 100.0, f"{symbol}: {ff} is not a valid percentage"


# ---------------------------------------------------------------------------
# fetch_all_freefloat — bulk fetch with yfinance fallback
# ---------------------------------------------------------------------------

class TestFetchAllFreefloatLive:

    @nse_available
    def test_covers_majority_of_known_large_caps(self):
        tickers = ["RELIANCE.NS", "INFY.NS", "WIPRO.NS", "HDFCBANK.NS", "BAJFINANCE.NS"]
        result = fetch_all_freefloat(tickers, yf_info={})
        assert len(result) >= 4, (
            f"Expected ≥4/5 large caps to have data, got {len(result)}: {result}"
        )

    @nse_available
    def test_all_returned_values_are_valid_percentages(self):
        tickers = ["ICICIBANK.NS", "POLYCAB.NS", "IRFC.NS"]
        result = fetch_all_freefloat(tickers, yf_info={})
        for ticker, ff in result.items():
            assert 0.0 < ff <= 100.0, f"{ticker}: {ff} is not a valid percentage"

    @nse_available
    def test_result_keys_use_yf_ticker_format(self):
        """Result dict must be keyed by the .NS ticker format, not bare NSE symbols."""
        tickers = ["RELIANCE.NS", "INFY.NS"]
        result = fetch_all_freefloat(tickers, yf_info={})
        for key in result:
            assert key.endswith(".NS"), f"Key {key!r} is not in yfinance .NS format"

    @nse_available
    def test_yfinance_fallback_covers_nse_gap(self):
        """ZOMATO and TATAMOTORS consistently return [] from NSE API.
        The yfinance heldPercentInsiders fallback should fill the gap."""
        # Simulate having yfinance data for these tickers in stock_info
        yf_info = {
            "ZOMATO.NS":    {"heldPercentInsiders": 0.10},   # → 90 % float
            "TATAMOTORS.NS": {"heldPercentInsiders": 0.42},  # → 58 % float
        }
        result = fetch_all_freefloat(
            ["ZOMATO.NS", "TATAMOTORS.NS"], yf_info=yf_info
        )
        # At least one should be covered via fallback
        covered = [t for t in ("ZOMATO.NS", "TATAMOTORS.NS") if t in result]
        assert len(covered) >= 1, "Expected at least one ticker covered via yf fallback"
        for ticker in covered:
            assert 0.0 < result[ticker] <= 100.0
