"""
Integration tests for the NSE sector map feature.

These tests hit the live NSE API — they require internet access and are
intentionally slow (0.8 s/ticker).  Run with:

    pytest tests/integration/test_nse_sector_integration.py -v -m integration

Mark with:  pytest.ini  →  markers = integration: live-network tests
"""

import os
import pickle
import time

import pytest
import requests

from src.nse_fetcher import fetch_nse_sector_map
from src.data_fetcher import enrich_with_nse_sectors


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sector_map_cache(tmp_path, monkeypatch):
    """Redirect cache writes to a temp directory."""
    cache_file = str(tmp_path / "nse_sector_map.pkl")
    monkeypatch.setattr("src.nse_fetcher.NSE_SECTOR_MAP_FILE", cache_file)
    monkeypatch.setattr("src.nse_fetcher.DATA_DIR", str(tmp_path))
    return cache_file


# ---------------------------------------------------------------------------
# Live API — known liquid NSE stocks (should always return industryInfo)
# ---------------------------------------------------------------------------

KNOWN_TICKERS = ["RELIANCE.NS", "INFY.NS", "HDFCBANK.NS"]


@pytest.mark.integration
class TestLiveNseApiResponse:

    def test_api_reachable(self):
        """Basic connectivity check — warm-up endpoint must return 200."""
        try:
            resp = requests.get(
                "https://www.nseindia.com/",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15,
            )
            assert resp.status_code == 200
        except requests.RequestException as exc:
            pytest.skip(f"NSE unreachable: {exc}")

    def test_returns_dict_for_known_tickers(self, sector_map_cache):
        result = fetch_nse_sector_map(KNOWN_TICKERS, force_refresh=True)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_each_result_has_required_keys(self, sector_map_cache):
        result = fetch_nse_sector_map(KNOWN_TICKERS, force_refresh=True)
        for ticker in KNOWN_TICKERS:
            if ticker in result:
                entry = result[ticker]
                assert "sector"         in entry, f"Missing 'sector' for {ticker}"
                assert "industry"       in entry, f"Missing 'industry' for {ticker}"
                assert "basic_industry" in entry, f"Missing 'basic_industry' for {ticker}"

    def test_sector_values_are_non_empty_strings(self, sector_map_cache):
        result = fetch_nse_sector_map(KNOWN_TICKERS, force_refresh=True)
        for ticker, entry in result.items():
            assert isinstance(entry["sector"], str) and entry["sector"], \
                f"Blank/None sector for {ticker}"

    def test_reliance_sector_is_financial_or_energy(self, sector_map_cache):
        """RELIANCE is classified under Energy or Financial Services by NSE."""
        result = fetch_nse_sector_map(["RELIANCE.NS"], force_refresh=True)
        if "RELIANCE.NS" in result:
            sector = result["RELIANCE.NS"]["sector"]
            assert sector in ("Energy", "Financial Services", "Basic Materials"), \
                f"Unexpected sector for RELIANCE: {sector}"

    def test_infy_sector_is_technology(self, sector_map_cache):
        result = fetch_nse_sector_map(["INFY.NS"], force_refresh=True)
        if "INFY.NS" in result:
            assert result["INFY.NS"]["sector"] == "Technology"

    def test_hdfcbank_sector_is_financial_services(self, sector_map_cache):
        result = fetch_nse_sector_map(["HDFCBANK.NS"], force_refresh=True)
        if "HDFCBANK.NS" in result:
            assert result["HDFCBANK.NS"]["sector"] == "Financial Services"


# ---------------------------------------------------------------------------
# Cache persistence
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestCachePersistence:

    def test_cache_file_created_after_fetch(self, sector_map_cache):
        fetch_nse_sector_map(["INFY.NS"], force_refresh=True)
        assert os.path.exists(sector_map_cache)

    def test_cached_data_loadable_as_pickle(self, sector_map_cache):
        fetch_nse_sector_map(["INFY.NS"], force_refresh=True)
        with open(sector_map_cache, "rb") as f:
            data = pickle.load(f)
        assert isinstance(data, dict)

    def test_second_call_uses_cache_not_network(self, sector_map_cache, monkeypatch):
        """After a fresh fetch, the same tickers should be served from cache."""
        monkeypatch.setattr("src.nse_fetcher.NSE_SECTOR_CACHE_DAYS", 30)

        # First call populates cache
        fetch_nse_sector_map(["INFY.NS"], force_refresh=True)

        # Second call should not hit the network (no session created)
        from unittest.mock import patch
        with patch("src.nse_fetcher._make_nse_session") as mock_fn:
            result = fetch_nse_sector_map(["INFY.NS"])

        mock_fn.assert_not_called()
        assert "INFY.NS" in result


# ---------------------------------------------------------------------------
# End-to-end enrichment: fetch → enrich → verify improvement
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestEnrichmentEndToEnd:

    def test_unknown_sectors_reduced_after_enrichment(self, sector_map_cache):
        """
        Given a stock_info dict where known NSE stocks are labelled 'Unknown',
        enrichment via NSE sector map must reduce the Unknown count.
        """
        tickers = ["INFY.NS", "HDFCBANK.NS"]
        stock_info = {t: {"sector": "Unknown", "industry": "Unknown"} for t in tickers}

        nse_map    = fetch_nse_sector_map(tickers, force_refresh=True)
        enriched   = enrich_with_nse_sectors(stock_info, nse_map)

        unknown_after = sum(
            1 for info in enriched.values()
            if info.get("sector") in ("Unknown", None, "")
        )
        assert unknown_after < len(tickers), \
            "Enrichment did not fix any Unknown sectors for known NSE stocks"

    def test_known_sectors_not_overwritten_by_enrichment(self, sector_map_cache):
        """Stocks that already have a known sector must not be changed."""
        tickers = ["INFY.NS"]
        stock_info = {"INFY.NS": {"sector": "Technology", "industry": "IT Services"}}

        nse_map   = fetch_nse_sector_map(tickers, force_refresh=True)
        enriched  = enrich_with_nse_sectors(stock_info, nse_map)

        assert enriched["INFY.NS"]["sector"] == "Technology"
