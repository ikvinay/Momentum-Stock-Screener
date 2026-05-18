"""
Unit tests for NSE sector map — all HTTP calls are mocked.
No network access required.
"""

import os
import pickle
import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.nse_fetcher import fetch_nse_sector_map
from src.data_fetcher import enrich_with_nse_sectors
from config import NSE_MACRO_TO_SECTOR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_session(json_data=None, status_code=200, raise_exc=None):
    session = MagicMock()
    resp = MagicMock()
    resp.status_code = status_code
    if raise_exc:
        session.get.side_effect = raise_exc
    else:
        resp.json.return_value = json_data
        session.get.return_value = resp
    return session


def _nse_payload(macro="Information Technology", industry="IT Services & Consulting",
                 basic="IT Consulting & Software"):
    return {
        "industryInfo": {
            "macro": macro,
            "industry": industry,
            "basicIndustry": basic,
        }
    }


# ---------------------------------------------------------------------------
# NSE_MACRO_TO_SECTOR mapping
# ---------------------------------------------------------------------------

class TestMacroToSectorMapping:

    def test_information_technology_maps_to_technology(self):
        assert NSE_MACRO_TO_SECTOR["Information Technology"] == "Technology"

    def test_financial_services_maps_to_financial_services(self):
        assert NSE_MACRO_TO_SECTOR["Financial Services"] == "Financial Services"

    def test_consumer_discretionary_maps_to_consumer_cyclical(self):
        assert NSE_MACRO_TO_SECTOR["Consumer Discretionary"] == "Consumer Cyclical"

    def test_healthcare_maps_correctly(self):
        assert NSE_MACRO_TO_SECTOR["Healthcare"] == "Healthcare"

    def test_commodities_maps_to_basic_materials(self):
        assert NSE_MACRO_TO_SECTOR["Commodities"] == "Basic Materials"

    def test_services_maps_to_communication_services(self):
        assert NSE_MACRO_TO_SECTOR["Services"] == "Communication Services"

    def test_diversified_maps_to_industrials(self):
        assert NSE_MACRO_TO_SECTOR["Diversified"] == "Industrials"

    def test_all_values_are_non_empty_strings(self):
        for macro, sector in NSE_MACRO_TO_SECTOR.items():
            assert isinstance(sector, str) and sector, f"Empty sector for macro '{macro}'"


# ---------------------------------------------------------------------------
# fetch_nse_sector_map — parsing and caching
# ---------------------------------------------------------------------------

class TestFetchNseSectorMap:

    def _run(self, tickers, json_data=None, status_code=200, raise_exc=None,
             tmp_path=None, monkeypatch=None):
        """Run fetch_nse_sector_map with mocked session and optional temp cache dir."""
        if tmp_path and monkeypatch:
            monkeypatch.setattr("src.nse_fetcher.NSE_SECTOR_MAP_FILE",
                                str(tmp_path / "nse_sector_map.pkl"))
            monkeypatch.setattr("src.nse_fetcher.DATA_DIR", str(tmp_path))

        session = _mock_session(json_data=json_data, status_code=status_code,
                                raise_exc=raise_exc)
        with patch("src.nse_fetcher._make_nse_session", return_value=session), \
             patch("src.nse_fetcher.time.sleep"):
            return fetch_nse_sector_map(tickers, force_refresh=True)

    def test_sector_mapped_from_macro(self, tmp_path, monkeypatch):
        result = self._run(
            ["INFY.NS"],
            json_data=_nse_payload("Information Technology"),
            tmp_path=tmp_path, monkeypatch=monkeypatch,
        )
        assert result["INFY.NS"]["sector"] == "Technology"

    def test_industry_returned_as_is(self, tmp_path, monkeypatch):
        result = self._run(
            ["INFY.NS"],
            json_data=_nse_payload(industry="IT Services & Consulting"),
            tmp_path=tmp_path, monkeypatch=monkeypatch,
        )
        assert result["INFY.NS"]["industry"] == "IT Services & Consulting"

    def test_basic_industry_included(self, tmp_path, monkeypatch):
        result = self._run(
            ["INFY.NS"],
            json_data=_nse_payload(basic="IT Consulting & Software"),
            tmp_path=tmp_path, monkeypatch=monkeypatch,
        )
        assert result["INFY.NS"]["basic_industry"] == "IT Consulting & Software"

    def test_ns_suffix_stripped_in_api_url(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.nse_fetcher.NSE_SECTOR_MAP_FILE",
                            str(tmp_path / "nse_sector_map.pkl"))
        monkeypatch.setattr("src.nse_fetcher.DATA_DIR", str(tmp_path))
        session = _mock_session(json_data=_nse_payload())
        with patch("src.nse_fetcher._make_nse_session", return_value=session), \
             patch("src.nse_fetcher.time.sleep"):
            fetch_nse_sector_map(["INFY.NS"], force_refresh=True)
        url = session.get.call_args[0][0]
        assert ".NS" not in url
        assert "INFY" in url

    def test_result_keyed_by_yf_ticker_with_ns(self, tmp_path, monkeypatch):
        result = self._run(
            ["RELIANCE.NS"],
            json_data=_nse_payload("Financial Services"),
            tmp_path=tmp_path, monkeypatch=monkeypatch,
        )
        assert "RELIANCE.NS" in result
        assert "RELIANCE" not in result

    def test_unknown_macro_falls_back_to_macro_string(self, tmp_path, monkeypatch):
        """Macros not in the mapping are returned verbatim (better than 'Unknown')."""
        result = self._run(
            ["X.NS"],
            json_data=_nse_payload("FuturisticSector"),
            tmp_path=tmp_path, monkeypatch=monkeypatch,
        )
        assert result["X.NS"]["sector"] == "FuturisticSector"

    def test_empty_macro_becomes_unknown(self, tmp_path, monkeypatch):
        result = self._run(
            ["X.NS"],
            json_data={"industryInfo": {"macro": "", "industry": "", "basicIndustry": ""}},
            tmp_path=tmp_path, monkeypatch=monkeypatch,
        )
        assert result["X.NS"]["sector"] == "Unknown"

    def test_http_error_skips_ticker_gracefully(self, tmp_path, monkeypatch):
        result = self._run(
            ["BAD.NS"],
            status_code=404,
            tmp_path=tmp_path, monkeypatch=monkeypatch,
        )
        assert "BAD.NS" not in result

    def test_network_exception_skips_ticker_gracefully(self, tmp_path, monkeypatch):
        result = self._run(
            ["BAD.NS"],
            raise_exc=requests.ConnectionError("timeout"),
            tmp_path=tmp_path, monkeypatch=monkeypatch,
        )
        assert "BAD.NS" not in result

    def test_partial_failure_does_not_abort_rest(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.nse_fetcher.NSE_SECTOR_MAP_FILE",
                            str(tmp_path / "nse_sector_map.pkl"))
        monkeypatch.setattr("src.nse_fetcher.DATA_DIR", str(tmp_path))

        responses = [
            {"status_code": 404, "json": None},
            {"status_code": 200, "json": _nse_payload("Information Technology")},
        ]
        call_count = [0]

        def side_effect(*args, **kwargs):
            idx = call_count[0] % len(responses)
            call_count[0] += 1
            r = responses[idx]
            resp = MagicMock()
            resp.status_code = r["status_code"]
            resp.json.return_value = r["json"]
            return resp

        session = MagicMock()
        session.get.side_effect = side_effect

        with patch("src.nse_fetcher._make_nse_session", return_value=session), \
             patch("src.nse_fetcher.time.sleep"):
            result = fetch_nse_sector_map(["FAIL.NS", "INFY.NS"], force_refresh=True)

        assert "INFY.NS" in result

    def test_empty_input_returns_empty_dict(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.nse_fetcher.NSE_SECTOR_MAP_FILE",
                            str(tmp_path / "nse_sector_map.pkl"))
        monkeypatch.setattr("src.nse_fetcher.DATA_DIR", str(tmp_path))
        with patch("src.nse_fetcher._make_nse_session"), \
             patch("src.nse_fetcher.time.sleep"):
            result = fetch_nse_sector_map([], force_refresh=True)
        assert result == {}

    def test_sleep_called_per_ticker(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.nse_fetcher.NSE_SECTOR_MAP_FILE",
                            str(tmp_path / "nse_sector_map.pkl"))
        monkeypatch.setattr("src.nse_fetcher.DATA_DIR", str(tmp_path))
        session = _mock_session(json_data=_nse_payload())
        with patch("src.nse_fetcher._make_nse_session", return_value=session), \
             patch("src.nse_fetcher.time.sleep") as mock_sleep:
            fetch_nse_sector_map(["A.NS", "B.NS", "C.NS"], force_refresh=True)
        assert mock_sleep.call_count == 3


# ---------------------------------------------------------------------------
# fetch_nse_sector_map — cache roundtrip
# ---------------------------------------------------------------------------

class TestNseSectorMapCache:

    def test_cache_is_written_after_fetch(self, tmp_path, monkeypatch):
        cache_file = str(tmp_path / "nse_sector_map.pkl")
        monkeypatch.setattr("src.nse_fetcher.NSE_SECTOR_MAP_FILE", cache_file)
        monkeypatch.setattr("src.nse_fetcher.DATA_DIR", str(tmp_path))

        session = _mock_session(json_data=_nse_payload())
        with patch("src.nse_fetcher._make_nse_session", return_value=session), \
             patch("src.nse_fetcher.time.sleep"):
            fetch_nse_sector_map(["INFY.NS"], force_refresh=True)

        assert os.path.exists(cache_file)
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        assert "INFY.NS" in cached

    def test_fresh_cache_skips_network(self, tmp_path, monkeypatch):
        cache_file = str(tmp_path / "nse_sector_map.pkl")
        monkeypatch.setattr("src.nse_fetcher.NSE_SECTOR_MAP_FILE", cache_file)
        monkeypatch.setattr("src.nse_fetcher.DATA_DIR", str(tmp_path))
        monkeypatch.setattr("src.nse_fetcher.NSE_SECTOR_CACHE_DAYS", 30)

        existing = {"INFY.NS": {"sector": "Technology", "industry": "IT", "basic_industry": "SW"}}
        with open(cache_file, "wb") as f:
            pickle.dump(existing, f)

        with patch("src.nse_fetcher._make_nse_session") as mock_session_fn, \
             patch("src.nse_fetcher.time.sleep"):
            result = fetch_nse_sector_map(["INFY.NS"])

        mock_session_fn.assert_not_called()
        assert result["INFY.NS"]["sector"] == "Technology"

    def test_stale_cache_triggers_refetch(self, tmp_path, monkeypatch):
        cache_file = str(tmp_path / "nse_sector_map.pkl")
        monkeypatch.setattr("src.nse_fetcher.NSE_SECTOR_MAP_FILE", cache_file)
        monkeypatch.setattr("src.nse_fetcher.DATA_DIR", str(tmp_path))
        monkeypatch.setattr("src.nse_fetcher.NSE_SECTOR_CACHE_DAYS", 30)

        old_data = {"INFY.NS": {"sector": "OLD", "industry": "OLD", "basic_industry": ""}}
        with open(cache_file, "wb") as f:
            pickle.dump(old_data, f)
        # Age the file to 31 days
        old_time = time.time() - (31 * 86400)
        os.utime(cache_file, (old_time, old_time))

        session = _mock_session(json_data=_nse_payload("Information Technology"))
        with patch("src.nse_fetcher._make_nse_session", return_value=session), \
             patch("src.nse_fetcher.time.sleep"):
            result = fetch_nse_sector_map(["INFY.NS"])

        assert result["INFY.NS"]["sector"] == "Technology"

    def test_corrupt_cache_is_discarded_and_refetched(self, tmp_path, monkeypatch):
        cache_file = str(tmp_path / "nse_sector_map.pkl")
        monkeypatch.setattr("src.nse_fetcher.NSE_SECTOR_MAP_FILE", cache_file)
        monkeypatch.setattr("src.nse_fetcher.DATA_DIR", str(tmp_path))

        with open(cache_file, "wb") as f:
            f.write(b"not valid pickle!!!")

        session = _mock_session(json_data=_nse_payload("Information Technology"))
        with patch("src.nse_fetcher._make_nse_session", return_value=session), \
             patch("src.nse_fetcher.time.sleep"):
            result = fetch_nse_sector_map(["INFY.NS"], force_refresh=False)

        assert "INFY.NS" in result

    def test_missing_tickers_fetched_incrementally(self, tmp_path, monkeypatch):
        """Tickers already in cache are not re-fetched."""
        cache_file = str(tmp_path / "nse_sector_map.pkl")
        monkeypatch.setattr("src.nse_fetcher.NSE_SECTOR_MAP_FILE", cache_file)
        monkeypatch.setattr("src.nse_fetcher.DATA_DIR", str(tmp_path))
        monkeypatch.setattr("src.nse_fetcher.NSE_SECTOR_CACHE_DAYS", 30)

        existing = {"INFY.NS": {"sector": "Technology", "industry": "IT", "basic_industry": ""}}
        with open(cache_file, "wb") as f:
            pickle.dump(existing, f)

        session = _mock_session(json_data=_nse_payload("Financial Services"))
        with patch("src.nse_fetcher._make_nse_session", return_value=session), \
             patch("src.nse_fetcher.time.sleep"):
            result = fetch_nse_sector_map(["INFY.NS", "RELIANCE.NS"])

        # INFY from cache (unchanged), RELIANCE freshly fetched
        assert result["INFY.NS"]["sector"] == "Technology"
        assert result["RELIANCE.NS"]["sector"] == "Financial Services"
        # Only one network call was made (for RELIANCE)
        assert session.get.call_count == 1


# ---------------------------------------------------------------------------
# enrich_with_nse_sectors
# ---------------------------------------------------------------------------

class TestEnrichWithNseSectors:

    def _stock(self, sector="Unknown", industry="Unknown"):
        return {"sector": sector, "industry": industry}

    def _nse_entry(self, sector="Technology", industry="IT Services"):
        return {"sector": sector, "industry": industry, "basic_industry": "SW"}

    def test_unknown_sector_overridden(self):
        stock_info = {"INFY.NS": self._stock(sector="Unknown")}
        nse_map    = {"INFY.NS": self._nse_entry(sector="Technology")}
        result = enrich_with_nse_sectors(stock_info, nse_map)
        assert result["INFY.NS"]["sector"] == "Technology"

    def test_unknown_industry_overridden(self):
        stock_info = {"INFY.NS": self._stock(industry="Unknown")}
        nse_map    = {"INFY.NS": self._nse_entry(industry="IT Services")}
        result = enrich_with_nse_sectors(stock_info, nse_map)
        assert result["INFY.NS"]["industry"] == "IT Services"

    def test_known_sector_not_overridden(self):
        stock_info = {"TCS.NS": self._stock(sector="Technology")}
        nse_map    = {"TCS.NS": self._nse_entry(sector="Information Technology")}
        result = enrich_with_nse_sectors(stock_info, nse_map)
        assert result["TCS.NS"]["sector"] == "Technology"

    def test_none_sector_treated_as_unknown(self):
        stock_info = {"X.NS": {"sector": None, "industry": "Unknown"}}
        nse_map    = {"X.NS": self._nse_entry(sector="Healthcare")}
        result = enrich_with_nse_sectors(stock_info, nse_map)
        assert result["X.NS"]["sector"] == "Healthcare"

    def test_empty_string_sector_treated_as_unknown(self):
        stock_info = {"X.NS": {"sector": "", "industry": ""}}
        nse_map    = {"X.NS": self._nse_entry(sector="Energy", industry="Oil & Gas")}
        result = enrich_with_nse_sectors(stock_info, nse_map)
        assert result["X.NS"]["sector"] == "Energy"
        assert result["X.NS"]["industry"] == "Oil & Gas"

    def test_ticker_absent_from_nse_map_left_unchanged(self):
        stock_info = {"MISS.NS": self._stock(sector="Unknown")}
        nse_map    = {}
        result = enrich_with_nse_sectors(stock_info, nse_map)
        assert result["MISS.NS"]["sector"] == "Unknown"

    def test_nse_unknown_does_not_override_existing_known(self):
        stock_info = {"X.NS": self._stock(sector="Unknown")}
        nse_map    = {"X.NS": {"sector": "Unknown", "industry": "Unknown", "basic_industry": ""}}
        result = enrich_with_nse_sectors(stock_info, nse_map)
        assert result["X.NS"]["sector"] == "Unknown"

    def test_empty_inputs_return_empty_dict(self):
        assert enrich_with_nse_sectors({}, {}) == {}

    def test_all_known_input_remains_unchanged(self):
        stock_info = {
            "RELIANCE.NS": self._stock(sector="Energy", industry="Oil Refining"),
            "TCS.NS":      self._stock(sector="Technology", industry="IT Services"),
        }
        nse_map = {
            "RELIANCE.NS": self._nse_entry(sector="Commodities", industry="Refining"),
            "TCS.NS":      self._nse_entry(sector="Information Technology"),
        }
        result = enrich_with_nse_sectors(stock_info, nse_map)
        assert result["RELIANCE.NS"]["sector"] == "Energy"
        assert result["TCS.NS"]["sector"] == "Technology"

    def test_modifies_stock_info_in_place(self):
        stock_info = {"INFY.NS": self._stock(sector="Unknown")}
        nse_map    = {"INFY.NS": self._nse_entry(sector="Technology")}
        result = enrich_with_nse_sectors(stock_info, nse_map)
        assert result is stock_info  # same dict object
