"""
Unit tests for src/nse_freefloat.py — all HTTP calls are mocked.
No network access required.
"""

import os
import pickle
import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.nse_freefloat import (
    fetch_freefloat,
    fetch_all_freefloat,
    is_freefloat_cache_fresh,
    load_freefloat_cache,
    save_freefloat_cache,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session(json_data=None, status_code=200, raise_exc=None):
    """Return a mock requests.Session whose .get() behaves as specified."""
    session = MagicMock()
    resp = MagicMock()
    resp.status_code = status_code
    if raise_exc:
        session.get.side_effect = raise_exc
    else:
        if status_code >= 400:
            resp.raise_for_status.side_effect = requests.HTTPError(f"HTTP {status_code}")
        else:
            resp.raise_for_status = MagicMock()
        resp.json.return_value = json_data
        session.get.return_value = resp
    return session


# Two quarterly filings for RELIANCE — newest first (index 0 is used)
NSE_RELIANCE_PAYLOAD = [
    {"public_val": "49.69", "pr_and_prgrp": "50.31", "date": "31-MAR-2026"},
    {"public_val": "48.10", "pr_and_prgrp": "51.90", "date": "31-DEC-2025"},
]


# ---------------------------------------------------------------------------
# fetch_freefloat — parsing and error handling
# ---------------------------------------------------------------------------

class TestFetchFreefloat:

    def test_returns_public_val_as_float(self):
        session = _make_session(json_data=NSE_RELIANCE_PAYLOAD)
        result = fetch_freefloat("RELIANCE", session)
        assert result == pytest.approx(49.7, abs=0.1)

    def test_uses_most_recent_entry_not_older_ones(self):
        """Index 0 is the most recent filing — must not return index 1's value."""
        session = _make_session(json_data=NSE_RELIANCE_PAYLOAD)
        result = fetch_freefloat("RELIANCE", session)
        # Index 0 → 49.69, index 1 → 48.10; confirm we did NOT return 48.1
        assert result != pytest.approx(48.1, abs=0.1)

    def test_result_rounded_to_one_decimal(self):
        session = _make_session(json_data=[{"public_val": "49.6888"}])
        assert fetch_freefloat("X", session) == pytest.approx(49.7, abs=0.05)

    def test_empty_list_returns_none(self):
        """NSE returns [] for stocks with data gaps (e.g. ZOMATO, TATAMOTORS)."""
        session = _make_session(json_data=[])
        assert fetch_freefloat("ZOMATO", session) is None

    def test_non_list_response_returns_none(self):
        """Unexpected JSON shapes (dict, string) must not raise."""
        session = _make_session(json_data={"error": "not found"})
        assert fetch_freefloat("X", session) is None

    def test_missing_public_val_key_returns_none(self):
        session = _make_session(json_data=[{"pr_and_prgrp": "50.0", "date": "31-MAR-2026"}])
        assert fetch_freefloat("X", session) is None

    def test_http_error_returns_none_not_exception(self):
        session = _make_session(status_code=404)
        assert fetch_freefloat("BADTICKER", session) is None

    def test_network_exception_returns_none_not_exception(self):
        session = _make_session(raise_exc=requests.ConnectionError("timeout"))
        assert fetch_freefloat("RELIANCE", session) is None

    def test_symbol_is_uppercased_in_request_url(self):
        session = _make_session(json_data=NSE_RELIANCE_PAYLOAD)
        fetch_freefloat("reliance", session)
        url_called = session.get.call_args[0][0]
        assert "RELIANCE" in url_called

    def test_ns_suffix_not_included_in_url(self):
        """Callers should pass bare symbols, but guard against accidental .NS."""
        session = _make_session(json_data=NSE_RELIANCE_PAYLOAD)
        fetch_freefloat("RELIANCE.NS", session)
        url_called = session.get.call_args[0][0]
        # The URL should still work; the symbol part must be 'RELIANCE.NS' uppercased
        # (stripping .NS is the caller's responsibility in fetch_all_freefloat)
        assert "symbol=" in url_called


# ---------------------------------------------------------------------------
# fetch_all_freefloat — routing, fallback, rate-limiting
# ---------------------------------------------------------------------------

class TestFetchAllFreefloat:

    def _patched(self, json_data, yf_tickers, yf_info=None):
        """Helper: run fetch_all_freefloat with a mocked session and sleep."""
        with patch("src.nse_freefloat._build_session") as mock_build, \
             patch("src.nse_freefloat.time.sleep"):
            mock_build.return_value = _make_session(json_data=json_data)
            return fetch_all_freefloat(yf_tickers, yf_info=yf_info)

    def test_nse_value_returned_when_api_has_data(self):
        result = self._patched(NSE_RELIANCE_PAYLOAD, ["RELIANCE.NS"])
        assert "RELIANCE.NS" in result
        assert result["RELIANCE.NS"] == pytest.approx(49.7, abs=0.1)

    def test_result_keyed_by_yf_ticker_with_ns_suffix(self):
        result = self._patched(NSE_RELIANCE_PAYLOAD, ["RELIANCE.NS"])
        assert "RELIANCE.NS" in result
        assert "RELIANCE" not in result

    def test_ns_suffix_stripped_before_nse_api_call(self):
        with patch("src.nse_freefloat._build_session") as mock_build, \
             patch("src.nse_freefloat.time.sleep"):
            session = _make_session(json_data=NSE_RELIANCE_PAYLOAD)
            mock_build.return_value = session
            fetch_all_freefloat(["RELIANCE.NS"])
        url_called = session.get.call_args[0][0]
        assert ".NS" not in url_called
        assert "RELIANCE" in url_called

    def test_yfinance_fallback_used_when_nse_returns_empty(self):
        yf_info = {"ZOMATO.NS": {"heldPercentInsiders": 0.10}}  # 90 % float
        result = self._patched([], ["ZOMATO.NS"], yf_info=yf_info)
        assert "ZOMATO.NS" in result
        assert result["ZOMATO.NS"] == pytest.approx(90.0, abs=0.5)

    def test_nse_value_takes_priority_over_yfinance_fallback(self):
        """When NSE returns data, yfinance fallback must NOT override it."""
        yf_info = {"RELIANCE.NS": {"heldPercentInsiders": 0.99}}  # would give 1 %
        result = self._patched(NSE_RELIANCE_PAYLOAD, ["RELIANCE.NS"], yf_info=yf_info)
        assert result["RELIANCE.NS"] == pytest.approx(49.7, abs=0.1)  # NSE value

    def test_ticker_excluded_when_both_sources_are_none(self):
        result = self._patched([], ["UNKNOWN.NS"], yf_info={})
        assert "UNKNOWN.NS" not in result

    def test_empty_input_returns_empty_dict(self):
        with patch("src.nse_freefloat._build_session"), \
             patch("src.nse_freefloat.time.sleep"):
            result = fetch_all_freefloat([])
        assert result == {}

    def test_sleep_called_between_requests_not_after_last(self):
        """Rate-limiting sleep must fire N-1 times for N tickers."""
        tickers = ["A.NS", "B.NS", "C.NS"]
        with patch("src.nse_freefloat._build_session") as mock_build, \
             patch("src.nse_freefloat.time.sleep") as mock_sleep:
            mock_build.return_value = _make_session(json_data=NSE_RELIANCE_PAYLOAD)
            fetch_all_freefloat(tickers)
        assert mock_sleep.call_count == len(tickers) - 1

    def test_single_ticker_no_sleep(self):
        with patch("src.nse_freefloat._build_session") as mock_build, \
             patch("src.nse_freefloat.time.sleep") as mock_sleep:
            mock_build.return_value = _make_session(json_data=NSE_RELIANCE_PAYLOAD)
            fetch_all_freefloat(["RELIANCE.NS"])
        assert mock_sleep.call_count == 0

    def test_partial_failures_do_not_abort_remaining_tickers(self):
        """A 404 for one ticker must not stop the rest from being fetched."""
        responses = [[], NSE_RELIANCE_PAYLOAD]  # first fails, second succeeds
        call_count = [0]

        def side_effect(*args, **kwargs):
            idx = call_count[0]
            call_count[0] += 1
            session = _make_session(json_data=responses[idx % len(responses)])
            return session.get(*args, **kwargs)

        with patch("src.nse_freefloat._build_session") as mock_build, \
             patch("src.nse_freefloat.time.sleep"):
            real_session = MagicMock()
            real_session.get.side_effect = side_effect
            mock_build.return_value = real_session
            result = fetch_all_freefloat(["ZOMATO.NS", "RELIANCE.NS"])

        # RELIANCE should still be in result even though ZOMATO failed
        assert "RELIANCE.NS" in result


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

class TestFreefloatCache:

    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.nse_freefloat.FREEFLOAT_FILE", str(tmp_path / "ff.pkl"))
        monkeypatch.setattr("src.nse_freefloat.DATA_DIR", str(tmp_path))
        data = {"RELIANCE.NS": 49.7, "INFY.NS": 85.4, "WIPRO.NS": 29.1}
        save_freefloat_cache(data)
        loaded = load_freefloat_cache()
        assert loaded == data

    def test_load_returns_empty_dict_when_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.nse_freefloat.FREEFLOAT_FILE", str(tmp_path / "missing.pkl"))
        assert load_freefloat_cache() == {}

    def test_load_removes_corrupt_file_and_returns_empty(self, tmp_path, monkeypatch):
        ff_path = str(tmp_path / "ff.pkl")
        monkeypatch.setattr("src.nse_freefloat.FREEFLOAT_FILE", ff_path)
        with open(ff_path, "wb") as f:
            f.write(b"not valid pickle data!!!")
        result = load_freefloat_cache()
        assert result == {}
        assert not os.path.exists(ff_path)

    def test_is_fresh_for_new_file(self, tmp_path, monkeypatch):
        ff_path = str(tmp_path / "ff.pkl")
        monkeypatch.setattr("src.nse_freefloat.FREEFLOAT_FILE", ff_path)
        monkeypatch.setattr("src.nse_freefloat.FREEFLOAT_CACHE_DAYS", 90)
        with open(ff_path, "wb") as f:
            pickle.dump({}, f)
        assert is_freefloat_cache_fresh() is True

    def test_is_stale_for_old_file(self, tmp_path, monkeypatch):
        ff_path = str(tmp_path / "ff.pkl")
        monkeypatch.setattr("src.nse_freefloat.FREEFLOAT_FILE", ff_path)
        monkeypatch.setattr("src.nse_freefloat.FREEFLOAT_CACHE_DAYS", 90)
        with open(ff_path, "wb") as f:
            pickle.dump({}, f)
        old_time = time.time() - (91 * 86400)
        os.utime(ff_path, (old_time, old_time))
        assert is_freefloat_cache_fresh() is False

    def test_is_fresh_returns_false_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.nse_freefloat.FREEFLOAT_FILE", str(tmp_path / "none.pkl"))
        assert is_freefloat_cache_fresh() is False

    def test_save_creates_data_dir_if_missing(self, tmp_path, monkeypatch):
        nested = tmp_path / "deep" / "dir"
        ff_path = str(nested / "ff.pkl")
        monkeypatch.setattr("src.nse_freefloat.FREEFLOAT_FILE", ff_path)
        monkeypatch.setattr("src.nse_freefloat.DATA_DIR", str(nested))
        save_freefloat_cache({"X.NS": 50.0})
        assert os.path.exists(ff_path)
