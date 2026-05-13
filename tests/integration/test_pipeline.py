"""
Integration tests — full pipeline with synthetic data (no network calls).

These tests exercise complete data flows:
  - screener filter + score → snapshot → forward returns
  - index screener → index snapshot → forward returns
  - pipeline status helpers

Slow (~1–5 s each).  Run with:  pytest tests/integration/ -v
"""

import os
import pickle
import tempfile
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.screener import run_screener
from src.tracker import save_snapshot, load_snapshots, compute_forward_returns
from src.index_screener import run_index_screener
from src.index_tracker import save_index_snapshot, load_index_snapshots, compute_index_forward_returns


# ---------------------------------------------------------------------------
# Shared synthetic data factory
# ---------------------------------------------------------------------------

def _ohlcv(n: int = 260, trend: float = 0.003, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1 + rng.normal(trend, 0.008, n))
    high  = close * (1 + rng.uniform(0.001, 0.015, n))
    low   = close * (1 - rng.uniform(0.001, 0.015, n))
    vol   = rng.integers(500_000, 5_000_000, n).astype(float)
    df = pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close, "Volume": vol})
    df.index = pd.date_range("2023-01-01", periods=n, freq="B")
    return df


def _make_stock_universe(n: int = 8) -> tuple[dict, dict]:
    price_data = {}
    stock_info = {}
    for i in range(n):
        ticker = f"STOCK{i:02d}.NS"
        price_data[ticker] = _ohlcv(n=260, trend=0.003, seed=i)
        stock_info[ticker] = {
            "sector":    "Technology" if i % 2 == 0 else "Pharma",
            "industry":  "Software" if i % 2 == 0 else "Drug",
            "pe":        20.0 + i,
            "market_cap": 50_000 * 1_00_00_000,
            "company":   f"Company {i}",
        }
    return price_data, stock_info


def _make_index_universe(n: int = 6) -> dict:
    return {f"Index {i}": _ohlcv(n=260, trend=0.003, seed=i + 100) for i in range(n)}


# ---------------------------------------------------------------------------
# Stock screener → tracker pipeline
# ---------------------------------------------------------------------------

class TestStockScreenerPipeline:

    def test_screener_returns_dataframe(self):
        price_data, stock_info = _make_stock_universe()
        result = run_screener(price_data, stock_info, {}, None)
        assert isinstance(result, pd.DataFrame)

    def test_snapshot_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SNAPSHOT_OVERRIDE", str(tmp_path / "snapshots.pkl"))
        monkeypatch.setattr("src.tracker.SNAPSHOTS_FILE", str(tmp_path / "snapshots.pkl"))
        monkeypatch.setattr("src.tracker.DATA_DIR", str(tmp_path))

        price_data, stock_info = _make_stock_universe()
        results = run_screener(price_data, stock_info, {}, None)

        today = date.today().isoformat()
        n = save_snapshot(results, capture_date=today, force=False)
        assert n == len(results)

        loaded = load_snapshots()
        assert not loaded.empty
        assert "Capture Date" in loaded.columns
        assert today in loaded["Capture Date"].values

    def test_forward_returns_computed(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.tracker.SNAPSHOTS_FILE", str(tmp_path / "snapshots.pkl"))
        monkeypatch.setattr("src.tracker.DATA_DIR", str(tmp_path))

        price_data, stock_info = _make_stock_universe()
        results = run_screener(price_data, stock_info, {}, None)

        # Capture yesterday so future prices exist in our data
        yesterday = (date.today() - timedelta(days=5)).isoformat()
        save_snapshot(results, capture_date=yesterday, force=True)

        snapshots = load_snapshots()
        df_with_ret = compute_forward_returns(snapshots, price_data)

        assert "Ret 1D %" in df_with_ret.columns or "Ret 1W %" in df_with_ret.columns

    def test_no_duplicate_snapshots(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.tracker.SNAPSHOTS_FILE", str(tmp_path / "snapshots.pkl"))
        monkeypatch.setattr("src.tracker.DATA_DIR", str(tmp_path))

        price_data, stock_info = _make_stock_universe()
        results = run_screener(price_data, stock_info, {}, None)
        today = date.today().isoformat()

        save_snapshot(results, capture_date=today, force=False)
        n2 = save_snapshot(results, capture_date=today, force=False)  # should be skipped
        assert n2 == 0

        loaded = load_snapshots()
        assert len(loaded[loaded["Capture Date"] == today]) == len(results)

    def test_force_replaces_snapshot(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.tracker.SNAPSHOTS_FILE", str(tmp_path / "snapshots.pkl"))
        monkeypatch.setattr("src.tracker.DATA_DIR", str(tmp_path))

        price_data, stock_info = _make_stock_universe()
        results = run_screener(price_data, stock_info, {}, None)
        today = date.today().isoformat()

        save_snapshot(results, capture_date=today, force=False)
        # Modify results slightly and force-replace
        modified = results.copy()
        if not modified.empty:
            modified = modified.head(max(1, len(modified) - 1))
        save_snapshot(modified, capture_date=today, force=True)

        loaded = load_snapshots()
        assert len(loaded[loaded["Capture Date"] == today]) == len(modified)


# ---------------------------------------------------------------------------
# Index screener → tracker pipeline
# ---------------------------------------------------------------------------

class TestIndexScreenerPipeline:

    def test_index_screener_returns_dataframe(self):
        ohlcv = _make_index_universe()
        result = run_index_screener(ohlcv)
        assert isinstance(result, pd.DataFrame)

    def test_index_snapshot_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.index_tracker.INDEX_SNAPSHOTS_FILE", str(tmp_path / "idx_snap.pkl"))
        monkeypatch.setattr("src.index_tracker.DATA_DIR", str(tmp_path))

        ohlcv = _make_index_universe()
        results = run_index_screener(ohlcv)
        today = date.today().isoformat()

        n = save_index_snapshot(results, capture_date=today, force=False)
        assert n == len(results)

        loaded = load_index_snapshots()
        assert not loaded.empty
        assert today in loaded["Capture Date"].values

    def test_index_forward_returns_computed(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.index_tracker.INDEX_SNAPSHOTS_FILE", str(tmp_path / "idx_snap.pkl"))
        monkeypatch.setattr("src.index_tracker.INDEX_OHLCV_FILE",     str(tmp_path / "idx_ohlcv.pkl"))
        monkeypatch.setattr("src.index_tracker.DATA_DIR", str(tmp_path))

        ohlcv = _make_index_universe()

        # Save OHLCV so the forward-return computation can load it
        with open(str(tmp_path / "idx_ohlcv.pkl"), "wb") as f:
            pickle.dump(ohlcv, f)

        results = run_index_screener(ohlcv)
        yesterday = (date.today() - timedelta(days=5)).isoformat()
        save_index_snapshot(results, capture_date=yesterday, force=True)

        snapshots = load_index_snapshots()
        df_ret = compute_index_forward_returns(snapshots)

        # At least the return columns should exist (values may be NaN if data not aligned)
        assert "Ret 1W %" in df_ret.columns

    def test_index_scores_ordered(self):
        ohlcv = _make_index_universe(n=10)
        result = run_index_screener(ohlcv)
        if len(result) > 1:
            scores = result["Score"].tolist()
            assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Pipeline status helpers
# ---------------------------------------------------------------------------

class TestPipelineStatus:

    def test_write_and_read_status(self, tmp_path, monkeypatch):
        status_file = str(tmp_path / "status.json")
        monkeypatch.setattr("src.pipeline.REFRESH_STATUS_FILE", status_file)
        monkeypatch.setattr("src.pipeline.DATA_DIR", str(tmp_path))

        from src.pipeline import write_status, read_status
        write_status("fetch", "running", "Fetching…")
        s = read_status()
        assert s["fetch"]["state"] == "running"
        assert s["fetch"]["message"] == "Fetching…"

    def test_read_status_defaults_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.pipeline.REFRESH_STATUS_FILE", str(tmp_path / "nonexistent.json"))

        from src.pipeline import read_status
        s = read_status()
        assert "fetch" in s
        assert "screener" in s
