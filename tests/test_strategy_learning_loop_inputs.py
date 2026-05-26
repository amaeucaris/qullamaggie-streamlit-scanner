from pathlib import Path

import pandas as pd

from scripts.run_strategy_learning_loop import latest_watchlist_path, normalize_learning_signals


def test_learning_loop_prefers_daily_shortlist_snapshot(tmp_path: Path):
    steve = tmp_path / "steve_algo_watchlist_2026-05-22.csv"
    daily = tmp_path / "daily_shortlist_2026-05-22.csv"
    older_daily = tmp_path / "daily_shortlist_2026-05-21.csv"
    steve.write_text("Date,Ticker\n2026-05-22,AAA\n")
    daily.write_text("Signal Date,Ticker,SteveAlgo Bucket\n2026-05-22,BBB,Entry\n")
    older_daily.write_text("Signal Date,Ticker\n2026-05-21,CCC\n")
    assert latest_watchlist_path(tmp_path) == daily


def test_normalize_learning_signals_accepts_daily_shortlist_columns(tmp_path: Path):
    path = tmp_path / "daily_shortlist_2026-05-22.csv"
    df = pd.DataFrame([{"Signal Date": "2026-05-22", "Ticker": "MU", "SteveAlgo Bucket": "Entry"}])
    normalized, source = normalize_learning_signals(df, path)
    assert source == "DailyShortlist"
    assert normalized.loc[0, "Date"] == "2026-05-22"
    assert normalized.loc[0, "Strategy"] == "DailyShortlist"
    assert normalized.loc[0, "SteveAlgo Primary Bucket"] == "Entry"
