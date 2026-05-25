from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.run_strategy_learning_loop import latest_watchlist_path, run_learning_loop


def test_latest_watchlist_path_picks_most_recent_name(tmp_path: Path):
    old = tmp_path / "steve_algo_watchlist_2026-01-01.csv"
    new = tmp_path / "steve_algo_watchlist_2026-01-02.csv"
    old.write_text("Date,Ticker\n2026-01-01,A\n")
    new.write_text("Date,Ticker\n2026-01-02,B\n")

    assert latest_watchlist_path(tmp_path) == new


def test_run_learning_loop_writes_journal_outcomes_and_review(tmp_path: Path):
    watch_dir = tmp_path / "watchlists"
    watch_dir.mkdir()
    watchlist = pd.DataFrame(
        {
            "Date": ["2026-01-02"],
            "Ticker": ["AAA"],
            "SteveAlgo Primary Bucket": ["Entry"],
            "Darvas Lower": [9.0],
        }
    )
    watchlist.to_csv(watch_dir / "steve_algo_watchlist_2026-01-02.csv", index=False)
    history = pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-02", periods=25, freq="D"),
            "Ticker": ["AAA"] * 25,
            "Open": [10.0] * 25,
            "High": [10.5] * 25,
            "Low": [9.5] * 25,
            "Close": [10.2] * 25,
            "Volume": [1000] * 25,
            "EMA20": [9.5] * 25,
            "ATR20": [1.0] * 25,
        }
    )
    history_path = tmp_path / "history.parquet"
    history.to_parquet(history_path)

    result = run_learning_loop(
        watchlist_dir=watch_dir,
        history_path=history_path,
        output_dir=tmp_path / "exports",
        max_hold_bars=3,
        min_sample=1,
    )

    assert result["journal_rows"] == 1
    assert result["closed_outcomes"] == 1
    assert (tmp_path / "exports" / "strategy_signal_journal.csv").exists()
    assert (tmp_path / "exports" / "strategy_paper_outcomes.csv").exists()
    assert (tmp_path / "exports" / "strategy_self_review.json").exists()
