from __future__ import annotations

from pathlib import Path

import pandas as pd

from qull_scanner.strategy_learning import append_signal_journal, resolve_paper_outcomes


def test_append_signal_journal_is_idempotent_and_adds_guardrail_columns(tmp_path: Path):
    watchlist = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-01-02", "2026-01-02"]),
            "Ticker": ["AAA", "BBB"],
            "SteveAlgo Primary Bucket": ["Entry", "White Up"],
            "Price": [10.0, 20.0],
        }
    )
    path = tmp_path / "signal_journal.csv"

    first = append_signal_journal(watchlist, path, config={"rule_version": "v1"})
    second = append_signal_journal(watchlist, path, config={"rule_version": "v1"})

    assert len(first) == 2
    assert len(second) == 2
    assert set(second["Paper Status"]) == {"OPEN"}
    assert set(second["Capital Authorized"]) == {"0%"}
    assert second["event_id"].is_unique


def test_resolve_paper_outcomes_closes_mature_events_and_keeps_immature_open():
    journal = pd.DataFrame(
        {
            "event_id": ["e1", "e2"],
            "Date": pd.to_datetime(["2026-01-02", "2026-01-08"]),
            "Ticker": ["AAA", "AAA"],
            "SteveAlgo Primary Bucket": ["Entry", "Entry"],
            "Darvas Lower": [9.0, 9.0],
            "Paper Status": ["OPEN", "OPEN"],
        }
    )
    dates = pd.date_range("2026-01-02", periods=8, freq="D")
    hist = pd.DataFrame(
        {
            "Open": [10, 10, 10, 10, 10, 10, 10, 10],
            "High": [10.5, 11, 11.5, 12, 14, 11, 11, 11],
            "Low": [9.8, 9.7, 9.6, 9.5, 9.4, 9.7, 9.7, 9.7],
            "Close": [10.2, 10.8, 11, 11.5, 13, 10.5, 10.6, 10.7],
            "EMA20": [9.5] * 8,
            "ATR20": [1.0] * 8,
        },
        index=dates,
    )

    outcomes = resolve_paper_outcomes(journal, {"AAA": hist}, max_hold_bars=3, target_r=3.0)

    closed = outcomes.set_index("event_id").loc["e1"]
    immature = outcomes.set_index("event_id").loc["e2"]
    assert closed["Paper Status"] == "CLOSED"
    assert closed["R"] > 0
    assert immature["Paper Status"] == "OPEN"
    assert pd.isna(immature["R"])
