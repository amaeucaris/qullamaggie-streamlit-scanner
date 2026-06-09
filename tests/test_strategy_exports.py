from __future__ import annotations

from pathlib import Path

import pandas as pd

from qull_scanner.strategy_exports import (
    DataFreshness,
    add_weekly_effectiveness,
    build_daily_shortlist,
    classify_extension_risk,
    scanner_outputs,
    write_daily_watchlist_note,
    write_weekly_effectiveness_note,
)


def _metrics() -> pd.DataFrame:
    rows = []
    base = {
        "Date": pd.Timestamp("2026-05-22"),
        "Price": 20.0,
        "Return 1W %": 5.0,
        "Return 1M %": 25.0,
        "Return 3M %": 60.0,
        "Return 6M %": 90.0,
        "ADR 20D %": 5.0,
        "Volume": 10_000_000,
        "Prev Volume": 5_000_000,
        "Avg Volume 20D": 2_000_000,
        "Daily $ Volume 20D": 60_000_000,
        "Daily Return %": 5.0,
        "SMA10": 18.0,
        "SMA20": 17.0,
        "SMA50": 15.0,
        "SMA200": 10.0,
        "EMA10": 18.0,
        "EMA20": 17.0,
        "EMA50": 15.0,
        "EMA10 Rising": True,
        "ATR20": 1.0,
        "ATR Extension SMA50": 2.0,
        "ATR Extension EMA10": 0.5,
        "ATR Extension EMA20": 1.0,
        "DCR %": 70.0,
        "Darvas Upper": 19.0,
        "Darvas Lower": 17.5,
        "Breakout Level": 19.0,
        "Market Cap": None,
        "Price > SMA10": True,
        "Price > SMA20": True,
        "Minervini Trend Template": True,
        "Green Candle": True,
        "Breakout Above Lookback High": True,
        "Momentum Rank": 95.0,
        "Universe Percentile": 95.0,
    }
    for i, ticker in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE"]):
        row = dict(base)
        row["Ticker"] = ticker
        row["Return 1M %"] = 30 - i
        row["Return 3M %"] = 70 - i
        row["Return 6M %"] = 100 - i
        row["Momentum Rank"] = 99 - i
        rows.append(row)
    return pd.DataFrame(rows)


def test_daily_shortlist_and_obsidian_note(tmp_path: Path):
    metrics = _metrics()
    sugar = pd.DataFrame([{"Ticker": "AAA", "Date": pd.Timestamp("2026-05-22"), "SB Score": 10}])
    outputs = scanner_outputs(metrics, sugar)
    shortlist = build_daily_shortlist(outputs, limit=3)
    assert not shortlist.empty
    assert shortlist.iloc[0]["Ticker"] == "AAA"
    assert "Capital Authorized" in shortlist.columns
    freshness = DataFreshness("FRESH", "2026-05-22", "2026-05-23 01:00 UTC", 1, "ok")
    paths = write_daily_watchlist_note(shortlist, outputs, freshness, tmp_path, tmp_path / "exports", as_of="2026-05-22")
    content = paths["markdown"].read_text()
    assert "Daily Strategy Lab Watchlist" in content
    assert "Capital authorized: 0%" in content
    assert paths["csv"].exists()
    assert (tmp_path / "queries" / "latest-strategy-lab-watchlist.md").exists()


def test_classify_extension_risk_zones():
    assert classify_extension_risk(4.99) == ("NORMAL", "ATR/SMA50 <5x — no extension warning")
    assert classify_extension_risk(5.0) == ("EXTENDED", "CAUTION_EXTENDED: ATR/SMA50 5.00x >=5x")
    assert classify_extension_risk(7.0) == ("OVEREXTENDED", "REJECT_EXTENDED_REVIEW: ATR/SMA50 7.00x >=7x")
    assert classify_extension_risk(11.0) == ("HYPEREXTENDED", "REJECT_HYPEREXTENDED_REVIEW: ATR/SMA50 11.00x >=11x")
    assert classify_extension_risk(None) == ("N/D", "ATR/SMA50 extension N/D")


def test_daily_shortlist_marks_overextended_candidates():
    metrics = _metrics()
    metrics.loc[metrics["Ticker"] == "AAA", "ATR Extension SMA50"] = 7.2
    outputs = scanner_outputs(metrics, pd.DataFrame([{"Ticker": "AAA", "Date": pd.Timestamp("2026-05-22"), "SB Score": 10}]))
    shortlist = build_daily_shortlist(outputs, limit=3)
    aaa = shortlist[shortlist["Ticker"] == "AAA"].iloc[0]
    assert aaa["Extension Risk Zone"] == "OVEREXTENDED"
    assert "REJECT_EXTENDED_REVIEW" in aaa["Extension Risk Reason"]
    assert aaa["Trade Readiness"] == "REJECT_EXTENDED_REVIEW"
    assert "REJECT_EXTENDED_REVIEW" in aaa["Reason"]


def test_daily_watchlist_note_includes_extension_risk_reason(tmp_path: Path):
    metrics = _metrics()
    metrics.loc[metrics["Ticker"] == "AAA", "ATR Extension SMA50"] = 11.5
    outputs = scanner_outputs(metrics, pd.DataFrame([{"Ticker": "AAA", "Date": pd.Timestamp("2026-05-22"), "SB Score": 10}]))
    shortlist = build_daily_shortlist(outputs, limit=1)
    freshness = DataFreshness("FRESH", "2026-05-22", "2026-05-23 01:00 UTC", 1, "ok")
    paths = write_daily_watchlist_note(shortlist, outputs, freshness, tmp_path, tmp_path / "exports", as_of="2026-05-22")
    content = paths["markdown"].read_text()
    assert "extension HYPEREXTENDED" in content
    assert "REJECT_HYPEREXTENDED_REVIEW" in content


def test_weekly_effectiveness_review(tmp_path: Path):
    snapshots = pd.DataFrame(
        [
            {
                "Ticker": "AAA",
                "Signal Date": "2026-05-20",
                "Score": 8,
                "Trade Readiness": "PRIORITY REVIEW",
                "Frameworks": "Qullamaggie Strict, Stockbee 4%",
                "Price": 10,
                "Breakout Level": 10.5,
            }
        ]
    )
    history = pd.DataFrame(
        [
            {"Ticker": "AAA", "Date": pd.Timestamp("2026-05-20"), "Open": 9, "High": 10.2, "Low": 9.8, "Close": 10, "Volume": 1_000_000},
            {"Ticker": "AAA", "Date": pd.Timestamp("2026-05-21"), "Open": 10, "High": 11.5, "Low": 9.9, "Close": 11, "Volume": 1_000_000},
        ]
    )
    current = _metrics().head(1).copy()
    current["Ticker"] = "AAA"
    review = add_weekly_effectiveness(snapshots, history, current)
    assert bool(review.iloc[0]["Breakout Verified"]) is True
    assert review.iloc[0]["Current Phase"]
    paths = write_weekly_effectiveness_note(review, tmp_path, tmp_path / "reviews", as_of="2026-05-24")
    assert "Weekly Selection Effectiveness" in paths["markdown"].read_text()
    assert paths["csv"].exists()
