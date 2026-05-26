from pathlib import Path

import pandas as pd

from qull_scanner.strategy_exports import (
    DataFreshness,
    build_feature_effectiveness_review,
    build_stale_data_alert,
    write_feature_effectiveness_note,
    write_stale_data_alert_note,
)


def test_stale_data_alert_includes_resolution_runbook_and_paper_only(tmp_path: Path):
    freshness = DataFreshness(
        status="STALE",
        last_market_date="2026-05-22",
        last_update="2026-05-26 01:42 UTC",
        age_days=4,
        message="stale",
    )
    alert = build_stale_data_alert(freshness, previous_last_market_date="2026-05-22", expected_last_market_date="2026-05-26")

    assert alert["alert"] is True
    assert alert["severity"] == "HIGH"
    assert alert["resolution_status"] == "UNRESOLVED"
    assert "data refresh ran but market date did not advance" in alert["message"]
    assert "rerun data update workflow" in " ".join(alert["resolution_steps"]).lower()
    assert alert["capital_authorized"] == "0%"
    assert alert["proposal_policy"] == "PAPER_ONLY"

    paths = write_stale_data_alert_note(alert, vault=tmp_path, as_of="2026-05-26")
    text = paths["markdown"].read_text()
    assert "DATA STALE" in text
    assert "Resolution runbook" in text
    assert "PAPER_ONLY" in text
    assert "Capital authorized: 0%" in text


def test_stale_data_alert_resolves_when_market_date_advances():
    freshness = DataFreshness(
        status="FRESH",
        last_market_date="2026-05-27",
        last_update="2026-05-27 22:30 UTC",
        age_days=0,
        message="fresh",
    )
    alert = build_stale_data_alert(freshness, previous_last_market_date="2026-05-22", expected_last_market_date="2026-05-27")

    assert alert["alert"] is False
    assert alert["resolution_status"] == "RESOLVED"
    assert "advanced" in alert["message"]


def test_feature_effectiveness_review_ranks_frameworks_false_positives_extension_and_drawdown(tmp_path: Path):
    review = pd.DataFrame(
        [
            {
                "Ticker": "AAA",
                "Frameworks": "Qullamaggie Strict, Stockbee 4%",
                "Breakout Verified": True,
                "Max Return %": 12.0,
                "Max Drawdown %": -3.0,
                "ATR Extension SMA50": 2.5,
                "Score": 9,
                "Evaluation Status": "EVALUATED",
            },
            {
                "Ticker": "BBB",
                "Frameworks": "Steve-style KQ",
                "Breakout Verified": False,
                "Max Return %": 1.0,
                "Max Drawdown %": -11.0,
                "ATR Extension SMA50": 6.2,
                "Score": 2,
                "Evaluation Status": "EVALUATED",
            },
            {
                "Ticker": "CCC",
                "Frameworks": "Qullamaggie Strict",
                "Breakout Verified": True,
                "Max Return %": 8.0,
                "Max Drawdown %": -4.0,
                "ATR Extension SMA50": 4.0,
                "Score": 7,
                "Evaluation Status": "EVALUATED",
            },
        ]
    )
    result = build_feature_effectiveness_review(review, min_weeks=2, lookback_weeks=4)

    assert result["proposal_policy"] == "PAPER_ONLY"
    assert result["production_change_allowed"] is False
    assert result["framework_rows"][0]["Framework"] == "Qullamaggie Strict"
    false_positive = next(row for row in result["false_positive_rows"] if row["Ticker"] == "BBB")
    assert false_positive["False Positive Reason"] == "no breakout + weak max return"
    assert result["overextension_rows"][0]["Ticker"] == "BBB"
    assert result["drawdown_rows"][0]["Ticker"] == "BBB"

    paths = write_feature_effectiveness_note(result, vault=tmp_path, export_dir=tmp_path / "exports", as_of="2026-05-26")
    text = paths["markdown"].read_text()
    assert "2–4 Week Feature Effectiveness Review" in text
    assert "PAPER_ONLY" in text
    assert "No production rule changes allowed" in text
    assert "Qullamaggie Strict" in text
    assert "BBB" in text
