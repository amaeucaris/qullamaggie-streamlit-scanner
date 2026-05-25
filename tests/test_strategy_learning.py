from __future__ import annotations

import pandas as pd

from qull_scanner.strategy_learning import (
    build_outcome_summary,
    feature_attribution,
    merge_events_with_outcomes,
    propose_rule_candidates,
    stable_event_id,
)


def test_stable_event_id_is_deterministic_and_sensitive_to_config():
    row = {"strategy": "SteveAlgo", "Ticker": "AAPL", "Date": "2026-01-02", "SteveAlgo Primary Bucket": "Entry"}
    config = {"min_rs": 90, "target_r": 3.0}

    first = stable_event_id(row, config)
    second = stable_event_id(dict(reversed(list(row.items()))), dict(reversed(list(config.items()))))
    changed = stable_event_id(row, {"min_rs": 95, "target_r": 3.0})

    assert first == second
    assert first != changed
    assert len(first) == 16


def test_merge_events_with_outcomes_marks_unresolved_events_without_dropping_them():
    events = pd.DataFrame(
        {
            "event_id": ["e1", "e2", "e3"],
            "Ticker": ["AAA", "BBB", "CCC"],
            "Date": pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-04"]),
        }
    )
    outcomes = pd.DataFrame({"event_id": ["e1", "e3"], "R": [2.0, -1.0], "Exit Reason": ["target", "stop"]})

    merged = merge_events_with_outcomes(events, outcomes)

    assert len(merged) == 3
    assert merged.set_index("event_id").loc["e2", "Outcome Status"] == "OPEN"
    assert merged.set_index("event_id").loc["e1", "Outcome Status"] == "CLOSED"


def test_build_outcome_summary_counts_open_events_and_core_trade_stats():
    events = pd.DataFrame({"event_id": ["e1", "e2", "e3", "e4"]})
    outcomes = pd.DataFrame(
        {
            "event_id": ["e1", "e2", "e3"],
            "R": [2.0, -1.0, 0.5],
            "MFE R": [3.0, 0.2, 1.0],
            "MAE R": [-0.5, -1.2, -0.4],
        }
    )

    summary = build_outcome_summary(events, outcomes)

    assert summary["events"] == 4
    assert summary["closed_trades"] == 3
    assert summary["open_events"] == 1
    assert summary["expectancy_r"] == 0.5
    assert summary["win_rate_pct"] == 66.67
    assert summary["profit_factor"] == 2.5
    assert summary["median_mfe_r"] == 1.0
    assert summary["median_mae_r"] == -0.5


def test_feature_attribution_flags_insufficient_sample():
    merged = pd.DataFrame({"R": [1.0, -1.0], "Momentum Rank": [99, 80], "Bucket": ["Entry", "White Up"]})

    attribution = feature_attribution(merged, min_sample=10)

    assert set(attribution["status"]) == {"INSUFFICIENT_SAMPLE"}


def test_feature_attribution_reports_numeric_and_categorical_associations():
    merged = pd.DataFrame(
        {
            "R": [2.0, 1.0, -1.0, -0.5, 3.0, -1.0],
            "Momentum Rank": [99, 95, 70, 72, 98, 65],
            "Reward-Risk": [4.0, 3.5, 2.0, 2.2, 5.0, 1.8],
            "Bucket": ["Entry", "Entry", "Yellow", "Yellow", "Entry", "Yellow"],
        }
    )

    attribution = feature_attribution(merged, min_sample=3)

    by_feature = attribution.set_index("feature")
    assert by_feature.loc["Momentum Rank", "status"] == "OK"
    assert by_feature.loc["Momentum Rank", "winner_median"] > by_feature.loc["Momentum Rank", "loser_median"]
    assert by_feature.loc["Bucket=Entry", "expectancy"] == 2.0
    assert by_feature.loc["Bucket=Yellow", "expectancy"] < 0


def test_feature_attribution_ignores_boolean_columns_for_numeric_quantiles():
    merged = pd.DataFrame(
        {
            "R": [1.0, -1.0, 2.0, -0.5],
            "Qullamaggie Overlap": [True, False, True, False],
            "Momentum Rank": [90, 70, 95, 75],
        }
    )

    attribution = feature_attribution(merged, min_sample=2)

    assert "Momentum Rank" in set(attribution["feature"])
    assert "Qullamaggie Overlap" not in set(attribution["feature"])


def test_propose_rule_candidates_rejects_weak_or_unvalidated_improvements():
    attribution = pd.DataFrame(
        {
            "feature": ["Bucket=Yellow", "Momentum Rank"],
            "sample_size": [80, 120],
            "expectancy": [-0.4, pd.NA],
            "difference": [pd.NA, 12.0],
            "status": ["OK", "OK"],
        }
    )
    baseline = {"expectancy_r": 0.10, "profit_factor": 1.25, "max_drawdown_r": -12.0}
    weak_oos = {"expectancy_r": 0.11, "profit_factor": 1.10, "max_drawdown_r": -11.0, "trades": 45}
    random_summary = {"expectancy_r": 0.09, "profit_factor": 1.15}

    proposals = propose_rule_candidates(attribution, baseline, weak_oos, random_summary)

    assert not proposals.empty
    assert set(proposals["promotion_status"]) == {"REJECTED"}
    assert all("Antonio" not in status for status in proposals["promotion_status"])


def test_propose_rule_candidates_can_mark_watch_but_never_auto_approve():
    attribution = pd.DataFrame(
        {
            "feature": ["Bucket=Yellow"],
            "sample_size": [150],
            "expectancy": [-0.5],
            "difference": [pd.NA],
            "status": ["OK"],
        }
    )
    baseline = {"expectancy_r": 0.10, "profit_factor": 1.25, "max_drawdown_r": -12.0}
    strong_oos = {"expectancy_r": 0.20, "profit_factor": 1.35, "max_drawdown_r": -13.0, "trades": 70}
    random_summary = {"expectancy_r": 0.12, "profit_factor": 1.15}

    proposals = propose_rule_candidates(attribution, baseline, strong_oos, random_summary)

    assert proposals.iloc[0]["promotion_status"] == "WATCH"
    assert "exclude Yellow" in proposals.iloc[0]["hypothesis"]
