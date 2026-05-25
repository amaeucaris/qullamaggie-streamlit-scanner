from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.run_strategy_self_review import build_self_review, write_self_review


def test_build_self_review_emits_no_production_change_and_watch_proposals(tmp_path: Path):
    events = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-01-02"] * 6),
            "Ticker": ["A", "B", "C", "D", "E", "F"],
            "SteveAlgo Primary Bucket": ["Entry", "Entry", "Yellow", "Yellow", "Entry", "Yellow"],
            "Momentum Rank": [99, 95, 70, 72, 98, 65],
            "Reward-Risk": [4.0, 3.5, 2.0, 2.2, 5.0, 1.8],
        }
    )
    trades = pd.DataFrame(
        {
            "Ticker": ["A", "B", "C", "D", "E", "F"],
            "Signal Date": pd.to_datetime(["2026-01-02"] * 6),
            "Bucket": ["Entry", "Entry", "Yellow", "Yellow", "Entry", "Yellow"],
            "R": [2.0, 1.0, -1.0, -0.5, 3.0, -1.0],
        }
    )

    review = build_self_review(events, trades, min_sample=3)

    assert review["verdict"] == "NO_PRODUCTION_CHANGE"
    assert review["capital_authorized_pct"] == 0
    assert review["baseline_summary"]["closed_trades"] == 6
    assert review["attribution_rows"] >= 1
    assert review["proposal_rows"] >= 1


def test_write_self_review_creates_json_and_markdown(tmp_path: Path):
    review = {
        "verdict": "NO_PRODUCTION_CHANGE",
        "capital_authorized_pct": 0,
        "baseline_summary": {"closed_trades": 3, "expectancy_r": 0.2, "profit_factor": 1.5},
        "attribution": [{"feature": "Momentum Rank", "status": "OK", "sample_size": 3}],
        "proposals": [{"hypothesis": "tighten min RS", "promotion_status": "REJECTED", "reason": "weak OOS"}],
        "warnings": ["research only"],
    }

    paths = write_self_review(review, tmp_path)

    assert paths["json"].exists()
    assert paths["markdown"].exists()
    parsed = json.loads(paths["json"].read_text())
    assert parsed["capital_authorized_pct"] == 0
    md = paths["markdown"].read_text()
    assert "NO_PRODUCTION_CHANGE" in md
    assert "tighten min RS" in md
