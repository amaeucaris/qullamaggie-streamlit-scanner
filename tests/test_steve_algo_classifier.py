import pandas as pd

from qull_scanner.steve_algo import (
    SteveAlgoThresholds,
    classify_steve_algo_row,
    apply_steve_algo_watchlists,
)


def base_row(**overrides):
    row = {
        "Ticker": "AAA",
        "Price": 100.0,
        "Market Cap": 2_000_000_000,
        "Daily $ Volume 20D": 100_000_000,
        "EMA10": 95.0,
        "EMA20": 90.0,
        "EMA50": 80.0,
        "SMA50": 85.0,
        "SMA100": 75.0,
        "SMA200": 70.0,
        "Return 1M %": 12.0,
        "Return 3M %": 25.0,
        "Return 6M %": 50.0,
        "Momentum Rank": 95.0,
        "DCR %": 75.0,
        "Darvas Upper": 99.0,
        "Darvas Lower": 98.0,
        "ATR20": 2.0,
        "ATR Extension EMA10": 1.0,
        "ATR Extension EMA20": 1.2,
        "ATR Extension SMA50": 2.0,
        "EMA10 Rising": True,
    }
    row.update(overrides)
    return row


def test_entry_bucket_requires_trend_momentum_and_reward_risk():
    result = classify_steve_algo_row(base_row())

    assert "Entry" in result["SteveAlgo Buckets"]
    assert result["SteveAlgo Primary Bucket"] == "Entry"
    assert result["Capital Authorized"] == "0%"


def test_white_up_pilot_bucket_for_early_breakout_not_full_entry():
    row = base_row(
        EMA20=97,
        EMA50=98,
        SMA50=110,
        Return_1M=0,
        **{"Return 1M %": -1.0, "Return 3M %": 4.0, "Return 6M %": 8.0},
    )

    result = classify_steve_algo_row(row)

    assert result["SteveAlgo Primary Bucket"] == "White Up"
    assert result["SteveAlgo Status"] == "Pilot"


def test_yellow_marks_entry_that_is_extended():
    result = classify_steve_algo_row(base_row(**{"ATR Extension EMA20": 2.5}))

    assert result["SteveAlgo Primary Bucket"] == "Yellow"
    assert result["SteveAlgo Status"] == "Buy Ext"


def test_avoids_low_liquidity_and_explains_reason():
    result = classify_steve_algo_row(base_row(**{"Daily $ Volume 20D": 1_000_000}))

    assert result["SteveAlgo Primary Bucket"] == "Avoid"
    assert "dollar volume" in result["SteveAlgo Reason"]


def test_market_cap_gate_can_be_disabled_for_missing_metadata():
    result = classify_steve_algo_row(base_row(**{"Market Cap": float("nan")}), SteveAlgoThresholds(min_market_cap=0))

    assert result["SteveAlgo Primary Bucket"] == "Entry"


def test_apply_watchlists_filters_avoids_and_sorts_entries_first():
    df = pd.DataFrame(
        [
            base_row(Ticker="WHITE", **{"Return 1M %": -2.0, "EMA20": 97, "EMA50": 98, "SMA50": 110}),
            base_row(Ticker="ENTRY"),
            base_row(Ticker="AVOID", **{"Daily $ Volume 20D": 1_000_000}),
        ]
    )

    out = apply_steve_algo_watchlists(df)

    assert out["Ticker"].tolist()[0] == "ENTRY"
    assert "AVOID" not in out["Ticker"].tolist()
    assert set(out["SteveAlgo Primary Bucket"]) == {"Entry", "White Up"}
