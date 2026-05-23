import pandas as pd

from qull_scanner.filters import (
    ScannerThresholds,
    apply_guru_filter,
    apply_minervini_filter,
    apply_qullamaggie_filter,
    apply_steve_style_qullamaggie_filter,
    apply_stockbee_9m_movers_filter,
    apply_stockbee_filter,
)


def thresholds():
    return ScannerThresholds(
        min_price=5,
        min_avg_volume=200_000,
        top_percent=25,
        min_breakout_pct=4,
        stockbee_min_price=5,
        stockbee_min_volume=1_000_000,
        min_dollar_volume=150_000_000,
        min_adr_pct=3.5,
        max_extension_atr=5,
    )


def metrics_frame():
    return pd.DataFrame(
        [
            {
                "Ticker": "STRICT",
                "Price": 50.0,
                "Return 1M %": 100,
                "Return 3M %": 300,
                "Return 6M %": 600,
                "ADR 20D %": 6.0,
                "Price > SMA10": True,
                "Price > SMA20": True,
                "Avg Volume 20D": 4_000_000,
                "Daily $ Volume 20D": 200_000_000,
                "Momentum Rank": 99,
                "ATR Extension SMA50": 2.0,
                "Daily Return %": 5.0,
                "Volume": 2_000_000,
                "Prev Volume": 1_000_000,
                "Minervini Trend Template": True,
                "Green Candle": True,
            },
            {
                "Ticker": "BROAD_ONLY",
                "Price": 40.0,
                "Return 1M %": 20,
                "Return 3M %": 90,
                "Return 6M %": 170,
                "ADR 20D %": 6.0,
                "Price > SMA10": True,
                "Price > SMA20": True,
                "Avg Volume 20D": 3_000_000,
                "Daily $ Volume 20D": 80_000_000,
                "Momentum Rank": 91,
                "ATR Extension SMA50": 3.0,
                "Daily Return %": 3.0,
                "Volume": 2_000_000,
                "Prev Volume": 3_000_000,
                "Minervini Trend Template": False,
                "Green Candle": True,
            },
            {
                "Ticker": "ILLIQUID",
                "Price": 30.0,
                "Return 1M %": 90,
                "Return 3M %": 250,
                "Return 6M %": 500,
                "ADR 20D %": 8.0,
                "Price > SMA10": True,
                "Price > SMA20": True,
                "Avg Volume 20D": 50_000,
                "Daily $ Volume 20D": 1_500_000,
                "Momentum Rank": 98,
                "ATR Extension SMA50": 2.0,
                "Daily Return %": 5.0,
                "Volume": 50_000,
                "Prev Volume": 25_000,
                "Minervini Trend Template": True,
                "Green Candle": True,
            },
            {
                "Ticker": "LOW_ADR",
                "Price": 60.0,
                "Return 1M %": 80,
                "Return 3M %": 200,
                "Return 6M %": 400,
                "ADR 20D %": 1.0,
                "Price > SMA10": True,
                "Price > SMA20": True,
                "Avg Volume 20D": 4_000_000,
                "Daily $ Volume 20D": 240_000_000,
                "Momentum Rank": 97,
                "ATR Extension SMA50": 2.0,
                "Daily Return %": 3.0,
                "Volume": 2_000_000,
                "Prev Volume": 1_000_000,
                "Minervini Trend Template": True,
                "Green Candle": False,
            },
        ]
    )


def test_strict_qullamaggie_filter_requires_intersection_liquidity_adr_and_trend():
    result = apply_qullamaggie_filter(metrics_frame(), thresholds())

    assert list(result["Ticker"]) == ["STRICT"]
    assert {"Top 25% 1M", "Top 25% 3M", "Top 25% 6M"}.issubset(result.columns)
    assert "Strict Q Lineage" in result.columns
    assert "Dollar volume 20D" in result.iloc[0]["Strict Q Lineage"]


def test_steve_style_filter_keeps_broad_candidates_separate_from_strict():
    result = apply_steve_style_qullamaggie_filter(metrics_frame(), thresholds())

    assert set(result["Ticker"]) == {"STRICT", "BROAD_ONLY"}
    strict = result.loc[result["Ticker"] == "STRICT"].iloc[0]
    broad = result.loc[result["Ticker"] == "BROAD_ONLY"].iloc[0]
    assert strict["Strict Qullamaggie Overlap"] is True
    assert broad["Strict Qullamaggie Overlap"] is False
    assert "Steve-style KQ Score" in result.columns


def test_stockbee_minervini_and_guru_filters_are_independent_helpers():
    df = metrics_frame()

    assert set(apply_stockbee_filter(df, thresholds())["Ticker"]) == {"STRICT"}
    assert set(apply_minervini_filter(df)["Ticker"]) == {"STRICT", "ILLIQUID"}
    assert set(apply_guru_filter(apply_qullamaggie_filter(df, thresholds()), apply_minervini_filter(df))["Ticker"]) == {"STRICT"}


def test_stockbee_9m_movers_means_nine_million_volume_not_nine_month_return():
    df = pd.DataFrame(
        [
            {"Ticker": "NINE_MOVER", "Daily Return %": 5.0, "Volume": 9_100_000, "Prev Volume": 8_000_000, "Price": 12.0, "Return 9M %": -20.0},
            {"Ticker": "LOW_VOLUME", "Daily Return %": 8.0, "Volume": 8_000_000, "Prev Volume": 7_000_000, "Price": 12.0, "Return 9M %": 500.0},
            {"Ticker": "NO_VOLUME_EXPANSION", "Daily Return %": 6.0, "Volume": 10_000_000, "Prev Volume": 11_000_000, "Price": 12.0, "Return 9M %": 500.0},
            {"Ticker": "LOW_DAILY_MOVE", "Daily Return %": 3.0, "Volume": 12_000_000, "Prev Volume": 10_000_000, "Price": 12.0, "Return 9M %": 500.0},
        ]
    )

    result = apply_stockbee_9m_movers_filter(df, thresholds())

    assert list(result["Ticker"]) == ["NINE_MOVER"]
