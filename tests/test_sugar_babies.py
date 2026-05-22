import pandas as pd

from qull_scanner.sugar_babies import (
    SUGAR_BABIES_PERIODS,
    build_sugar_babies_metrics,
    sugar_baby_counts,
    sugar_baby_event,
)


def make_history(closes, volumes):
    return pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", periods=len(closes), freq="B"),
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": volumes,
        }
    )


def test_sugar_baby_event_requires_four_percent_upday_higher_volume_and_nine_million_volume():
    history = make_history(
        closes=[100, 104, 109, 112, 117, 122],
        volumes=[8_000_000, 9_000_000, 8_000_000, 10_000_000, 10_000_000, 12_000_000],
    )

    event = sugar_baby_event(history)

    assert event.tolist() == [False, True, False, False, False, True]


def test_sugar_baby_counts_match_tc2000_counttrue_windows_without_lookahead():
    history = make_history(
        closes=[100, 104, 103, 108, 113, 112, 118],
        volumes=[9_000_000, 10_000_000, 11_000_000, 12_000_000, 13_000_000, 14_000_000, 15_000_000],
    )

    counts = sugar_baby_counts(history, periods=[3, 5, 7])

    assert counts == {"SB 9/3": 2, "SB 9/5": 3, "SB 9/7": 4}


def test_build_sugar_babies_metrics_unions_top_n_per_period_and_tracks_hit_windows_and_best_rank():
    history = {
        "LONG_LEADER": make_history([100, 104, 108.2, 112.6, 117.2, 122.0], [9_000_000, 10_000_000, 11_000_000, 12_000_000, 13_000_000, 14_000_000]),
        "RECENT_LEADER": make_history([100, 101, 102, 106.2, 110.5, 115.0], [9_000_000, 9_100_000, 9_200_000, 10_000_000, 11_000_000, 12_000_000]),
        "OLD_ONLY": make_history([100, 104, 108.2, 112.6, 117.2, 116], [9_000_000, 10_000_000, 11_000_000, 12_000_000, 13_000_000, 14_000_000]),
        "NOISE": make_history([100, 101, 102, 103, 104, 105], [9_000_000, 10_000_000, 11_000_000, 12_000_000, 13_000_000, 14_000_000]),
    }

    result = build_sugar_babies_metrics(history, periods=[6, 3], top_n=2)

    assert set(result["Ticker"]) == {"LONG_LEADER", "RECENT_LEADER", "OLD_ONLY"}
    assert result.loc[result["Ticker"] == "LONG_LEADER", "SB 9/6"].iloc[0] == 5
    assert result.loc[result["Ticker"] == "RECENT_LEADER", "SB 9/3"].iloc[0] == 3
    assert result.loc[result["Ticker"] == "LONG_LEADER", "SB Hit Windows"].iloc[0] == 2
    assert result.loc[result["Ticker"] == "OLD_ONLY", "SB Best Rank"].iloc[0] == 2


def test_sugar_baby_counts_accepts_single_ticker_yfinance_multiindex_shape():
    history = pd.DataFrame(
        {
            ("Close", "MARA"): [100, 104, 103, 108],
            ("Volume", "MARA"): [8_000_000, 9_000_000, 10_000_000, 11_000_000],
        }
    )

    counts = sugar_baby_counts(history, periods=[3])

    assert counts == {"SB 9/3": 2}


def test_sugar_babies_periods_match_stockbee_tc2000_columns():
    assert SUGAR_BABIES_PERIODS == [1450, 1260, 1008, 756, 504, 252, 126, 50, 20, 10, 5]
