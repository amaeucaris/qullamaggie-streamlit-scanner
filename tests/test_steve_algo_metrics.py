import math

import pandas as pd

from qull_scanner.metrics import (
    daily_close_range,
    gap_pct,
    open_to_close_pct,
    true_range,
    atr,
    darvas_levels,
    rolling_52w_position,
)


def test_daily_close_range_maps_low_mid_high_to_0_50_100():
    high = pd.Series([10.0, 10.0, 10.0])
    low = pd.Series([0.0, 0.0, 0.0])
    close = pd.Series([0.0, 5.0, 10.0])

    result = daily_close_range(high, low, close)

    assert result.tolist() == [0.0, 50.0, 100.0]


def test_gap_and_open_to_close_percent_use_correct_denominators():
    assert gap_pct(105.0, 100.0) == 5.0
    assert open_to_close_pct(100.0, 110.0) == 10.0
    assert math.isnan(gap_pct(105.0, 0.0))


def test_true_range_includes_previous_close_gap():
    high = pd.Series([10.0, 13.0, 12.0])
    low = pd.Series([8.0, 11.0, 7.0])
    close = pd.Series([9.0, 12.0, 8.0])

    result = true_range(high, low, close)

    assert result.tolist() == [2.0, 4.0, 5.0]


def test_atr_is_rolling_mean_of_true_range_for_backtest_determinism():
    high = pd.Series([10.0, 13.0, 12.0])
    low = pd.Series([8.0, 11.0, 7.0])
    close = pd.Series([9.0, 12.0, 8.0])

    result = atr(high, low, close, length=2)

    assert math.isnan(result.iloc[0])
    assert result.iloc[1] == 3.0
    assert result.iloc[2] == 4.5


def test_darvas_levels_are_shifted_to_avoid_same_bar_lookahead():
    high = pd.Series([10.0, 11.0, 12.0, 13.0])
    low = pd.Series([7.0, 6.0, 5.0, 4.0])

    upper, lower = darvas_levels(high, low, length=2)

    assert math.isnan(upper.iloc[0])
    assert math.isnan(upper.iloc[1])
    assert upper.iloc[2] == 11.0
    assert lower.iloc[2] == 6.0
    assert upper.iloc[3] == 12.0
    assert lower.iloc[3] == 5.0


def test_rolling_52w_position_is_percent_between_prior_low_and_high():
    close = pd.Series([10.0, 20.0, 15.0, 18.0])

    result = rolling_52w_position(close, length=3)

    assert math.isnan(result.iloc[0])
    assert math.isnan(result.iloc[1])
    assert math.isnan(result.iloc[2])
    assert result.iloc[3] == 80.0
