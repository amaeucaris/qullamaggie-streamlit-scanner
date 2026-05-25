import pandas as pd

from qull_scanner.backtest_steve_algo import (
    SteveBacktestConfig,
    simulate_steve_algo_trades,
    summarize_steve_algo_backtest,
)


def test_signal_enters_next_open_and_exits_on_target_or_timeout():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    history = {
        "AAA": pd.DataFrame(
            {
                "Open": [10.0, 11.0, 12.0, 13.0],
                "High": [10.5, 12.5, 13.5, 14.0],
                "Low": [9.5, 10.8, 11.8, 12.8],
                "Close": [10.0, 12.0, 13.0, 13.5],
                "Darvas Lower": [9.0, 9.0, 9.0, 9.0],
                "EMA20": [9.5, 10.0, 11.0, 12.0],
                "ATR20": [1.0, 1.0, 1.0, 1.0],
            },
            index=dates,
        )
    }
    events = pd.DataFrame([{"Ticker": "AAA", "Date": dates[0], "SteveAlgo Primary Bucket": "Entry"}])

    trades = simulate_steve_algo_trades(events, history, SteveBacktestConfig(max_hold_bars=2, target_r=1.0))

    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["Entry Date"] == dates[1]
    assert trade["Entry Price"] == 11.0
    assert trade["Exit Reason"] == "target"
    assert trade["R"] == 1.0


def test_stop_hit_intraday_uses_stop_exit_and_negative_one_r():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    history = {
        "AAA": pd.DataFrame(
            {
                "Open": [10.0, 11.0, 10.0],
                "High": [10.5, 11.2, 10.5],
                "Low": [9.5, 8.8, 9.0],
                "Close": [10.0, 9.0, 10.0],
                "Darvas Lower": [9.0, 9.0, 9.0],
                "EMA20": [9.5, 9.5, 9.5],
                "ATR20": [1.0, 1.0, 1.0],
            },
            index=dates,
        )
    }
    events = pd.DataFrame([{"Ticker": "AAA", "Date": dates[0], "SteveAlgo Primary Bucket": "White Up"}])

    trades = simulate_steve_algo_trades(events, history, SteveBacktestConfig(max_hold_bars=5))

    assert trades.iloc[0]["Exit Reason"] == "stop"
    assert trades.iloc[0]["R"] == -1.0


def test_overlapping_signals_same_ticker_are_ignored_until_prior_exit():
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    history = {
        "AAA": pd.DataFrame(
            {
                "Open": [10.0, 11.0, 11.0, 11.0, 11.0, 11.0],
                "High": [10.5, 11.2, 11.2, 11.2, 11.2, 11.2],
                "Low": [9.5, 10.5, 10.5, 10.5, 10.5, 10.5],
                "Close": [10.0, 11.0, 11.0, 11.0, 11.0, 11.0],
                "Darvas Lower": [9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
                "EMA20": [9.5, 10.0, 10.0, 10.0, 10.0, 10.0],
                "ATR20": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            },
            index=dates,
        )
    }
    events = pd.DataFrame(
        [
            {"Ticker": "AAA", "Date": dates[0], "SteveAlgo Primary Bucket": "Entry"},
            {"Ticker": "AAA", "Date": dates[1], "SteveAlgo Primary Bucket": "Entry"},
        ]
    )

    trades = simulate_steve_algo_trades(events, history, SteveBacktestConfig(max_hold_bars=3, target_r=None))

    assert len(trades) == 1


def test_random_benchmark_samples_same_dates_and_trade_count():
    from qull_scanner.backtest_steve_algo import build_random_event_benchmark

    dates = pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"])
    eligible = pd.DataFrame(
        {
            "Date": dates,
            "Ticker": ["AAA", "BBB", "CCC", "DDD"],
            "Momentum Rank": [90, 80, 70, 60],
        }
    )
    actual = pd.DataFrame(
        {
            "Signal Date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "Ticker": ["AAA", "CCC"],
        }
    )

    sampled = build_random_event_benchmark(actual, eligible, seed=7)

    assert len(sampled) == 2
    assert sampled["Date"].tolist() == actual["Signal Date"].tolist()
    assert set(sampled["Ticker"]).issubset(set(eligible["Ticker"]))


def test_oos_summary_splits_by_signal_date():
    from qull_scanner.backtest_steve_algo import summarize_in_sample_out_of_sample

    trades = pd.DataFrame(
        {
            "Signal Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "R": [1.0, 1.0, -1.0, -1.0],
            "Bucket": ["Entry", "Entry", "Entry", "Entry"],
            "Exit Reason": ["target", "target", "stop", "stop"],
        }
    )

    split = summarize_in_sample_out_of_sample(trades, split_fraction=0.5)

    assert split["split_date"] == "2024-01-03"
    assert split["in_sample"]["expectancy_r"] == 1.0
    assert split["out_of_sample"]["expectancy_r"] == -1.0


def test_summary_reports_expectancy_profit_factor_and_drawdown():
    trades = pd.DataFrame({"R": [1.0, -1.0, 2.0], "Bucket": ["Entry", "Entry", "White Up"]})

    summary = summarize_steve_algo_backtest(trades)

    assert summary["trades"] == 3
    assert summary["expectancy_r"] == 0.6667
    assert summary["profit_factor"] == 3.0
    assert summary["max_drawdown_r"] == -1.0
