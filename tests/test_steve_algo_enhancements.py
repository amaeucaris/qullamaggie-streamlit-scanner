from __future__ import annotations

from pathlib import Path

import pandas as pd

from qull_scanner.backtest_steve_algo import (
    apply_regime_filter,
    build_config_random_benchmark,
    compare_with_qullamaggie_candidates,
    export_daily_watchlist,
    load_market_metadata,
    simulate_capital_curve,
)


def test_load_market_metadata_normalizes_ticker_and_market_cap(tmp_path: Path):
    csv = tmp_path / "market_metadata.csv"
    csv.write_text("symbol,marketCap,sector\naapl,2500000000000,Technology\nmsft,,Technology\n")

    meta = load_market_metadata(csv)

    assert meta.loc[meta["Ticker"] == "AAPL", "Market Cap"].iloc[0] == 2_500_000_000_000
    assert pd.isna(meta.loc[meta["Ticker"] == "MSFT", "Market Cap"].iloc[0])
    assert "Market Cap Source" in meta.columns


def test_config_random_benchmark_preserves_trade_count_by_signal_date():
    actual_trades = pd.DataFrame(
        {
            "Signal Date": pd.to_datetime(["2026-01-02", "2026-01-02", "2026-01-03"]),
            "Ticker": ["AAA", "BBB", "CCC"],
        }
    )
    eligible = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-01-02", "2026-01-02", "2026-01-03", "2026-01-04"]),
            "Ticker": ["X", "Y", "Z", "Q"],
            "Price": [10, 11, 12, 13],
        }
    )

    sampled = build_config_random_benchmark(actual_trades, eligible, seed=7)

    assert len(sampled) == 3
    assert sampled["Date"].dt.strftime("%Y-%m-%d").value_counts().to_dict() == {"2026-01-02": 2, "2026-01-03": 1}
    assert set(sampled["SteveAlgo Primary Bucket"]) == {"Random Eligible"}


def test_apply_regime_filter_keeps_events_only_when_spy_and_qqq_above_sma():
    events = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-04"]),
            "Ticker": ["AAA", "BBB", "CCC"],
        }
    )
    regime = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-04"]),
            "SPY Regime OK": [True, False, True],
            "QQQ Regime OK": [True, True, False],
        }
    )

    filtered = apply_regime_filter(events, regime)

    assert filtered["Ticker"].tolist() == ["AAA"]
    assert filtered["Regime Filter"].tolist() == ["SPY_AND_QQQ_OK"]


def test_export_daily_watchlist_writes_csv_and_obsidian_markdown(tmp_path: Path):
    watchlist = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-01-02", "2026-01-02"]),
            "Ticker": ["AAA", "BBB"],
            "SteveAlgo Primary Bucket": ["Entry", "White Up"],
            "Price": [10.5, 20.0],
            "Momentum Rank": [99, 95],
            "Reward-Risk": [4.2, 3.1],
            "SteveAlgo Reason": ["strong", "pivot"],
        }
    )

    paths = export_daily_watchlist(watchlist, tmp_path, tmp_path / "obsidian", as_of="2026-01-02")

    assert paths["csv"].exists()
    assert paths["markdown"].exists()
    md = paths["markdown"].read_text()
    assert "AAA" in md and "Capital authorized: 0%" in md


def test_compare_with_qullamaggie_candidates_marks_overlap():
    steve = pd.DataFrame({"Ticker": ["AAA", "BBB"], "SteveAlgo Primary Bucket": ["Entry", "White Up"]})
    qull = pd.DataFrame({"Ticker": ["BBB", "CCC"]})

    out = compare_with_qullamaggie_candidates(steve, qull)

    assert out.set_index("Ticker").loc["AAA", "Qullamaggie Overlap"] is False
    assert out.set_index("Ticker").loc["BBB", "Qullamaggie Overlap"] is True


def test_simulate_capital_curve_compounds_r_multiple_with_fixed_risk_fraction():
    trades = pd.DataFrame(
        {
            "Exit Date": pd.to_datetime(["2026-01-03", "2026-01-04", "2026-01-05"]),
            "R": [1.0, -1.0, 2.0],
        }
    )

    summary = simulate_capital_curve(trades, initial_capital=10_000, risk_fraction=0.01)

    assert summary["initial_capital"] == 10_000
    assert summary["risk_fraction_pct"] == 1.0
    assert summary["final_capital"] == 10198.98
    assert summary["total_return_pct"] == 1.99
