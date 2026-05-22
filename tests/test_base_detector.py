import pandas as pd

from qull_scanner.setup import detect_base_setup, add_base_setup_columns


def make_base_history():
    dates = pd.date_range("2025-01-01", periods=80, freq="B")
    closes = [20 + i * 0.5 for i in range(40)]  # prior move: 20 -> 39.5
    closes += [39.0, 40.0, 41.5, 42.0, 41.0, 40.5, 42.5, 43.0, 42.0, 41.2]
    closes += [42.8, 43.5, 44.0, 43.2, 42.4, 43.8, 44.5, 44.0, 43.5, 44.8]
    closes += [45.0] * 19 + [46.0]
    highs = [c * 1.01 for c in closes]
    lows = [c * 0.99 for c in closes]
    # Force the consolidation pivot/base low to known values in bars -40:-1.
    for idx in range(40, 60):
        highs[idx] = min(max(highs[idx], 40.0), 45.0)
        lows[idx] = max(min(lows[idx], 44.0), 40.0)
    highs[58] = 45.0
    lows[44] = 40.0
    for idx in range(60, 80):
        highs[idx] = 45.0
        lows[idx] = 44.0
    lows[64] = 40.0
    return pd.DataFrame(
        {
            "Open": closes,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": [1_000_000] * len(closes),
        },
        index=dates,
    )


def test_detect_base_setup_returns_pivot_base_low_depth_and_distance_without_lookahead():
    history = make_base_history()

    setup = detect_base_setup(history, base_window=20, prior_window=40)

    assert setup is not None
    assert setup.pivot == 45.0
    assert setup.base_low == 40.0
    assert setup.base_depth_pct == 11.11
    assert setup.distance_to_pivot_pct == 2.22
    assert setup.prior_move_pct > 30.0


def test_add_base_setup_columns_enriches_rows_from_history_map():
    history = make_base_history()
    metrics = pd.DataFrame([{"Ticker": "TEST", "Price": 45.0, "Breakout Level": 44.0}])

    enriched = add_base_setup_columns(metrics, {"TEST": history}, base_window=20, prior_window=40)
    row = enriched.iloc[0]

    assert row["Base Pivot"] == 45.0
    assert row["Base Low"] == 40.0
    assert row["Base Depth %"] == 11.11
    assert row["Distance to Pivot %"] == 2.22
    assert row["Prior Move %"] > 30.0
    assert bool(row["MA Surfing 10/20"]) is True
