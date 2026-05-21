import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd


class _StreamlitStub(types.SimpleNamespace):
    def set_page_config(self, *args, **kwargs):
        return None

    def cache_data(self, *args, **kwargs):
        def decorator(fn):
            return fn
        return decorator


class _ColumnConfigStub(types.SimpleNamespace):
    def NumberColumn(self, *args, **kwargs):
        return None


st_stub = _StreamlitStub(column_config=_ColumnConfigStub())
sys.modules.setdefault("streamlit", st_stub)
sys.modules.setdefault("yfinance", types.SimpleNamespace(download=lambda *args, **kwargs: pd.DataFrame()))
plotly_module = types.ModuleType("plotly")
go_module = types.ModuleType("plotly.graph_objects")
go_module.Figure = lambda *args, **kwargs: types.SimpleNamespace(
    add_trace=lambda *a, **k: None,
    add_hline=lambda *a, **k: None,
    add_vline=lambda *a, **k: None,
    update_layout=lambda *a, **k: None,
)
sys.modules.setdefault("plotly", plotly_module)
sys.modules.setdefault("plotly.graph_objects", go_module)

spec = importlib.util.spec_from_file_location("app", Path(__file__).resolve().parents[1] / "app.py")
app = importlib.util.module_from_spec(spec)
sys.modules["app"] = app
spec.loader.exec_module(app)


def make_history(closes=None, volumes=None, rows=260):
    if closes is None:
        closes = [20.0] * rows
    if volumes is None:
        volumes = [1_000_000] * rows
    dates = pd.date_range("2025-01-01", periods=rows, freq="B")
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c * 1.04 for c in closes],
            "Low": [c * 0.96 for c in closes],
            "Close": closes,
            "Volume": volumes,
        },
        index=dates,
    )


def test_calculate_metrics_outputs_true_rolling_dollar_volume_20d_not_last_price_times_avg_volume():
    closes = [10.0] * 240 + list(range(11, 31))
    volumes = [1_000_000] * 260
    history = {"TEST": make_history(closes=closes, volumes=volumes)}

    metrics, _ = app.calculate_metrics(history, breakout_lookback=20)

    row = metrics.loc[metrics["Ticker"] == "TEST"].iloc[0]
    expected = sum(c * v for c, v in zip(closes[-20:], volumes[-20:])) / 20
    fallback = row["Price"] * row["Avg Volume 20D"]
    assert row["Daily $ Volume 20D"] == expected
    assert row["Daily $ Volume 20D"] != fallback


def test_steve_dashboard_fields_preserve_true_rolling_dollar_volume():
    metrics = pd.DataFrame(
        [
            {
                "Ticker": "TEST",
                "Price": 30.0,
                "Avg Volume 20D": 1_000_000,
                "Daily $ Volume 20D": 20_500_000.0,
                "Momentum Rank": 90.0,
                "ATR Extension SMA50": 2.0,
                "Daily Return %": 1.0,
                "Prev Close": 29.5,
                "52W High": 40.0,
                "SMA10": 29.0,
                "SMA20": 28.0,
                "SMA50": 25.0,
                "SMA150": 20.0,
                "SMA200": 18.0,
            }
        ]
    )

    enriched = app.add_steve_dashboard_fields(metrics)

    assert enriched.iloc[0]["Daily $ Volume 20D"] == 20_500_000.0
    assert enriched.iloc[0]["Daily $ Volume 20D"] != 30_000_000.0


def test_stop_to_adr_ratio_classifies_trade_risk_buckets():
    from qull_scanner.trade_plan import classify_stop_to_adr

    assert classify_stop_to_adr(0.70) == "A+"
    assert classify_stop_to_adr(1.00) == "OK"
    assert classify_stop_to_adr(1.25) == "Wide"
    assert classify_stop_to_adr(1.51) == "Reject"


def test_add_trade_plan_columns_enriches_scanner_candidates_for_export():
    from qull_scanner.trade_plan import add_trade_plan_columns

    candidates = pd.DataFrame(
        [
            {
                "Ticker": "TEST",
                "Price": 100.0,
                "Breakout Level": 99.0,
                "Base Low": 94.0,
                "ADR 20D %": 4.0,
                "Daily $ Volume 20D": 180_000_000.0,
                "ATR Extension SMA50": 2.5,
            }
        ]
    )

    enriched = add_trade_plan_columns(candidates, setup_type="Strict Q Breakout")
    row = enriched.iloc[0]

    assert row["Trade Setup Type"] == "Strict Q Breakout"
    assert row["Trade Entry Trigger"] == 100.10
    assert row["Trade Stop"] == 94.00
    assert row["Trade Risk %"] == 6.09
    assert row["Stop / ADR"] == 1.52
    assert row["Stop Bucket"] == "Reject"


def test_add_trade_plan_columns_handles_missing_base_low_with_sma20_fallback():
    from qull_scanner.trade_plan import add_trade_plan_columns

    candidates = pd.DataFrame(
        [
            {
                "Ticker": "NOBASE",
                "Price": 50.0,
                "Breakout Level": 51.0,
                "Base Low": pd.NA,
                "SMA20": 47.0,
                "ADR 20D %": 5.0,
                "Daily $ Volume 20D": 200_000_000.0,
            }
        ]
    )

    enriched = add_trade_plan_columns(candidates, setup_type="Strict Q Breakout")

    assert enriched.iloc[0]["Trade Stop"] == 47.0
    assert enriched.iloc[0]["Trade Risk %"] > 0


def test_format_output_keeps_trade_plan_columns_visible_near_breakout_level():
    output = app.format_output(
        pd.DataFrame(
            [
                {
                    "Ticker": "TEST",
                    "Price": 100.0,
                    "Trade Setup Type": "Strict Q Breakout",
                    "Trade Entry Trigger": 100.10,
                    "Trade Stop": 94.00,
                    "Trade Risk %": 6.09,
                    "Stop / ADR": 1.52,
                    "Stop Bucket": "Reject",
                    "Breakout Level": 99.0,
                }
            ]
        )
    )

    columns = list(output.columns)
    assert columns[columns.index("Trade Setup Type") : columns.index("Breakout Level") + 1] == [
        "Trade Setup Type",
        "Trade Entry Trigger",
        "Trade Stop",
        "Trade Risk %",
        "Stop / ADR",
        "Stop Bucket",
        "Breakout Level",
    ]


def test_lineage_explains_strict_qullamaggie_pass_conditions():
    from qull_scanner.lineage import strict_qullamaggie_lineage

    row = {
        "Ticker": "TEST",
        "Top 2% 1M": True,
        "Top 2% 3M": False,
        "Top 2% 6M": True,
        "ADR 20D %": 4.2,
        "Daily $ Volume 20D": 175_000_000,
        "Price > SMA10": True,
        "Price > SMA20": True,
        "ATR Extension SMA50": 2.7,
    }

    lines = strict_qullamaggie_lineage(row, min_adr_pct=3.5, min_dollar_volume=150_000_000)

    assert "Top 2% 3M: FAIL" in lines
    assert "ADR20: 4.20% >= 3.50%: PASS" in lines
    assert "Dollar volume 20D: $175.0M >= $150.0M: PASS" in lines
