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


def test_merge_sugar_babies_with_metrics_adds_current_scanner_context():
    sugar_babies = pd.DataFrame(
        [
            {"Ticker": "MARA", "SB Hit Windows": 11, "SB Best Rank": 1, "SB 9/1450": 264, "SB 9/50": 11},
            {"Ticker": "RIOT", "SB Hit Windows": 10, "SB Best Rank": 2, "SB 9/1450": 246, "SB 9/50": 13},
        ]
    )
    metrics = pd.DataFrame(
        [
            {"Ticker": "MARA", "Price": 20.0, "ADR 20D %": 7.1, "Momentum Rank": 95.0},
            {"Ticker": "OTHER", "Price": 10.0, "ADR 20D %": 3.0, "Momentum Rank": 50.0},
        ]
    )

    merged = app.merge_sugar_babies_with_metrics(sugar_babies, metrics)

    mara = merged.loc[merged["Ticker"] == "MARA"].iloc[0]
    riot = merged.loc[merged["Ticker"] == "RIOT"].iloc[0]
    assert mara["Price"] == 20.0
    assert mara["ADR 20D %"] == 7.1
    assert pd.isna(riot["Price"])
    assert list(merged["Ticker"]) == ["MARA", "RIOT"]


def test_sort_sugar_babies_view_supports_actionable_and_tc2000_orders():
    sugar_babies = pd.DataFrame(
        [
            {"Ticker": "A", "SB Hit Windows": 8, "SB Best Rank": 2, "SB 9/1450": 300, "SB 9/252": 10, "SB 9/50": 3},
            {"Ticker": "B", "SB Hit Windows": 11, "SB Best Rank": 5, "SB 9/1450": 200, "SB 9/252": 20, "SB 9/50": 6},
            {"Ticker": "C", "SB Hit Windows": 11, "SB Best Rank": 1, "SB 9/1450": 100, "SB 9/252": 12, "SB 9/50": 8},
        ]
    )

    actionable = app.sort_sugar_babies_view(sugar_babies, "Actionable SB")
    tc2000 = app.sort_sugar_babies_view(sugar_babies, "Replica TC2000 9/1450")

    assert list(actionable["Ticker"]) == ["C", "B", "A"]
    assert list(tc2000["Ticker"]) == ["A", "B", "C"]
