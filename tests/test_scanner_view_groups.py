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


def test_scanner_groups_separate_qullamaggie_and_stockbee_progression():
    assert app.SCANNER_GROUPS == ["Qullamaggie", "Stockbee"]

    qullamaggie_views = app.view_options_for_scanner_group("Qullamaggie")
    stockbee_views = app.view_options_for_scanner_group("Stockbee")

    assert "Qullamaggie Top 2%" in qullamaggie_views
    assert "Steve-style KQ" in qullamaggie_views
    assert "Backtest Q" in qullamaggie_views
    assert "Stockbee 4% Breakout" not in qullamaggie_views
    assert "Sugar Babies SB" not in qullamaggie_views

    assert stockbee_views == ["Stockbee 4% Breakout", "Sugar Babies SB"]
