import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


def test_scanner_groups_separate_frameworks_by_operational_role():
    assert app.framework_options() == ["Dashboard", "Qullamaggie", "SteveAlgo", "Stockbee", "Quality Filters"]

    dashboard_views = app.view_options_for_scanner_group("Dashboard")
    qullamaggie_views = app.view_options_for_scanner_group("Qullamaggie")
    steve_views = app.view_options_for_scanner_group("SteveAlgo")
    stockbee_views = app.view_options_for_scanner_group("Stockbee")
    quality_views = app.view_options_for_scanner_group("Quality Filters")

    assert dashboard_views == ["Daily Dashboard", "Strategy Learning Lab"]
    assert qullamaggie_views == ["Qullamaggie Top 2%", "Backtest Q"]
    assert "Steve-style KQ" in steve_views
    assert "Steve Dashboard" in steve_views
    assert "Stockbee 4% Breakout" not in qullamaggie_views
    assert "Sugar Babies SB" not in qullamaggie_views
    assert "Minervini" not in qullamaggie_views

    assert stockbee_views == ["Stockbee 4% Breakout", "Sugar Babies SB", "Stockbee + Sugar Baby Overlap"]
    assert "Minervini" in quality_views


def test_scanner_frameworks_can_be_overridden_from_app_state():
    framework_map = {
        "Qullamaggie": ["Steve Dashboard", "Chart"],
        "Stockbee": ["Stockbee 4% Breakout"],
        "Minervini": ["Minervini", "Guru Q x Minervini"],
    }

    assert app.framework_options(framework_map) == ["Qullamaggie", "Stockbee", "Minervini"]
    assert app.view_options_for_scanner_group("Minervini", framework_map) == ["Minervini", "Guru Q x Minervini"]
    assert app.view_options_for_scanner_group("Qullamaggie", framework_map) == ["Steve Dashboard", "Chart"]


def test_steve_dashboard_context_can_filter_non_gurus_sections_without_mutating_sources():
    steve_all = pd.DataFrame({"Ticker": ["Q", "SB_ONLY", "OTHER"], "Value": [1, 2, 3]})
    stockbee = pd.DataFrame({"Ticker": ["Q", "SB_ONLY"], "Value": [10, 20]})
    q_screen = pd.DataFrame({"Ticker": ["Q"]})

    context_steve, context_stockbee = app.steve_dashboard_context_frames(
        steve_all,
        stockbee,
        q_screen,
        strict_q_context=True,
    )
    assert list(context_steve["Ticker"]) == ["Q"]
    assert list(context_stockbee["Ticker"]) == ["Q"]

    unfiltered_steve, unfiltered_stockbee = app.steve_dashboard_context_frames(
        steve_all,
        stockbee,
        q_screen,
        strict_q_context=False,
    )
    assert list(unfiltered_steve["Ticker"]) == ["Q", "SB_ONLY", "OTHER"]
    assert list(unfiltered_stockbee["Ticker"]) == ["Q", "SB_ONLY"]


def test_scanner_framework_config_drops_empty_and_unknown_views():
    dirty_map = {
        "": ["Steve Dashboard"],
        "Stockbee": ["Sugar Babies SB", "Not a real scanner"],
        "Empty": [],
    }

    assert app.normalize_scanner_frameworks(dirty_map) == {"Stockbee": ["Sugar Babies SB"]}


def test_steve_algo_metric_column_guard_detects_stale_precomputed_metrics():
    stale_metrics = pd.DataFrame({"Ticker": ["AAPL"], "Price": [100.0], "Momentum Rank": [95.0]})
    fresh_metrics = stale_metrics.assign(
        EMA10=99.0,
        EMA20=98.0,
        EMA50=97.0,
        ATR20=2.0,
        **{
            "DCR %": 80.0,
            "Darvas Upper": 101.0,
            "Darvas Lower": 90.0,
            "ATR Extension EMA10": 0.5,
            "ATR Extension EMA20": 1.0,
            "ATR Extension SMA50": 1.5,
            "EMA10 Rising": True,
        },
    )

    assert not app.has_steve_algo_metric_columns(stale_metrics)
    assert app.has_steve_algo_metric_columns(fresh_metrics)
