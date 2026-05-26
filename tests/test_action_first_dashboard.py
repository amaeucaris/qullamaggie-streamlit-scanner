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


sys.modules.setdefault("streamlit", _StreamlitStub(column_config=_ColumnConfigStub()))
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

spec = importlib.util.spec_from_file_location("app", REPO_ROOT / "app.py")
app = importlib.util.module_from_spec(spec)
sys.modules["app"] = app
spec.loader.exec_module(app)


def test_data_freshness_marks_market_data_stale_after_two_days():
    status = app.compute_data_freshness_status(
        updated_at="2026-05-23 01:41 UTC",
        last_market_date="2026-05-22",
        now=pd.Timestamp("2026-05-26 08:00:00Z"),
    )

    assert status["status"] == "STALE"
    assert status["last_market_date"] == "2026-05-22"
    assert status["last_update"] == "2026-05-23 01:41 UTC"
    assert "DATA STALE" in status["message"]


def test_data_freshness_marks_recent_market_data_fresh():
    status = app.compute_data_freshness_status(
        updated_at="2026-05-25T23:30:00Z",
        last_market_date="2026-05-25",
        now=pd.Timestamp("2026-05-26 08:00:00Z"),
    )

    assert status["status"] == "FRESH"
    assert "operativa" in status["message"]


def test_daily_shortlist_prioritizes_overlap_and_action_labels():
    metrics = pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB", "CCC"],
            "Price": [10.0, 20.0, 30.0],
            "Daily Return %": [5.0, 1.0, 2.0],
            "Momentum Rank": [95.0, 88.0, 80.0],
            "ADR 20D %": [6.0, 4.0, 3.0],
            "Daily $ Volume 20D": [200_000_000, 80_000_000, 70_000_000],
            "ATR Extension SMA50": [1.0, 8.0, 2.0],
            "Reward-Risk": [4.0, 2.0, 3.5],
        }
    )
    q = pd.DataFrame({"Ticker": ["AAA"]})
    stockbee = pd.DataFrame({"Ticker": ["AAA", "BBB"]})
    sugar = pd.DataFrame({"Ticker": ["AAA", "CCC"]})
    minervini = pd.DataFrame({"Ticker": ["AAA", "CCC"]})
    steve_algo = pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB", "CCC"],
            "SteveAlgo Primary Bucket": ["Entry", "Yellow", "White Up"],
            "SteveAlgo Status": ["Entry", "Watch", "Watch"],
            "SteveAlgo Reason": ["entry reason", "yellow reason", "white reason"],
        }
    )

    shortlist = app.build_daily_shortlist(
        metrics=metrics,
        q_screen=q,
        steve_style_kq_screen=pd.DataFrame({"Ticker": ["AAA", "CCC"]}),
        minervini_screen=minervini,
        stockbee_screen=stockbee,
        sugar_babies=sugar,
        steve_algo_watchlist=steve_algo,
        limit=10,
    )

    assert list(shortlist["Ticker"])[0] == "AAA"
    aaa = shortlist[shortlist["Ticker"] == "AAA"].iloc[0]
    bbb = shortlist[shortlist["Ticker"] == "BBB"].iloc[0]
    assert aaa["Next Action"] == "Review chart"
    assert "SteveAlgo Entry" in aaa["Frameworks confirmed"]
    assert "Overlap" in aaa["Reason"]
    assert bbb["Next Action"] == "Skip"
    assert bbb["Trade Readiness"] == "REJECT_RR"


def test_default_frameworks_are_action_first_and_not_mixed():
    assert app.framework_options() == ["Dashboard", "Qullamaggie", "SteveAlgo", "Stockbee", "Quality Filters"]
    assert app.view_options_for_scanner_group("Dashboard") == ["Daily Dashboard", "Strategy Learning Lab"]
    assert app.view_options_for_scanner_group("Stockbee") == [
        "Stockbee 4% Breakout",
        "Sugar Babies SB",
        "Stockbee + Sugar Baby Overlap",
    ]
    assert "Steve Dashboard" not in app.view_options_for_scanner_group("Qullamaggie")
    assert "Minervini" not in app.view_options_for_scanner_group("Qullamaggie")
