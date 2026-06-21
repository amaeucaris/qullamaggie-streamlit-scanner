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
    assert app.framework_options() == ["Dashboard", "Strategy Lab", "Breadth", "Qullamaggie", "SteveAlgo", "Stockbee", "Quality Filters"]

    dashboard_views = app.view_options_for_scanner_group("Dashboard")
    strategy_lab_views = app.view_options_for_scanner_group("Strategy Lab")
    breadth_views = app.view_options_for_scanner_group("Breadth")
    qullamaggie_views = app.view_options_for_scanner_group("Qullamaggie")
    steve_views = app.view_options_for_scanner_group("SteveAlgo")
    stockbee_views = app.view_options_for_scanner_group("Stockbee")
    quality_views = app.view_options_for_scanner_group("Quality Filters")

    assert dashboard_views == ["Daily Dashboard", "Strategy Learning Lab"]
    assert strategy_lab_views == ["Theme Hits Board", "Breadth / Breath Board"]
    assert breadth_views == ["Breadth / Breath Board"]
    assert qullamaggie_views == ["Qullamaggie Top 2%", "Backtest Q"]
    assert "Steve-style KQ" in steve_views
    assert "Steve Dashboard" in steve_views
    assert "Stockbee 4% Breakout" not in qullamaggie_views
    assert "Sugar Babies SB" not in qullamaggie_views
    assert "Minervini" not in qullamaggie_views

    assert stockbee_views == ["Stockbee 4% Breakout", "Sugar Babies SB", "Stockbee + Sugar Baby Overlap"]
    assert "Minervini" in quality_views


def test_theme_hits_report_splits_themes_setup_timing_and_pivot_distance():
    metrics = pd.DataFrame(
        {
            "Ticker": ["MRNA", "AMN", "EVC", "SSRM", "WEAK", "ANTS", "LOOSE"],
            "Date": pd.to_datetime(["2026-06-16"] * 7),
            "Price": [55.40, 31.88, 9.15, 31.80, 10.0, 24.08, 31.00],
            "Prev Close": [52.10, 30.89, 9.18, 28.91, 10.6, 24.00, 30.30],
            "Close 2D Ago": [51.0, 30.4, 9.11, 27.5, 10.8, 24.15, 30.10],
            "Daily Return %": [6.3, 3.2, -0.3, 10.0, -5.0, 0.33, 0.25],
            "Volume Ratio 20D": [2.2, 1.0, 0.8, 1.6, 2.0, 1.15, 1.2],
            "Daily $ Volume 20D": [50_000_000, 20_000_000, 15_000_000, 30_000_000, 20_000_000, 18_000_000, 18_000_000],
            "ATR Extension SMA50": [1.8, 4.4, 3.0, 0.7, -1.0, 2.0, 2.0],
            "DCR %": [95, 80, 65, 92, 12, 62, 40],
            "ADR 20D %": [5, 4, 3, 6, 4, 3, 3],
            "Breakout Level": [52.13, 31.69, 9.93, 31.22, 11.0, 24.70, 31.50],
            "Breakout Above Lookback High": [True, True, False, True, False, False, False],
            "Minervini Trend Template": [True, True, True, True, False, True, True],
            "Price > SMA10": [True, True, True, True, False, True, True],
            "Price > SMA20": [True, True, True, True, False, True, True],
            "Price > SMA50": [True, True, True, True, False, True, True],
            "Universe Percentile": [95, 90, 80, 70, 10, 92, 92],
            "Return 1M %": [20, 10, 8, 12, -10, 18, 18],
            "Return 3M %": [25, 12, 6, 15, -20, 30, 30],
            "Return 6M %": [30, 20, 10, 18, -30, 35, 35],
        }
    )
    metadata = pd.DataFrame(
        {
            "Ticker": ["MRNA", "AMN", "EVC", "SSRM", "WEAK", "ANTS", "LOOSE"],
            "Sector": ["Healthcare", "Healthcare", "Communication Services", "Basic Materials", "Technology", "Technology", "Technology"],
            "Industry": ["Biotechnology", "Medical Care Facilities", "Advertising Agencies", "Gold", "Semiconductors", "Semiconductors", "Semiconductors"],
        }
    )

    report = app.build_theme_hits_report(metrics, metadata)

    assert report["as_of"] == "2026-06-16"
    assert report["kpis"]["confirmed_breakouts"] == 1
    assert report["kpis"]["early_breakouts"] == 1
    assert report["kpis"]["base_watch"] == 0
    assert report["kpis"]["stockbee_ants"] == 2
    assert report["kpis"]["chase_risk"] == 1
    assert report["setup_lists"]["confirmed_breakouts"].iloc[0]["Ticker"] == "MRNA"
    assert report["setup_lists"]["confirmed_breakouts"].iloc[0]["Pivot Position"] == "EXTENDED_FROM_PIVOT"
    assert report["setup_lists"]["early_breakouts"].iloc[0]["Ticker"] == "AMN"
    ants_row = report["setup_lists"]["stockbee_ants"].iloc[0]
    assert ants_row["Ticker"] == "ANTS"
    assert ants_row["Setup Status"] == "STOCKBEE_ANTS"
    assert ants_row["Stockbee Net Change"] == 0.08
    assert ants_row["Stockbee 3D Close Range %"] < 1.0
    assert "Biotech Momentum" in set(report["green_themes"]["theme"])
    assert "Semiconductors" in set(report["green_themes"]["theme"])


def _history_frame(dates, closes):
    close = pd.Series(closes, index=dates, dtype="float64")
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": 10_000_000,
        },
        index=dates,
    )


def test_breadth_report_builds_ariel_style_indicators_and_group_drilldown():
    dates = pd.bdate_range("2026-03-02", "2026-06-18")
    base = [100.0] * len(dates)
    strong = base.copy()
    strong[-64:] = [100.0] * 30 + [112.0] * 13 + [120.0] * 20 + [130.0]
    strong[-22] = 100.0
    stronger = base.copy()
    stronger[-64:] = [102.0] * 30 + [114.0] * 13 + [121.0] * 20 + [132.0]
    stronger[-22] = 102.0
    weak = base.copy()
    weak[-64:] = [100.0] * 30 + [90.0] * 13 + [80.0] * 20 + [74.0]
    weak[-22] = 100.0
    history = {
        "BIO1": _history_frame(dates, strong),
        "SEMI1": _history_frame(dates, stronger),
        "WEAK1": _history_frame(dates, weak),
    }
    metadata = pd.DataFrame(
        {
            "Ticker": ["BIO1", "SEMI1", "WEAK1"],
            "Sector": ["Healthcare", "Technology", "Technology"],
            "Industry": ["Biotechnology", "Semiconductors", "Software - Infrastructure"],
        }
    )

    report = app.build_breadth_report(history, metadata, lookback_dates=5)
    latest = report["table"].iloc[0]

    assert report["as_of"] == "2026-06-18"
    assert latest["Stocks Up 4%+ Today"] == 2
    assert latest["Stocks Down 4%+ Today"] == 1
    assert latest["Up 25%+ Quarter"] == 2
    assert latest["Down 25%+ Quarter"] == 1
    assert latest["Up 25%+ Month"] == 2
    assert latest["Down 25%+ Month"] == 1
    assert latest[">50dma"] == "66.7%"

    groups = app.breadth_group_drilldown(report["signals"], metadata, "2026-06-18", "Up 25%+ Quarter")
    assert list(groups["Group / Tickers"]) == ["Biotechnology", "Semiconductors"]
    assert list(groups["Count / Change %"]) == [1, 1]


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
