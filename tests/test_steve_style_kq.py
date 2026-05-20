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


def make_filters():
    return app.ScanFilters(
        min_price=5,
        min_avg_volume=200_000,
        top_percent=2,
        only_non_extended=False,
        min_extension_atr=0,
        moderate_extension_atr=3,
        max_extension_atr=5,
        high_extension_atr=7,
        hyper_extension_atr=11,
        min_breakout_pct=4,
        stockbee_min_price=5,
        stockbee_min_volume=1_000_000,
        breakout_lookback=20,
        min_dollar_volume=150_000_000,
        min_adr_pct=3.5,
    )


def test_steve_style_kq_includes_broad_kq_candidate_excluded_by_strict_filter():
    metrics = pd.DataFrame(
        [
            {
                "Ticker": "STRICT",
                "Date": "2026-05-20",
                "Price": 50.0,
                "Return 1M %": 300.0,
                "Return 3M %": 300.0,
                "Return 6M %": 300.0,
                "ADR 20D %": 6.0,
                "Price > SMA10": True,
                "Price > SMA20": True,
                "Avg Volume 20D": 4_000_000,
                "Daily $ Volume 20D": 200_000_000,
                "ATR Extension SMA50": 2.0,
                "Momentum Rank": 99.0,
            },
            {
                "Ticker": "BKSY_LIKE",
                "Date": "2026-05-20",
                "Price": 45.0,
                "Return 1M %": 20.7,
                "Return 3M %": 105.0,
                "Return 6M %": 242.0,
                "ADR 20D %": 12.6,
                "Price > SMA10": True,
                "Price > SMA20": True,
                "Avg Volume 20D": 2_365_890,
                "Daily $ Volume 20D": 106_867_200,
                "ATR Extension SMA50": 2.5,
                "Momentum Rank": 94.0,
            },
            {
                "Ticker": "LOW_DOLLAR_VOLUME",
                "Date": "2026-05-20",
                "Price": 10.0,
                "Return 1M %": 40.0,
                "Return 3M %": 120.0,
                "Return 6M %": 250.0,
                "ADR 20D %": 7.0,
                "Price > SMA10": True,
                "Price > SMA20": True,
                "Avg Volume 20D": 1_000_000,
                "Daily $ Volume 20D": 10_000_000,
                "ATR Extension SMA50": 2.0,
                "Momentum Rank": 98.0,
            },
        ]
    )

    strict = app.apply_qullamaggie_filter(metrics, make_filters())
    steve_style = app.apply_steve_style_qullamaggie_filter(metrics, make_filters())

    assert set(strict["Ticker"]) == {"STRICT"}
    assert set(steve_style["Ticker"]) == {"STRICT", "BKSY_LIKE"}
    assert "Steve-style KQ Score" in steve_style.columns
    assert "Steve-style KQ Reason" in steve_style.columns


def test_steve_style_kq_is_separate_from_strict_and_marks_overlap():
    metrics = pd.DataFrame(
        [
            {
                "Ticker": "STRICT",
                "Date": "2026-05-20",
                "Price": 50.0,
                "Return 1M %": 300.0,
                "Return 3M %": 300.0,
                "Return 6M %": 300.0,
                "ADR 20D %": 6.0,
                "Price > SMA10": True,
                "Price > SMA20": True,
                "Avg Volume 20D": 4_000_000,
                "Daily $ Volume 20D": 200_000_000,
                "ATR Extension SMA50": 2.0,
                "Momentum Rank": 99.0,
            },
            {
                "Ticker": "IRDM_LIKE",
                "Date": "2026-05-20",
                "Price": 43.0,
                "Return 1M %": 4.3,
                "Return 3M %": 90.0,
                "Return 6M %": 167.0,
                "ADR 20D %": 6.4,
                "Price > SMA10": True,
                "Price > SMA20": True,
                "Avg Volume 20D": 2_338_890,
                "Daily $ Volume 20D": 101_900_000,
                "ATR Extension SMA50": 3.5,
                "Momentum Rank": 91.0,
            },
        ]
    )

    steve_style = app.apply_steve_style_qullamaggie_filter(metrics, make_filters())
    overlap_by_ticker = dict(zip(steve_style["Ticker"], steve_style["Strict Qullamaggie Overlap"]))

    assert set(steve_style["Ticker"]) == {"STRICT", "IRDM_LIKE"}
    assert overlap_by_ticker["STRICT"] is True
    assert overlap_by_ticker["IRDM_LIKE"] is False
