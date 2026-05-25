from __future__ import annotations

import math
import os
import time
import json
from dataclasses import dataclass
from html import escape
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from qull_scanner.filters import (
    ScannerThresholds,
    apply_guru_filter as core_apply_guru_filter,
    apply_minervini_filter as core_apply_minervini_filter,
    apply_qullamaggie_filter as core_apply_qullamaggie_filter,
    apply_steve_style_qullamaggie_filter as core_apply_steve_style_qullamaggie_filter,
    apply_stockbee_9m_movers_filter as core_apply_stockbee_9m_movers_filter,
    apply_stockbee_filter as core_apply_stockbee_filter,
)
from qull_scanner.backtest_steve_algo import SteveBacktestConfig, simulate_steve_algo_trades, summarize_steve_algo_backtest
from qull_scanner.metrics import (
    daily_close_range,
    darvas_levels,
    gap_pct,
    open_to_close_pct,
    rolling_52w_position,
    rolling_dollar_volume,
)
from qull_scanner.setup import add_base_setup_columns
from qull_scanner.steve_algo import SteveAlgoThresholds, apply_steve_algo_watchlists
from qull_scanner.sugar_babies import SUGAR_BABIES_PERIODS
from qull_scanner.trade_plan import add_trade_plan_columns


NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
EXPORT_DIR = Path("exports")
DATA_DIR = Path("data")
HISTORY_FILE = DATA_DIR / "history_prices.parquet"
METRICS_FILE = DATA_DIR / "scanner_metrics.parquet"
SUGAR_BABIES_FILE = DATA_DIR / "sugar_babies.parquet"
SUGAR_BABIES_METADATA_FILE = DATA_DIR / "sugar_babies_metadata.json"
METADATA_FILE = DATA_DIR / "metadata.json"
DEFAULT_BREAKOUT_LOOKBACK = 20
APP_BUILD_MARKER = "2026-05-23-steve-gurus-unfiltered-stockbee"
RETURN_WINDOWS = {
    "1W": 5,
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "9M": 189,
}
QULLAMAGGIE_VIEWS = [
    "Steve Dashboard",
    "Steve Algo Watchlist",
    "Steve Algo Backtest",
    "Steve-style KQ",
    "Qullamaggie Top 2%",
    "Backtest Q",
    "Guru Q x Minervini",
    "Minervini",
    "Extension Map",
    "Universo",
    "Chart",
]
STOCKBEE_VIEWS = ["Stockbee 4% Breakout", "Sugar Babies SB"]
DEFAULT_SCANNER_FRAMEWORKS = {
    "Qullamaggie": QULLAMAGGIE_VIEWS,
    "Stockbee": STOCKBEE_VIEWS,
}
SCANNER_GROUPS = list(DEFAULT_SCANNER_FRAMEWORKS)
ALL_SCANNER_VIEWS = list(dict.fromkeys(QULLAMAGGIE_VIEWS + STOCKBEE_VIEWS))


def normalize_scanner_frameworks(frameworks: dict[str, list[str]]) -> dict[str, list[str]]:
    """Return a clean framework -> scanner map editable from Streamlit state."""
    normalized: dict[str, list[str]] = {}
    for framework, views in frameworks.items():
        framework_name = str(framework).strip()
        if not framework_name:
            continue
        valid_views = [view for view in views if view in ALL_SCANNER_VIEWS]
        if valid_views:
            normalized[framework_name] = list(dict.fromkeys(valid_views))
    return normalized


def framework_options(frameworks: dict[str, list[str]] | None = None) -> list[str]:
    return list(normalize_scanner_frameworks(frameworks or DEFAULT_SCANNER_FRAMEWORKS))


def view_options_for_scanner_group(
    scanner_group: str,
    frameworks: dict[str, list[str]] | None = None,
) -> list[str]:
    scanner_frameworks = normalize_scanner_frameworks(frameworks or DEFAULT_SCANNER_FRAMEWORKS)
    return scanner_frameworks.get(scanner_group, QULLAMAGGIE_VIEWS).copy()


def scanner_framework_editor() -> dict[str, list[str]]:
    """Always-visible main-page UI to change framework grouping without editing app.py."""
    st.markdown("### ⚙️ Configura framework scanner")
    st.caption("Qui imposti quali scanner compaiono dentro ogni framework. Non e un segnale operativo.")
    base_frameworks = list(DEFAULT_SCANNER_FRAMEWORKS)
    custom_framework = st.text_input(
        "Nuovo framework opzionale",
        value="",
        placeholder="es. Minervini",
        help="Aggiunge un framework temporaneo nella sessione dell'app.",
    ).strip()
    framework_pool = base_frameworks.copy()
    if custom_framework and custom_framework not in framework_pool:
        framework_pool.append(custom_framework)

    frameworks = st.multiselect(
        "Framework visibili",
        options=framework_pool,
        default=framework_pool,
        help="Usalo per mostrare/nascondere framework nella radio principale.",
    )
    configured: dict[str, list[str]] = {}
    for framework in frameworks:
        configured[framework] = st.multiselect(
            f"Scanner in {framework}",
            options=ALL_SCANNER_VIEWS,
            default=DEFAULT_SCANNER_FRAMEWORKS.get(framework, []),
            key=f"scanner_views_{framework}",
        )
    return normalize_scanner_frameworks(configured) or DEFAULT_SCANNER_FRAMEWORKS.copy()


def export_section(name: str, df: pd.DataFrame, filename: str) -> None:
    csv_data = format_output(df).to_csv(index=False).encode("utf-8")
    export_bytes(name, csv_data, filename)


def export_raw_section(name: str, df: pd.DataFrame, filename: str) -> None:
    csv_data = df.to_csv(index=False).encode("utf-8")
    export_bytes(name, csv_data, filename)


def export_bytes(name: str, csv_data: bytes, filename: str) -> None:
    st.download_button(
        f"Download CSV {name}",
        csv_data,
        filename,
        "text/csv",
    )

    if st.button(f"Salva CSV {name} in exports/", key=f"save_{filename}", use_container_width=False):
        EXPORT_DIR.mkdir(exist_ok=True)
        path = EXPORT_DIR / filename
        path.write_bytes(csv_data)
        st.success(f"File salvato: {path.resolve()}")


@dataclass(frozen=True)
class ScanFilters:
    min_price: float
    min_avg_volume: int
    top_percent: float
    only_non_extended: bool
    min_extension_atr: float
    moderate_extension_atr: float
    max_extension_atr: float
    high_extension_atr: float
    hyper_extension_atr: float
    min_breakout_pct: float
    stockbee_min_price: float
    stockbee_min_volume: int
    breakout_lookback: int
    # Qullamaggie mandatory filters (per video timestamp 01:28:00)
    min_dollar_volume: int = 150_000_000  # $150M default; $15M small-account mode
    min_adr_pct: float = 3.5              # ADR 20D default; small-account mode often uses 5%


st.set_page_config(
    page_title="Qullamaggie NASDAQ Scanner",
    layout="wide",
)


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_symbols(universe: str, include_etfs: bool) -> pd.DataFrame:
    raw = pd.read_csv(NASDAQ_LISTED_URL, sep="|")
    raw = raw[raw["Symbol"].notna()]
    raw = raw[raw["Symbol"] != "File Creation Time"]
    raw = raw[raw["Test Issue"] == "N"].copy()
    raw = raw.rename(columns={"Security Name": "Name"})
    raw["Exchange"] = "NASDAQ"

    if not include_etfs and "ETF" in raw.columns:
        raw = raw[raw["ETF"] == "N"]

    frames = [raw[["Symbol", "Name", "Exchange", "ETF"]]]

    if universe == "All US listed":
        other = pd.read_csv(OTHER_LISTED_URL, sep="|")
        other = other[other["ACT Symbol"].notna()]
        other = other[other["ACT Symbol"] != "File Creation Time"]
        other = other[other["Test Issue"] == "N"].copy()

        if not include_etfs and "ETF" in other.columns:
            other = other[other["ETF"] == "N"]

        other = other.rename(
            columns={
                "ACT Symbol": "Symbol",
                "Security Name": "Name",
                "Exchange": "Exchange",
            }
        )
        frames.append(other[["Symbol", "Name", "Exchange", "ETF"]])

    symbols = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["Symbol"])
    symbols["yahoo_symbol"] = symbols["Symbol"].str.replace(".", "-", regex=False)
    return symbols.sort_values("Symbol")


@st.cache_data(ttl=60 * 30, show_spinner=False)
def download_price_history(
    tickers: tuple[str, ...],
    period: str,
    interval: str,
    chunk_size: int,
    pause_seconds: float,
) -> dict[str, pd.DataFrame]:
    history: dict[str, pd.DataFrame] = {}
    chunks = math.ceil(len(tickers) / chunk_size)

    for chunk_number in range(chunks):
        start = chunk_number * chunk_size
        batch = tickers[start : start + chunk_size]

        try:
            data = yf.download(
                tickers=list(batch),
                period=period,
                interval=interval,
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception:
            continue

        if data.empty:
            continue

        if isinstance(data.columns, pd.MultiIndex):
            for ticker in batch:
                if ticker not in data.columns.get_level_values(0):
                    continue
                frame = data[ticker].dropna(how="all").copy()
                if not frame.empty:
                    history[ticker] = frame
        else:
            ticker = batch[0]
            frame = data.dropna(how="all").copy()
            if not frame.empty:
                history[ticker] = frame

        if pause_seconds > 0 and chunk_number < chunks - 1:
            time.sleep(pause_seconds)

    return history


@st.cache_data(ttl=60 * 15, show_spinner=False)
def load_precomputed_history(path: str) -> dict[str, pd.DataFrame]:
    data = pd.read_parquet(path)
    if data.empty:
        return {}

    data["Date"] = pd.to_datetime(data["Date"])
    history: dict[str, pd.DataFrame] = {}
    for ticker, frame in data.groupby("Ticker", sort=False):
        ticker_frame = frame.drop(columns=["Ticker"]).set_index("Date").sort_index()
        history[str(ticker)] = ticker_frame
    return history


def save_history_to_parquet(history: dict[str, pd.DataFrame], path: Path = HISTORY_FILE) -> None:
    rows: list[pd.DataFrame] = []
    for ticker, frame in history.items():
        if frame.empty:
            continue
        output = frame.reset_index(names="Date").copy()
        output["Ticker"] = ticker
        rows.append(output)

    if not rows:
        raise ValueError("No price history to save.")

    path.parent.mkdir(exist_ok=True)
    pd.concat(rows, ignore_index=True).to_parquet(path, index=False)


@st.cache_data(ttl=60 * 15, show_spinner=False)
def load_precomputed_metrics(path: str) -> pd.DataFrame:
    metrics = pd.read_parquet(path)
    if "Date" in metrics.columns:
        metrics["Date"] = pd.to_datetime(metrics["Date"])
    return metrics


def save_metrics_to_parquet(metrics: pd.DataFrame, path: Path = METRICS_FILE) -> None:
    if metrics.empty:
        raise ValueError("No scanner metrics to save.")

    path.parent.mkdir(exist_ok=True)
    metrics.to_parquet(path, index=False)


@st.cache_data(ttl=60 * 15, show_spinner=False)
def load_precomputed_sugar_babies(path: str = str(SUGAR_BABIES_FILE)) -> pd.DataFrame:
    sugar_babies = pd.read_parquet(path)
    if "Updated At" in sugar_babies.columns:
        sugar_babies["Updated At"] = pd.to_datetime(sugar_babies["Updated At"])
    return sugar_babies


def merge_sugar_babies_with_metrics(sugar_babies: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    if sugar_babies.empty or metrics.empty:
        return sugar_babies.copy()

    context_columns = [
        "Ticker",
        "Date",
        "Price",
        "Momentum Rank",
        "Return 1M %",
        "Return 3M %",
        "Return 6M %",
        "ADR 20D %",
        "Daily Return %",
        "Volume",
        "Avg Volume 20D",
        "Daily $ Volume 20D",
        "ATR Extension SMA50",
        "% Extension SMA50",
        "Price > SMA10",
        "Price > SMA20",
        "Minervini Trend Template",
    ]
    available_context = [column for column in context_columns if column in metrics.columns]
    if available_context == ["Ticker"]:
        return sugar_babies.copy()

    return sugar_babies.merge(metrics[available_context], on="Ticker", how="left")


def sort_sugar_babies_view(sugar_babies: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    if sugar_babies.empty:
        return sugar_babies

    df = sugar_babies.copy()
    sort_options = {
        "Actionable SB": (["SB Hit Windows", "SB Best Rank", "SB 9/252", "SB 9/50", "SB 9/1450", "Ticker"], [False, True, False, False, False, True]),
        "Replica TC2000 9/1450": (["SB 9/1450", "SB 9/1260", "SB 9/1008", "Ticker"], [False, False, False, True]),
        "Recent SB 9/50": (["SB 9/50", "SB 9/20", "SB 9/10", "SB Best Rank", "Ticker"], [False, False, False, True, True]),
    }
    columns, ascending = sort_options.get(sort_by, sort_options["Actionable SB"])
    available = [(column, asc) for column, asc in zip(columns, ascending) if column in df.columns]
    if not available:
        return df
    return df.sort_values([column for column, _ in available], ascending=[asc for _, asc in available]).reset_index(drop=True)


def safe_return(close: pd.Series, days: int) -> float:
    if len(close.dropna()) <= days:
        return np.nan
    return (close.iloc[-1] / close.iloc[-days - 1] - 1) * 100


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()


def calculate_metrics(
    history: dict[str, pd.DataFrame],
    breakout_lookback: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows: list[dict[str, float | str | bool | pd.Timestamp]] = []
    enriched: dict[str, pd.DataFrame] = {}

    for ticker, frame in history.items():
        required = {"Open", "High", "Low", "Close", "Volume"}
        if frame.empty or not required.issubset(frame.columns):
            continue

        df = frame[list(required)].copy().dropna(subset=["Close"])
        if len(df) < 130:
            continue

        df["SMA10"] = sma(df["Close"], length=10)
        df["SMA20"] = sma(df["Close"], length=20)
        df["SMA50"] = sma(df["Close"], length=50)
        df["SMA150"] = sma(df["Close"], length=150)
        df["SMA200"] = sma(df["Close"], length=200)
        df["EMA5"] = df["Close"].ewm(span=5, min_periods=5, adjust=False).mean()
        df["EMA10"] = df["Close"].ewm(span=10, min_periods=10, adjust=False).mean()
        df["EMA20"] = df["Close"].ewm(span=20, min_periods=20, adjust=False).mean()
        df["EMA50"] = df["Close"].ewm(span=50, min_periods=50, adjust=False).mean()
        df["ATR14"] = atr(df["High"], df["Low"], df["Close"], length=14)
        df["ATR20"] = atr(df["High"], df["Low"], df["Close"], length=20)
        df["AVG_VOL20"] = sma(df["Volume"], length=20)
        df["DOLLAR_VOL20"] = rolling_dollar_volume(df["Close"], df["Volume"], length=20)
        df["ADR_PCT"] = ((df["High"] / df["Low"]) - 1).replace([np.inf, -np.inf], np.nan) * 100
        df["ADR20_PCT"] = sma(df["ADR_PCT"], length=20)
        df["RET_1D_PCT"] = df["Close"].pct_change() * 100
        df["DCR_PCT"] = daily_close_range(df["High"], df["Low"], df["Close"])
        df["GAP_PCT"] = ((df["Open"] / df["Close"].shift(1)) - 1).replace([np.inf, -np.inf], np.nan) * 100
        df["OPEN_CHANGE_PCT"] = ((df["Close"] / df["Open"]) - 1).replace([np.inf, -np.inf], np.nan) * 100
        df["HIGH_LOOKBACK"] = df["Close"].shift(1).rolling(breakout_lookback).max()
        df["DARVAS_UPPER"], df["DARVAS_LOWER"] = darvas_levels(df["High"], df["Low"], length=20)
        df["VOL_RATIO20"] = df["Volume"] / df["AVG_VOL20"]
        # FIX #5: 52W High — shift(1) to avoid lookahead bias (consistent with HIGH_LOOKBACK)
        df["HIGH_52W"] = df["High"].shift(1).rolling(252, min_periods=200).max()
        df["LOW_52W"] = df["Low"].shift(1).rolling(252, min_periods=200).min()
        df["ATR_EXTENSION_SMA50"] = (df["Close"] - df["SMA50"]) / df["ATR14"]
        df["ATR_EXTENSION_EMA10"] = (df["Close"] - df["EMA10"]) / df["ATR20"]
        df["ATR_EXTENSION_EMA20"] = (df["Close"] - df["EMA20"]) / df["ATR20"]
        df["PCT_EXTENSION_SMA50"] = (df["Close"] / df["SMA50"] - 1) * 100
        enriched[ticker] = df

        last = df.iloc[-1]
        previous = df.iloc[-2]
        sma200_1m_ago = df["SMA200"].iloc[-22] if len(df) >= 222 else np.nan
        minervini_trend_template = bool(
            pd.notna(last["SMA50"])
            and pd.notna(last["SMA150"])
            and pd.notna(last["SMA200"])
            and pd.notna(sma200_1m_ago)
            and pd.notna(last["HIGH_52W"])
            and pd.notna(last["LOW_52W"])
            and last["Close"] > last["SMA50"]
            and last["Close"] > last["SMA150"]
            and last["Close"] > last["SMA200"]
            and last["SMA50"] > last["SMA150"]
            and last["SMA50"] > last["SMA200"]
            and last["SMA150"] > last["SMA200"]
            and last["SMA200"] > sma200_1m_ago
            and last["Close"] >= 1.3 * last["LOW_52W"]
            and last["Close"] >= 0.75 * last["HIGH_52W"]
        )
        green_candle = bool(last["Close"] >= previous["Close"] and last["Close"] >= last["Open"])
        rows.append(
            {
                "Ticker": ticker,
                "Date": df.index[-1],
                "Price": last["Close"],
                "Return 1W %": safe_return(df["Close"], RETURN_WINDOWS["1W"]),
                "Return 1M %": safe_return(df["Close"], RETURN_WINDOWS["1M"]),
                "Return 3M %": safe_return(df["Close"], RETURN_WINDOWS["3M"]),
                "Return 6M %": safe_return(df["Close"], RETURN_WINDOWS["6M"]),
                "Return 9M %": safe_return(df["Close"], RETURN_WINDOWS["9M"]),
                "ADR 20D %": last["ADR20_PCT"],
                "Volume": last["Volume"],
                "Avg Volume 20D": last["AVG_VOL20"],
                "Daily $ Volume 20D": last["DOLLAR_VOL20"],
                "Volume Ratio 20D": last["VOL_RATIO20"],
                "Prev Volume": previous["Volume"],
                "SMA10": last["SMA10"],
                "SMA20": last["SMA20"],
                "SMA50": last["SMA50"],
                "SMA150": last["SMA150"],
                "SMA200": last["SMA200"],
                "EMA5": last["EMA5"],
                "EMA10": last["EMA10"],
                "EMA20": last["EMA20"],
                "EMA50": last["EMA50"],
                "EMA10 Rising": bool(last["EMA10"] > df["EMA10"].iloc[-2]),
                "ATR14": last["ATR14"],
                "ATR20": last["ATR20"],
                "ATR Extension SMA50": last["ATR_EXTENSION_SMA50"],
                "ATR Extension EMA10": last["ATR_EXTENSION_EMA10"],
                "ATR Extension EMA20": last["ATR_EXTENSION_EMA20"],
                "% Extension SMA50": last["PCT_EXTENSION_SMA50"],
                "52W High": last["HIGH_52W"],
                "52W Low": last["LOW_52W"],
                "52W Range %": ((last["Close"] - last["LOW_52W"]) / (last["HIGH_52W"] - last["LOW_52W"]) * 100)
                if pd.notna(last["HIGH_52W"]) and pd.notna(last["LOW_52W"]) and last["HIGH_52W"] != last["LOW_52W"]
                else np.nan,
                "DCR %": last["DCR_PCT"],
                "Gap %": last["GAP_PCT"],
                "Open Change %": last["OPEN_CHANGE_PCT"],
                "Darvas Upper": last["DARVAS_UPPER"],
                "Darvas Lower": last["DARVAS_LOWER"],
                "Market Cap": np.nan,
                "Price > SMA10": bool(last["Close"] > last["SMA10"]),
                "Price > SMA20": bool(last["Close"] > last["SMA20"]),
                "Minervini Trend Template": minervini_trend_template,
                "Green Candle": green_candle,
                "Price > 5": bool(last["Close"] > 5),
                "Avg Volume > 200k": bool(last["AVG_VOL20"] > 200_000),
                "Daily Return %": last["RET_1D_PCT"],
                "Prev Close": previous["Close"],
                "Breakout Level": last["HIGH_LOOKBACK"],
                "Breakout Above Lookback High": bool(last["Close"] > last["HIGH_LOOKBACK"]),
            }
        )

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        return metrics, enriched

    rank_columns = ["Return 1M %", "Return 3M %", "Return 6M %"]
    for column in rank_columns:
        metrics[f"{column} Rank"] = metrics[column].rank(pct=True, ascending=True) * 100

    metrics["Momentum Rank"] = metrics[[f"{column} Rank" for column in rank_columns]].mean(axis=1)
    metrics["Universe Rank"] = metrics["Momentum Rank"].rank(method="min", ascending=False)
    metrics["Universe Percentile"] = metrics["Momentum Rank"].rank(pct=True, ascending=True) * 100
    return metrics.sort_values("Momentum Rank", ascending=False), enriched


def scanner_thresholds(filters: ScanFilters) -> ScannerThresholds:
    return ScannerThresholds(
        min_price=filters.min_price,
        min_avg_volume=filters.min_avg_volume,
        top_percent=filters.top_percent,
        min_breakout_pct=filters.min_breakout_pct,
        stockbee_min_price=filters.stockbee_min_price,
        stockbee_min_volume=filters.stockbee_min_volume,
        min_dollar_volume=filters.min_dollar_volume,
        min_adr_pct=filters.min_adr_pct,
        max_extension_atr=filters.max_extension_atr,
    )


def apply_qullamaggie_filter(metrics: pd.DataFrame, filters: ScanFilters) -> pd.DataFrame:
    return core_apply_qullamaggie_filter(metrics, scanner_thresholds(filters))


def apply_steve_style_qullamaggie_filter(metrics: pd.DataFrame, filters: ScanFilters) -> pd.DataFrame:
    return core_apply_steve_style_qullamaggie_filter(metrics, scanner_thresholds(filters))


def apply_stockbee_filter(metrics: pd.DataFrame, filters: ScanFilters) -> pd.DataFrame:
    return core_apply_stockbee_filter(metrics, scanner_thresholds(filters))


def apply_stockbee_9m_movers_filter(metrics: pd.DataFrame, filters: ScanFilters) -> pd.DataFrame:
    return core_apply_stockbee_9m_movers_filter(metrics, scanner_thresholds(filters))


def apply_minervini_filter(metrics: pd.DataFrame) -> pd.DataFrame:
    return core_apply_minervini_filter(metrics)


def apply_guru_filter(qullamaggie_screen: pd.DataFrame, minervini_screen: pd.DataFrame) -> pd.DataFrame:
    return core_apply_guru_filter(qullamaggie_screen, minervini_screen)


def apply_extension_filter(df: pd.DataFrame, filters: ScanFilters) -> pd.DataFrame:
    if df.empty:
        return df

    filtered = df[df["ATR Extension SMA50"] >= filters.min_extension_atr].copy()

    if not filters.only_non_extended:
        return filtered

    # FIX #4: "non-extended" means <= moderate_extension_atr (3x default), NOT max_extension_atr (5x).
    # "Extended" (3x-5x) is a core Qullamaggie setup — excluding it is wrong.
    return filtered[filtered["ATR Extension SMA50"] <= filters.moderate_extension_atr].copy()


def classify_extension(extension: pd.Series, filters: ScanFilters) -> pd.Series:
    return pd.Series(
        np.select(
            [
                extension < 0,
                extension <= filters.max_extension_atr,
                extension < filters.high_extension_atr,
                extension < filters.hyper_extension_atr,
            ],
            [
                "Below SMA50",
                "Non extended",
                "Extended",
                "Very extended",
            ],
            default="Hyper extended",
        ),
        index=extension.index,
    )


def apply_extension_zone_filter(df: pd.DataFrame, selected_zones: list[str]) -> pd.DataFrame:
    if df.empty or not selected_zones:
        return df

    return df[df["Extension Zone"].isin(selected_zones)].copy()


def add_extension_buckets(df: pd.DataFrame, filters: ScanFilters) -> pd.DataFrame:
    if df.empty:
        return df

    output = df.copy()
    output["Non Extended"] = output["ATR Extension SMA50"] <= filters.max_extension_atr
    output["Extension Zone"] = classify_extension(output["ATR Extension SMA50"], filters)
    return output.sort_values("ATR Extension SMA50", ascending=True)


def format_output(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    ordered = [
        "Ticker",
        "Date",
        "Price",
        "Momentum Rank",
        "Universe Rank",
        "SB Hit Windows",
        "SB Best Rank",
        "SB Score",
        *[f"SB 9/{period}" for period in SUGAR_BABIES_PERIODS],
        *[column for column in df.columns if column.startswith("SB Rank 9/")],
        *[column for column in df.columns if column.startswith("Top ")],
        "Strict Q Lineage",
        "Strict Qullamaggie Overlap",
        "Steve-style KQ Score",
        "Steve-style KQ Reason",
        "Return 1W %",
        "Return 1M %",
        "Return 3M %",
        "Return 6M %",
        "Return 9M %",
        "ADR 20D %",
        "ATR Extension SMA50",
        "% Extension SMA50",
        "Extension Zone",
        "Non Extended",
        "Daily Return %",
        "Volume",
        "Prev Volume",
        "Avg Volume 20D",
        "Daily $ Volume 20D",
        "Volume Ratio 20D",
        "SMA10",
        "SMA20",
        "SMA50",
        "SMA150",
        "SMA200",
        "Minervini Trend Template",
        "Green Candle",
        "Base Pivot",
        "Base Low",
        "Base Depth %",
        "Distance to Pivot %",
        "Prior Move %",
        "Base Bars",
        "MA Surfing 10/20",
        "Trade Setup Type",
        "Trade Entry Trigger",
        "Trade Stop",
        "Trade Risk %",
        "Stop / ADR",
        "Stop Bucket",
        "Breakout Level",
    ]
    visible = [column for column in ordered if column in df.columns]
    return df[visible].copy()


def candle_color(row: pd.Series) -> str:
    if bool(row.get("Green Candle", False)):
        return "#15803d"
    if row.get("Daily Return %", 0) < 0 and row.get("Price", 0) < row.get("Prev Close", np.inf):
        return "#b91c1c"
    return "#334155"


def classify_price_structure(row: pd.Series) -> str:
    price = row["Price"]
    sma10 = row["SMA10"]
    sma20 = row["SMA20"]
    sma50 = row["SMA50"]
    sma150 = row["SMA150"]
    sma200 = row["SMA200"]

    if pd.isna([price, sma10, sma20, sma50, sma150, sma200]).any():
        return "Unclassified"
    if price > sma10 > sma20 > sma50 > sma150 > sma200:
        return "2A: Close > 10 > 20 > 50 > 150 > 200"
    if price > sma10 > sma20 > sma50:
        return "2B: Close > 10 > 20 > 50"
    if price > sma20 > sma50 and sma50 > sma150:
        return "2C: Above 20/50"
    if price > sma50:
        return "1B/2: Above SMA50"
    if price > sma200:
        return "3A/B: Above SMA200"
    return "4A/B: Negative trend"


def calculate_trend_strength(row: pd.Series) -> float:
    checks = [
        row["Price"] > row["SMA10"],
        row["Price"] > row["SMA20"],
        row["Price"] > row["SMA50"],
        row["Price"] > row["SMA150"],
        row["Price"] > row["SMA200"],
        row["SMA10"] > row["SMA20"],
        row["SMA20"] > row["SMA50"],
        row["SMA50"] > row["SMA150"],
        row["SMA150"] > row["SMA200"],
        row["Price"] >= 0.75 * row["52W High"] if pd.notna(row["52W High"]) else False,
    ]
    return float(np.mean(checks) * 100)


def rts_grade(score: float) -> str:
    if score >= 97:
        return "A+"
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 65:
        return "C"
    return "D"


def add_steve_dashboard_fields(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics

    output = metrics.copy()
    if "Daily $ Volume 20D" not in output.columns:
        output["Daily $ Volume 20D"] = output["Price"] * output["Avg Volume 20D"]
    output["Price Structure"] = output.apply(classify_price_structure, axis=1)
    output["Negative Structure"] = output["Price Structure"].str.contains("Negative", na=False)
    output["Trend Strength"] = output.apply(calculate_trend_strength, axis=1)
    output["RTS Score"] = (output["Momentum Rank"] + output["Trend Strength"]) / 2
    output["RTS Grade"] = output["RTS Score"].apply(rts_grade)
    output["Extended 5x+"] = output["ATR Extension SMA50"] >= 5
    output["Over Extended 7x+"] = output["ATR Extension SMA50"] >= 7
    output["Hyper Extended 11x+"] = output["ATR Extension SMA50"] >= 11
    return output


def build_grouped_ticker_matrix(
    df: pd.DataFrame,
    group_column: str,
    sort_column: str,
    max_rows: int = 40,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    if df.empty:
        return pd.DataFrame(), {}

    groups = [group for group in df[group_column].dropna().unique().tolist()]
    ordered_groups = sorted(groups)
    columns: dict[str, list[str]] = {}
    ticker_meta: dict[str, dict[str, object]] = {}

    for group in ordered_groups:
        group_df = df[df[group_column] == group].sort_values(sort_column, ascending=False).head(max_rows)
        tickers = group_df["Ticker"].tolist()
        columns[group] = tickers + [""] * (max_rows - len(tickers))

        for _, row in group_df.iterrows():
            ticker_meta[str(row["Ticker"])] = {
                "color": candle_color(row),
                "negative": bool(row.get("Negative Structure", False)),
                "extended": bool(row.get("Extended 5x+", False)),
                "over_extended": bool(row.get("Over Extended 7x+", False)),
            }

    return pd.DataFrame(columns), ticker_meta


def style_ticker_matrix(matrix: pd.DataFrame, ticker_meta: dict[str, dict[str, object]]) -> pd.io.formats.style.Styler:
    def style_cell(value: object) -> str:
        ticker = str(value)
        if not ticker:
            return "background-color: #ffffff;"

        meta = ticker_meta.get(ticker, {})
        color = meta.get("color", "#334155")
        background = "#e5e7eb" if meta.get("negative") else "#ffffff"
        weight = "800" if meta.get("extended") else "700"
        if meta.get("over_extended"):
            color = "#b91c1c"
        elif meta.get("extended"):
            color = "#7c3aed"

        return f"color: {color}; background-color: {background}; font-weight: {weight}; text-align: center;"

    return matrix.style.map(style_cell)


def steve_css() -> str:
    return """
    <style>
    .steve-board {
        background: #0b0f17;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 14px 14px 22px;
        margin: 4px 0 22px;
        color: #e5e7eb;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .steve-nav {
        display: flex;
        gap: 24px;
        align-items: center;
        border-bottom: 2px solid #222936;
        margin-bottom: 28px;
        white-space: nowrap;
        overflow-x: auto;
    }
    .steve-nav span {
        color: #d1d5db;
        font-size: 15px;
        line-height: 36px;
    }
    .steve-nav .active {
        color: #ff4d4d;
        border-bottom: 2px solid #ff4d4d;
        margin-bottom: -2px;
    }
    .steve-controls {
        display: flex;
        gap: 42px;
        align-items: end;
        margin-bottom: 20px;
        color: #e5e7eb;
    }
    .steve-control-label {
        font-size: 14px;
        color: #d1d5db;
        margin-bottom: 8px;
    }
    .steve-select {
        min-width: 220px;
        background: #2a2933;
        border-radius: 8px;
        color: #f5f5f5;
        padding: 12px 16px;
        font-size: 16px;
    }
    .steve-radio {
        display: grid;
        gap: 8px;
        font-size: 17px;
    }
    .steve-radio span::before {
        content: "";
        display: inline-block;
        width: 16px;
        height: 16px;
        border: 1px solid #465264;
        border-radius: 50%;
        margin-right: 10px;
        vertical-align: -2px;
    }
    .steve-radio .selected::before {
        background: #ff4d4d;
        border: 4px solid #ff4d4d;
        box-shadow: inset 0 0 0 4px #ffe1e1;
    }
    .steve-filter-bar {
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 11px 16px;
        color: #e5e7eb;
        margin-bottom: 18px;
        font-size: 17px;
    }
    .steve-grid {
        display: grid;
        grid-template-columns: repeat(5, minmax(150px, 1fr));
        gap: 18px;
        align-items: start;
    }
    .signal-header {
        border-radius: 5px;
        text-align: center;
        padding: 10px 8px 9px;
        margin-bottom: 10px;
        color: #111827;
        font-weight: 800;
        min-height: 64px;
    }
    .signal-header .count {
        font-size: 20px;
        line-height: 22px;
    }
    .signal-header .label {
        font-size: 12px;
        letter-spacing: .01em;
        margin-top: 4px;
    }
    .header-yellow { background: #ffd400; }
    .header-green { background: #2bc79a; }
    .signal-tile {
        position: relative;
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 8px;
        align-items: center;
        background: #182236;
        border-radius: 5px;
        min-height: 36px;
        padding: 7px 10px 7px 14px;
        margin-bottom: 4px;
        overflow: hidden;
    }
    .signal-tile::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: var(--bar-color);
    }
    .ticker-line {
        min-width: 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        font-size: 13px;
        font-weight: 800;
        color: #e5e7eb;
    }
    .ticker-line .pct {
        color: #00c46a;
        font-size: 12px;
        font-weight: 700;
        margin-left: 6px;
    }
    .ticker-line .weekly {
        color: #2dd4bf;
        font-size: 11px;
        margin-left: 6px;
    }
    .signal-badge {
        border: 2px solid var(--badge-color);
        color: var(--badge-color);
        border-radius: 7px;
        min-width: 22px;
        height: 22px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: 800;
    }
    @media (max-width: 1100px) {
        .steve-grid { grid-template-columns: repeat(2, minmax(150px, 1fr)); }
    }
    @media (max-width: 680px) {
        .steve-grid { grid-template-columns: 1fr; }
        .steve-controls { display: block; }
        .steve-control { margin-bottom: 16px; }
    }
    </style>
    """


def signal_pct(row: pd.Series, pct_column: str) -> str:
    value = row.get(pct_column, np.nan)
    if pd.isna(value):
        return ""
    prefix = "+" if value >= 0 else ""
    return f"{prefix}{value:.1f}%"


def signal_badge(row: pd.Series) -> tuple[str, str, str]:
    extension = row.get("ATR Extension SMA50", np.nan)
    if pd.isna(extension):
        return "0", "#6b7280", "#6b7280"

    rounded = int(max(0, round(float(extension))))
    badge_color = "#facc15" if rounded >= 4 else "#00c46a"
    bar_color = "#facc15" if rounded >= 5 else "#00d46a"
    if extension < 0:
        bar_color = "#9ca3af"
        badge_color = "#9ca3af"
    return str(rounded), badge_color, bar_color


def signal_tile_html(row: pd.Series, pct_column: str, tile_style: str) -> str:
    ticker = escape(str(row["Ticker"]))
    pct = escape(signal_pct(row, pct_column))
    badge, badge_color, bar_color = signal_badge(row)
    weekly = ""
    if pct_column == "Return 1W %":
        weekly = " <span class='weekly'>1W</span>"

    detail = ""
    if tile_style == "Detailed":
        price = row.get("Price", np.nan)
        adr = row.get("ADR 20D %", np.nan)
        price_text = f" ${price:.2f}" if pd.notna(price) else ""
        adr_text = f" ADR {adr:.1f}%" if pd.notna(adr) else ""
        detail = f"<span class='weekly'>{escape(price_text + adr_text)}</span>"

    return (
        f"<div class='signal-tile' style='--badge-color:{badge_color};--bar-color:{bar_color};'>"
        f"<div class='ticker-line'>{ticker}{weekly} <span class='pct'>{pct}</span>{detail}</div>"
        f"<div class='signal-badge'>{escape(badge)}</div>"
        "</div>"
    )


def sort_signal_df(df: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    if df.empty:
        return df

    sort_map = {
        "ATR Extension": "ATR Extension SMA50",
        "Daily Return": "Daily Return %",
        "Momentum Rank": "Momentum Rank",
        "Dollar Volume": "Daily $ Volume 20D",
    }
    column = sort_map.get(sort_by, "ATR Extension SMA50")
    if column not in df.columns:
        return df
    return df.sort_values(column, ascending=False)


def prepare_signal_df(source: pd.DataFrame, steve: pd.DataFrame) -> pd.DataFrame:
    if source.empty or steve.empty:
        return pd.DataFrame(columns=steve.columns)
    return steve[steve["Ticker"].isin(set(source["Ticker"]))].copy()


def signal_column_html(
    title: str,
    code: str,
    df: pd.DataFrame,
    pct_column: str,
    sort_by: str,
    tile_style: str,
    header_class: str,
    max_items: int,
) -> str:
    visible_df = sort_signal_df(df, sort_by).head(max_items)
    tiles = "".join(signal_tile_html(row, pct_column, tile_style) for _, row in visible_df.iterrows())
    return (
        "<div class='signal-column'>"
        f"<div class='signal-header {header_class}'>"
        f"<div class='count'>{len(df)}</div>"
        f"<div class='label'>{escape(title)} &nbsp;·&nbsp; {escape(code)}</div>"
        "</div>"
        f"{tiles}"
        "</div>"
    )


def render_steve_signals_board(
    steve: pd.DataFrame,
    q_screen: pd.DataFrame,
    steve_style_kq_screen: pd.DataFrame,
    minervini_screen: pd.DataFrame,
    stockbee_screen: pd.DataFrame,
    filters: ScanFilters,
    sort_by: str,
    tile_style: str,
) -> None:
    if steve.empty:
        st.info("Nessun dato disponibile per la dashboard Steve.")
        return

    kq = prepare_signal_df(steve_style_kq_screen, steve)
    mm = prepare_signal_df(minervini_screen, steve)
    sb4 = prepare_signal_df(stockbee_screen, steve)
    sb9 = prepare_signal_df(apply_stockbee_9m_movers_filter(steve, filters), steve)
    base_liquid = steve[(steve["Price"] > 5) & (steve["Avg Volume 20D"] > 200_000)].copy()

    sbw = base_liquid[base_liquid["Return 1W %"] >= 20].copy()

    columns = [
        signal_column_html("Qullamaggie", "KQ", kq, "Daily Return %", sort_by, tile_style, "header-yellow", 26),
        signal_column_html("Minervini", "MM", mm, "Daily Return %", sort_by, tile_style, "header-yellow", 26),
        signal_column_html("9 Million Movers", "SB9M", sb9, "Daily Return %", sort_by, tile_style, "header-green", 26),
        signal_column_html("20% Weekly", "SBW", sbw, "Return 1W %", sort_by, tile_style, "header-green", 26),
        signal_column_html("4% Daily", "SB4", sb4, "Daily Return %", sort_by, tile_style, "header-green", 32),
    ]

    board = (
        steve_css()
        + "<div class='steve-board'>"
        + "<div class='steve-nav'>"
        + "<span>Signals</span><span>Liquid Leaders</span><span>Stage Analysis</span><span>Heatmap</span>"
        + "<span>Industry RS</span><span>Quadrant Stocks</span><span class='active'>Gurus</span>"
        + "</div>"
        + "<div class='steve-controls'>"
        + "<div class='steve-control'><div class='steve-control-label'>Sort by</div>"
        + f"<div class='steve-select'>{escape(sort_by)} ˅</div></div>"
        + "<div class='steve-control'><div class='steve-control-label'>Tile style</div>"
        + "<div class='steve-radio'>"
        + f"<span class='{'' if tile_style == 'Compact' else 'selected'}'>Detailed</span>"
        + f"<span class='{'selected' if tile_style == 'Compact' else ''}'>Compact</span>"
        + "</div></div></div>"
        + "<div class='steve-filter-bar'>› &nbsp; Filters</div>"
        + "<div class='steve-grid'>"
        + "".join(columns)
        + "</div></div>"
    )
    st.markdown(board, unsafe_allow_html=True)


def render_market_signals(steve: pd.DataFrame, stockbee_screen: pd.DataFrame) -> None:
    if steve.empty:
        st.info("Nessun dato market signals disponibile.")
        return

    above_sma50 = (steve["Price"] > steve["SMA50"]).mean() * 100
    above_sma200 = (steve["Price"] > steve["SMA200"]).mean() * 100
    stage2 = (
        (steve["Price"] > steve["SMA50"])
        & (steve["SMA50"] > steve["SMA150"])
        & (steve["SMA150"] > steve["SMA200"])
    ).mean() * 100
    positive_day = (steve["Daily Return %"] > 0).mean() * 100

    cols = st.columns(5)
    cols[0].metric("Stocks > SMA50", f"{above_sma50:.0f}%")
    cols[1].metric("Stocks > SMA200", f"{above_sma200:.0f}%")
    cols[2].metric("Stage 2 proxy", f"{stage2:.0f}%")
    cols[3].metric("Positive day", f"{positive_day:.0f}%")
    cols[4].metric("4% Daily", f"{len(stockbee_screen):,}")

    breadth = pd.DataFrame(
        {
            "Signal": ["Above SMA50", "Above SMA200", "Stage 2", "Positive day"],
            "Value": [above_sma50, above_sma200, stage2, positive_day],
        }
    )
    fig = go.Figure(go.Bar(x=breadth["Signal"], y=breadth["Value"], marker_color=["#22c55e", "#2dd4bf", "#facc15", "#38bdf8"]))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), yaxis=dict(range=[0, 100], ticksuffix="%"))
    st.plotly_chart(fig, use_container_width=True)


def render_stage_analysis(steve: pd.DataFrame) -> None:
    if steve.empty:
        st.info("Nessun dato stage disponibile.")
        return

    counts = steve["Price Structure"].value_counts().reset_index()
    counts.columns = ["Price Structure", "Count"]
    fig = go.Figure(go.Bar(x=counts["Price Structure"], y=counts["Count"], marker_color="#2bc79a"))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=80), xaxis_tickangle=-25)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        steve.sort_values(["Trend Strength", "Momentum Rank"], ascending=False)[
            ["Ticker", "Price Structure", "Trend Strength", "Momentum Rank", "ATR Extension SMA50", "Daily Return %"]
        ].head(120).round(2),
        use_container_width=True,
        hide_index=True,
    )


def render_heatmap(steve: pd.DataFrame) -> None:
    if steve.empty:
        st.info("Nessun dato heatmap disponibile.")
        return

    top = steve.sort_values("Momentum Rank", ascending=False).head(60)
    heatmap_data = top.set_index("Ticker")[["Return 1W %", "Return 1M %", "Return 3M %", "Return 6M %", "Return 9M %"]]
    fig = go.Figure(
        go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale="RdYlGn",
            zmid=0,
            colorbar=dict(title="%"),
        )
    )
    fig.update_layout(height=760, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)


def render_quadrant_stocks(steve: pd.DataFrame) -> None:
    if steve.empty:
        st.info("Nessun dato quadrant disponibile.")
        return

    plot_df = steve.dropna(subset=["Momentum Rank", "ATR Extension SMA50", "Daily Return %"]).copy()
    fig = go.Figure(
        go.Scatter(
            x=plot_df["Momentum Rank"],
            y=plot_df["ATR Extension SMA50"],
            mode="markers",
            text=plot_df["Ticker"],
            marker=dict(
                size=np.clip(plot_df["ADR 20D %"].fillna(3) * 2.5, 5, 22),
                color=plot_df["Daily Return %"],
                colorscale="RdYlGn",
                cmid=0,
                showscale=True,
                colorbar=dict(title="Daily %"),
            ),
            hovertemplate="<b>%{text}</b><br>Momentum %{x:.1f}<br>ATR Ext %{y:.2f}<extra></extra>",
        )
    )
    fig.add_hline(y=3, line_dash="dash", line_color="#facc15")
    fig.add_hline(y=7, line_dash="dash", line_color="#ef4444")
    fig.add_vline(x=90, line_dash="dash", line_color="#22c55e")
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Momentum Rank", yaxis_title="ATR Extension SMA50")
    st.plotly_chart(fig, use_container_width=True)


def render_industry_rs_placeholder() -> None:
    st.info(
        "Industry RS richiede dati settoriali/industry per ticker. Il dataset daily precomputato contiene prezzi e volumi; "
        "possiamo aggiungere industry con un secondo job di metadata, evitando chiamate live lente a yfinance."
    )


def steve_dashboard_context_frames(
    steve_all: pd.DataFrame,
    stockbee_screen: pd.DataFrame,
    q_screen: pd.DataFrame,
    strict_q_context: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return contextual Steve dashboard frames; Gurus columns stay unfiltered separately."""
    if not strict_q_context:
        return steve_all, stockbee_screen
    q_tickers = set(q_screen["Ticker"]) if not q_screen.empty else set()
    return (
        steve_all[steve_all["Ticker"].isin(q_tickers)].copy(),
        stockbee_screen[stockbee_screen["Ticker"].isin(q_tickers)].copy(),
    )


def render_steve_dashboard(
    metrics: pd.DataFrame,
    q_screen: pd.DataFrame,
    steve_style_kq_screen: pd.DataFrame,
    minervini_screen: pd.DataFrame,
    guru_screen: pd.DataFrame,
    stockbee_screen: pd.DataFrame,
    filters: ScanFilters,
) -> None:
    steve_all = add_steve_dashboard_fields(metrics)
    strict_q_context = st.toggle(
        "Steve Dashboard: contesto solo Qullamaggie Top 2%",
        value=False,
        help=(
            "ON = le sezioni contestuali Liquid Leaders / Stage / Heatmap / RTS / Quadrant mostrano solo i ticker "
            "che passano lo scanner Qullamaggie puro. OFF = dashboard sull'universo completo. "
            "La sezione Gurus resta sempre non filtrata, come nello screenshot di Steve: KQ, MM, SB9M, SBW e SB4 "
            "devono essere liste indipendenti, non intersezioni con Qullamaggie."
        ),
    )
    steve, stockbee_context = steve_dashboard_context_frames(steve_all, stockbee_screen, q_screen, strict_q_context)
    if strict_q_context:
        st.caption(
            "Filtro contesto Steve attivo: Liquid Leaders / Stage / Heatmap / RTS / Quadrant mostrano solo ticker "
            "già presenti nello scanner Qullamaggie Top 2%. La sezione Gurus resta non filtrata."
        )
    else:
        st.caption(
            "Filtro contesto Steve disattivato: dashboard sull'universo completo. La sezione Gurus mostra liste "
            "indipendenti KQ / MM / SB9M / SBW / SB4."
        )

    steve_view = st.radio(
        "Steve section",
        ["Gurus", "Signals", "Liquid Leaders", "Stage Analysis", "Heatmap", "Industry RS", "Quadrant Stocks"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if steve_view == "Gurus":
        control_cols = st.columns([1, 1, 3])
        sort_by = control_cols[0].selectbox(
            "Sort by",
            ["ATR Extension", "Daily Return", "Momentum Rank", "Dollar Volume"],
            index=0,
            key="steve_sort_by",
        )
        tile_style = control_cols[1].radio(
            "Tile style",
            ["Detailed", "Compact"],
            index=1,
            horizontal=False,
            key="steve_tile_style",
        )
        render_steve_signals_board(
            steve_all,
            q_screen,
            steve_style_kq_screen,
            minervini_screen,
            stockbee_screen,
            filters,
            sort_by,
            tile_style,
        )

    liquid = steve[steve["Daily $ Volume 20D"] >= 500_000_000].copy()
    rts = steve[(steve["Daily $ Volume 20D"] >= 50_000_000) & (steve["ADR 20D %"] >= steve["ADR 20D %"].median())].copy()

    if steve_view == "Signals":
        render_market_signals(steve, stockbee_context)

    if steve_view == "Liquid Leaders":
        st.subheader("Liquid Leaders")
        st.caption(
            "Daily dollar volume 20D >= $500M, ticker raggruppati per price structure delle medie mobili."
        )
        liquid_matrix, liquid_meta = build_grouped_ticker_matrix(
            liquid,
            group_column="Price Structure",
            sort_column="Daily $ Volume 20D",
            max_rows=35,
        )
        if liquid_matrix.empty:
            st.info("Nessun Liquid Leader con daily $ volume 20D >= $500M.")
        else:
            st.dataframe(style_ticker_matrix(liquid_matrix, liquid_meta), use_container_width=True, hide_index=True)
            export_raw_section(
                "Steve Liquid Leaders",
                liquid.sort_values("Daily $ Volume 20D", ascending=False),
                "steve_liquid_leaders.csv",
            )

    if steve_view == "Stage Analysis":
        render_stage_analysis(steve)

    if steve_view == "Heatmap":
        render_heatmap(steve)

    if steve_view == "Industry RS":
        render_industry_rs_placeholder()

    if steve_view == "Quadrant Stocks":
        render_quadrant_stocks(steve)

    with st.expander("Relative Trend Strength ed Extension Monitor", expanded=False):
        signal_cols = st.columns(6)
        signal_cols[0].metric("Signals", f"{len(q_screen):,}", "Qullamaggie")
        signal_cols[1].metric("Q x M", f"{len(guru_screen):,}", "Guru")
        signal_cols[2].metric("Stockbee", f"{len(stockbee_screen):,}", "4%+")
        signal_cols[3].metric("Liquid Leaders", f"{len(liquid):,}", "$500M+ D$V")
        signal_cols[4].metric("Extended", f"{int(steve['Over Extended 7x+'].sum()):,}", "7x+ ATR")
        signal_cols[5].metric("Hyper", f"{int(steve['Hyper Extended 11x+'].sum()):,}", "11x+ ATR")

        st.subheader("Relative Trend Strength")
        rts_top = rts.sort_values(["RTS Score", "Momentum Rank"], ascending=False).head(80)
        rts_matrix, rts_meta = build_grouped_ticker_matrix(
            rts_top,
            group_column="RTS Grade",
            sort_column="RTS Score",
            max_rows=25,
        )
        if rts_matrix.empty:
            st.info("Nessun dato RTS disponibile.")
        else:
            st.dataframe(style_ticker_matrix(rts_matrix, rts_meta), use_container_width=True, hide_index=True)
            st.dataframe(
                rts_top[
                    [
                        "Ticker",
                        "RTS Grade",
                        "RTS Score",
                        "Momentum Rank",
                        "Trend Strength",
                        "ADR 20D %",
                        "Daily $ Volume 20D",
                        "ATR Extension SMA50",
                        "Price Structure",
                    ]
                ].round(2),
                use_container_width=True,
                hide_index=True,
            )
            export_raw_section("Steve RTS", rts_top, "steve_relative_trend_strength.csv")

        st.subheader("Extension Monitor")
        extension_cols = st.columns(3)
        extension_cols[0].dataframe(
            steve[steve["Extended 5x+"]].sort_values("ATR Extension SMA50", ascending=False).head(30)[
                ["Ticker", "ATR Extension SMA50", "Price", "Daily $ Volume 20D", "Price Structure"]
            ].round(2),
            use_container_width=True,
            hide_index=True,
        )
        extension_cols[1].dataframe(
            steve[steve["Over Extended 7x+"]].sort_values("ATR Extension SMA50", ascending=False).head(30)[
                ["Ticker", "ATR Extension SMA50", "Price", "Daily $ Volume 20D", "Price Structure"]
            ].round(2),
            use_container_width=True,
            hide_index=True,
        )
        extension_cols[2].dataframe(
            steve[steve["Hyper Extended 11x+"]].sort_values("ATR Extension SMA50", ascending=False).head(30)[
                ["Ticker", "ATR Extension SMA50", "Price", "Daily $ Volume 20D", "Price Structure"]
            ].round(2),
            use_container_width=True,
            hide_index=True,
        )


def build_backtest_panel(history: dict[str, pd.DataFrame], breakout_lookback: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for ticker, frame in history.items():
        required = {"Open", "High", "Low", "Close", "Volume"}
        if frame.empty or not required.issubset(frame.columns):
            continue

        df = frame[list(required)].copy().dropna(subset=["Close"])
        if len(df) < RETURN_WINDOWS["6M"] + 25:
            continue

        df["Ticker"] = ticker
        df["SMA10"] = sma(df["Close"], length=10)
        df["SMA20"] = sma(df["Close"], length=20)
        df["SMA50"] = sma(df["Close"], length=50)
        df["ATR14"] = atr(df["High"], df["Low"], df["Close"], length=14)
        df["AVG_VOL20"] = sma(df["Volume"], length=20)
        df["ADR20_PCT"] = sma(((df["High"] / df["Low"]) - 1) * 100, length=20)
        df["ATR Extension SMA50"] = (df["Close"] - df["SMA50"]) / df["ATR14"]
        df["Return 1M %"] = df["Close"].pct_change(RETURN_WINDOWS["1M"]) * 100
        df["Return 3M %"] = df["Close"].pct_change(RETURN_WINDOWS["3M"]) * 100
        df["Return 6M %"] = df["Close"].pct_change(RETURN_WINDOWS["6M"]) * 100
        df["Breakout Level"] = df["Close"].shift(1).rolling(breakout_lookback).max()
        df["Next Open"] = df["Open"].shift(-1)
        frames.append(df.reset_index(names="Date"))

    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames, ignore_index=True).dropna(
        subset=["Return 1M %", "Return 3M %", "Return 6M %", "SMA10", "SMA20", "AVG_VOL20", "Next Open"]
    )
    rank_columns = ["Return 1M %", "Return 3M %", "Return 6M %"]
    for column in rank_columns:
        panel[f"{column} Rank"] = panel.groupby("Date")[column].rank(pct=True, ascending=True) * 100

    panel["Momentum Rank"] = panel[[f"{column} Rank" for column in rank_columns]].mean(axis=1)
    return panel


def run_qullamaggie_backtest(
    history: dict[str, pd.DataFrame],
    filters: ScanFilters,
    hold_days: int,
    max_trades_per_day: int,
    use_non_extended: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = build_backtest_panel(history, filters.breakout_lookback)
    if panel.empty:
        return pd.DataFrame(), pd.DataFrame()

    top_cutoff_by_date = panel.groupby("Date")["Ticker"].transform("count") * filters.top_percent / 100
    top_cutoff_by_date = np.ceil(top_cutoff_by_date).clip(lower=1)
    for window in ["1M", "3M", "6M"]:
        panel[f"Top {filters.top_percent:g}% {window}"] = (
            panel.groupby("Date")[f"Return {window} %"].rank(method="min", ascending=False) <= top_cutoff_by_date
        )

    top_columns = [f"Top {filters.top_percent:g}% 1M", f"Top {filters.top_percent:g}% 3M", f"Top {filters.top_percent:g}% 6M"]
    signal = panel[
        # Keep the backtest aligned with the live Qullamaggie scanner:
        # intersection across all three momentum windows, not union.
        panel[top_columns].all(axis=1)
        & (panel["Close"] > panel["SMA10"])
        & (panel["Close"] > panel["SMA20"])
        & (panel["AVG_VOL20"] > filters.min_avg_volume)
        & (panel["Close"] > filters.min_price)
        & (panel["ADR20_PCT"] >= filters.min_adr_pct)
        & ((panel["Close"] * panel["AVG_VOL20"]) >= filters.min_dollar_volume)
        & (panel["ATR Extension SMA50"] >= filters.min_extension_atr)
    ].copy()

    if use_non_extended:
        signal = signal[signal["ATR Extension SMA50"] <= filters.moderate_extension_atr]

    if signal.empty:
        return pd.DataFrame(), pd.DataFrame()

    signal = signal.sort_values(["Date", "Momentum Rank"], ascending=[True, False])
    signal = signal.groupby("Date").head(max_trades_per_day)

    trades: list[dict[str, float | str | pd.Timestamp]] = []
    history_by_ticker = {ticker: frame.copy() for ticker, frame in history.items()}

    for _, row in signal.iterrows():
        ticker = row["Ticker"]
        df = history_by_ticker.get(ticker)
        if df is None or row["Date"] not in df.index:
            continue

        signal_idx = df.index.get_loc(row["Date"])
        entry_idx = signal_idx + 1
        exit_idx = entry_idx + hold_days
        if exit_idx >= len(df):
            continue

        entry_price = df["Open"].iloc[entry_idx]
        exit_price = df["Close"].iloc[exit_idx]
        if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
            continue

        trades.append(
            {
                "Signal Date": row["Date"],
                "Entry Date": df.index[entry_idx],
                "Exit Date": df.index[exit_idx],
                "Ticker": ticker,
                "Entry": entry_price,
                "Exit": exit_price,
                "Return %": (exit_price / entry_price - 1) * 100,
                "Momentum Rank": row["Momentum Rank"],
                "ADR 20D %": row["ADR20_PCT"],
                "ATR Extension SMA50": row["ATR Extension SMA50"],
            }
        )

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df, pd.DataFrame()

    trades_df = trades_df.sort_values("Entry Date")
    equity = (1 + trades_df["Return %"] / 100).cumprod()
    summary = pd.DataFrame(
        [
            {
                "Trades": len(trades_df),
                "Win Rate %": (trades_df["Return %"] > 0).mean() * 100,
                "Avg Return %": trades_df["Return %"].mean(),
                "Median Return %": trades_df["Return %"].median(),
                "Best Trade %": trades_df["Return %"].max(),
                "Worst Trade %": trades_df["Return %"].min(),
                "Compounded Return %": (equity.iloc[-1] - 1) * 100,
            }
        ]
    )
    return trades_df, summary


def enrich_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["Open", "High", "Low", "Close", "Volume"]
    df = frame[required].copy().dropna(subset=["Close"])
    df["SMA10"] = sma(df["Close"], length=10)
    df["SMA20"] = sma(df["Close"], length=20)
    df["SMA50"] = sma(df["Close"], length=50)
    df["SMA150"] = sma(df["Close"], length=150)
    df["SMA200"] = sma(df["Close"], length=200)
    df["ATR14"] = atr(df["High"], df["Low"], df["Close"], length=14)
    return df


def draw_chart(
    ticker: str,
    enriched_history: dict[str, pd.DataFrame],
    history: dict[str, pd.DataFrame] | None = None,
) -> None:
    df = enriched_history.get(ticker)
    if (df is None or df.empty) and history is not None and ticker in history:
        df = enrich_price_frame(history[ticker])
    if df is None or df.empty:
        st.info("Nessun dato chart disponibile per il ticker selezionato.")
        return

    chart_df = df.tail(180)
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=chart_df.index,
            open=chart_df["Open"],
            high=chart_df["High"],
            low=chart_df["Low"],
            close=chart_df["Close"],
            name=ticker,
        )
    )
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["SMA10"], name="SMA10", line=dict(width=1.4)))
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["SMA20"], name="SMA20", line=dict(width=1.4)))
    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=35, b=10),
        xaxis_rangeslider_visible=False,
        title=f"{ticker} daily",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def sidebar_controls(symbols: pd.DataFrame) -> tuple[list[str], ScanFilters, list[str], int, float]:
    st.sidebar.header("Scanner")
    max_symbols = st.sidebar.number_input(
        "Numero massimo ticker",
        min_value=50,
        max_value=max(50, len(symbols)),
        value=len(symbols),
        step=50,
        help="Riduci questo valore per test rapidi. L'intero NASDAQ puo richiedere vari minuti.",
    )
    selected_symbols = symbols["yahoo_symbol"].head(int(max_symbols)).tolist()

    min_price = st.sidebar.number_input("Prezzo minimo", min_value=0.0, value=5.0, step=0.5)
    min_avg_volume = st.sidebar.number_input("Avg volume minimo", min_value=0, value=200_000, step=50_000)
    top_percent = st.sidebar.slider("Top momentum %", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
    only_non_extended = st.sidebar.toggle("Steve mode: solo non-extended", value=False)
    min_extension_atr = st.sidebar.slider("Estensione minima ATR da SMA50", -5.0, 12.0, -5.0, 0.25)
    moderate_extension_atr = st.sidebar.slider("Extended da ATR", 1.0, 8.0, 3.0, 0.25)
    max_extension_atr = moderate_extension_atr
    high_extension_atr = st.sidebar.slider("Very extended da ATR", 2.0, 12.0, 5.0, 0.25)
    hyper_extension_atr = st.sidebar.slider("Hyper extended da ATR", 4.0, 15.0, 7.0, 0.25)
    selected_extension_zones = st.sidebar.multiselect(
        "Mostra zone estensione",
        ["Below SMA50", "Non extended", "Extended", "Very extended", "Hyper extended"],
        default=["Below SMA50", "Non extended", "Extended", "Very extended", "Hyper extended"],
    )
    min_breakout_pct = st.sidebar.slider("Stockbee breakout minimo %", 1.0, 15.0, 4.0, 0.5)
    stockbee_min_price = st.sidebar.number_input("Stockbee prezzo minimo", min_value=0.0, value=3.0, step=0.5)
    stockbee_min_volume = st.sidebar.number_input("Stockbee volume minimo", min_value=0, value=100_000, step=50_000)
    breakout_lookback = st.sidebar.slider("Lookback high contesto", 10, 100, DEFAULT_BREAKOUT_LOOKBACK, 5)
    # FIX #2 & #3: Qullamaggie mandatory filters from video timestamp 01:28:00
    # Dollar Volume > $150M (or $15M small account), ADR 20D > 3.5% (or >5% small account)
    min_dollar_volume = st.sidebar.number_input(
        "Dollar Volume min ($M)", min_value=0, value=150, step=10,
        help="Qullamaggie usa $150M. Small account: $15M."
    ) * 1_000_000
    min_adr_pct = st.sidebar.slider(
        "ADR minimo %", 1.0, 10.0, 3.5, 0.5,
        help="Qullamaggie usa 3.5%. Small account: 5%."
    )
    chunk_size = st.sidebar.slider("Ticker per batch", 25, 250, 100, 25)
    pause_seconds = st.sidebar.slider("Pausa tra batch", 0.0, 2.0, 0.2, 0.1)

    filters = ScanFilters(
        min_price=min_price,
        min_avg_volume=int(min_avg_volume),
        top_percent=top_percent,
        only_non_extended=only_non_extended,
        min_extension_atr=min_extension_atr,
        moderate_extension_atr=moderate_extension_atr,
        max_extension_atr=max_extension_atr,
        high_extension_atr=high_extension_atr,
        hyper_extension_atr=hyper_extension_atr,
        min_breakout_pct=min_breakout_pct,
        stockbee_min_price=stockbee_min_price,
        stockbee_min_volume=int(stockbee_min_volume),
        breakout_lookback=breakout_lookback,
        # FIX #2 & #3: new Qullamaggie mandatory filters
        min_dollar_volume=min_dollar_volume,
        min_adr_pct=min_adr_pct,
    )
    return selected_symbols, filters, selected_extension_zones, chunk_size, pause_seconds


def steve_algo_thresholds_from_filters(filters: ScanFilters, enforce_market_cap: bool = False) -> SteveAlgoThresholds:
    return SteveAlgoThresholds(
        min_market_cap=1_000_000_000 if enforce_market_cap else 0,
        min_dollar_volume=max(50_000_000, min(filters.min_dollar_volume, 150_000_000)),
        min_price=filters.min_price,
        min_rs=85,
        min_trend_strength=80,
        min_reward_risk=3,
    )


def render_steve_algo_watchlist(metrics: pd.DataFrame, filters: ScanFilters) -> pd.DataFrame:
    st.caption(
        "Replica trasparente v0 dello Swing Trading Algo Watchlist di SteveDJacobs: White Up / Entry / Yellow. "
        "Formula proprietaria ignota: qui ogni riga espone ragione e metriche. Research only, capitale autorizzato 0%."
    )
    enforce_market_cap = st.toggle(
        "Applica hard gate Market Cap >= $1B",
        value=False,
        help="OFF perché il parquet/precomputed corrente non contiene market cap affidabile. ON esclude righe con Market Cap N/D.",
    )
    watchlist = apply_steve_algo_watchlists(metrics, steve_algo_thresholds_from_filters(filters, enforce_market_cap))
    if watchlist.empty:
        st.warning("Nessun ticker SteveAlgo con i filtri correnti.")
        return watchlist

    counts = watchlist["SteveAlgo Primary Bucket"].value_counts()
    cols = st.columns(4)
    cols[0].metric("White Up", int(counts.get("White Up", 0)))
    cols[1].metric("Entry", int(counts.get("Entry", 0)))
    cols[2].metric("Yellow", int(counts.get("Yellow", 0)))
    cols[3].metric("Capital authorized", "0%")

    display_cols = [
        "Ticker", "SteveAlgo Primary Bucket", "SteveAlgo Status", "Price", "Daily Return %", "DCR %",
        "Gap %", "Open Change %", "Return 1M %", "Return 3M %", "Return 6M %", "Momentum Rank",
        "SteveAlgo Trend Strength", "Reward-Risk", "ATR Extension EMA20", "ATR Extension SMA50",
        "Daily $ Volume 20D", "Market Cap", "Capital Authorized", "SteveAlgo Reason",
    ]
    available = [c for c in display_cols if c in watchlist.columns]
    for bucket in ["White Up", "Entry", "Yellow"]:
        bucket_df = watchlist[watchlist["SteveAlgo Primary Bucket"] == bucket].copy()
        st.subheader(f"{bucket} — {len(bucket_df)} stocks")
        if bucket_df.empty:
            st.info(f"Nessun {bucket}.")
            continue
        st.dataframe(bucket_df[available].round(2), use_container_width=True, hide_index=True)
    export_raw_section("Steve Algo Watchlist", watchlist, "steve_algo_watchlist.csv")
    return watchlist


def build_steve_algo_backtest_events(
    history: dict[str, pd.DataFrame],
    filters: ScanFilters,
    enforce_market_cap: bool = False,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    enriched: dict[str, pd.DataFrame] = {}
    for ticker, frame in history.items():
        required = {"Open", "High", "Low", "Close", "Volume"}
        if frame.empty or not required.issubset(frame.columns):
            continue
        df = frame[list(required)].copy().dropna(subset=["Close"])
        if len(df) < 220:
            continue
        df["EMA10"] = df["Close"].ewm(span=10, min_periods=10, adjust=False).mean()
        df["EMA20"] = df["Close"].ewm(span=20, min_periods=20, adjust=False).mean()
        df["EMA50"] = df["Close"].ewm(span=50, min_periods=50, adjust=False).mean()
        df["SMA50"] = sma(df["Close"], 50)
        df["SMA200"] = sma(df["Close"], 200)
        df["ATR20"] = atr(df["High"], df["Low"], df["Close"], 20)
        df["DCR %"] = daily_close_range(df["High"], df["Low"], df["Close"])
        df["Daily $ Volume 20D"] = rolling_dollar_volume(df["Close"], df["Volume"], 20)
        df["Return 1M %"] = df["Close"].pct_change(21) * 100
        df["Return 3M %"] = df["Close"].pct_change(63) * 100
        df["Return 6M %"] = df["Close"].pct_change(126) * 100
        df["Darvas Upper"], df["Darvas Lower"] = darvas_levels(df["High"], df["Low"], 20)
        df["ATR Extension EMA10"] = (df["Close"] - df["EMA10"]) / df["ATR20"]
        df["ATR Extension EMA20"] = (df["Close"] - df["EMA20"]) / df["ATR20"]
        df["ATR Extension SMA50"] = (df["Close"] - df["SMA50"]) / df["ATR20"]
        df["EMA10 Rising"] = df["EMA10"] > df["EMA10"].shift(1)
        df["Breakout Above Lookback High"] = df["Close"] > df["Close"].shift(1).rolling(filters.breakout_lookback).max()
        df["Price"] = df["Close"]
        df["Daily Return %"] = df["Close"].pct_change() * 100
        enriched[ticker] = df
        tmp = df.reset_index(names="Date")
        tmp["Ticker"] = ticker
        rows.extend(tmp.to_dict("records"))
    panel = pd.DataFrame(rows)
    if panel.empty:
        return pd.DataFrame(), enriched
    panel = panel.dropna(subset=["Return 1M %", "Return 3M %", "Return 6M %", "EMA10", "EMA20", "EMA50", "SMA50", "SMA200", "ATR20"])
    for col in ["Return 1M %", "Return 3M %", "Return 6M %"]:
        panel[f"{col} Rank"] = panel.groupby("Date")[col].rank(pct=True, ascending=True) * 100
    panel["Momentum Rank"] = panel[["Return 1M % Rank", "Return 3M % Rank", "Return 6M % Rank"]].mean(axis=1)
    panel["Market Cap"] = np.nan if "Market Cap" not in panel.columns else panel["Market Cap"]
    panel["Reward-Risk"] = np.nan
    classified = apply_steve_algo_watchlists(panel, steve_algo_thresholds_from_filters(filters, enforce_market_cap))
    return classified[classified["SteveAlgo Primary Bucket"].isin(["White Up", "Entry", "Yellow"])], enriched


def render_steve_algo_backtest(history: dict[str, pd.DataFrame], selected_symbols: list[str], filters: ScanFilters) -> None:
    st.caption(
        "Backtest event-based SteveAlgo v0: segnale a close, entrata next open, stop Darvas/EMA20-ATR, target R, max hold. "
        "No same-bar execution. Research only; capitale autorizzato 0%."
    )
    cols = st.columns(4)
    max_hold = cols[0].number_input("Max hold bars", min_value=2, max_value=80, value=20, step=1)
    target_r = cols[1].number_input("Target R", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
    slippage_bps = cols[2].number_input("Slippage bps", min_value=0.0, max_value=100.0, value=10.0, step=5.0)
    enforce_market_cap = cols[3].toggle("Gate $1B", value=False)
    if st.button("Esegui backtest SteveAlgo", use_container_width=True):
        if not history and HISTORY_FILE.exists():
            with st.spinner("Carico storico prezzi precomputato..."):
                all_history = load_precomputed_history(str(HISTORY_FILE))
                history = {ticker: all_history[ticker] for ticker in selected_symbols if ticker in all_history}
        with st.spinner("Genero eventi e simulo trade SteveAlgo..."):
            events, enriched = build_steve_algo_backtest_events(history, filters, enforce_market_cap)
            trades = simulate_steve_algo_trades(
                events,
                enriched,
                SteveBacktestConfig(max_hold_bars=int(max_hold), target_r=float(target_r), slippage_bps=float(slippage_bps)),
            )
            summary = summarize_steve_algo_backtest(trades)
        if trades.empty:
            st.warning("Nessun trade generato con i parametri correnti.")
            return
        st.json(summary)
        st.line_chart(trades["R"].cumsum())
        st.dataframe(trades.round(3), use_container_width=True, hide_index=True)
        export_raw_section("SteveAlgo Backtest Trades", trades, "steve_algo_backtest_trades.csv")


def render_sugar_babies_view(sugar_babies: pd.DataFrame, metrics: pd.DataFrame) -> None:
    st.caption(
        "Sugar Babies SB replica la Stockbee Sugar Babies List: conta quante volte ogni ticker ha fatto "
        "close/prev_close >= 1.04 con volume > volume precedente e volume >= 8.9M. "
        "La watchlist e la union dei top 25 per finestre 1450/1260/1008/756/504/252/126/50/20/10/5. "
        "Nota: e diversa dallo Stockbee 4% Breakout, che mostra solo il trigger giornaliero corrente."
    )

    if sugar_babies.empty:
        st.warning(
            "File Sugar Babies non trovato o vuoto. Esegui: "
            "python update_sugar_babies.py --universe 'All US listed' --period 6y"
        )
        return

    enriched = merge_sugar_babies_with_metrics(sugar_babies, metrics)
    cols = st.columns(4)
    cols[0].metric("SB watchlist", f"{len(enriched):,}")
    if "SB 9/1450" in enriched.columns:
        cols[1].metric("Max 9/1450", f"{int(enriched['SB 9/1450'].max()):,}")
    if "SB Hit Windows" in enriched.columns:
        cols[2].metric("Max hit windows", f"{int(enriched['SB Hit Windows'].max()):,}")
    if "Updated At" in enriched.columns and enriched["Updated At"].notna().any():
        cols[3].metric("Aggiornato", str(pd.to_datetime(enriched["Updated At"]).max().date()))

    sort_by = st.selectbox(
        "Ordinamento SB",
        ["Actionable SB", "Replica TC2000 9/1450", "Recent SB 9/50"],
        index=0,
        help="Actionable privilegia presenza multi-finestra e rank migliore; TC2000 replica la colonna storica 9/1450.",
    )
    view_df = sort_sugar_babies_view(enriched, sort_by)
    st.dataframe(
        format_output(view_df),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Price": st.column_config.NumberColumn(format="$%.2f"),
            "Momentum Rank": st.column_config.NumberColumn(format="%.1f"),
            "ADR 20D %": st.column_config.NumberColumn(format="%.1f%%"),
            "Daily Return %": st.column_config.NumberColumn(format="%.1f%%"),
            "ATR Extension SMA50": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    export_raw_section("Sugar Babies", view_df, "sugar_babies_sb.csv")


def main() -> None:
    st.title("Qullamaggie NASDAQ Scanner")
    st.caption(f"Build: {APP_BUILD_MARKER}")
    scanner_frameworks = scanner_framework_editor()

    controls = st.columns([1, 1, 2, 3])
    with controls[0]:
        include_etfs = st.toggle("Includi ETF", value=False)
    with controls[1]:
        if st.button("Refresh dati", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with controls[2]:
        data_options = ["Precomputed", "Live yfinance"] if HISTORY_FILE.exists() else ["Live yfinance", "Precomputed"]
        data_mode = st.radio(
            "Dati",
            data_options,
            horizontal=True,
            index=0,
            help="Per Streamlit Cloud usa Precomputed: legge data/history_prices.parquet senza chiamare Yahoo.",
        )
    with controls[3]:
        universe = st.radio(
            "Universo",
            ["NASDAQ", "All US listed"],
            horizontal=True,
            index=0,
            help="Steve Jacobs sembra usare US stocks ampio; NASDAQ-only esclude NYSE/AMEX.",
        )

    symbols = load_symbols(universe=universe, include_etfs=include_etfs)
    selected_symbols, filters, selected_extension_zones, chunk_size, pause_seconds = sidebar_controls(symbols)

    st.caption(
        f"Universo caricato: {len(symbols):,} strumenti ({universe}). "
        f"Scanner corrente: {len(selected_symbols):,} ticker daily."
    )

    metrics = pd.DataFrame()
    sugar_babies = pd.DataFrame()
    enriched_history: dict[str, pd.DataFrame] = {}
    history: dict[str, pd.DataFrame] = {}

    if data_mode == "Precomputed" and HISTORY_FILE.exists():
        with st.spinner("Carico dati precomputati..."):
            if METRICS_FILE.exists() and filters.breakout_lookback == DEFAULT_BREAKOUT_LOOKBACK:
                all_metrics = load_precomputed_metrics(str(METRICS_FILE))
                metrics = all_metrics[all_metrics["Ticker"].isin(selected_symbols)].copy()
            else:
                all_history = load_precomputed_history(str(HISTORY_FILE))
                history = {ticker: all_history[ticker] for ticker in selected_symbols if ticker in all_history}
        if METADATA_FILE.exists():
            try:
                metadata = json.loads(METADATA_FILE.read_text())
                st.caption(
                    f"Dati precomputati aggiornati: {metadata.get('updated_at', 'n/d')} - "
                    f"{metadata.get('tickers_with_data', len(metrics) or len(history)):,} ticker con dati."
                )
            except json.JSONDecodeError:
                pass
    else:
        if data_mode == "Precomputed" and not HISTORY_FILE.exists():
            st.warning("Nessun file precomputato trovato in data/history_prices.parquet. Uso yfinance live.")
        with st.spinner("Scarico dati daily da yfinance e calcolo ranking..."):
            history = download_price_history(
                tickers=tuple(selected_symbols),
                period="1y",
                interval="1d",
                chunk_size=chunk_size,
                pause_seconds=pause_seconds,
            )

    if metrics.empty:
        if not history and data_mode == "Precomputed" and HISTORY_FILE.exists():
            with st.spinner("Carico storico prezzi per calcolo custom..."):
                all_history = load_precomputed_history(str(HISTORY_FILE))
                history = {ticker: all_history[ticker] for ticker in selected_symbols if ticker in all_history}
        with st.spinner("Calcolo metriche scanner..."):
            metrics, enriched_history = calculate_metrics(history, filters.breakout_lookback)

    if metrics.empty:
        st.warning("Nessun dato valido ricevuto. Prova a ridurre i batch o premere Refresh dati.")
        return

    if SUGAR_BABIES_FILE.exists():
        try:
            sugar_babies = load_precomputed_sugar_babies(str(SUGAR_BABIES_FILE))
        except Exception as exc:
            st.warning(f"Sugar Babies non caricato: {exc}")

    if SUGAR_BABIES_METADATA_FILE.exists():
        try:
            sb_metadata = json.loads(SUGAR_BABIES_METADATA_FILE.read_text())
            st.caption(
                f"Sugar Babies aggiornato: {sb_metadata.get('updated_at', 'n/d')} - "
                f"{sb_metadata.get('sugar_babies_rows', len(sugar_babies)):,} ticker in watchlist."
            )
        except json.JSONDecodeError:
            pass

    base_history = enriched_history if enriched_history else history
    if not base_history and data_mode == "Precomputed" and HISTORY_FILE.exists():
        with st.spinner("Carico storico prezzi per base detector..."):
            all_history = load_precomputed_history(str(HISTORY_FILE))
            history = {ticker: all_history[ticker] for ticker in selected_symbols if ticker in all_history}
            base_history = history

    q_candidates = apply_extension_zone_filter(
        add_extension_buckets(apply_extension_filter(apply_qullamaggie_filter(metrics, filters), filters), filters),
        selected_extension_zones,
    )
    q_screen = add_trade_plan_columns(
        add_base_setup_columns(q_candidates, base_history),
        setup_type="Strict Q Breakout",
    )
    steve_style_kq_screen = apply_extension_zone_filter(
        add_extension_buckets(apply_steve_style_qullamaggie_filter(metrics, filters), filters),
        selected_extension_zones,
    )
    minervini_screen = apply_extension_zone_filter(
        add_extension_buckets(apply_extension_filter(apply_minervini_filter(metrics), filters), filters),
        selected_extension_zones,
    )
    guru_screen = apply_extension_zone_filter(
        add_extension_buckets(apply_extension_filter(apply_guru_filter(q_screen, minervini_screen), filters), filters),
        selected_extension_zones,
    )
    extension_screen = apply_extension_zone_filter(add_extension_buckets(metrics, filters), selected_extension_zones)
    stockbee_screen = apply_stockbee_filter(metrics, filters)

    kpis = st.columns(7)
    kpis[0].metric("Ticker con dati", f"{len(metrics):,}")
    kpis[1].metric("Q strict", f"{len(q_screen):,}")
    kpis[2].metric("Steve-style KQ", f"{len(steve_style_kq_screen):,}")
    kpis[3].metric("Q x Minervini", f"{len(guru_screen):,}")
    kpis[4].metric("Stockbee 4%", f"{len(stockbee_screen):,}")
    kpis[5].metric("Sugar Babies", f"{len(sugar_babies):,}")
    kpis[6].metric("Ultima data", str(pd.to_datetime(metrics["Date"]).max().date()))

    scanner_group = st.radio(
        "Framework",
        framework_options(scanner_frameworks),
        horizontal=True,
        help="Separa gli scanner per framework. Puoi cambiare la composizione dalla sidebar: Framework scanner.",
    )
    view = st.radio(
        "Scanner",
        view_options_for_scanner_group(scanner_group, scanner_frameworks),
        horizontal=True,
    )

    if view == "Steve Dashboard":
        render_steve_dashboard(metrics, q_screen, steve_style_kq_screen, minervini_screen, guru_screen, stockbee_screen, filters)

    elif view == "Steve Algo Watchlist":
        render_steve_algo_watchlist(metrics, filters)

    elif view == "Steve Algo Backtest":
        render_steve_algo_backtest(history, selected_symbols, filters)

    elif view == "Steve-style KQ":
        st.caption(
            "Vista ampia separata per replicare meglio la colonna Steve Jacobs Qullamaggie · KQ. "
            "Non sostituisce Qullamaggie strict: qui basta momentum forte su almeno una dimensione o momentum composito alto, "
            "con ADR/trend/liquidità e ATR extension <= soglia max. Usa $50M come cap rilassato del dollar volume quando il default strict è $150M."
        )
        st.dataframe(
            format_output(steve_style_kq_screen),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Price": st.column_config.NumberColumn(format="$%.2f"),
                "Momentum Rank": st.column_config.NumberColumn(format="%.1f"),
                "Steve-style KQ Score": st.column_config.NumberColumn(format="%.1f"),
                "Return 1M %": st.column_config.NumberColumn(format="%.1f%%"),
                "Return 3M %": st.column_config.NumberColumn(format="%.1f%%"),
                "Return 6M %": st.column_config.NumberColumn(format="%.1f%%"),
                "ADR 20D %": st.column_config.NumberColumn(format="%.1f%%"),
                "Daily Return %": st.column_config.NumberColumn(format="%.1f%%"),
                "ATR Extension SMA50": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        export_section("Steve-style KQ", steve_style_kq_screen, "steve_style_kq_scan.csv")

    elif view == "Qullamaggie Top 2%":
        st.caption(
            "Scanner Qullamaggie principale: top 2% contemporaneamente su 1M + 3M + 6M, ADR 20D, "
            "dollar volume, prezzo sopra SMA10/SMA20, avg volume e prezzo minimo. Le zone di estensione "
            "servono per evitare titoli troppo tirati."
        )
        st.dataframe(
            format_output(q_screen),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Price": st.column_config.NumberColumn(format="$%.2f"),
                "Momentum Rank": st.column_config.NumberColumn(format="%.1f"),
                "Return 1M %": st.column_config.NumberColumn(format="%.1f%%"),
                "Return 3M %": st.column_config.NumberColumn(format="%.1f%%"),
                "Return 6M %": st.column_config.NumberColumn(format="%.1f%%"),
                "ADR 20D %": st.column_config.NumberColumn(format="%.1f%%"),
                "Daily Return %": st.column_config.NumberColumn(format="%.1f%%"),
                "ATR Extension SMA50": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        export_section("Qullamaggie", q_screen, "qullamaggie_scan.csv")

    elif view == "Backtest Q":
        st.caption(
            "Backtest meccanico del solo scanner Qullamaggie: top 2% contemporaneamente su 1M + 3M + 6M, "
            "stessi filtri principali di ADR/liquidità/medie/estensione dello scanner live, segnale a fine giornata, "
            "entrata il giorno dopo in apertura, uscita dopo N sedute. Non replica l'entry discrezionale ORH di Qullamaggie."
        )
        bt_cols = st.columns(3)
        hold_days = bt_cols[0].number_input("Hold giorni", min_value=1, max_value=60, value=10, step=1)
        max_trades_per_day = bt_cols[1].number_input("Max trade/giorno", min_value=1, max_value=50, value=5, step=1)
        bt_non_extended = bt_cols[2].toggle("Solo non-extended", value=True)

        if st.button("Esegui backtest Qullamaggie", use_container_width=True):
            if not history and HISTORY_FILE.exists():
                with st.spinner("Carico storico prezzi per backtest..."):
                    all_history = load_precomputed_history(str(HISTORY_FILE))
                    history = {ticker: all_history[ticker] for ticker in selected_symbols if ticker in all_history}
            with st.spinner("Calcolo backtest Qullamaggie..."):
                trades_df, summary_df = run_qullamaggie_backtest(
                    history,
                    filters,
                    hold_days=int(hold_days),
                    max_trades_per_day=int(max_trades_per_day),
                    use_non_extended=bt_non_extended,
                )

            if trades_df.empty:
                st.warning("Nessun trade generato con i parametri correnti.")
            else:
                st.dataframe(summary_df.round(2), use_container_width=True, hide_index=True)
                st.line_chart((1 + trades_df["Return %"] / 100).cumprod())
                st.dataframe(trades_df.round(3), use_container_width=True, hide_index=True)
                export_raw_section("Backtest Trades", trades_df, "qullamaggie_backtest_trades.csv")

    elif view == "Guru Q x Minervini":
        st.caption(
            "Ispirato ai tweet di Steve Jacobs: intersezione Qullamaggie x Minervini, ordinata per "
            "ATR-to-SMA50 extension. Attiva 'Steve mode' nella sidebar per mostrare solo i non-extended."
        )
        st.dataframe(
            format_output(guru_screen),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Price": st.column_config.NumberColumn(format="$%.2f"),
            "ATR Extension SMA50": st.column_config.NumberColumn(format="%.2f"),
                "% Extension SMA50": st.column_config.NumberColumn(format="%.1f%%"),
                "Return 1M %": st.column_config.NumberColumn(format="%.1f%%"),
                "Return 3M %": st.column_config.NumberColumn(format="%.1f%%"),
                "Return 6M %": st.column_config.NumberColumn(format="%.1f%%"),
                "ADR 20D %": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )
        export_section("Guru", guru_screen, "guru_qullamaggie_minervini_scan.csv")

    elif view == "Minervini":
        st.caption(
            "Minervini inspired: Trend Template classico piu green candle. Ordinato per ATR-to-SMA50 extension."
        )
        st.dataframe(format_output(minervini_screen), use_container_width=True, hide_index=True)
        export_section("Minervini", minervini_screen, "minervini_trend_template_scan.csv")

    elif view == "Extension Map":
        st.caption(
            "Mappa estensione ispirata a Steve Jacobs: ATR-to-SMA50 extension. "
            "Hyper extended parte da 7x ATR di default."
        )
        zone_counts = (
            extension_screen["Extension Zone"]
            .value_counts()
            .reindex(["Below SMA50", "Non extended", "Extended", "Very extended", "Hyper extended"])
            .fillna(0)
            .astype(int)
        )
        st.bar_chart(zone_counts)
        st.dataframe(format_output(extension_screen), use_container_width=True, hide_index=True)
        export_section("Extension Map", extension_screen, "extension_map_scan.csv")

    elif view == "Stockbee 4% Breakout":
        st.caption("Stockbee 4%: close / previous close >= 1.04, volume > volume ieri, volume >= soglia.")
        st.dataframe(format_output(stockbee_screen), use_container_width=True, hide_index=True)
        export_section("Stockbee", stockbee_screen, "stockbee_4pct_breakouts.csv")

    elif view == "Sugar Babies SB":
        render_sugar_babies_view(sugar_babies, metrics)

    elif view == "Universo":
        st.dataframe(format_output(metrics), use_container_width=True, hide_index=True)
        export_section("Universo", metrics, "universe_metrics.csv")

    elif view == "Chart":
        candidates = (
            guru_screen["Ticker"].tolist()
            or q_screen["Ticker"].tolist()
            or stockbee_screen["Ticker"].tolist()
            or metrics["Ticker"].head(50).tolist()
        )
        ticker = st.selectbox("Ticker", candidates)
        if st.button("Mostra chart", use_container_width=True):
            if not history and HISTORY_FILE.exists():
                with st.spinner("Carico storico prezzi per chart..."):
                    all_history = load_precomputed_history(str(HISTORY_FILE))
                    history = {ticker_key: all_history[ticker_key] for ticker_key in selected_symbols if ticker_key in all_history}
            draw_chart(ticker, enriched_history, history)
        else:
            st.caption("Seleziona un ticker e premi Mostra chart per caricare lo storico prezzi solo quando serve.")


if __name__ == "__main__":
    main()
