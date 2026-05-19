from __future__ import annotations

import math
import os
import time
import json
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import numpy as np
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
EXPORT_DIR = Path("exports")
DATA_DIR = Path("data")
HISTORY_FILE = DATA_DIR / "history_prices.parquet"
METADATA_FILE = DATA_DIR / "metadata.json"
RETURN_WINDOWS = {
    "1M": 21,
    "3M": 63,
    "6M": 126,
}


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


def safe_return(close: pd.Series, days: int) -> float:
    if len(close.dropna()) <= days:
        return np.nan
    return (close.iloc[-1] / close.iloc[-days - 1] - 1) * 100


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

        df["SMA10"] = ta.sma(df["Close"], length=10)
        df["SMA20"] = ta.sma(df["Close"], length=20)
        df["SMA50"] = ta.sma(df["Close"], length=50)
        df["SMA150"] = ta.sma(df["Close"], length=150)
        df["SMA200"] = ta.sma(df["Close"], length=200)
        df["ATR14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
        df["AVG_VOL20"] = ta.sma(df["Volume"], length=20)
        df["ADR_PCT"] = ((df["High"] / df["Low"]) - 1).replace([np.inf, -np.inf], np.nan) * 100
        df["ADR20_PCT"] = ta.sma(df["ADR_PCT"], length=20)
        df["RET_1D_PCT"] = df["Close"].pct_change() * 100
        df["HIGH_LOOKBACK"] = df["Close"].shift(1).rolling(breakout_lookback).max()
        df["VOL_RATIO20"] = df["Volume"] / df["AVG_VOL20"]
        df["HIGH_52W"] = df["High"].rolling(252, min_periods=200).max()
        df["LOW_52W"] = df["Low"].rolling(252, min_periods=200).min()
        df["ATR_EXTENSION_SMA50"] = (df["Close"] - df["SMA50"]) / df["ATR14"]
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
                "Return 1M %": safe_return(df["Close"], RETURN_WINDOWS["1M"]),
                "Return 3M %": safe_return(df["Close"], RETURN_WINDOWS["3M"]),
                "Return 6M %": safe_return(df["Close"], RETURN_WINDOWS["6M"]),
                "ADR 20D %": last["ADR20_PCT"],
                "Volume": last["Volume"],
                "Avg Volume 20D": last["AVG_VOL20"],
                "Volume Ratio 20D": last["VOL_RATIO20"],
                "Prev Volume": previous["Volume"],
                "SMA10": last["SMA10"],
                "SMA20": last["SMA20"],
                "SMA50": last["SMA50"],
                "SMA150": last["SMA150"],
                "SMA200": last["SMA200"],
                "ATR14": last["ATR14"],
                "ATR Extension SMA50": last["ATR_EXTENSION_SMA50"],
                "% Extension SMA50": last["PCT_EXTENSION_SMA50"],
                "52W High": last["HIGH_52W"],
                "52W Low": last["LOW_52W"],
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


def apply_qullamaggie_filter(metrics: pd.DataFrame, filters: ScanFilters) -> pd.DataFrame:
    if metrics.empty:
        return metrics

    top_cutoff = max(1, math.ceil(len(metrics) * filters.top_percent / 100))
    top_rank_columns = [
        f"Top {filters.top_percent:g}% 1M",
        f"Top {filters.top_percent:g}% 3M",
        f"Top {filters.top_percent:g}% 6M",
    ]
    q_metrics = metrics.copy()
    q_metrics[top_rank_columns[0]] = q_metrics["Return 1M %"].rank(method="min", ascending=False) <= top_cutoff
    q_metrics[top_rank_columns[1]] = q_metrics["Return 3M %"].rank(method="min", ascending=False) <= top_cutoff
    q_metrics[top_rank_columns[2]] = q_metrics["Return 6M %"].rank(method="min", ascending=False) <= top_cutoff

    return q_metrics[
        (q_metrics[top_rank_columns].any(axis=1))
        & (q_metrics["ADR 20D %"].notna())
        & (q_metrics["Price > SMA10"])
        & (q_metrics["Price > SMA20"])
        & (q_metrics["Avg Volume 20D"] > filters.min_avg_volume)
        & (q_metrics["Price"] > filters.min_price)
    ].copy()


def apply_stockbee_filter(metrics: pd.DataFrame, filters: ScanFilters) -> pd.DataFrame:
    if metrics.empty:
        return metrics

    return metrics[
        (metrics["Daily Return %"] >= filters.min_breakout_pct)
        & (metrics["Volume"] > metrics["Prev Volume"])
        & (metrics["Volume"] >= filters.stockbee_min_volume)
        & (metrics["Price"] > filters.stockbee_min_price)
    ].copy()


def apply_minervini_filter(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics

    return metrics[
        (metrics["Minervini Trend Template"])
        & (metrics["Green Candle"])
    ].copy()


def apply_guru_filter(qullamaggie_screen: pd.DataFrame, minervini_screen: pd.DataFrame) -> pd.DataFrame:
    if qullamaggie_screen.empty or minervini_screen.empty:
        return pd.DataFrame(columns=qullamaggie_screen.columns)

    minervini_tickers = set(minervini_screen["Ticker"])
    return qullamaggie_screen[qullamaggie_screen["Ticker"].isin(minervini_tickers)].copy()


def apply_extension_filter(df: pd.DataFrame, filters: ScanFilters) -> pd.DataFrame:
    if df.empty:
        return df

    filtered = df[df["ATR Extension SMA50"] >= filters.min_extension_atr].copy()

    if not filters.only_non_extended:
        return filtered

    return filtered[filtered["ATR Extension SMA50"] <= filters.max_extension_atr].copy()


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
        *[column for column in df.columns if column.startswith("Top ")],
        "Return 1M %",
        "Return 3M %",
        "Return 6M %",
        "ADR 20D %",
        "ATR Extension SMA50",
        "% Extension SMA50",
        "Extension Zone",
        "Non Extended",
        "Daily Return %",
        "Volume",
        "Prev Volume",
        "Avg Volume 20D",
        "Volume Ratio 20D",
        "SMA10",
        "SMA20",
        "SMA50",
        "SMA150",
        "SMA200",
        "Minervini Trend Template",
        "Green Candle",
        "Breakout Level",
    ]
    visible = [column for column in ordered if column in df.columns]
    return df[visible].copy()


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
        df["SMA10"] = ta.sma(df["Close"], length=10)
        df["SMA20"] = ta.sma(df["Close"], length=20)
        df["SMA50"] = ta.sma(df["Close"], length=50)
        df["ATR14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
        df["AVG_VOL20"] = ta.sma(df["Volume"], length=20)
        df["ADR20_PCT"] = ta.sma(((df["High"] / df["Low"]) - 1) * 100, length=20)
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
        panel[top_columns].any(axis=1)
        & (panel["Close"] > panel["SMA10"])
        & (panel["Close"] > panel["SMA20"])
        & (panel["AVG_VOL20"] > filters.min_avg_volume)
        & (panel["Close"] > filters.min_price)
        & panel["ADR20_PCT"].notna()
    ].copy()

    if use_non_extended:
        signal = signal[signal["ATR Extension SMA50"] <= filters.max_extension_atr]

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


def draw_chart(ticker: str, enriched_history: dict[str, pd.DataFrame]) -> None:
    df = enriched_history.get(ticker)
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
    breakout_lookback = st.sidebar.slider("Lookback high contesto", 10, 100, 20, 5)
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
    )
    return selected_symbols, filters, selected_extension_zones, chunk_size, pause_seconds


def main() -> None:
    st.title("Qullamaggie NASDAQ Scanner")

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

    if data_mode == "Precomputed" and HISTORY_FILE.exists():
        with st.spinner("Carico dati precomputati e calcolo ranking..."):
            all_history = load_precomputed_history(str(HISTORY_FILE))
            history = {ticker: all_history[ticker] for ticker in selected_symbols if ticker in all_history}
        if METADATA_FILE.exists():
            try:
                metadata = json.loads(METADATA_FILE.read_text())
                st.caption(
                    f"Dati precomputati aggiornati: {metadata.get('updated_at', 'n/d')} - "
                    f"{metadata.get('tickers_with_data', len(history)):,} ticker con dati."
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

    with st.spinner("Calcolo metriche scanner..."):
        metrics, enriched_history = calculate_metrics(history, filters.breakout_lookback)

    if metrics.empty:
        st.warning("Nessun dato valido ricevuto. Prova a ridurre i batch o premere Refresh dati.")
        return

    q_screen = apply_extension_zone_filter(
        add_extension_buckets(apply_extension_filter(apply_qullamaggie_filter(metrics, filters), filters), filters),
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

    kpis = st.columns(5)
    kpis[0].metric("Ticker con dati", f"{len(metrics):,}")
    kpis[1].metric("Qullamaggie top", f"{len(q_screen):,}")
    kpis[2].metric("Q x Minervini", f"{len(guru_screen):,}")
    kpis[3].metric("Stockbee 4%", f"{len(stockbee_screen):,}")
    kpis[4].metric("Ultima data", str(pd.to_datetime(metrics["Date"]).max().date()))

    (
        tab_qullamaggie,
        tab_backtest,
        tab_guru,
        tab_minervini,
        tab_extension,
        tab_stockbee,
        tab_universe,
        tab_chart,
    ) = st.tabs(
        [
            "Qullamaggie Top 2%",
            "Backtest Q",
            "Guru Q x Minervini",
            "Minervini",
            "Extension Map",
            "Stockbee 4% Breakout",
            "Universo",
            "Chart",
        ]
    )

    with tab_qullamaggie:
        st.caption(
            "Scanner Qullamaggie principale: top momentum per 1M/3M/6M, ADR 20D, prezzo sopra SMA10/SMA20, "
            "avg volume e prezzo minimo. Le zone di estensione servono per evitare titoli troppo tirati."
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

    with tab_backtest:
        st.caption(
            "Backtest meccanico del solo scanner Qullamaggie: segnale a fine giornata, entrata il giorno dopo "
            "in apertura, uscita dopo N sedute. Non replica l'entry discrezionale ORH di Qullamaggie."
        )
        bt_cols = st.columns(3)
        hold_days = bt_cols[0].number_input("Hold giorni", min_value=1, max_value=60, value=10, step=1)
        max_trades_per_day = bt_cols[1].number_input("Max trade/giorno", min_value=1, max_value=50, value=5, step=1)
        bt_non_extended = bt_cols[2].toggle("Solo non-extended", value=True)

        if st.button("Esegui backtest Qullamaggie", use_container_width=True):
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

    with tab_guru:
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

    with tab_minervini:
        st.caption(
            "Minervini inspired: Trend Template classico piu green candle. Ordinato per ATR-to-SMA50 extension."
        )
        st.dataframe(format_output(minervini_screen), use_container_width=True, hide_index=True)
        export_section("Minervini", minervini_screen, "minervini_trend_template_scan.csv")

    with tab_extension:
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

    with tab_stockbee:
        st.caption("Stockbee 4%: close / previous close >= 1.04, volume > volume ieri, volume >= soglia.")
        st.dataframe(format_output(stockbee_screen), use_container_width=True, hide_index=True)
        export_section("Stockbee", stockbee_screen, "stockbee_4pct_breakouts.csv")

    with tab_universe:
        st.dataframe(format_output(metrics), use_container_width=True, hide_index=True)
        export_section("Universo", metrics, "universe_metrics.csv")

    with tab_chart:
        candidates = (
            guru_screen["Ticker"].tolist()
            or q_screen["Ticker"].tolist()
            or stockbee_screen["Ticker"].tolist()
            or metrics["Ticker"].head(50).tolist()
        )
        ticker = st.selectbox("Ticker", candidates)
        draw_chart(ticker, enriched_history)


if __name__ == "__main__":
    main()
