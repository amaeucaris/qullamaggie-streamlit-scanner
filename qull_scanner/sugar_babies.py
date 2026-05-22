from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import pandas as pd

SUGAR_BABIES_PERIODS = [1450, 1260, 1008, 756, 504, 252, 126, 50, 20, 10, 5]
SUGAR_BABIES_MIN_VOLUME = 8_900_000
SUGAR_BABIES_MIN_RETURN = 0.04


def _as_numeric_series(value: pd.Series | pd.DataFrame) -> pd.Series:
    """Return a 1-D numeric series from normal or single-ticker yfinance columns."""
    if isinstance(value, pd.DataFrame):
        if value.empty:
            return pd.Series(dtype="float64")
        value = value.iloc[:, 0]
    return pd.to_numeric(value, errors="coerce")


def sugar_baby_event(
    history: pd.DataFrame,
    min_daily_return: float = SUGAR_BABIES_MIN_RETURN,
    min_volume: int = SUGAR_BABIES_MIN_VOLUME,
) -> pd.Series:
    """Replicate Stockbee/TC2000 SB event: c/c1>=1.04 and v>v1 and v>=8.9M."""
    if history.empty or "Close" not in history.columns or "Volume" not in history.columns:
        return pd.Series(False, index=history.index, dtype=bool)

    close = _as_numeric_series(history["Close"])
    volume = _as_numeric_series(history["Volume"])
    event = (close / close.shift(1) >= 1 + min_daily_return) & (volume > volume.shift(1)) & (volume >= min_volume)
    return event.fillna(False).astype(bool)


def sugar_baby_counts(
    history: pd.DataFrame,
    periods: Iterable[int] = SUGAR_BABIES_PERIODS,
    min_daily_return: float = SUGAR_BABIES_MIN_RETURN,
    min_volume: int = SUGAR_BABIES_MIN_VOLUME,
) -> dict[str, int]:
    event = sugar_baby_event(history, min_daily_return=min_daily_return, min_volume=min_volume)
    return {f"SB 9/{int(period)}": int(event.tail(int(period)).sum()) for period in periods}


def build_sugar_babies_metrics(
    history: Mapping[str, pd.DataFrame],
    periods: Iterable[int] = SUGAR_BABIES_PERIODS,
    top_n: int = 25,
    min_daily_return: float = SUGAR_BABIES_MIN_RETURN,
    min_volume: int = SUGAR_BABIES_MIN_VOLUME,
) -> pd.DataFrame:
    """Build Sugar Babies watchlist: union of top N tickers for each CountTrue period."""
    periods = [int(period) for period in periods]
    count_columns = [f"SB 9/{period}" for period in periods]
    rows: list[dict[str, object]] = []

    for ticker, df in history.items():
        if df is None or df.empty:
            continue
        counts = sugar_baby_counts(df, periods=periods, min_daily_return=min_daily_return, min_volume=min_volume)
        latest = df.dropna(subset=["Close"]).iloc[-1] if "Close" in df.columns and not df.dropna(subset=["Close"]).empty else None
        row: dict[str, object] = {"Ticker": ticker, **counts}
        if latest is not None:
            row["Date"] = latest.get("Date", latest.name)
            row["Price"] = float(latest["Close"])
            if "Volume" in latest:
                row["Volume"] = float(latest["Volume"])
        rows.append(row)

    all_counts = pd.DataFrame(rows)
    if all_counts.empty:
        return all_counts

    selected: set[str] = set()
    rank_frames: list[pd.DataFrame] = []
    for period, column in zip(periods, count_columns):
        ranked = all_counts.sort_values([column, "Ticker"], ascending=[False, True]).copy()
        ranked[f"SB Rank {period}"] = range(1, len(ranked) + 1)
        top = ranked.head(max(0, int(top_n)))
        selected.update(top["Ticker"].tolist())
        rank_frames.append(ranked[["Ticker", f"SB Rank {period}"]])

    output = all_counts[all_counts["Ticker"].isin(selected)].copy()
    for ranks in rank_frames:
        output = output.merge(ranks, on="Ticker", how="left")

    rank_columns = [f"SB Rank {period}" for period in periods]
    output["SB Hit Windows"] = output[rank_columns].le(top_n).sum(axis=1).astype(int)
    output["SB Best Rank"] = output[rank_columns].min(axis=1).astype(int)
    weighted_score = pd.Series(0.0, index=output.index)
    for weight, column in zip(range(len(count_columns), 0, -1), count_columns):
        weighted_score += output[column].fillna(0).astype(float) * math.sqrt(weight)
    output["SB Score"] = weighted_score.round(2)

    sort_columns = ["SB Hit Windows", "SB Best Rank"]
    ascending = [False, True]
    if "SB 9/252" in output.columns:
        sort_columns.append("SB 9/252")
        ascending.append(False)
    if "SB 9/50" in output.columns:
        sort_columns.append("SB 9/50")
        ascending.append(False)
    if "SB 9/1450" in output.columns:
        sort_columns.append("SB 9/1450")
        ascending.append(False)
    sort_columns.append("Ticker")
    ascending.append(True)
    return output.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)
