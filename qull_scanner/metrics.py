from __future__ import annotations

import math

import numpy as np
import pandas as pd


def rolling_dollar_volume(close: pd.Series, volume: pd.Series, length: int = 20) -> pd.Series:
    """Rolling mean of daily dollar volume, i.e. mean(Close * Volume)."""
    return (close * volume).rolling(length).mean()


def latest_rolling_dollar_volume(close: pd.Series, volume: pd.Series, length: int = 20) -> float:
    return float(rolling_dollar_volume(close, volume, length).iloc[-1])


def daily_close_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Close location in the daily range, 0=low, 100=high."""
    denominator = (high - low).replace(0, np.nan)
    return ((close - low) / denominator * 100).replace([np.inf, -np.inf], np.nan)


def gap_pct(open_: float, prev_close: float) -> float:
    """Opening gap percentage versus previous close."""
    if prev_close is None or not math.isfinite(float(prev_close)) or float(prev_close) == 0:
        return math.nan
    return round((float(open_) / float(prev_close) - 1) * 100, 10)


def open_to_close_pct(open_: float, close: float) -> float:
    """Intraday open-to-close percentage change."""
    if open_ is None or not math.isfinite(float(open_)) or float(open_) == 0:
        return math.nan
    return round((float(close) / float(open_) - 1) * 100, 10)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Classic true range using previous close; first bar falls back to high-low."""
    previous_close = close.shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Deterministic simple moving-average ATR for scanner/backtest features."""
    return true_range(high, low, close).rolling(length, min_periods=length).mean()


def darvas_levels(high: pd.Series, low: pd.Series, length: int = 20) -> tuple[pd.Series, pd.Series]:
    """Prior rolling high/low levels, shifted one bar to avoid lookahead."""
    upper = high.shift(1).rolling(length, min_periods=length).max()
    lower = low.shift(1).rolling(length, min_periods=length).min()
    return upper, lower


def rolling_52w_position(close: pd.Series, length: int = 252) -> pd.Series:
    """Current close as percent of prior rolling range, shifted to avoid lookahead."""
    prior_high = close.shift(1).rolling(length, min_periods=length).max()
    prior_low = close.shift(1).rolling(length, min_periods=length).min()
    denominator = (prior_high - prior_low).replace(0, np.nan)
    return ((close - prior_low) / denominator * 100).replace([np.inf, -np.inf], np.nan)
