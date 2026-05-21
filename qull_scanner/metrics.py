from __future__ import annotations

import pandas as pd


def rolling_dollar_volume(close: pd.Series, volume: pd.Series, length: int = 20) -> pd.Series:
    """Rolling mean of daily dollar volume, i.e. mean(Close * Volume)."""
    return (close * volume).rolling(length).mean()


def latest_rolling_dollar_volume(close: pd.Series, volume: pd.Series, length: int = 20) -> float:
    return float(rolling_dollar_volume(close, volume, length).iloc[-1])
