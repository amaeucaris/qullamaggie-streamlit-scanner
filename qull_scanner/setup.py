from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PriorMove:
    low_price: float
    high_price: float
    bars: int

    @property
    def gain_pct(self) -> float:
        return ((self.high_price / self.low_price) - 1.0) * 100 if self.low_price > 0 else 0.0

    def is_valid(self, min_gain_pct: float = 30.0) -> bool:
        return self.gain_pct >= min_gain_pct


@dataclass(frozen=True)
class Consolidation:
    high: float
    low: float
    bars: int
    has_higher_lows: bool
    has_tightening_range: bool

    @property
    def range_pct(self) -> float:
        return ((self.high / self.low) - 1.0) * 100 if self.low > 0 else 0.0

    def is_valid(self, min_bars: int = 7, max_bars: int = 42) -> bool:
        return min_bars <= self.bars <= max_bars and self.has_higher_lows and self.has_tightening_range


@dataclass(frozen=True)
class MASurfing:
    price_above_sma10: bool
    price_above_sma20: bool
    sma10_rising: bool = True
    sma20_rising: bool = True

    def is_valid(self) -> bool:
        return self.price_above_sma10 and self.price_above_sma20 and self.sma10_rising and self.sma20_rising


@dataclass(frozen=True)
class BreakoutSetup:
    prior_move: PriorMove
    consolidation: Consolidation
    ma_surfing: MASurfing

    def is_valid(self) -> bool:
        return self.prior_move.is_valid() and self.consolidation.is_valid() and self.ma_surfing.is_valid()


@dataclass(frozen=True)
class DetectedBaseSetup:
    pivot: float
    base_low: float
    base_depth_pct: float
    distance_to_pivot_pct: float
    prior_move_pct: float
    base_bars: int
    price_above_sma10: bool
    price_above_sma20: bool


def detect_base_setup(history: pd.DataFrame, base_window: int = 20, prior_window: int = 40) -> DetectedBaseSetup | None:
    """Detect a simple consolidation base using only bars before the current bar.

    This is deliberately conservative: the current bar is excluded from the base
    window so the detector can be used for scan/export metadata without leaking
    today's breakout bar into the pivot/base-low calculation.
    """
    required = {"High", "Low", "Close"}
    if history.empty or not required.issubset(history.columns):
        return None
    if len(history) < base_window + prior_window + 1:
        return None

    frame = history.dropna(subset=["High", "Low", "Close"]).copy()
    if len(frame) < base_window + prior_window + 1:
        return None

    base_slice = frame.iloc[-base_window - 1 : -1]
    prior_slice = frame.iloc[-base_window - prior_window - 1 : -base_window - 1]
    if base_slice.empty or prior_slice.empty:
        return None

    pivot = float(base_slice["High"].max())
    base_low = float(base_slice["Low"].min())
    current_close = float(frame["Close"].iloc[-1])
    prior_low = float(prior_slice["Low"].min())
    if pivot <= 0 or base_low <= 0 or prior_low <= 0:
        return None

    sma10 = frame["Close"].rolling(10).mean().iloc[-1]
    sma20 = frame["Close"].rolling(20).mean().iloc[-1]
    return DetectedBaseSetup(
        pivot=round(pivot, 2),
        base_low=round(base_low, 2),
        base_depth_pct=round(((pivot - base_low) / pivot) * 100, 2),
        distance_to_pivot_pct=round(((current_close / pivot) - 1.0) * 100, 2),
        prior_move_pct=round(((pivot / prior_low) - 1.0) * 100, 2),
        base_bars=len(base_slice),
        price_above_sma10=bool(pd.notna(sma10) and current_close > sma10),
        price_above_sma20=bool(pd.notna(sma20) and current_close > sma20),
    )


def add_base_setup_columns(
    metrics: pd.DataFrame,
    history_by_ticker: dict[str, pd.DataFrame],
    base_window: int = 20,
    prior_window: int = 40,
) -> pd.DataFrame:
    """Append base/pivot metadata to scanner rows from a ticker->history map."""
    if metrics.empty:
        return metrics

    output = metrics.copy()
    setups = {
        str(ticker): detect_base_setup(history, base_window=base_window, prior_window=prior_window)
        for ticker, history in history_by_ticker.items()
    }

    output["Base Pivot"] = output["Ticker"].map(lambda ticker: setups.get(str(ticker)).pivot if setups.get(str(ticker)) else pd.NA)
    output["Base Low"] = output["Ticker"].map(lambda ticker: setups.get(str(ticker)).base_low if setups.get(str(ticker)) else pd.NA)
    output["Base Depth %"] = output["Ticker"].map(lambda ticker: setups.get(str(ticker)).base_depth_pct if setups.get(str(ticker)) else pd.NA)
    output["Distance to Pivot %"] = output["Ticker"].map(lambda ticker: setups.get(str(ticker)).distance_to_pivot_pct if setups.get(str(ticker)) else pd.NA)
    output["Prior Move %"] = output["Ticker"].map(lambda ticker: setups.get(str(ticker)).prior_move_pct if setups.get(str(ticker)) else pd.NA)
    output["Base Bars"] = output["Ticker"].map(lambda ticker: setups.get(str(ticker)).base_bars if setups.get(str(ticker)) else pd.NA)
    output["MA Surfing 10/20"] = output["Ticker"].map(
        lambda ticker: (
            setups.get(str(ticker)).price_above_sma10 and setups.get(str(ticker)).price_above_sma20
            if setups.get(str(ticker))
            else pd.NA
        )
    )
    return output
