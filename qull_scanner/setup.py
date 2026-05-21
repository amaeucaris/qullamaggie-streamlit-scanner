from __future__ import annotations

from dataclasses import dataclass


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
