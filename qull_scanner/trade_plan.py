from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Any

import pandas as pd


@dataclass(frozen=True)
class TradePlan:
    ticker: str
    setup_type: str
    entry_trigger: float
    breakout_level: float
    stop: float
    risk_pct: float
    stop_to_adr_ratio: float
    stop_bucket: str
    adr_20d_pct: float
    dollar_volume_20d: float
    distance_to_pivot_pct: float | None = None
    extension_atr_sma50: float | None = None
    reason: str = ""


def classify_stop_to_adr(stop_to_adr_ratio: float) -> str:
    """Classify planned stop width relative to ADR20.

    Buckets are intentionally coarse and human-readable for the decision panel:
    - A+: stop <= 0.75x ADR
    - OK: stop <= 1.00x ADR
    - Wide: stop <= 1.50x ADR
    - Reject: stop > 1.50x ADR or invalid
    """
    if not math.isfinite(stop_to_adr_ratio) or stop_to_adr_ratio < 0:
        return "Reject"
    if stop_to_adr_ratio <= 0.75:
        return "A+"
    if stop_to_adr_ratio <= 1.00:
        return "OK"
    if stop_to_adr_ratio <= 1.50:
        return "Wide"
    return "Reject"


def build_breakout_trade_plan(row: Mapping[str, Any], setup_type: str = "Breakout") -> TradePlan:
    """Build a decision-oriented trade plan from a scanner candidate row.

    This does not place trades. It just turns scanner evidence into a structured
    risk card for review.
    """
    ticker = str(row.get("Ticker", ""))
    price = float(row["Price"])
    breakout_level = float(row.get("Breakout Level", price))
    base_low = float(row.get("Base Low", row.get("SMA20", price)))
    entry = max(price, breakout_level) * 1.001
    stop = base_low
    risk_pct = ((entry - stop) / entry) * 100 if entry > 0 else math.inf
    adr = float(row.get("ADR 20D %", 0.0))
    stop_to_adr = risk_pct / adr if adr > 0 else math.inf
    return TradePlan(
        ticker=ticker,
        setup_type=setup_type,
        entry_trigger=round(entry, 2),
        breakout_level=round(breakout_level, 2),
        stop=round(stop, 2),
        risk_pct=round(risk_pct, 2),
        stop_to_adr_ratio=round(stop_to_adr, 2) if math.isfinite(stop_to_adr) else math.inf,
        stop_bucket=classify_stop_to_adr(stop_to_adr),
        adr_20d_pct=adr,
        dollar_volume_20d=float(row.get("Daily $ Volume 20D", 0.0)),
        distance_to_pivot_pct=float(row.get("Distance to Pivot %", 0.0)) if "Distance to Pivot %" in row else None,
        extension_atr_sma50=float(row.get("ATR Extension SMA50", 0.0)) if "ATR Extension SMA50" in row else None,
        reason=str(row.get("Reason", "")),
    )


def add_trade_plan_columns(candidates: pd.DataFrame, setup_type: str = "Breakout") -> pd.DataFrame:
    """Append non-execution trade-plan columns to scanner candidates.

    The function is intentionally deterministic and side-effect-free: it does not
    place trades, size positions, or decide entries. It only exposes the risk
    card needed for manual review/export.
    """
    if candidates.empty:
        return candidates

    output = candidates.copy()
    plans = [build_breakout_trade_plan(row, setup_type=setup_type) for row in output.to_dict("records")]
    output["Trade Setup Type"] = [plan.setup_type for plan in plans]
    output["Trade Entry Trigger"] = [plan.entry_trigger for plan in plans]
    output["Trade Stop"] = [plan.stop for plan in plans]
    output["Trade Risk %"] = [plan.risk_pct for plan in plans]
    output["Stop / ADR"] = [plan.stop_to_adr_ratio for plan in plans]
    output["Stop Bucket"] = [plan.stop_bucket for plan in plans]
    return output
