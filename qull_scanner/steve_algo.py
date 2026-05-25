from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True)
class SteveAlgoThresholds:
    min_market_cap: float = 1_000_000_000
    min_dollar_volume: float = 50_000_000
    min_price: float = 5.0
    min_rs: float = 85.0
    min_trend_strength: float = 80.0
    min_reward_risk: float = 3.0
    min_dcr_white_up: float = 60.0
    min_dcr_entry: float = 50.0
    max_white_up_ema10_atr: float = 1.5
    yellow_ema20_atr: float = 1.75
    yellow_sma50_atr: float = 3.5


def _float(row: Mapping[str, Any], key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, default)
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(row: Mapping[str, Any], key: str) -> bool:
    return bool(row.get(key, False))


def _passes_universe(row: Mapping[str, Any], thresholds: SteveAlgoThresholds) -> tuple[bool, list[str]]:
    failures: list[str] = []
    market_cap = _float(row, "Market Cap")
    dollar_volume = _float(row, "Daily $ Volume 20D", 0.0)
    price = _float(row, "Price", 0.0)
    if thresholds.min_market_cap > 0 and (not math.isfinite(market_cap) or market_cap < thresholds.min_market_cap):
        failures.append("market cap < $1B or N/D")
    if dollar_volume < thresholds.min_dollar_volume:
        failures.append("dollar volume below threshold")
    if price < thresholds.min_price:
        failures.append("price below threshold")
    return not failures, failures


def compute_trend_strength(row: Mapping[str, Any]) -> float:
    price = _float(row, "Price")
    checks = [
        price > _float(row, "EMA10"),
        price > _float(row, "EMA20"),
        price > _float(row, "EMA50"),
        price > _float(row, "SMA50"),
        price > _float(row, "SMA200"),
        _float(row, "EMA10") >= _float(row, "EMA20"),
        _float(row, "EMA20") >= _float(row, "EMA50"),
        _float(row, "SMA50") >= _float(row, "SMA200"),
        _float(row, "Return 1M %", 0.0) > 0,
        _float(row, "Return 3M %", 0.0) > 0,
        _float(row, "Return 6M %", 0.0) > 0,
        _float(row, "DCR %", 0.0) >= 50,
    ]
    return round(sum(bool(x) for x in checks) / len(checks) * 100, 2)


def compute_reward_risk(row: Mapping[str, Any]) -> float:
    if "Reward-Risk" in row and math.isfinite(_float(row, "Reward-Risk")):
        return _float(row, "Reward-Risk")
    price = _float(row, "Price")
    darvas_lower = _float(row, "Darvas Lower")
    atr20 = _float(row, "ATR20")
    ema20 = _float(row, "EMA20")
    stop_candidates = [darvas_lower]
    if math.isfinite(ema20) and math.isfinite(atr20):
        stop_candidates.append(ema20 - atr20)
    valid_stops = [s for s in stop_candidates if math.isfinite(s) and s < price]
    if not valid_stops or not math.isfinite(price) or atr20 <= 0:
        return math.nan
    stop = max(valid_stops)
    stop_risk = price - stop
    upside = atr20 * 5
    return round(upside / stop_risk, 2) if stop_risk > 0 else math.nan


def classify_steve_algo_row(row: Mapping[str, Any], thresholds: SteveAlgoThresholds | None = None) -> dict[str, Any]:
    thresholds = thresholds or SteveAlgoThresholds()
    universe_ok, failures = _passes_universe(row, thresholds)
    if not universe_ok:
        return {
            "SteveAlgo Buckets": "Avoid",
            "SteveAlgo Primary Bucket": "Avoid",
            "SteveAlgo Status": "Avoid",
            "SteveAlgo Trend Strength": compute_trend_strength(row),
            "SteveAlgo Reason": "; ".join(failures),
            "Capital Authorized": "0%",
        }

    trend_strength = compute_trend_strength(row)
    reward_risk = compute_reward_risk(row)
    rs = _float(row, "Momentum Rank", _float(row, "Universe Percentile", 0.0))
    dcr = _float(row, "DCR %", 0.0)
    price = _float(row, "Price")
    ema10 = _float(row, "EMA10")
    ema20 = _float(row, "EMA20")
    ema50 = _float(row, "EMA50")
    sma50 = _float(row, "SMA50")
    sma200 = _float(row, "SMA200")
    darvas_upper = _float(row, "Darvas Upper")
    ema10_ext = _float(row, "ATR Extension EMA10", 0.0)
    ema20_ext = _float(row, "ATR Extension EMA20", 0.0)
    sma50_ext = _float(row, "ATR Extension SMA50", 0.0)

    upward_pivot = (
        (price > darvas_upper or _bool(row, "Breakout Above Lookback High"))
        and _bool(row, "EMA10 Rising")
        and price > ema10
        and dcr >= thresholds.min_dcr_white_up
        and ema10_ext < thresholds.max_white_up_ema10_atr
    )
    ma_aligned = price >= ema10 >= ema20 >= ema50 and price >= sma50 >= sma200
    returns_positive = all(_float(row, key, 0.0) > 0 for key in ["Return 1M %", "Return 3M %", "Return 6M %"])
    entry = (
        ma_aligned
        and returns_positive
        and rs >= thresholds.min_rs
        and trend_strength >= thresholds.min_trend_strength
        and dcr >= thresholds.min_dcr_entry
        and math.isfinite(reward_risk)
        and reward_risk >= thresholds.min_reward_risk
    )
    extended = ema20_ext >= thresholds.yellow_ema20_atr or sma50_ext >= thresholds.yellow_sma50_atr

    buckets: list[str] = []
    reasons: list[str] = []
    if upward_pivot:
        buckets.append("White Up")
        reasons.append("upward pivot: prior high/Darvas break, EMA10 rising, strong DCR")
    if entry:
        buckets.append("Entry")
        reasons.append(f"Entry: MA alignment, RS {rs:.1f}, trend {trend_strength:.1f}, R/R {reward_risk:.2f}")
    if entry and extended:
        buckets.append("Yellow")
        reasons.append(f"extended: EMA20 {ema20_ext:.2f} ATR or SMA50 {sma50_ext:.2f} ATR")

    if not buckets:
        buckets = ["Avoid"]
        status = "Avoid"
        primary = "Avoid"
        reasons.append("no SteveAlgo bucket passed")
    elif "Yellow" in buckets:
        status = "Buy Ext"
        primary = "Yellow"
    elif "Entry" in buckets:
        status = "Buy"
        primary = "Entry"
    else:
        status = "Pilot"
        primary = "White Up"

    return {
        "SteveAlgo Buckets": ", ".join(buckets),
        "SteveAlgo Primary Bucket": primary,
        "SteveAlgo Status": status,
        "SteveAlgo Trend Strength": trend_strength,
        "Reward-Risk": round(reward_risk, 2) if math.isfinite(reward_risk) else math.nan,
        "SteveAlgo Reason": "; ".join(reasons),
        "Capital Authorized": "0%",
    }


def apply_steve_algo_watchlists(
    metrics: pd.DataFrame,
    thresholds: SteveAlgoThresholds | None = None,
    include_avoids: bool = False,
) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    thresholds = thresholds or SteveAlgoThresholds()
    output = metrics.copy()
    classifications = [classify_steve_algo_row(row, thresholds) for row in output.to_dict("records")]
    class_df = pd.DataFrame(classifications, index=output.index)
    for column in class_df.columns:
        if column == "Reward-Risk" and column in output.columns:
            output[column] = class_df[column].combine_first(output[column])
        else:
            output[column] = class_df[column]
    if not include_avoids:
        output = output[output["SteveAlgo Primary Bucket"] != "Avoid"].copy()
    bucket_order = {"Entry": 0, "White Up": 1, "Yellow": 2, "Avoid": 3}
    output["SteveAlgo Bucket Sort"] = output["SteveAlgo Primary Bucket"].map(bucket_order).fillna(9)
    sort_cols = [c for c in ["SteveAlgo Bucket Sort", "SteveAlgo Trend Strength", "Momentum Rank"] if c in output.columns]
    return output.sort_values(sort_cols, ascending=[True, False, False][: len(sort_cols)]).drop(columns=["SteveAlgo Bucket Sort"])
