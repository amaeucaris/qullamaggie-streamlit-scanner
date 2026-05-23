from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from qull_scanner.lineage import strict_qullamaggie_lineage


@dataclass(frozen=True)
class ScannerThresholds:
    min_price: float
    min_avg_volume: int
    top_percent: float
    min_breakout_pct: float
    stockbee_min_price: float
    stockbee_min_volume: int
    min_dollar_volume: int = 150_000_000
    min_adr_pct: float = 3.5
    max_extension_atr: float = 5.0


def _ensure_dollar_volume(metrics: pd.DataFrame) -> pd.DataFrame:
    q_metrics = metrics.copy()
    if "Daily $ Volume 20D" not in q_metrics.columns:
        q_metrics["Daily $ Volume 20D"] = q_metrics["Price"] * q_metrics["Avg Volume 20D"]
    return q_metrics


def _top_rank_columns(top_percent: float) -> list[str]:
    return [
        f"Top {top_percent:g}% 1M",
        f"Top {top_percent:g}% 3M",
        f"Top {top_percent:g}% 6M",
    ]


def add_top_timeframe_flags(metrics: pd.DataFrame, top_percent: float) -> pd.DataFrame:
    q_metrics = metrics.copy()
    top_cutoff = max(1, math.ceil(len(q_metrics) * top_percent / 100))
    top_rank_columns = _top_rank_columns(top_percent)
    q_metrics[top_rank_columns[0]] = q_metrics["Return 1M %"].rank(method="min", ascending=False) <= top_cutoff
    q_metrics[top_rank_columns[1]] = q_metrics["Return 3M %"].rank(method="min", ascending=False) <= top_cutoff
    q_metrics[top_rank_columns[2]] = q_metrics["Return 6M %"].rank(method="min", ascending=False) <= top_cutoff
    return q_metrics


def apply_qullamaggie_filter(metrics: pd.DataFrame, thresholds: ScannerThresholds) -> pd.DataFrame:
    """Strict Qullamaggie screen: top cohort in 1M AND 3M AND 6M plus hard gates."""
    if metrics.empty:
        return metrics

    q_metrics = add_top_timeframe_flags(_ensure_dollar_volume(metrics), thresholds.top_percent)
    top_rank_columns = _top_rank_columns(thresholds.top_percent)

    passed = q_metrics[
        (q_metrics[top_rank_columns].all(axis=1))
        & (q_metrics["ADR 20D %"] >= thresholds.min_adr_pct)
        & (q_metrics["Price > SMA10"])
        & (q_metrics["Price > SMA20"])
        & (q_metrics["Avg Volume 20D"] > thresholds.min_avg_volume)
        & (q_metrics["Price"] > thresholds.min_price)
        & (q_metrics["Daily $ Volume 20D"] >= thresholds.min_dollar_volume)
    ].copy()
    passed["Strict Q Lineage"] = passed.apply(
        lambda row: " | ".join(
            strict_qullamaggie_lineage(
                row,
                min_adr_pct=thresholds.min_adr_pct,
                min_dollar_volume=thresholds.min_dollar_volume,
                top_percent=thresholds.top_percent,
            )
        ),
        axis=1,
    )
    return passed


def apply_steve_style_qullamaggie_filter(metrics: pd.DataFrame, thresholds: ScannerThresholds) -> pd.DataFrame:
    """Broader SteveDJacobs-style KQ candidate board, separate from strict Q."""
    if metrics.empty:
        return metrics

    q_metrics = add_top_timeframe_flags(_ensure_dollar_volume(metrics), thresholds.top_percent)
    top_rank_columns = _top_rank_columns(thresholds.top_percent)

    strict_overlap = q_metrics[top_rank_columns].all(axis=1) & (q_metrics["Daily $ Volume 20D"] >= thresholds.min_dollar_volume)
    top_hits = q_metrics[top_rank_columns].sum(axis=1)
    steve_min_dollar_volume = min(thresholds.min_dollar_volume, 50_000_000)

    broad_momentum = (
        (top_hits >= 1)
        | (q_metrics["Momentum Rank"] >= 85)
        | ((q_metrics["Return 3M %"] >= 50) & (q_metrics["Return 6M %"] >= 50))
        | (q_metrics["Return 1M %"] >= 20)
    )
    liquid_trending = (
        (q_metrics["Price"] > thresholds.min_price)
        & (q_metrics["Avg Volume 20D"] > thresholds.min_avg_volume)
        & (q_metrics["Daily $ Volume 20D"] >= steve_min_dollar_volume)
        & (q_metrics["ADR 20D %"] >= thresholds.min_adr_pct)
        & (q_metrics["Price > SMA10"])
        & (q_metrics["Price > SMA20"])
    )
    extension_ok = q_metrics["ATR Extension SMA50"] <= thresholds.max_extension_atr

    q_metrics["Strict Qullamaggie Overlap"] = strict_overlap.astype(object)
    q_metrics["Steve-style KQ Score"] = (
        top_hits.astype(float) * 25
        + q_metrics["Momentum Rank"].fillna(0) * 0.35
        + q_metrics["ADR 20D %"].fillna(0) * 1.5
        + np.minimum(q_metrics["Return 3M %"].fillna(0), 150) * 0.12
        + np.minimum(q_metrics["Return 6M %"].fillna(0), 250) * 0.08
    )
    q_metrics["Steve-style KQ Reason"] = np.select(
        [strict_overlap, top_hits >= 1, q_metrics["Momentum Rank"] >= 85, q_metrics["Return 1M %"] >= 20],
        ["Strict Q overlap", "Top 2% on at least one timeframe", "High composite momentum", "Strong 1M momentum"],
        default="3M/6M momentum continuation",
    )

    return q_metrics[broad_momentum & liquid_trending & extension_ok].sort_values(
        ["Steve-style KQ Score", "Momentum Rank"], ascending=False
    ).copy()


def apply_stockbee_filter(metrics: pd.DataFrame, thresholds: ScannerThresholds) -> pd.DataFrame:
    if metrics.empty:
        return metrics

    return metrics[
        (metrics["Daily Return %"] >= thresholds.min_breakout_pct)
        & (metrics["Volume"] > metrics["Prev Volume"])
        & (metrics["Volume"] >= thresholds.stockbee_min_volume)
        & (metrics["Price"] > thresholds.stockbee_min_price)
    ].copy()


def apply_stockbee_9m_movers_filter(metrics: pd.DataFrame, thresholds: ScannerThresholds) -> pd.DataFrame:
    """Stockbee 9M Movers: 4%+ daily mover with volume expansion and >= 9M shares traded."""
    if metrics.empty:
        return metrics

    return metrics[
        (metrics["Daily Return %"] >= thresholds.min_breakout_pct)
        & (metrics["Volume"] > metrics["Prev Volume"])
        & (metrics["Volume"] >= 8_900_000)
        & (metrics["Price"] > thresholds.stockbee_min_price)
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
