from __future__ import annotations

from typing import Mapping, Any


def _pass_fail(value: bool) -> str:
    return "PASS" if bool(value) else "FAIL"


def _money_millions(value: float) -> str:
    return f"${value / 1_000_000:.1f}M"


def strict_qullamaggie_lineage(
    row: Mapping[str, Any],
    *,
    min_adr_pct: float,
    min_dollar_volume: float,
    top_percent: float = 2,
) -> list[str]:
    """Explain the hard-gate strict Qullamaggie filter for a candidate row."""
    prefix = f"Top {top_percent:g}%"
    lines = []
    for window in ("1M", "3M", "6M"):
        key = f"{prefix} {window}"
        lines.append(f"{key}: {_pass_fail(bool(row.get(key, False)))}")

    adr = float(row.get("ADR 20D %", 0.0))
    dollar_volume = float(row.get("Daily $ Volume 20D", 0.0))
    price_above_sma10 = bool(row.get("Price > SMA10", False))
    price_above_sma20 = bool(row.get("Price > SMA20", False))

    lines.append(f"ADR20: {adr:.2f}% >= {min_adr_pct:.2f}%: {_pass_fail(adr >= min_adr_pct)}")
    lines.append(
        "Dollar volume 20D: "
        f"{_money_millions(dollar_volume)} >= {_money_millions(min_dollar_volume)}: "
        f"{_pass_fail(dollar_volume >= min_dollar_volume)}"
    )
    lines.append(f"Price > SMA10: {_pass_fail(price_above_sma10)}")
    lines.append(f"Price > SMA20: {_pass_fail(price_above_sma20)}")

    if "ATR Extension SMA50" in row:
        lines.append(f"ATR extension SMA50: {float(row['ATR Extension SMA50']):.2f}x")

    return lines
