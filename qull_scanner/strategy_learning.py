from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_object_dtype, is_string_dtype


EVENT_ID_LENGTH = 16
MIN_PROMOTION_SAMPLE = 100
MIN_OOS_TRADES = 50
MIN_EXPECTANCY_EDGE = 0.05
MIN_OOS_PROFIT_FACTOR = 1.20
MAX_DD_DETERIORATION = 1.20


def _json_default(value: Any) -> str:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _pick(row: Mapping[str, Any], *names: str, default: Any = "N/D") -> Any:
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    return default


def stable_event_id(row: Mapping[str, Any], config: Mapping[str, Any] | None = None) -> str:
    """Build a stable source-trace ID for one strategy signal event.

    The ID deliberately includes the config, because the same ticker/date can be
    selected by different rule versions. This prevents later learning reports
    from mixing outcomes across incompatible strategy definitions.
    """
    payload = {
        "strategy": _pick(row, "strategy", "Strategy", default="N/D"),
        "ticker": str(_pick(row, "Ticker", "ticker", default="N/D")).upper(),
        "date": str(_pick(row, "Date", "Signal Date", "signal_date", default="N/D")),
        "bucket": _pick(row, "SteveAlgo Primary Bucket", "Bucket", "bucket", default="N/D"),
        "config": dict(config or {}),
    }
    raw = json.dumps(payload, sort_keys=True, default=_json_default, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:EVENT_ID_LENGTH]


def _initial_stop_from_signal(signal_row: Mapping[str, Any], entry_bar: pd.Series, entry_price: float) -> float:
    darvas = signal_row.get("Darvas Lower", entry_bar.get("Darvas Lower", math.nan))
    ema20 = entry_bar.get("EMA20", math.nan)
    atr20 = entry_bar.get("ATR20", math.nan)
    candidates: list[float] = []
    for value in [darvas, (float(ema20) - float(atr20)) if pd.notna(ema20) and pd.notna(atr20) else math.nan]:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value < entry_price:
            candidates.append(value)
    return max(candidates) if candidates else entry_price * 0.92


def append_signal_journal(
    signals: pd.DataFrame,
    journal_path: str | Path,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Append paper-tracked signal events idempotently.

    This is the forward-learning journal: every candidate becomes an immutable
    research event with capital authorization locked at 0%.
    """
    journal_path = Path(journal_path)
    config = dict(config or {"strategy": "SteveAlgo", "rule_version": "self_learning_v1"})
    if signals.empty:
        existing = pd.read_csv(journal_path) if journal_path.exists() else pd.DataFrame()
        return existing
    required = {"Date", "Ticker"}
    missing = required - set(signals.columns)
    if missing:
        raise ValueError(f"signals missing required columns: {sorted(missing)}")
    incoming = signals.copy()
    incoming["Date"] = pd.to_datetime(incoming["Date"]).dt.date.astype(str)
    incoming["Ticker"] = incoming["Ticker"].astype(str).str.upper().str.strip()
    incoming["Strategy"] = incoming.get("Strategy", "SteveAlgo")
    incoming["event_id"] = [stable_event_id(row, config) for row in incoming.to_dict("records")]
    incoming["Rule Config JSON"] = json.dumps(config, sort_keys=True, default=_json_default)
    incoming["Paper Status"] = incoming.get("Paper Status", "OPEN")
    incoming["Capital Authorized"] = "0%"
    incoming["Created At UTC"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    existing = pd.read_csv(journal_path) if journal_path.exists() else pd.DataFrame()
    combined = pd.concat([existing, incoming], ignore_index=True, sort=False) if not existing.empty else incoming
    combined = combined.drop_duplicates("event_id", keep="first").sort_values(["Date", "Ticker"]).reset_index(drop=True)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(journal_path, index=False)
    return combined


def resolve_paper_outcomes(
    journal: pd.DataFrame,
    history_by_ticker: Mapping[str, pd.DataFrame],
    max_hold_bars: int = 20,
    target_r: float | None = 3.0,
    slippage_bps: float = 0.0,
) -> pd.DataFrame:
    """Resolve mature paper events into R outcomes while leaving immature events open."""
    rows: list[dict[str, Any]] = []
    if journal.empty:
        return pd.DataFrame(columns=["event_id", "Paper Status", "R"])
    for event in journal.to_dict("records"):
        event_id = event.get("event_id")
        ticker = str(event.get("Ticker", "")).upper()
        signal_date = pd.Timestamp(event.get("Date"))
        hist = history_by_ticker.get(ticker)
        base = {"event_id": event_id, "Ticker": ticker, "Signal Date": signal_date.date().isoformat(), "Paper Status": "OPEN", "R": np.nan}
        if hist is None or hist.empty:
            base["Outcome Note"] = "missing history"
            rows.append(base)
            continue
        h = hist.copy().sort_index()
        if "Date" in h.columns:
            h = h.set_index(pd.to_datetime(h["Date"]))
        else:
            h.index = pd.to_datetime(h.index)
        future_positions = h.index[h.index > signal_date]
        if len(future_positions) < max_hold_bars:
            base["Outcome Note"] = f"immature: {len(future_positions)}/{max_hold_bars} future bars"
            rows.append(base)
            continue
        entry_date = future_positions[0]
        entry_loc = h.index.get_loc(entry_date)
        entry_bar = h.iloc[entry_loc]
        entry = float(entry_bar["Open"]) * (1 + slippage_bps / 10_000)
        stop = _initial_stop_from_signal(event, entry_bar, entry)
        risk = entry - stop
        if risk <= 0 or not math.isfinite(risk):
            base["Paper Status"] = "SKIPPED"
            base["Outcome Note"] = "invalid risk"
            rows.append(base)
            continue
        target = entry + risk * target_r if target_r else math.inf
        exit_date = entry_date
        exit_raw = h.iloc[min(entry_loc + max_hold_bars - 1, len(h) - 1)]["Close"]
        exit_reason = "timeout"
        bars_held = 0
        mfe_r = -math.inf
        mae_r = math.inf
        for bar_offset in range(max_hold_bars):
            loc = entry_loc + bar_offset
            bar = h.iloc[loc]
            exit_date = h.index[loc]
            bars_held = bar_offset + 1
            mfe_r = max(mfe_r, (float(bar["High"]) - entry) / risk)
            mae_r = min(mae_r, (float(bar["Low"]) - entry) / risk)
            if float(bar["Low"]) <= stop:
                exit_reason = "stop"
                exit_raw = stop
                break
            if float(bar["High"]) >= target:
                exit_reason = "target"
                exit_raw = target
                break
            if pd.notna(bar.get("EMA20", math.nan)) and float(bar["Close"]) < float(bar["EMA20"]):
                exit_reason = "ema20_close"
                exit_raw = bar["Close"]
                break
            exit_raw = bar["Close"]
        exit_px = float(exit_raw) * (1 - slippage_bps / 10_000)
        rows.append(
            {
                **base,
                "Paper Status": "CLOSED",
                "Entry Date": pd.Timestamp(entry_date).date().isoformat(),
                "Exit Date": pd.Timestamp(exit_date).date().isoformat(),
                "Entry Price": round(entry, 4),
                "Stop": round(stop, 4),
                "Exit Price": round(exit_px, 4),
                "Exit Reason": exit_reason,
                "Bars Held": bars_held,
                "R": round((exit_px - entry) / risk, 4),
                "MFE R": round(float(mfe_r), 4),
                "MAE R": round(float(mae_r), 4),
                "Outcome Note": "resolved from price history",
            }
        )
    return pd.DataFrame(rows)


def merge_events_with_outcomes(events: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Left-join outcomes to events and keep unresolved/open events visible."""
    if events.empty:
        return events.copy()
    if "event_id" not in events.columns:
        raise ValueError("events must include event_id")
    out = outcomes.copy()
    if out.empty:
        merged = events.copy()
    else:
        if "event_id" not in out.columns:
            raise ValueError("outcomes must include event_id")
        overlap = [c for c in out.columns if c in events.columns and c != "event_id"]
        out = out.rename(columns={c: f"Outcome {c}" for c in overlap})
        merged = events.merge(out, on="event_id", how="left")
    r_col = "R" if "R" in merged.columns else "Outcome R" if "Outcome R" in merged.columns else None
    merged["Outcome Status"] = "OPEN"
    if r_col:
        merged.loc[pd.to_numeric(merged[r_col], errors="coerce").notna(), "Outcome Status"] = "CLOSED"
    return merged


def _profit_factor(r_values: pd.Series) -> float | str:
    wins = r_values[r_values > 0].sum()
    losses = -r_values[r_values < 0].sum()
    if losses == 0:
        return "N/D" if wins == 0 else math.inf
    return round(float(wins / losses), 4)


def _max_drawdown_r(r_values: pd.Series) -> float:
    if r_values.empty:
        return 0.0
    cumulative = r_values.cumsum()
    peak = cumulative.cummax()
    dd = cumulative - peak
    return round(float(dd.min()), 4)


def build_outcome_summary(events: pd.DataFrame, outcomes: pd.DataFrame) -> dict[str, Any]:
    """Summarize closed trade outcomes while accounting for unresolved events."""
    merged = merge_events_with_outcomes(events, outcomes) if not events.empty else outcomes.copy()
    r_col = "R" if "R" in merged.columns else "Outcome R" if "Outcome R" in merged.columns else None
    r = pd.to_numeric(merged[r_col], errors="coerce").dropna() if r_col else pd.Series(dtype=float)
    events_count = int(len(events)) if not events.empty else int(len(outcomes))
    closed = int(len(r))
    wins = int((r > 0).sum())

    mfe_col = next((c for c in ["MFE R", "Outcome MFE R", "max_favorable_r"] if c in merged.columns), None)
    mae_col = next((c for c in ["MAE R", "Outcome MAE R", "max_adverse_r"] if c in merged.columns), None)
    mfe = pd.to_numeric(merged[mfe_col], errors="coerce").dropna() if mfe_col else pd.Series(dtype=float)
    mae = pd.to_numeric(merged[mae_col], errors="coerce").dropna() if mae_col else pd.Series(dtype=float)

    return {
        "events": events_count,
        "closed_trades": closed,
        "open_events": max(events_count - closed, 0),
        "expectancy_r": round(float(r.mean()), 4) if closed else "N/D",
        "win_rate_pct": round(float(wins / closed * 100), 2) if closed else "N/D",
        "profit_factor": _profit_factor(r),
        "max_drawdown_r": _max_drawdown_r(r),
        "median_mfe_r": round(float(mfe.median()), 4) if not mfe.empty else "N/D",
        "median_mae_r": round(float(mae.median()), 4) if not mae.empty else "N/D",
    }


def _r_column(df: pd.DataFrame) -> str:
    for col in ["R", "Outcome R", "r_multiple"]:
        if col in df.columns:
            return col
    raise ValueError("dataframe needs an R/outcome column")


def feature_attribution(events_with_outcomes: pd.DataFrame, min_sample: int = 30) -> pd.DataFrame:
    """Compute non-causal winner/loser feature associations.

    Output is explicitly descriptive. It does not claim predictive power.
    """
    if events_with_outcomes.empty:
        return pd.DataFrame(columns=["feature", "sample_size", "status"])
    r_col = _r_column(events_with_outcomes)
    df = events_with_outcomes.copy()
    df["_R"] = pd.to_numeric(df[r_col], errors="coerce")
    df = df[df["_R"].notna()]
    rows: list[dict[str, Any]] = []
    excluded = {r_col, "_R", "event_id", "Date", "Signal Date", "Entry Date", "Exit Date", "Ticker"}
    status = "OK" if len(df) >= min_sample else "INSUFFICIENT_SAMPLE"

    for col in df.columns:
        if col in excluded or is_bool_dtype(df[col]):
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if is_bool_dtype(numeric):
            continue
        if numeric.notna().sum() >= 2:
            winners = numeric[df["_R"] > 0].dropna()
            losers = numeric[df["_R"] <= 0].dropna()
            if winners.empty or losers.empty:
                continue
            q80 = numeric.quantile(0.8)
            q20 = numeric.quantile(0.2)
            rows.append(
                {
                    "feature": col,
                    "sample_size": int(numeric.notna().sum()),
                    "winner_median": round(float(winners.median()), 4),
                    "loser_median": round(float(losers.median()), 4),
                    "difference": round(float(winners.median() - losers.median()), 4),
                    "expectancy_top_quantile": round(float(df.loc[numeric >= q80, "_R"].mean()), 4),
                    "expectancy_bottom_quantile": round(float(df.loc[numeric <= q20, "_R"].mean()), 4),
                    "status": status,
                    "association_only": True,
                }
            )
        elif is_object_dtype(df[col]) or is_string_dtype(df[col]) or isinstance(df[col].dtype, pd.CategoricalDtype):
            for value, group in df.groupby(col, dropna=True):
                if len(group) < 1:
                    continue
                rows.append(
                    {
                        "feature": f"{col}={value}",
                        "sample_size": int(len(group)),
                        "expectancy": round(float(group["_R"].mean()), 4),
                        "win_rate_pct": round(float((group["_R"] > 0).mean() * 100), 2),
                        "status": status,
                        "association_only": True,
                    }
                )
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["feature", "sample_size", "status"])


def _as_float(mapping: Mapping[str, Any], key: str, default: float = math.nan) -> float:
    try:
        return float(mapping.get(key, default))
    except (TypeError, ValueError):
        return default


def _promotion_gate(baseline: Mapping[str, Any], oos: Mapping[str, Any], random_summary: Mapping[str, Any]) -> tuple[str, str]:
    oos_trades = _as_float(oos, "trades", _as_float(oos, "closed_trades", 0))
    baseline_exp = _as_float(baseline, "expectancy_r")
    oos_exp = _as_float(oos, "expectancy_r")
    random_exp = _as_float(random_summary, "expectancy_r")
    oos_pf = _as_float(oos, "profit_factor")
    baseline_dd = abs(_as_float(baseline, "max_drawdown_r", 0))
    oos_dd = abs(_as_float(oos, "max_drawdown_r", 0))
    reasons = []
    if oos_trades < MIN_OOS_TRADES:
        reasons.append("OOS trades below minimum")
    if not math.isfinite(oos_exp) or not math.isfinite(baseline_exp) or oos_exp - baseline_exp < MIN_EXPECTANCY_EDGE:
        reasons.append("OOS expectancy edge too small")
    if not math.isfinite(random_exp) or oos_exp - random_exp < MIN_EXPECTANCY_EDGE:
        reasons.append("does not beat matched random by required edge")
    if not math.isfinite(oos_pf) or oos_pf < MIN_OOS_PROFIT_FACTOR:
        reasons.append("OOS profit factor below threshold")
    if baseline_dd and oos_dd > baseline_dd * MAX_DD_DETERIORATION:
        reasons.append("drawdown deterioration too large")
    if reasons:
        return "REJECTED", "; ".join(reasons)
    return "WATCH", "passes initial research gates; requires forward paper trades and Antonio approval"


def propose_rule_candidates(
    attribution: pd.DataFrame,
    baseline_summary: Mapping[str, Any],
    oos_summary: Mapping[str, Any],
    random_summary: Mapping[str, Any],
) -> pd.DataFrame:
    """Generate conservative rule hypotheses from attribution rows.

    The function never returns an approved production rule. It can only reject or
    mark a hypothesis for watch/paper review.
    """
    proposals: list[dict[str, Any]] = []
    if attribution.empty:
        return pd.DataFrame(columns=["rule_id", "hypothesis", "promotion_status", "reason"])
    for _, row in attribution.iterrows():
        if row.get("status") != "OK":
            continue
        feature = str(row.get("feature"))
        sample_size = int(row.get("sample_size", 0) or 0)
        if sample_size < 1:
            continue
        hypothesis = None
        changed: dict[str, Any] = {}
        if feature == "Bucket=Yellow" and pd.notna(row.get("expectancy")) and float(row.get("expectancy")) < 0:
            hypothesis = "exclude Yellow bucket from tradable SteveAlgo events"
            changed = {"allow_yellow": False}
        elif feature == "Momentum Rank" and pd.notna(row.get("difference")) and float(row.get("difference")) > 5:
            hypothesis = "tighten minimum relative-strength / Momentum Rank threshold"
            changed = {"min_rs": "increase_by_5_points"}
        elif feature == "Reward-Risk" and pd.notna(row.get("difference")) and float(row.get("difference")) > 0.5:
            hypothesis = "tighten minimum Reward-Risk threshold"
            changed = {"min_reward_risk": "increase_by_0.5R"}
        if not hypothesis:
            continue
        status, reason = _promotion_gate(baseline_summary, oos_summary, random_summary)
        if sample_size < MIN_OOS_TRADES:
            status = "REJECTED"
            reason = f"sample below proposal evidence minimum ({sample_size} < {MIN_OOS_TRADES}); " + reason
        payload = {"hypothesis": hypothesis, "changed": changed, "feature": feature}
        proposals.append(
            {
                "rule_id": hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12],
                "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "hypothesis": hypothesis,
                "changed_parameters_json": json.dumps(changed, sort_keys=True),
                "source_feature": feature,
                "sample_size": sample_size,
                "promotion_status": status,
                "reason": reason,
                "capital_authorized_pct": 0,
            }
        )
    return pd.DataFrame(proposals)
