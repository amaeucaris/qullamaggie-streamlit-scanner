from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qull_scanner.backtest_steve_algo import summarize_in_sample_out_of_sample, summarize_steve_algo_backtest
from qull_scanner.strategy_learning import (
    build_outcome_summary,
    feature_attribution,
    merge_events_with_outcomes,
    propose_rule_candidates,
    stable_event_id,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENTS = ROOT / "exports" / "steve_algo_backtest_events.csv"
DEFAULT_TRADES = ROOT / "exports" / "steve_algo_backtest_trades.csv"
DEFAULT_OUT = ROOT / "exports"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (pd.isna(value)):
        return None
    return value


def _records(df: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    if df.empty:
        return []
    if limit is not None:
        df = df.head(limit)
    records = []
    for record in df.to_dict("records"):
        records.append({k: _json_safe(v) for k, v in record.items()})
    return records


def _ensure_event_ids(events: pd.DataFrame, trades: pd.DataFrame, config: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = config or {"strategy": "SteveAlgo", "rule_version": "self_review_v1"}
    e = events.copy()
    t = trades.copy()
    if not e.empty and "event_id" not in e.columns:
        e["strategy"] = e.get("strategy", "SteveAlgo")
        e["event_id"] = [stable_event_id(row, config) for row in e.to_dict("records")]
    if not t.empty and "event_id" not in t.columns:
        synth_events = []
        for row in t.to_dict("records"):
            synth_events.append(
                {
                    "strategy": row.get("Strategy", "SteveAlgo"),
                    "Ticker": row.get("Ticker"),
                    "Date": row.get("Signal Date", row.get("Date")),
                    "SteveAlgo Primary Bucket": row.get("Bucket", row.get("SteveAlgo Primary Bucket", "N/D")),
                }
            )
        t["event_id"] = [stable_event_id(row, config) for row in synth_events]
    return e, t


def _normalize_trade_outcomes(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["event_id", "R"])
    out = trades.copy()
    rename = {}
    if "MFE R" not in out.columns and "max_favorable_r" in out.columns:
        rename["max_favorable_r"] = "MFE R"
    if "MAE R" not in out.columns and "max_adverse_r" in out.columns:
        rename["max_adverse_r"] = "MAE R"
    if rename:
        out = out.rename(columns=rename)
    keep = [c for c in ["event_id", "R", "MFE R", "MAE R", "Exit Reason", "Exit Date", "Entry Date"] if c in out.columns]
    return out[keep]


def _matched_random_summary(trades: pd.DataFrame) -> dict[str, Any]:
    """Conservative placeholder for self-review reports.

    The full backtest script already computes a matched random benchmark. If no
    random trades file is provided, self-review treats random as equal to baseline
    so proposals cannot pass by accident.
    """
    summary = summarize_steve_algo_backtest(trades) if not trades.empty else {"expectancy_r": 0.0, "profit_factor": 0.0, "trades": 0}
    return {"expectancy_r": summary.get("expectancy_r", 0.0), "profit_factor": summary.get("profit_factor", 0.0), "trades": summary.get("trades", 0)}


def build_self_review(events: pd.DataFrame, trades: pd.DataFrame, min_sample: int = 30) -> dict[str, Any]:
    events, trades = _ensure_event_ids(events, trades)
    outcomes = _normalize_trade_outcomes(trades)
    merged = merge_events_with_outcomes(events, outcomes) if not events.empty else trades.copy()
    if "R" not in merged.columns and "Outcome R" in merged.columns:
        merged["R"] = merged["Outcome R"]
    if "Bucket" not in merged.columns and "SteveAlgo Primary Bucket" in merged.columns:
        merged["Bucket"] = merged["SteveAlgo Primary Bucket"]

    baseline_summary = build_outcome_summary(events, outcomes) if not events.empty else summarize_steve_algo_backtest(trades)
    split = summarize_in_sample_out_of_sample(trades) if not trades.empty and "Signal Date" in trades.columns else {"out_of_sample": baseline_summary}
    oos_summary = split.get("out_of_sample", baseline_summary)
    random_summary = _matched_random_summary(trades)
    attribution = feature_attribution(merged, min_sample=min_sample) if not merged.empty and ("R" in merged.columns or "Outcome R" in merged.columns) else pd.DataFrame()
    proposals = propose_rule_candidates(attribution, baseline_summary, oos_summary, random_summary)

    warnings = [
        "Research only: self-improvement proposes hypotheses, not trades.",
        "Capital authorized remains 0% until Portfolio Risk Gate + Antonio approval.",
    ]
    if proposals.empty:
        warnings.append("No rule proposal passed minimum evidence gates.")
    if baseline_summary.get("open_events", 0):
        warnings.append(f"Open/unresolved events: {baseline_summary['open_events']}")

    return {
        "verdict": "NO_PRODUCTION_CHANGE",
        "capital_authorized_pct": 0,
        "baseline_summary": baseline_summary,
        "split_summary": split,
        "random_summary": random_summary,
        "attribution_rows": int(len(attribution)),
        "proposal_rows": int(len(proposals)),
        "attribution": _records(attribution, limit=100),
        "proposals": _records(proposals, limit=50),
        "warnings": warnings,
    }


def write_self_review(review: dict[str, Any], output_dir: str | Path = DEFAULT_OUT) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "strategy_self_review.json"
    md_path = output_dir / "strategy_self_review.md"
    json_path.write_text(json.dumps(review, indent=2, sort_keys=True, default=str))

    lines = [
        "# Strategy Self-Review",
        "",
        f"Verdict: `{review.get('verdict', 'N/D')}`",
        f"Capital authorized: `{review.get('capital_authorized_pct', 0)}%`",
        "",
        "## Baseline",
    ]
    for key, value in review.get("baseline_summary", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Warnings"])
    for warning in review.get("warnings", []):
        lines.append(f"- {warning}")
    lines.extend(["", "## Rule Proposals"])
    proposals = review.get("proposals", [])
    if not proposals:
        lines.append("- None. No production change.")
    for proposal in proposals:
        lines.append(
            f"- {proposal.get('promotion_status', 'N/D')}: {proposal.get('hypothesis', 'N/D')} "
            f"({proposal.get('reason', 'N/D')})"
        )
    lines.extend(["", "## Top Attribution Rows"])
    for row in review.get("attribution", [])[:20]:
        lines.append(f"- {row.get('feature', 'N/D')}: status={row.get('status', 'N/D')} sample={row.get('sample_size', 'N/D')}")
    md_path.write_text("\n".join(lines) + "\n")
    return {"json": json_path, "markdown": md_path}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic strategy self-review")
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-sample", type=int, default=30)
    args = parser.parse_args()

    events = _read_csv(args.events)
    trades = _read_csv(args.trades)
    review = build_self_review(events, trades, min_sample=args.min_sample)
    paths = write_self_review(review, args.out)
    print(f"Wrote {paths['json']} and {paths['markdown']}")
    print(f"VERDICT {review['verdict']} proposals={review['proposal_rows']} attribution={review['attribution_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
