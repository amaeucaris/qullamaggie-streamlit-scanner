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

from qull_scanner.strategy_learning import append_signal_journal, resolve_paper_outcomes
from scripts.run_strategy_self_review import build_self_review, write_self_review

DEFAULT_WATCHLIST_DIR = ROOT / "exports" / "watchlists"
DEFAULT_HISTORY = ROOT / "data" / "history_prices.parquet"
DEFAULT_OUTPUT = ROOT / "exports"


def latest_watchlist_path(watchlist_dir: str | Path = DEFAULT_WATCHLIST_DIR) -> Path | None:
    watchlist_dir = Path(watchlist_dir)
    candidates = sorted(watchlist_dir.glob("steve_algo_watchlist_*.csv"))
    return candidates[-1] if candidates else None


def load_history_by_ticker(history_path: str | Path) -> dict[str, pd.DataFrame]:
    history_path = Path(history_path)
    if not history_path.exists():
        return {}
    df = pd.read_parquet(history_path) if history_path.suffix.lower() == ".parquet" else pd.read_csv(history_path)
    if df.empty or "Ticker" not in df.columns:
        return {}
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    return {str(ticker).upper(): group.sort_values("Date").set_index("Date") for ticker, group in df.groupby("Ticker", sort=False)}


def run_learning_loop(
    watchlist_dir: str | Path = DEFAULT_WATCHLIST_DIR,
    history_path: str | Path = DEFAULT_HISTORY,
    output_dir: str | Path = DEFAULT_OUTPUT,
    max_hold_bars: int = 20,
    target_r: float = 3.0,
    slippage_bps: float = 10.0,
    min_sample: int = 30,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    watchlist_path = latest_watchlist_path(watchlist_dir)
    if watchlist_path is None:
        raise FileNotFoundError(f"No steve_algo_watchlist_*.csv in {watchlist_dir}")

    config = {
        "strategy": "SteveAlgo",
        "rule_version": "self_learning_v1",
        "max_hold_bars": int(max_hold_bars),
        "target_r": float(target_r),
        "slippage_bps": float(slippage_bps),
    }
    signals = pd.read_csv(watchlist_path)
    journal_path = output_dir / "strategy_signal_journal.csv"
    outcomes_path = output_dir / "strategy_paper_outcomes.csv"
    journal = append_signal_journal(signals, journal_path, config=config)

    history = load_history_by_ticker(history_path)
    outcomes = resolve_paper_outcomes(journal, history, max_hold_bars=max_hold_bars, target_r=target_r, slippage_bps=slippage_bps)
    outcomes.to_csv(outcomes_path, index=False)
    closed = outcomes[outcomes["Paper Status"] == "CLOSED"].copy() if not outcomes.empty else pd.DataFrame()
    historical_events_path = output_dir / "steve_algo_backtest_events.csv"
    historical_trades_path = output_dir / "steve_algo_backtest_trades.csv"
    historical_events = pd.read_csv(historical_events_path) if historical_events_path.exists() else pd.DataFrame()
    historical_trades = pd.read_csv(historical_trades_path) if historical_trades_path.exists() else pd.DataFrame()
    events_for_review = pd.concat([historical_events, journal], ignore_index=True, sort=False) if not historical_events.empty else journal
    trades_for_review = pd.concat([historical_trades, closed], ignore_index=True, sort=False) if not historical_trades.empty else closed
    review = build_self_review(events_for_review, trades_for_review, min_sample=min_sample)
    review["learning_loop"] = {
        "watchlist_path": str(watchlist_path),
        "journal_path": str(journal_path),
        "outcomes_path": str(outcomes_path),
        "history_path": str(history_path),
        "open_outcomes": int((outcomes.get("Paper Status", pd.Series(dtype=str)) == "OPEN").sum()) if not outcomes.empty else 0,
        "closed_outcomes": int((outcomes.get("Paper Status", pd.Series(dtype=str)) == "CLOSED").sum()) if not outcomes.empty else 0,
        "historical_event_rows": int(len(historical_events)),
        "historical_trade_rows": int(len(historical_trades)),
    }
    paths = write_self_review(review, output_dir)
    return {
        "watchlist_path": str(watchlist_path),
        "journal_rows": int(len(journal)),
        "outcome_rows": int(len(outcomes)),
        "closed_outcomes": int((outcomes.get("Paper Status", pd.Series(dtype=str)) == "CLOSED").sum()) if not outcomes.empty else 0,
        "open_outcomes": int((outcomes.get("Paper Status", pd.Series(dtype=str)) == "OPEN").sum()) if not outcomes.empty else 0,
        "review_json": str(paths["json"]),
        "review_markdown": str(paths["markdown"]),
        "verdict": review["verdict"],
        "proposal_rows": review["proposal_rows"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Append latest SteveAlgo watchlist, resolve paper outcomes, and rerun self-review")
    parser.add_argument("--watchlist-dir", type=Path, default=DEFAULT_WATCHLIST_DIR)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-hold-bars", type=int, default=20)
    parser.add_argument("--target-r", type=float, default=3.0)
    parser.add_argument("--slippage-bps", type=float, default=10.0)
    parser.add_argument("--min-sample", type=int, default=30)
    args = parser.parse_args()
    result = run_learning_loop(
        watchlist_dir=args.watchlist_dir,
        history_path=args.history,
        output_dir=args.out,
        max_hold_bars=args.max_hold_bars,
        target_r=args.target_r,
        slippage_bps=args.slippage_bps,
        min_sample=args.min_sample,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
