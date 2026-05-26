from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qull_scanner.strategy_exports import (
    add_weekly_effectiveness,
    load_history,
    obsidian_vault_path,
    write_weekly_effectiveness_note,
)

DATA = ROOT / "data"
WATCHLIST_DIR = ROOT / "exports" / "watchlists"
REVIEW_DIR = ROOT / "exports" / "reviews"


def load_recent_snapshots(watchlist_dir: Path, days: int = 10) -> pd.DataFrame:
    files = sorted(watchlist_dir.glob("daily_shortlist_*.csv"))
    if not files:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    cutoff = pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta(days=days)
    for path in files:
        df = pd.read_csv(path)
        if df.empty:
            continue
        if "Signal Date" in df.columns:
            dates = pd.to_datetime(df["Signal Date"], errors="coerce")
            df = df[dates >= cutoff].copy()
        if not df.empty:
            df["Snapshot File"] = str(path)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    subset = [c for c in ["Ticker", "Signal Date"] if c in out.columns]
    return out.drop_duplicates(subset=subset, keep="first") if subset else out


def run_weekly_review(vault: Path, watchlist_dir: Path = WATCHLIST_DIR, review_dir: Path = REVIEW_DIR, days: int = 10) -> dict[str, str | int | float]:
    snapshots = load_recent_snapshots(watchlist_dir, days=days)
    history = load_history(DATA / "history_prices.parquet")
    metrics = pd.read_parquet(DATA / "scanner_metrics.parquet")
    metrics["Date"] = pd.to_datetime(metrics["Date"])
    review = add_weekly_effectiveness(snapshots, history, metrics, lookahead_days=10)
    as_of = pd.Timestamp.utcnow().date().isoformat()
    paths = write_weekly_effectiveness_note(review, vault=vault, export_dir=review_dir, as_of=as_of)
    breakouts = int(review["Breakout Verified"].sum()) if not review.empty and "Breakout Verified" in review.columns else 0
    return {
        "review_rows": int(len(review)),
        "breakouts": breakouts,
        "breakout_rate_pct": round(breakouts / len(review) * 100, 2) if len(review) else 0.0,
        "markdown": str(paths["markdown"]),
        "latest": str(paths["latest"]),
        "csv": str(paths["csv"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Weekly effectiveness review for Daily Strategy Lab watchlists.")
    parser.add_argument("--vault", type=Path, default=obsidian_vault_path())
    parser.add_argument("--watchlist-dir", type=Path, default=WATCHLIST_DIR)
    parser.add_argument("--review-dir", type=Path, default=REVIEW_DIR)
    parser.add_argument("--days", type=int, default=10)
    args = parser.parse_args()
    result = run_weekly_review(vault=args.vault, watchlist_dir=args.watchlist_dir, review_dir=args.review_dir, days=args.days)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
