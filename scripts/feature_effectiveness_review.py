from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qull_scanner.strategy_exports import build_feature_effectiveness_review, obsidian_vault_path, write_feature_effectiveness_note

REVIEW_DIR = ROOT / "exports" / "reviews"


def load_recent_effectiveness_reviews(review_dir: Path = REVIEW_DIR, lookback_weeks: int = 4, now: pd.Timestamp | None = None) -> pd.DataFrame:
    now = now or pd.Timestamp.utcnow()
    cutoff = now.tz_localize(None) - pd.Timedelta(weeks=lookback_weeks)
    frames: list[pd.DataFrame] = []
    for path in sorted(review_dir.glob("weekly_selection_effectiveness_*.csv")):
        try:
            date_part = path.stem.replace("weekly_selection_effectiveness_", "")
            as_of = pd.Timestamp(date_part)
        except Exception:
            as_of = pd.Timestamp(path.stat().st_mtime, unit="s")
        if as_of < cutoff:
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["Review As Of"] = as_of.date().isoformat()
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def run_feature_effectiveness_review(
    review_dir: Path = REVIEW_DIR,
    vault: Path | None = None,
    export_dir: Path | None = None,
    lookback_weeks: int = 4,
    min_weeks: int = 2,
) -> dict[str, object]:
    vault = vault or obsidian_vault_path()
    export_dir = export_dir or review_dir
    data = load_recent_effectiveness_reviews(review_dir, lookback_weeks=lookback_weeks)
    result = build_feature_effectiveness_review(data, min_weeks=min_weeks, lookback_weeks=lookback_weeks)
    paths = write_feature_effectiveness_note(result, vault=vault, export_dir=export_dir)
    return {
        "evaluated_signals": result["evaluated_signals"],
        "frameworks": len(result["framework_rows"]),
        "false_positives": len(result["false_positive_rows"]),
        "overextended": len(result["overextension_rows"]),
        "drawdown_flags": len(result["drawdown_rows"]),
        "proposal_policy": result["proposal_policy"],
        "production_change_allowed": result["production_change_allowed"],
        "markdown": str(paths["markdown"]),
        "framework_csv": str(paths["framework_csv"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build 2–4 week PAPER_ONLY feature effectiveness review.")
    parser.add_argument("--review-dir", type=Path, default=REVIEW_DIR)
    parser.add_argument("--vault", type=Path, default=obsidian_vault_path())
    parser.add_argument("--export-dir", type=Path, default=REVIEW_DIR)
    parser.add_argument("--lookback-weeks", type=int, default=4)
    parser.add_argument("--min-weeks", type=int, default=2)
    args = parser.parse_args()
    result = run_feature_effectiveness_review(args.review_dir, args.vault, args.export_dir, args.lookback_weeks, args.min_weeks)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
