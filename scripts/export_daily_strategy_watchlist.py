from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qull_scanner.strategy_exports import (
    build_daily_shortlist,
    compute_data_freshness,
    load_json,
    obsidian_vault_path,
    scanner_outputs,
    write_daily_watchlist_note,
)

DATA = ROOT / "data"
EXPORT_DIR = ROOT / "exports" / "watchlists"


def maybe_git_pull(enabled: bool) -> None:
    if not enabled:
        return
    subprocess.run(["git", "pull", "--ff-only", "origin", "main"], cwd=ROOT, check=True, timeout=120)


def run_daily_export(vault: Path, export_dir: Path = EXPORT_DIR, git_pull: bool = False) -> dict[str, str | int]:
    maybe_git_pull(git_pull)
    metrics = pd.read_parquet(DATA / "scanner_metrics.parquet")
    metrics["Date"] = pd.to_datetime(metrics["Date"])
    sugar_path = DATA / "sugar_babies.parquet"
    sugar = pd.read_parquet(sugar_path) if sugar_path.exists() else pd.DataFrame()
    metadata = load_json(DATA / "metadata.json")
    last_market_date = metrics["Date"].max()
    freshness = compute_data_freshness(metadata.get("updated_at"), last_market_date)
    outputs = scanner_outputs(metrics, sugar)
    shortlist = build_daily_shortlist(outputs, limit=10)
    paths = write_daily_watchlist_note(shortlist, outputs, freshness, vault=vault, export_dir=export_dir, as_of=freshness.last_market_date)
    steve_algo_csv = export_dir / f"steve_algo_watchlist_{freshness.last_market_date}.csv"
    outputs.get("steve_algo", pd.DataFrame()).to_csv(steve_algo_csv, index=False)
    return {
        "status": freshness.status,
        "last_market_date": freshness.last_market_date,
        "last_update": freshness.last_update,
        "shortlist_rows": int(len(shortlist)),
        "steve_algo_rows": int(len(outputs.get("steve_algo", pd.DataFrame()))),
        "markdown": str(paths["markdown"]),
        "latest": str(paths["latest"]),
        "csv": str(paths["csv"]),
        "steve_algo_csv": str(steve_algo_csv),
        "scanner_counts": str(paths["scanner_counts"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Daily Strategy Lab watchlist to Obsidian and CSV snapshots.")
    parser.add_argument("--vault", type=Path, default=obsidian_vault_path())
    parser.add_argument("--export-dir", type=Path, default=EXPORT_DIR)
    parser.add_argument("--git-pull", action="store_true", help="Pull origin/main before reading data.")
    args = parser.parse_args()
    result = run_daily_export(vault=args.vault, export_dir=args.export_dir, git_pull=args.git_pull)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
