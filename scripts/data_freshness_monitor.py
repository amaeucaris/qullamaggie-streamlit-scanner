from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qull_scanner.strategy_exports import (
    build_stale_data_alert,
    compute_data_freshness,
    latest_market_frame,
    load_json,
    obsidian_vault_path,
    write_stale_data_alert_note,
)

DATA = ROOT / "data"
STATE_PATH = ROOT / "exports" / "data_freshness_state.json"


def expected_last_market_date(now: pd.Timestamp | None = None) -> str:
    now = now or pd.Timestamp.utcnow()
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    # Morning Europe job should expect the previous US trading day. Use the US
    # federal holiday calendar as a conservative proxy for NYSE full-day closures.
    candidate = now.normalize() - pd.Timedelta(days=1)
    calendar = USFederalHolidayCalendar()
    holidays = set(calendar.holidays(start=candidate - pd.Timedelta(days=10), end=now + pd.Timedelta(days=1)).date)
    while candidate.weekday() >= 5 or candidate.date() in holidays:
        candidate -= pd.Timedelta(days=1)
    return candidate.date().isoformat()


def load_state(path: Path = STATE_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_state(state: dict[str, Any], path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True))


def monitor_data_freshness(
    metadata_path: Path = DATA / "metadata.json",
    metrics_path: Path = DATA / "scanner_metrics.parquet",
    state_path: Path = STATE_PATH,
    vault: Path | None = None,
    now: pd.Timestamp | None = None,
) -> dict[str, Any]:
    metadata = load_json(metadata_path)
    metrics = pd.read_parquet(metrics_path)
    latest = latest_market_frame(metrics)
    last_market_date = str(pd.Timestamp(latest["Date"].max()).date()) if not latest.empty and "Date" in latest.columns else "N/D"
    freshness = compute_data_freshness(metadata.get("updated_at"), last_market_date, now=now)
    state = load_state(state_path)
    expected = expected_last_market_date(now)
    alert = build_stale_data_alert(freshness, previous_last_market_date=state.get("last_market_date"), expected_last_market_date=expected)
    vault = vault or obsidian_vault_path()
    paths: dict[str, str] = {}
    if alert["alert"] or alert["resolution_status"] == "RESOLVED":
        note_paths = write_stale_data_alert_note(alert, vault=vault, as_of=pd.Timestamp(now or pd.Timestamp.utcnow()).date().isoformat())
        paths = {key: str(value) for key, value in note_paths.items()}
    save_state(
        {
            "last_market_date": freshness.last_market_date,
            "last_update": freshness.last_update,
            "status": freshness.status,
            "expected_last_market_date": expected,
            "resolution_status": alert["resolution_status"],
            "checked_at_utc": pd.Timestamp.utcnow().isoformat(),
        },
        state_path,
    )
    return {**alert, "paths": paths}


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor scanner data freshness and emit alert/resolution notes.")
    parser.add_argument("--metadata", type=Path, default=DATA / "metadata.json")
    parser.add_argument("--metrics", type=Path, default=DATA / "scanner_metrics.parquet")
    parser.add_argument("--state", type=Path, default=STATE_PATH)
    parser.add_argument("--vault", type=Path, default=obsidian_vault_path())
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    result = monitor_data_freshness(args.metadata, args.metrics, args.state, args.vault)
    if result["alert"] or result["resolution_status"] == "RESOLVED" or args.verbose:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
