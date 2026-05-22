from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from app import DATA_DIR, SUGAR_BABIES_FILE, SUGAR_BABIES_METADATA_FILE, download_price_history, load_symbols
from qull_scanner.sugar_babies import SUGAR_BABIES_PERIODS, build_sugar_babies_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update Stockbee Sugar Babies aggregate watchlist data.")
    parser.add_argument("--universe", choices=["NASDAQ", "All US listed"], default="All US listed")
    parser.add_argument("--include-etfs", action="store_true")
    parser.add_argument("--period", default="6y")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--pause-seconds", type=float, default=2.0)
    parser.add_argument("--max-tickers", type=int, default=0, help="0 means all tickers.")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--top-n", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = load_symbols(args.universe, args.include_etfs)
    tickers = symbols["yahoo_symbol"].tolist()
    if args.max_tickers > 0:
        tickers = tickers[: args.max_tickers]

    history = {}
    remaining = tuple(tickers)
    for attempt in range(args.retries + 1):
        if not remaining:
            break

        batch_history = download_price_history(
            tickers=remaining,
            period=args.period,
            interval=args.interval,
            chunk_size=args.chunk_size,
            pause_seconds=args.pause_seconds,
        )
        history.update(batch_history)
        remaining = tuple(ticker for ticker in remaining if ticker not in history)

        if remaining and attempt < args.retries:
            sleep_seconds = args.pause_seconds * 5 * (attempt + 1)
            print(f"Retry {attempt + 1}: {len(remaining)} missing tickers. Sleeping {sleep_seconds:.1f}s.")
            time.sleep(sleep_seconds)

    sugar_babies = build_sugar_babies_metrics(
        history,
        periods=SUGAR_BABIES_PERIODS,
        top_n=args.top_n,
    )
    if sugar_babies.empty:
        raise ValueError("No Sugar Babies rows generated.")

    updated_at = datetime.now(timezone.utc).isoformat()
    sugar_babies["Updated At"] = updated_at
    DATA_DIR.mkdir(exist_ok=True)
    sugar_babies.to_parquet(SUGAR_BABIES_FILE, index=False)

    metadata = {
        "updated_at": updated_at,
        "universe": args.universe,
        "include_etfs": args.include_etfs,
        "period": args.period,
        "interval": args.interval,
        "requested_tickers": len(tickers),
        "tickers_with_data": len(history),
        "missing_tickers": len(remaining),
        "sugar_babies_file": str(Path(SUGAR_BABIES_FILE).as_posix()),
        "sugar_babies_rows": len(sugar_babies),
        "sugar_babies_periods": SUGAR_BABIES_PERIODS,
        "sugar_babies_top_n": args.top_n,
        "formula": "CountTrue(c/c1>=1.04 and v>v1 and v>=8900000, N)",
    }
    SUGAR_BABIES_METADATA_FILE.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
