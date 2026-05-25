from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

DATA = ROOT / "data" / "history_prices.parquet"
OUT = ROOT / "data" / "market_metadata.csv"
AUDIT = ROOT / "exports" / "market_metadata_audit.json"


def fetch_market_caps(tickers: list[str], limit: int | None = None, sleep_s: float = 0.05) -> pd.DataFrame:
    """Fetch current market-cap metadata via yfinance fast_info.

    This is current metadata, not historical point-in-time market cap. The output
    is source-traced and should be treated as a gate only when coverage is high.
    """
    import yfinance as yf

    rows = []
    selected = tickers[:limit] if limit else tickers
    for i, ticker in enumerate(selected, start=1):
        cap = None
        error = None
        try:
            info = yf.Ticker(ticker).fast_info
            cap = getattr(info, "market_cap", None) or info.get("market_cap")
        except Exception as exc:  # network/vendor failures are recorded, not hidden
            error = type(exc).__name__
        rows.append(
            {
                "Ticker": ticker,
                "Market Cap": cap,
                "Market Cap Source": "yfinance.fast_info.market_cap.current",
                "Metadata Timestamp UTC": pd.Timestamp.utcnow().isoformat(),
                "Metadata Error": error,
            }
        )
        if i % 100 == 0:
            print(f"fetched={i}/{len(selected)}")
        time.sleep(sleep_s)
    return pd.DataFrame(rows)


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    raw = pd.read_parquet(DATA, columns=["Ticker"])
    tickers = sorted(raw["Ticker"].dropna().astype(str).unique())
    meta = fetch_market_caps(tickers, limit=limit)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(meta.to_csv(index=False))
    AUDIT.parent.mkdir(exist_ok=True)
    audit = {
        "source": "yfinance.fast_info.market_cap.current",
        "rows": int(len(meta)),
        "coverage_non_null_pct": round(float(meta["Market Cap"].notna().mean() * 100), 2) if len(meta) else 0.0,
        "output": str(OUT),
        "warning": "Current market cap, not historical point-in-time. Use only when coverage is high; otherwise report N/D.",
    }
    AUDIT.write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
