from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from qull_scanner.filters import (
    ScannerThresholds,
    apply_guru_filter,
    apply_minervini_filter,
    apply_qullamaggie_filter,
    apply_steve_style_qullamaggie_filter,
    apply_stockbee_filter,
)
from qull_scanner.steve_algo import SteveAlgoThresholds, apply_steve_algo_watchlists, classify_steve_algo_row

DEFAULT_SCANNER_THRESHOLDS = ScannerThresholds(
    min_price=5,
    min_avg_volume=1_000_000,
    top_percent=2,
    min_breakout_pct=4,
    stockbee_min_price=5,
    stockbee_min_volume=9_000_000,
    min_dollar_volume=50_000_000,
    min_adr_pct=3.5,
    max_extension_atr=5,
)
DEFAULT_STEVE_THRESHOLDS = SteveAlgoThresholds(
    min_market_cap=0,
    min_dollar_volume=50_000_000,
    min_price=5,
    min_rs=85,
    min_trend_strength=80,
    min_reward_risk=3,
)

WATCHLIST_OBSIDIAN_DIR = Path("finance/strategy-lab/watchlists")
REVIEW_OBSIDIAN_DIR = Path("finance/strategy-lab/reviews")
QUERY_OBSIDIAN_DIR = Path("queries")


@dataclass(frozen=True)
class DataFreshness:
    status: str
    last_market_date: str
    last_update: str
    age_days: int | None
    message: str


def obsidian_vault_path(default: str | Path | None = None) -> Path:
    value = os.environ.get("OBSIDIAN_VAULT_PATH") or default or str(Path.home() / ".hermes" / "antonio-kb")
    return Path(value).expanduser()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def latest_market_frame(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty or "Date" not in metrics.columns:
        return metrics.copy()
    out = metrics.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    return out[out["Date"] == out["Date"].max()].copy()


def compute_data_freshness(updated_at: Any, last_market_date: Any, now: pd.Timestamp | None = None) -> DataFreshness:
    now = now or pd.Timestamp.utcnow()
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    try:
        last_market_day = pd.Timestamp(last_market_date).date()
    except Exception:
        return DataFreshness(
            status="STALE",
            last_market_date="N/D",
            last_update=format_timestamp(updated_at),
            age_days=None,
            message="⚠️ DATA STALE — data mercato non disponibile; usare solo per analisi storica, non per watchlist operativa.",
        )
    age_days = (now.date() - last_market_day).days
    status = "STALE" if age_days > 2 else "FRESH"
    message = (
        "⚠️ DATA STALE — usare solo per analisi storica, non per watchlist operativa."
        if status == "STALE"
        else "Dati abbastanza recenti per review operativa; resta obbligatoria conferma su chart."
    )
    return DataFreshness(status=status, last_market_date=str(last_market_day), last_update=format_timestamp(updated_at), age_days=age_days, message=message)


def format_timestamp(value: Any) -> str:
    if value is None or value == "":
        return "N/D"
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return str(value)


def scanner_outputs(metrics: pd.DataFrame, sugar_babies: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    latest = latest_market_frame(metrics)
    sugar = sugar_babies.copy() if sugar_babies is not None else pd.DataFrame()
    if not sugar.empty and "Date" in sugar.columns:
        sugar["Date"] = pd.to_datetime(sugar["Date"])
        sugar = sugar[sugar["Date"] == sugar["Date"].max()].copy()
    q = apply_qullamaggie_filter(latest, DEFAULT_SCANNER_THRESHOLDS)
    steve_kq = apply_steve_style_qullamaggie_filter(latest, DEFAULT_SCANNER_THRESHOLDS)
    stockbee = apply_stockbee_filter(latest, DEFAULT_SCANNER_THRESHOLDS)
    minervini = apply_minervini_filter(latest)
    guru = apply_guru_filter(q, minervini)
    steve = apply_steve_algo_watchlists(latest, DEFAULT_STEVE_THRESHOLDS)
    return {
        "latest": latest,
        "qullamaggie_strict": q,
        "steve_style_kq": steve_kq,
        "stockbee_4pct": stockbee,
        "minervini": minervini,
        "guru_q_minervini": guru,
        "sugar_babies": sugar,
        "steve_algo": steve,
    }


def _ticker_set(df: pd.DataFrame) -> set[str]:
    if df.empty or "Ticker" not in df.columns:
        return set()
    return set(df["Ticker"].astype(str).str.upper())


def build_daily_shortlist(outputs: dict[str, pd.DataFrame], limit: int = 10) -> pd.DataFrame:
    latest = outputs.get("latest", pd.DataFrame())
    if latest.empty or "Ticker" not in latest.columns:
        return pd.DataFrame()
    q = _ticker_set(outputs.get("qullamaggie_strict", pd.DataFrame()))
    kq = _ticker_set(outputs.get("steve_style_kq", pd.DataFrame()))
    stockbee = _ticker_set(outputs.get("stockbee_4pct", pd.DataFrame()))
    sugar = _ticker_set(outputs.get("sugar_babies", pd.DataFrame()))
    minervini = _ticker_set(outputs.get("minervini", pd.DataFrame()))
    guru = _ticker_set(outputs.get("guru_q_minervini", pd.DataFrame()))
    steve = outputs.get("steve_algo", pd.DataFrame())
    steve_by_ticker = {str(r["Ticker"]).upper(): r for r in steve.to_dict("records")} if not steve.empty and "Ticker" in steve.columns else {}
    candidates = q | kq | stockbee | sugar | minervini | guru | set(steve_by_ticker)
    rows: list[dict[str, Any]] = []
    latest_by_ticker = {str(r["Ticker"]).upper(): r for r in latest.to_dict("records")}
    for ticker in candidates:
        row = latest_by_ticker.get(ticker, {})
        frameworks: list[str] = []
        score = 0
        if ticker in q:
            frameworks.append("Qullamaggie Strict")
            score += 3
        if ticker in guru:
            frameworks.append("Guru Q x Minervini")
            score += 3
        if ticker in stockbee:
            frameworks.append("Stockbee 4%")
            score += 2
        if ticker in sugar:
            frameworks.append("Sugar Babies")
            score += 2
        if ticker in minervini:
            frameworks.append("Minervini")
            score += 1
        if ticker in kq:
            frameworks.append("Steve-style KQ")
            score += 1
        steve_row = steve_by_ticker.get(ticker, {})
        bucket = steve_row.get("SteveAlgo Primary Bucket", "")
        if bucket in {"Entry", "White Up", "Yellow"}:
            frameworks.append(f"SteveAlgo {bucket}")
            score += {"Entry": 4, "White Up": 3, "Yellow": 2}.get(str(bucket), 0)
        rr = _safe_float(steve_row.get("Reward-Risk", row.get("Reward-Risk")))
        if math.isfinite(rr) and rr >= 3:
            score += 1
        readiness = "CHART REVIEW"
        if score >= 7 and (bucket in {"Entry", "White Up"} or ticker in stockbee):
            readiness = "PRIORITY REVIEW"
        elif score <= 2:
            readiness = "MONITOR"
        rows.append(
            {
                "Ticker": ticker,
                "Score": score,
                "Trade Readiness": readiness,
                "Frameworks": ", ".join(frameworks) if frameworks else "N/D",
                "Price": row.get("Price"),
                "Daily Return %": row.get("Daily Return %"),
                "Momentum Rank": row.get("Momentum Rank"),
                "ATR Extension SMA50": row.get("ATR Extension SMA50"),
                "Reward-Risk": rr if math.isfinite(rr) else None,
                "SteveAlgo Bucket": bucket or "N/D",
                "Reason": steve_row.get("SteveAlgo Reason") or row.get("Steve-style KQ Reason") or "multi-framework candidate",
                "Capital Authorized": "0%",
                "Signal Date": str(pd.Timestamp(row.get("Date")).date()) if row.get("Date") is not None else "N/D",
                "Breakout Level": row.get("Breakout Level"),
                "Darvas Upper": row.get("Darvas Upper"),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["Score", "Momentum Rank", "Daily Return %"], ascending=[False, False, False]).head(limit).reset_index(drop=True)


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def write_daily_watchlist_note(
    shortlist: pd.DataFrame,
    outputs: dict[str, pd.DataFrame],
    freshness: DataFreshness,
    vault: Path,
    export_dir: Path,
    as_of: str | None = None,
) -> dict[str, Path]:
    as_of = as_of or freshness.last_market_date or pd.Timestamp.utcnow().date().isoformat()
    obsidian_dir = vault / WATCHLIST_OBSIDIAN_DIR
    query_dir = vault / QUERY_OBSIDIAN_DIR
    obsidian_dir.mkdir(parents=True, exist_ok=True)
    query_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)
    md_path = obsidian_dir / f"daily-shortlist-{as_of}.md"
    latest_path = query_dir / "latest-strategy-lab-watchlist.md"
    csv_path = export_dir / f"daily_shortlist_{as_of}.csv"
    scanner_csv = export_dir / f"scanner_counts_{as_of}.csv"
    shortlist.to_csv(csv_path, index=False)
    counts = pd.DataFrame(
        [
            {"Scanner": name, "Count": len(df)}
            for name, df in outputs.items()
            if name != "latest"
        ]
    )
    counts.to_csv(scanner_csv, index=False)
    lines = [
        f"# Daily Strategy Lab Watchlist — {as_of}",
        "",
        "Research only. Scanner output = chart review, non trade. Capital authorized: 0%.",
        "",
        "## Data status",
        f"- Data status: **{freshness.status}**",
        f"- Last market date: `{freshness.last_market_date}`",
        f"- Last update: `{freshness.last_update}`",
        f"- Warning: {freshness.message}",
        "",
        "## Scanner counts",
    ]
    for _, row in counts.iterrows():
        lines.append(f"- {row['Scanner']}: {int(row['Count'])}")
    lines += ["", "## Daily Shortlist — Top 10 Review Today"]
    if shortlist.empty:
        lines.append("- Nessun candidato con i filtri correnti.")
    else:
        for row in shortlist.to_dict("records"):
            lines.append(
                f"- **{row['Ticker']}** — {row['Trade Readiness']} | score {row['Score']} | "
                f"price {_fmt(row.get('Price'))} | daily {_fmt(row.get('Daily Return %'))}% | "
                f"momentum {_fmt(row.get('Momentum Rank'))} | frameworks: {row.get('Frameworks', 'N/D')} | "
                f"R/R {_fmt(row.get('Reward-Risk'))} | {row.get('Reason', '')}"
            )
    lines += [
        "",
        "## Files",
        f"- CSV shortlist: `{csv_path}`",
        f"- Scanner counts: `{scanner_csv}`",
        "",
        "## Operating rules",
        "- No automatic trade.",
        "- Check chart manually before action.",
        "- Portfolio Risk Gate + Antonio approval required before any capital > 0%.",
    ]
    md_path.write_text("\n".join(lines) + "\n")
    latest_path.write_text(f"# Latest Strategy Lab Watchlist\n\n![[{md_path.relative_to(vault)}]]\n")
    return {"markdown": md_path, "latest": latest_path, "csv": csv_path, "scanner_counts": scanner_csv}


def _fmt(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return "N/D"
        return f"{float(value):.2f}"
    except Exception:
        return str(value)


def load_history(history_path: Path) -> pd.DataFrame:
    history = pd.read_parquet(history_path)
    history["Date"] = pd.to_datetime(history["Date"])
    return history.sort_values(["Ticker", "Date"])


def classify_current_phase(row: pd.Series) -> str:
    close = _safe_float(row.get("Close"))
    ema10 = _safe_float(row.get("EMA10"))
    ema20 = _safe_float(row.get("EMA20"))
    ema50 = _safe_float(row.get("EMA50"))
    if math.isfinite(close) and math.isfinite(ema10) and close >= ema10:
        return "Phase 2 advance / above EMA10"
    if math.isfinite(close) and math.isfinite(ema20) and close >= ema20:
        return "Pullback orderly / above EMA20"
    if math.isfinite(close) and math.isfinite(ema50) and close >= ema50:
        return "Deep pullback / above EMA50"
    return "Broken / below EMA50 or insufficient MA data"


def add_weekly_effectiveness(
    snapshots: pd.DataFrame,
    history: pd.DataFrame,
    current_metrics: pd.DataFrame,
    lookahead_days: int = 10,
) -> pd.DataFrame:
    if snapshots.empty:
        return pd.DataFrame()
    hist = history.copy()
    current = latest_market_frame(current_metrics)
    current_by_ticker = {str(r["Ticker"]).upper(): r for r in current.to_dict("records")}
    rows: list[dict[str, Any]] = []
    for snap in snapshots.to_dict("records"):
        ticker = str(snap.get("Ticker", "")).upper()
        signal_date = pd.Timestamp(snap.get("Signal Date", snap.get("Date", pd.NaT)))
        if not ticker or pd.isna(signal_date):
            continue
        ticker_history = hist[hist["Ticker"].astype(str).str.upper() == ticker].copy()
        if ticker_history.empty:
            continue
        signal_rows = ticker_history[ticker_history["Date"] <= signal_date]
        signal_row = signal_rows.iloc[-1] if not signal_rows.empty else ticker_history.iloc[0]
        future = ticker_history[ticker_history["Date"] > signal_date].head(lookahead_days).copy()
        signal_close = _safe_float(signal_row.get("Close", snap.get("Price")))
        current_row = current_by_ticker.get(ticker, {})
        breakout_level = _safe_float(snap.get("Breakout Level", snap.get("Darvas Upper")))
        if not math.isfinite(breakout_level):
            breakout_level = _safe_float(current_row.get("Breakout Level", current_row.get("Darvas Upper")))
        if future.empty:
            max_return = math.nan
            drawdown = math.nan
            breakout = False
            last = signal_row
            evaluation_status = "PENDING — no post-signal bars yet"
        else:
            max_high = _safe_float(future["High"].max())
            min_low = _safe_float(future["Low"].min())
            last = future.iloc[-1]
            breakout = bool(math.isfinite(breakout_level) and (future["High"] > breakout_level).any())
            max_return = ((max_high / signal_close) - 1) * 100 if math.isfinite(signal_close) and signal_close else math.nan
            drawdown = ((min_low / signal_close) - 1) * 100 if math.isfinite(signal_close) and signal_close else math.nan
            evaluation_status = "EVALUATED"
        phase_source = pd.Series({**current_row, **last.to_dict()})
        rows.append(
            {
                "Ticker": ticker,
                "Signal Date": str(signal_date.date()),
                "Score": snap.get("Score"),
                "Trade Readiness": snap.get("Trade Readiness"),
                "Frameworks": snap.get("Frameworks"),
                "Signal Close": signal_close,
                "Last Close": _safe_float(last.get("Close")),
                "Max Return %": round(max_return, 2) if math.isfinite(max_return) else None,
                "Max Drawdown %": round(drawdown, 2) if math.isfinite(drawdown) else None,
                "Breakout Verified": breakout,
                "Breakout Level": breakout_level if math.isfinite(breakout_level) else None,
                "Current Phase": classify_current_phase(phase_source),
                "Evaluation Status": evaluation_status,
                "Capital Authorized": "0%",
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["Breakout Verified", "Max Return %"], ascending=[False, False]).reset_index(drop=True)


def write_weekly_effectiveness_note(review: pd.DataFrame, vault: Path, export_dir: Path, as_of: str | None = None) -> dict[str, Path]:
    as_of = as_of or pd.Timestamp.utcnow().date().isoformat()
    obsidian_dir = vault / REVIEW_OBSIDIAN_DIR
    query_dir = vault / QUERY_OBSIDIAN_DIR
    obsidian_dir.mkdir(parents=True, exist_ok=True)
    query_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)
    md_path = obsidian_dir / f"weekly-selection-effectiveness-{as_of}.md"
    latest_path = query_dir / "latest-selection-effectiveness.md"
    csv_path = export_dir / f"weekly_selection_effectiveness_{as_of}.csv"
    review.to_csv(csv_path, index=False)
    total = len(review)
    breakouts = int(review["Breakout Verified"].sum()) if total and "Breakout Verified" in review.columns else 0
    avg_max = float(review["Max Return %"].dropna().mean()) if total and "Max Return %" in review.columns and not review["Max Return %"].dropna().empty else math.nan
    lines = [
        f"# Weekly Selection Effectiveness — {as_of}",
        "",
        "Research only. This is process feedback, not trade authorization. Capital authorized: 0%.",
        "",
        "## Summary",
        f"- Candidates reviewed: {total}",
        f"- Breakout verified: {breakouts}",
        f"- Breakout rate: {(breakouts / total * 100):.1f}%" if total else "- Breakout rate: N/D",
        f"- Avg max return: {_fmt(avg_max)}%",
        "",
        "## Candidates",
    ]
    if review.empty:
        lines.append("- Nessun snapshot daily disponibile per la settimana.")
    else:
        for row in review.to_dict("records"):
            lines.append(
                f"- **{row['Ticker']}** — status: {row.get('Evaluation Status', 'N/D')} | breakout: {row['Breakout Verified']} | phase: {row['Current Phase']} | "
                f"max return {_fmt(row.get('Max Return %'))}% | drawdown {_fmt(row.get('Max Drawdown %'))}% | "
                f"signal {row['Signal Date']} | frameworks: {row.get('Frameworks', 'N/D')}"
            )
    lines += ["", "## File", f"- CSV review: `{csv_path}`"]
    md_path.write_text("\n".join(lines) + "\n")
    latest_path.write_text(f"# Latest Selection Effectiveness\n\n![[{md_path.relative_to(vault)}]]\n")
    return {"markdown": md_path, "latest": latest_path, "csv": csv_path}
