from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from qull_scanner.backtest_steve_algo import (
    SteveBacktestConfig,
    apply_market_metadata,
    apply_regime_filter,
    build_config_random_benchmark,
    compare_with_qullamaggie_candidates,
    export_daily_watchlist,
    load_market_metadata,
    simulate_capital_curve,
    simulate_steve_algo_trades,
    summarize_in_sample_out_of_sample,
    summarize_steve_algo_backtest,
)
from qull_scanner.filters import ScannerThresholds, apply_qullamaggie_filter
from qull_scanner.metrics import daily_close_range, darvas_levels, rolling_dollar_volume
from qull_scanner.steve_algo import SteveAlgoThresholds, apply_steve_algo_watchlists

DATA = ROOT / "data" / "history_prices.parquet"
MARKET_METADATA = ROOT / "data" / "market_metadata.csv"
OUT = ROOT / "exports" / "steve_algo_backtest_summary.json"
TRADES = ROOT / "exports" / "steve_algo_backtest_trades.csv"
EVENTS = ROOT / "exports" / "steve_algo_backtest_events.csv"
MATRIX = ROOT / "exports" / "steve_algo_retest_matrix.csv"
REPORT = ROOT / "exports" / "steve_algo_backtest_report.md"
WATCHLIST_DIR = ROOT / "exports" / "watchlists"
OBSIDIAN_WATCHLIST_DIR = Path.home() / ".hermes" / "antonio-kb" / "finance" / "strategy-lab" / "watchlists"


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()


def build_events(history: dict[str, pd.DataFrame], thresholds: SteveAlgoThresholds, metadata: pd.DataFrame | None = None):
    rows = []
    enriched = {}
    for ticker, df0 in history.items():
        df = df0[["Open", "High", "Low", "Close", "Volume"]].copy().dropna(subset=["Close"])
        if len(df) < 220:
            continue
        df["EMA10"] = df["Close"].ewm(span=10, min_periods=10, adjust=False).mean()
        df["EMA20"] = df["Close"].ewm(span=20, min_periods=20, adjust=False).mean()
        df["EMA50"] = df["Close"].ewm(span=50, min_periods=50, adjust=False).mean()
        df["SMA50"] = sma(df["Close"], 50)
        df["SMA200"] = sma(df["Close"], 200)
        df["ATR20"] = atr(df["High"], df["Low"], df["Close"], 20)
        df["DCR %"] = daily_close_range(df["High"], df["Low"], df["Close"])
        df["Daily $ Volume 20D"] = rolling_dollar_volume(df["Close"], df["Volume"], 20)
        df["Avg Volume 20D"] = df["Volume"].rolling(20, min_periods=20).mean()
        df["ADR 20D %"] = ((df["High"] / df["Low"] - 1) * 100).rolling(20, min_periods=20).mean()
        df["Return 1M %"] = df["Close"].pct_change(21) * 100
        df["Return 3M %"] = df["Close"].pct_change(63) * 100
        df["Return 6M %"] = df["Close"].pct_change(126) * 100
        df["Darvas Upper"], df["Darvas Lower"] = darvas_levels(df["High"], df["Low"], 20)
        df["ATR Extension EMA10"] = (df["Close"] - df["EMA10"]) / df["ATR20"]
        df["ATR Extension EMA20"] = (df["Close"] - df["EMA20"]) / df["ATR20"]
        df["ATR Extension SMA50"] = (df["Close"] - df["SMA50"]) / df["ATR20"]
        df["EMA10 Rising"] = df["EMA10"] > df["EMA10"].shift(1)
        df["Breakout Above Lookback High"] = df["Close"] > df["Close"].shift(1).rolling(20).max()
        df["Price"] = df["Close"]
        df["Daily Return %"] = df["Close"].pct_change() * 100
        df["Price > SMA10"] = df["Close"] > df["Close"].rolling(10, min_periods=10).mean()
        df["Price > SMA20"] = df["Close"] > df["Close"].rolling(20, min_periods=20).mean()
        enriched[ticker] = df
        tmp = df.reset_index(names="Date")
        tmp["Ticker"] = ticker
        rows.append(tmp)
    panel = pd.concat(rows, ignore_index=True)
    panel = panel.dropna(subset=["Return 1M %", "Return 3M %", "Return 6M %", "EMA10", "EMA20", "EMA50", "SMA50", "SMA200", "ATR20"])
    for col in ["Return 1M %", "Return 3M %", "Return 6M %"]:
        panel[f"{col} Rank"] = panel.groupby("Date")[col].rank(pct=True, ascending=True) * 100
    panel["Momentum Rank"] = panel[["Return 1M % Rank", "Return 3M % Rank", "Return 6M % Rank"]].mean(axis=1)
    panel = apply_market_metadata(panel, metadata if metadata is not None else pd.DataFrame())
    events = apply_steve_algo_watchlists(panel, thresholds)
    return events, enriched, panel


def build_regime_frame(history: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    source = "local"
    regime_history = dict(history)
    if "SPY" not in regime_history or "QQQ" not in regime_history:
        try:
            import yfinance as yf

            start = min(df.index.min() for df in history.values()).date().isoformat()
            end = (max(df.index.max() for df in history.values()) + pd.Timedelta(days=1)).date().isoformat()
            downloaded = yf.download(["SPY", "QQQ"], start=start, end=end, progress=False, auto_adjust=True, group_by="ticker", threads=True)
            for symbol in ["SPY", "QQQ"]:
                if symbol in regime_history:
                    continue
                if isinstance(downloaded.columns, pd.MultiIndex) and symbol in downloaded.columns.get_level_values(0):
                    part = downloaded[symbol].dropna(subset=["Close"])
                else:
                    part = downloaded.dropna(subset=["Close"])
                if not part.empty:
                    regime_history[symbol] = part[["Open", "High", "Low", "Close", "Volume"]].copy()
                    source = "yfinance"
        except Exception:
            source = "N/D"
    for symbol, col in [("SPY", "SPY Regime OK"), ("QQQ", "QQQ Regime OK")]:
        if symbol not in regime_history:
            continue
        df = regime_history[symbol].copy().sort_index()
        ok = df["Close"] > df["Close"].rolling(200, min_periods=200).mean()
        frame = ok.rename(col).reset_index()
        frame = frame.rename(columns={frame.columns[0]: "Date"})
        frame[f"{symbol} Regime Source"] = source
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["Date", "SPY Regime OK", "QQQ Regime OK"])
    regime = frames[0]
    for frame in frames[1:]:
        regime = regime.merge(frame, on="Date", how="outer")
    for col in ["SPY Regime OK", "QQQ Regime OK"]:
        if col not in regime.columns:
            regime[col] = False
    regime["Regime Source"] = source
    return regime.sort_values("Date")


def build_qullamaggie_candidates(panel: pd.DataFrame) -> pd.DataFrame:
    thresholds = ScannerThresholds(
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
    latest_by_date = []
    for _, day in panel.groupby("Date", sort=True):
        latest_by_date.append(apply_qullamaggie_filter(day, thresholds))
    return pd.concat(latest_by_date, ignore_index=True) if latest_by_date else pd.DataFrame()


def select_events(events: pd.DataFrame, buckets: list[str], max_per_day: int) -> pd.DataFrame:
    selected = events[events["SteveAlgo Primary Bucket"].isin(buckets)].sort_values(
        ["Date", "Momentum Rank"], ascending=[True, False]
    )
    return selected.groupby("Date").head(max_per_day).copy()


def compact_summary(label: str, trades: pd.DataFrame) -> dict[str, object]:
    summary = summarize_steve_algo_backtest(trades)
    return {
        "label": label,
        "trades": summary["trades"],
        "expectancy_r": summary["expectancy_r"],
        "win_rate": summary["win_rate"],
        "profit_factor": summary["profit_factor"],
        "max_drawdown_r": summary["max_drawdown_r"],
    }


def write_report(summary: dict[str, object], matrix: pd.DataFrame) -> None:
    best = matrix.sort_values(["edge_vs_random_r", "expectancy_r", "profit_factor"], ascending=[False, False, False]).head(10)
    cap = summary.get("capital_10k", {})
    cap_1 = cap.get("risk_1pct", {}) if isinstance(cap, dict) else {}
    lines = [
        "# SteveAlgo Watchlist Backtest Report",
        "",
        "Research only. Capitale autorizzato: 0% fino a validazione esterna/OOS più lunga.",
        "",
        f"- Selected config: {summary.get('selected_config')}",
        f"- Dataset: {summary['date_min']} → {summary['date_max']}",
        f"- Ticker con storico: {summary['tickers_with_history']}",
        f"- Eventi selezionati: {summary['events']}",
        f"- Trade: {summary['trades']}",
        f"- Expectancy R: {summary['expectancy_r']}",
        f"- Profit factor: {summary['profit_factor']}",
        f"- Max DD R: {summary['max_drawdown_r']}",
        f"- 10k capital result at 1% risk/trade: {cap_1.get('final_capital', 'N/D')} EUR ({cap_1.get('total_return_pct', 'N/D')}%), max DD {cap_1.get('max_drawdown_pct', 'N/D')}%",
        "",
        "## Market cap metadata",
        json.dumps(summary.get("market_cap_metadata", {}), indent=2),
        "",
        "## Regime filter",
        json.dumps(summary.get("regime_filter", {}), indent=2),
        "",
        "## Qullamaggie comparison",
        json.dumps(summary.get("qullamaggie_comparison", {}), indent=2),
        "",
        "## Bucket expectancy",
        json.dumps(summary.get("bucket_expectancy_r", {}), indent=2),
        "",
        "## OOS split",
        json.dumps(summary.get("oos_split", {}), indent=2),
        "",
        "## Random benchmark",
        json.dumps(summary.get("random_benchmark", {}), indent=2),
        "",
        "## Capital 10k sensitivity",
        json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "curve"} for k, v in cap.items() if isinstance(v, dict)}, indent=2),
        "",
        "## Top retest configs ranked by edge vs random",
        best.to_string(index=False),
        "",
        "## Hard conclusion",
        "Selected config is positive and beats same-date random on expectancy, but OOS edge is weak. It remains research/paper-trading only.",
    ]
    REPORT.write_text("\n".join(lines))


def main():
    raw = pd.read_parquet(DATA)
    raw["Date"] = pd.to_datetime(raw["Date"])
    history = {ticker: g.drop(columns=["Ticker"]).set_index("Date").sort_index() for ticker, g in raw.groupby("Ticker")}

    metadata = load_market_metadata(MARKET_METADATA)
    market_cap_gate = 1_000_000_000 if not metadata.empty and metadata["Market Cap"].notna().any() else 0
    thresholds = SteveAlgoThresholds(
        min_market_cap=market_cap_gate,
        min_dollar_volume=50_000_000,
        min_price=5,
        min_rs=85,
        min_trend_strength=80,
        min_reward_risk=3,
    )
    all_events, enriched, panel = build_events(history, thresholds, metadata)
    regime = build_regime_frame(history)
    regime_available = not regime.empty and regime[["SPY Regime OK", "QQQ Regime OK"]].any().any()
    filtered_events = apply_regime_filter(all_events, regime) if regime_available else all_events.copy()

    qull = build_qullamaggie_candidates(panel)
    filtered_events = compare_with_qullamaggie_candidates(filtered_events, qull)
    all_events = compare_with_qullamaggie_candidates(all_events, qull)

    # Selected production-candidate research configuration: conservative frequency, no warning bucket.
    selected_label = "no_yellow|max3|hold40|target3.0|regime" if regime_available else "no_yellow|max3|hold40|target3.0"
    events = select_events(filtered_events, ["White Up", "Entry"], max_per_day=3)
    selected_cfg = SteveBacktestConfig(max_hold_bars=40, target_r=3, slippage_bps=10)
    trades = simulate_steve_algo_trades(events, enriched, selected_cfg)

    eligible_random = panel[
        (panel["Price"] >= 5)
        & (panel["Daily $ Volume 20D"] >= 50_000_000)
        & (panel["Momentum Rank"] >= 85)
    ].copy()
    if regime_available:
        eligible_random = apply_regime_filter(eligible_random, regime)
    random_events = build_config_random_benchmark(trades, eligible_random, seed=42)
    random_trades = simulate_steve_algo_trades(random_events, enriched, selected_cfg)

    matrix_rows = []
    bucket_sets = {
        "entry_only": ["Entry"],
        "white_up_only": ["White Up"],
        "no_yellow": ["White Up", "Entry"],
        "all_buckets": ["White Up", "Entry", "Yellow"],
    }
    for label, buckets in bucket_sets.items():
        for max_per_day in [3, 5, 10]:
            for hold in [10, 20, 40]:
                for target in [2.0, 3.0, None]:
                    cfg_events = select_events(filtered_events, buckets, max_per_day=max_per_day)
                    cfg = SteveBacktestConfig(max_hold_bars=hold, target_r=target, slippage_bps=10)
                    cfg_trades = simulate_steve_algo_trades(cfg_events, enriched, cfg)
                    row = compact_summary(f"{label}|max{max_per_day}|hold{hold}|target{target or 'none'}", cfg_trades)
                    cfg_random_events = build_config_random_benchmark(cfg_trades, eligible_random, seed=42)
                    cfg_random_trades = simulate_steve_algo_trades(cfg_random_events, enriched, cfg)
                    random_summary = summarize_steve_algo_backtest(cfg_random_trades)
                    row.update(
                        {
                            "bucket_set": label,
                            "max_per_day": max_per_day,
                            "max_hold": hold,
                            "target_r": target or "none",
                            "random_trades": random_summary["trades"],
                            "random_expectancy_r": random_summary["expectancy_r"],
                            "edge_vs_random_r": round(row["expectancy_r"] - random_summary["expectancy_r"], 4),
                        }
                    )
                    matrix_rows.append(row)
    matrix = pd.DataFrame(matrix_rows)

    summary = summarize_steve_algo_backtest(trades)
    summary["selected_config"] = selected_label
    summary["events"] = int(len(events))
    summary["tickers_with_history"] = int(len(history))
    summary["date_min"] = str(raw["Date"].min().date())
    summary["date_max"] = str(raw["Date"].max().date())
    summary["market_cap_metadata"] = {
        "path": str(MARKET_METADATA),
        "loaded_rows": int(len(metadata)),
        "gate_applied": bool(market_cap_gate > 0),
        "min_market_cap": int(market_cap_gate),
        "coverage_pct_panel": round(float(panel["Market Cap"].notna().mean() * 100), 2) if "Market Cap" in panel.columns else 0.0,
    }
    summary["regime_filter"] = {
        "applied": bool(regime_available),
        "rule": "SPY close > SMA200 AND QQQ close > SMA200",
        "source": str(regime["Regime Source"].dropna().iloc[-1]) if "Regime Source" in regime.columns and not regime.empty else "N/D",
        "kept_events": int(len(filtered_events)),
        "raw_events": int(len(all_events)),
    }
    summary["qullamaggie_comparison"] = {
        "qullamaggie_candidates": int(len(qull)),
        "selected_events_with_qull_overlap": int(events["Qullamaggie Overlap"].sum()) if "Qullamaggie Overlap" in events.columns else 0,
        "selected_events_qull_overlap_pct": round(float(events["Qullamaggie Overlap"].mean() * 100), 2) if "Qullamaggie Overlap" in events.columns and len(events) else 0.0,
    }
    summary["oos_split"] = summarize_in_sample_out_of_sample(trades, split_fraction=0.7)
    summary["random_benchmark"] = summarize_steve_algo_backtest(random_trades)
    summary["capital_10k"] = {
        "assumption": "fixed fractional risk per trade, sequentially compounded by exit date; ignores broker margin/concurrency constraints",
        "risk_0_5pct": simulate_capital_curve(trades, initial_capital=10_000, risk_fraction=0.005),
        "risk_1pct": simulate_capital_curve(trades, initial_capital=10_000, risk_fraction=0.01),
        "risk_2pct": simulate_capital_curve(trades, initial_capital=10_000, risk_fraction=0.02),
    }
    summary["best_retest_configs"] = matrix.sort_values(["edge_vs_random_r", "expectancy_r", "profit_factor"], ascending=[False, False, False]).head(10).to_dict("records")

    latest_date = str(panel["Date"].max().date())
    latest_watchlist = select_events(all_events[all_events["Date"] == panel["Date"].max()], ["White Up", "Entry", "Yellow"], max_per_day=50)
    watchlist_paths = export_daily_watchlist(latest_watchlist, WATCHLIST_DIR, OBSIDIAN_WATCHLIST_DIR, as_of=latest_date)
    summary["daily_watchlist_exports"] = {k: str(v) for k, v in watchlist_paths.items()}

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, default=str))
    trades.to_csv(TRADES, index=False)
    events.head(5000).to_csv(EVENTS, index=False)
    matrix.to_csv(MATRIX, index=False)
    write_report(summary, matrix)
    print(json.dumps(summary, indent=2, default=str))
    print(f"trades={TRADES}")
    print(f"events={EVENTS}")
    print(f"matrix={MATRIX}")
    print(f"report={REPORT}")


if __name__ == "__main__":
    main()
