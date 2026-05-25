from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd
import numpy as np


@dataclass(frozen=True)
class SteveBacktestConfig:
    max_hold_bars: int = 20
    stop_mode: str = "darvas_or_ema20_atr"
    target_r: float | None = 3.0
    slippage_bps: float = 0.0


def _entry_price(open_price: float, slippage_bps: float) -> float:
    return float(open_price) * (1 + slippage_bps / 10_000)


def _exit_price(raw_price: float, slippage_bps: float) -> float:
    return float(raw_price) * (1 - slippage_bps / 10_000)


def _initial_stop(signal_row: Mapping[str, object], entry_bar: pd.Series, entry_price: float) -> float:
    darvas = signal_row.get("Darvas Lower", entry_bar.get("Darvas Lower", math.nan))
    ema20 = entry_bar.get("EMA20", math.nan)
    atr20 = entry_bar.get("ATR20", math.nan)
    candidates: list[float] = []
    for value in [darvas, (float(ema20) - float(atr20)) if pd.notna(ema20) and pd.notna(atr20) else math.nan]:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value < entry_price:
            candidates.append(value)
    if not candidates:
        return entry_price * 0.92
    return max(candidates)


def simulate_steve_algo_trades(
    events: pd.DataFrame,
    history_by_ticker: Mapping[str, pd.DataFrame],
    config: SteveBacktestConfig | None = None,
) -> pd.DataFrame:
    """Simulate next-open long trades from SteveAlgo signal events with no same-bar execution."""
    config = config or SteveBacktestConfig()
    trades: list[dict[str, object]] = []
    if events.empty:
        return pd.DataFrame(trades)

    available_after_by_ticker: dict[str, pd.Timestamp] = {}
    for _, event in events.sort_values("Date").iterrows():
        ticker = str(event["Ticker"])
        signal_date = pd.Timestamp(event["Date"])
        if ticker not in history_by_ticker:
            continue
        if signal_date <= available_after_by_ticker.get(ticker, pd.Timestamp.min):
            continue
        hist = history_by_ticker[ticker].sort_index()
        signal_date = pd.Timestamp(event["Date"])
        future_positions = hist.index[hist.index > signal_date]
        if len(future_positions) == 0:
            continue
        entry_date = future_positions[0]
        entry_loc = hist.index.get_loc(entry_date)
        entry_bar = hist.iloc[entry_loc]
        entry = _entry_price(entry_bar["Open"], config.slippage_bps)
        stop = _initial_stop(event.to_dict(), entry_bar, entry)
        risk = entry - stop
        if risk <= 0 or not math.isfinite(risk):
            continue
        target = entry + risk * config.target_r if config.target_r else math.inf
        exit_date = entry_date
        exit_reason = "timeout"
        exit_raw = hist.iloc[min(entry_loc + config.max_hold_bars - 1, len(hist) - 1)]["Close"]
        bars_held = 0
        for bar_offset in range(config.max_hold_bars):
            loc = entry_loc + bar_offset
            if loc >= len(hist):
                break
            bar = hist.iloc[loc]
            exit_date = hist.index[loc]
            bars_held = bar_offset + 1
            if float(bar["Low"]) <= stop:
                exit_reason = "stop"
                exit_raw = stop
                break
            if float(bar["High"]) >= target:
                exit_reason = "target"
                exit_raw = target
                break
            if pd.notna(bar.get("EMA20", math.nan)) and float(bar["Close"]) < float(bar["EMA20"]):
                exit_reason = "ema20_close"
                exit_raw = bar["Close"]
                break
            exit_raw = bar["Close"]
        exit_px = _exit_price(exit_raw, config.slippage_bps)
        r_mult = (exit_px - entry) / risk
        trades.append(
            {
                "Ticker": ticker,
                "Bucket": event.get("SteveAlgo Primary Bucket", "N/D"),
                "Signal Date": signal_date,
                "Entry Date": entry_date,
                "Exit Date": exit_date,
                "Entry Price": round(entry, 4),
                "Stop": round(stop, 4),
                "Exit Price": round(exit_px, 4),
                "Exit Reason": exit_reason,
                "Bars Held": bars_held,
                "R": round(r_mult, 4),
            }
        )
        available_after_by_ticker[ticker] = pd.Timestamp(exit_date)
    return pd.DataFrame(trades)


def build_random_event_benchmark(actual_trades: pd.DataFrame, eligible_events: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Sample one random eligible event for each actual trade date.

    This keeps the benchmark comparable by date/count instead of comparing the
    strategy against a different market regime or trade frequency.
    """
    if actual_trades.empty or eligible_events.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    sampled: list[pd.Series] = []
    eligible = eligible_events.copy()
    eligible["Date"] = pd.to_datetime(eligible["Date"])
    signal_dates = pd.to_datetime(actual_trades["Signal Date"])
    for signal_date in signal_dates:
        candidates = eligible[eligible["Date"] == signal_date]
        if candidates.empty:
            continue
        picked_idx = rng.choice(candidates.index.to_numpy())
        sampled.append(candidates.loc[picked_idx])
    return pd.DataFrame(sampled).reset_index(drop=True) if sampled else pd.DataFrame()


def build_config_random_benchmark(actual_trades: pd.DataFrame, eligible_events: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Build a same-date/random-eligible benchmark for one tested configuration."""
    sampled = build_random_event_benchmark(actual_trades, eligible_events, seed=seed)
    if not sampled.empty:
        sampled = sampled.copy()
        sampled["SteveAlgo Primary Bucket"] = "Random Eligible"
    return sampled


def load_market_metadata(path: str | Path) -> pd.DataFrame:
    """Load ticker metadata from CSV/Parquet and normalize market-cap columns."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["Ticker", "Market Cap", "Market Cap Source"])
    meta = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    meta = meta.copy()
    lower_map = {c.lower(): c for c in meta.columns}
    ticker_col = lower_map.get("ticker") or lower_map.get("symbol")
    cap_col = lower_map.get("market cap") or lower_map.get("marketcap") or lower_map.get("market_cap")
    if ticker_col is None:
        raise ValueError("market metadata needs a Ticker or symbol column")
    meta["Ticker"] = meta[ticker_col].astype(str).str.upper().str.strip()
    meta["Market Cap"] = pd.to_numeric(meta[cap_col], errors="coerce") if cap_col else np.nan
    if "Market Cap Source" not in meta.columns:
        meta["Market Cap Source"] = str(path)
    keep = ["Ticker", "Market Cap", "Market Cap Source"]
    extra = [c for c in ["Sector", "Industry", "Exchange"] if c in meta.columns]
    return meta[keep + extra].drop_duplicates("Ticker", keep="last")


def apply_market_metadata(panel: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Attach source-traced market cap metadata to an event/panel dataframe."""
    output = panel.copy()
    if metadata.empty:
        if "Market Cap" not in output.columns:
            output["Market Cap"] = np.nan
        output["Market Cap Source"] = "N/D"
        return output
    drop_cols = [c for c in ["Market Cap", "Market Cap Source"] if c in output.columns]
    output = output.drop(columns=drop_cols)
    return output.merge(metadata, on="Ticker", how="left")


def apply_regime_filter(events: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
    """Keep only events where both SPY and QQQ regime gates are true."""
    if events.empty or regime.empty:
        return events.iloc[0:0].copy()
    e = events.copy()
    r = regime.copy()
    e["Date"] = pd.to_datetime(e["Date"])
    r["Date"] = pd.to_datetime(r["Date"])
    merged = e.merge(r[["Date", "SPY Regime OK", "QQQ Regime OK"]], on="Date", how="left")
    ok = merged["SPY Regime OK"].fillna(False).astype(bool) & merged["QQQ Regime OK"].fillna(False).astype(bool)
    out = merged[ok].copy()
    out["Regime Filter"] = "SPY_AND_QQQ_OK"
    return out


def export_daily_watchlist(
    watchlist: pd.DataFrame,
    export_dir: str | Path,
    obsidian_dir: str | Path,
    as_of: str | None = None,
) -> dict[str, Path]:
    """Write the latest SteveAlgo watchlist to CSV and an Obsidian markdown note."""
    export_dir = Path(export_dir)
    obsidian_dir = Path(obsidian_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    obsidian_dir.mkdir(parents=True, exist_ok=True)
    if as_of is None:
        as_of = pd.Timestamp(watchlist["Date"].max()).date().isoformat() if not watchlist.empty and "Date" in watchlist.columns else pd.Timestamp.today().date().isoformat()
    csv_path = export_dir / f"steve_algo_watchlist_{as_of}.csv"
    md_path = obsidian_dir / f"steve-algo-watchlist-{as_of}.md"
    watchlist.to_csv(csv_path, index=False)
    lines = [f"# SteveAlgo Watchlist — {as_of}", "", "Research only. Capital authorized: 0%.", ""]
    for row in watchlist.head(100).to_dict("records"):
        lines.append(
            f"- {row.get('Ticker', 'N/D')} — {row.get('SteveAlgo Primary Bucket', 'N/D')} | "
            f"price {row.get('Price', 'N/D')} | momentum {row.get('Momentum Rank', 'N/D')} | "
            f"R/R {row.get('Reward-Risk', 'N/D')} | {row.get('SteveAlgo Reason', '')}"
        )
    md_path.write_text("\n".join(lines))
    return {"csv": csv_path, "markdown": md_path}


def compare_with_qullamaggie_candidates(steve_events: pd.DataFrame, qullamaggie_events: pd.DataFrame) -> pd.DataFrame:
    """Add overlap flags between SteveAlgo candidates and Qullamaggie candidates."""
    out = steve_events.copy()
    qtickers = set(qullamaggie_events.get("Ticker", pd.Series(dtype=str)).astype(str))
    out["Qullamaggie Overlap"] = out["Ticker"].astype(str).map(lambda t: bool(t in qtickers)).astype(object)
    return out


def simulate_capital_curve(trades: pd.DataFrame, initial_capital: float = 10_000.0, risk_fraction: float = 0.01) -> dict[str, object]:
    """Convert R-multiple trades into a compounded capital curve using fixed fractional risk."""
    equity = float(initial_capital)
    curve = []
    if trades.empty:
        return {
            "initial_capital": round(initial_capital, 2),
            "final_capital": round(equity, 2),
            "total_return_pct": 0.0,
            "risk_fraction_pct": round(risk_fraction * 100, 2),
            "max_drawdown_pct": 0.0,
            "trades": 0,
        }
    ordered = trades.sort_values([c for c in ["Exit Date", "Entry Date", "Signal Date"] if c in trades.columns]).copy()
    peak = equity
    max_dd = 0.0
    for _, trade in ordered.iterrows():
        pnl = equity * risk_fraction * float(trade["R"])
        equity += pnl
        peak = max(peak, equity)
        dd = (equity / peak - 1) * 100 if peak else 0.0
        max_dd = min(max_dd, dd)
        curve.append({"date": trade.get("Exit Date"), "equity": round(equity, 2), "pnl": round(pnl, 2), "R": float(trade["R"])})
    return {
        "initial_capital": round(initial_capital, 2),
        "final_capital": round(equity, 2),
        "total_return_pct": round((equity / initial_capital - 1) * 100, 2),
        "risk_fraction_pct": round(risk_fraction * 100, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "trades": int(len(ordered)),
        "curve": curve,
    }


def summarize_in_sample_out_of_sample(trades: pd.DataFrame, split_fraction: float = 0.7) -> dict[str, object]:
    """Summarize chronological in-sample/out-of-sample performance by signal date."""
    if trades.empty:
        return {
            "split_date": None,
            "in_sample": summarize_steve_algo_backtest(trades),
            "out_of_sample": summarize_steve_algo_backtest(trades),
        }
    ordered = trades.sort_values("Signal Date").copy()
    split_idx = max(1, min(len(ordered) - 1, int(len(ordered) * split_fraction))) if len(ordered) > 1 else 1
    split_date = pd.Timestamp(ordered["Signal Date"].iloc[split_idx]).date().isoformat() if len(ordered) > 1 else None
    return {
        "split_date": split_date,
        "in_sample": summarize_steve_algo_backtest(ordered.iloc[:split_idx]),
        "out_of_sample": summarize_steve_algo_backtest(ordered.iloc[split_idx:]),
    }


def summarize_steve_algo_backtest(trades: pd.DataFrame) -> dict[str, object]:
    if trades.empty:
        return {
            "trades": 0,
            "expectancy_r": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_r": 0.0,
        }
    r = trades["R"].astype(float)
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    equity = r.cumsum()
    drawdown = equity - equity.cummax()
    return {
        "trades": int(len(trades)),
        "expectancy_r": round(float(r.mean()), 4),
        "win_rate": round(float((r > 0).mean() * 100), 2),
        "profit_factor": round(float(gains / losses), 4) if losses > 0 else math.inf,
        "max_drawdown_r": round(float(drawdown.min()), 4),
        "bucket_expectancy_r": {k: round(float(v), 4) for k, v in trades.groupby("Bucket")["R"].mean().items()}
        if "Bucket" in trades.columns
        else {},
        "exit_reasons": trades["Exit Reason"].value_counts().to_dict() if "Exit Reason" in trades.columns else {},
    }
