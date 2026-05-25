# Strategy Self-Improvement Loop Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a controlled self-improvement layer for the SteveAlgo/Qullamaggie scanner that learns from outcomes without allowing autonomous trading or overfit parameter chasing.

**Architecture:** Implement a closed feedback loop: signal snapshot -> paper trade/outcome journal -> feature attribution -> candidate rule proposals -> walk-forward validation -> human approval gate. The system may recommend rule changes, but it must never auto-authorize capital or silently change production thresholds.

**Tech Stack:** Python, pandas, Streamlit, pytest, CSV/JSON artifacts, optional Obsidian markdown export.

---

## Non-Negotiable Safety Rules

1. Capital authorized remains `0%` unless Antonio explicitly approves a separate risk-gated playbook.
2. No autonomous parameter mutation in the live scanner.
3. Any proposed rule must beat the current baseline out-of-sample and against a date-matched random benchmark.
4. Every metric must be source-traced to a trade/event row. Missing data becomes `N/D`.
5. Optimize for robustness, not maximum backtest equity curve.

---

## Self-Improvement Definition

Allowed:
- Learn which features preceded winners/losers.
- Track false positives/false negatives.
- Propose threshold changes.
- Run walk-forward tests.
- Generate a weekly review note.
- Require Antonio approval before promoting a rule.

Forbidden:
- Auto-trade.
- Auto-increase position size.
- Auto-change production thresholds.
- Select the best historical config without OOS/random/stability checks.
- Hide losing cohorts.

---

## Data Model

### `StrategyEvent`

Minimum fields:
- `event_id`: stable hash of strategy, ticker, signal date, bucket, entry config.
- `strategy`: `SteveAlgo` / `Qullamaggie` / `SteveStyleKQ`.
- `ticker`
- `signal_date`
- `bucket`
- `features_json`: source-traced feature snapshot at signal date.
- `selected`: bool, whether paper-tracked.
- `notes`

### `StrategyOutcome`

Minimum fields:
- `event_id`
- `entry_date`
- `entry_price`
- `exit_date`
- `exit_price`
- `r_multiple`
- `max_favorable_r`
- `max_adverse_r`
- `exit_reason`
- `manual_grade`: optional Antonio grade.
- `post_trade_notes`

### `RuleCandidate`

Minimum fields:
- `rule_id`
- `created_at`
- `hypothesis`
- `changed_parameters_json`
- `training_period`
- `validation_period`
- `baseline_metrics_json`
- `candidate_metrics_json`
- `random_benchmark_json`
- `promotion_status`: `REJECTED`, `WATCH`, `PAPER_ONLY`, `APPROVED_BY_ANTONIO`.

---

## Task 1: Add outcome journal schema tests

**Objective:** Define the self-improvement artifacts before writing implementation.

**Files:**
- Create: `tests/test_strategy_learning.py`
- Create later: `qull_scanner/strategy_learning.py`

**Step 1: Write failing tests**

Tests:
- stable `event_id` is deterministic for same strategy/ticker/date/bucket/config.
- `build_outcome_summary()` returns expectancy, win rate, PF, max DD, median MFE/MAE.
- missing outcome rows are counted as open/unresolved, not dropped silently.

**Step 2: Verify RED**

Run:
```bash
pytest tests/test_strategy_learning.py -q
```
Expected: fails because `qull_scanner.strategy_learning` does not exist.

---

## Task 2: Implement immutable signal/outcome journal helpers

**Objective:** Add deterministic helpers for event IDs and summary stats.

**Files:**
- Create: `qull_scanner/strategy_learning.py`

Functions:
- `stable_event_id(row: Mapping[str, Any], config: Mapping[str, Any]) -> str`
- `build_outcome_summary(events: pd.DataFrame, outcomes: pd.DataFrame) -> dict`
- `merge_events_with_outcomes(events, outcomes) -> pd.DataFrame`

**Verification:**
```bash
pytest tests/test_strategy_learning.py -q
pytest tests -q
```

---

## Task 3: Add feature attribution tests

**Objective:** Determine which signal features separate winners from losers without claiming causality.

**Files:**
- Modify: `tests/test_strategy_learning.py`

Tests:
- numeric features produce winner/loser median differences.
- categorical features produce bucket win-rate/expectancy differences.
- tiny samples are flagged `INSUFFICIENT_SAMPLE`.

---

## Task 4: Implement feature attribution

**Files:**
- Modify: `qull_scanner/strategy_learning.py`

Functions:
- `feature_attribution(events_with_outcomes: pd.DataFrame, min_sample: int = 30) -> pd.DataFrame`

Output columns:
- `feature`
- `sample_size`
- `winner_median`
- `loser_median`
- `difference`
- `expectancy_top_quantile`
- `expectancy_bottom_quantile`
- `status`

Hard rule: label as association, not prediction.

---

## Task 5: Add rule-candidate proposal tests

**Objective:** Generate conservative candidate improvements, not optimized curve-fit rules.

Tests:
- proposal requires minimum sample size.
- proposal rejects changes that improve IS but fail OOS.
- proposal rejects candidates that do not beat random benchmark.
- proposal status defaults to `WATCH`, never `APPROVED`.

---

## Task 6: Implement rule-candidate generator

**Files:**
- Modify: `qull_scanner/strategy_learning.py`

Function:
- `propose_rule_candidates(attribution, baseline_summary, oos_summary, random_summary) -> pd.DataFrame`

Allowed proposal types:
- tighten/loosen `min_rs`
- tighten/loosen `min_reward_risk`
- exclude/allow `Yellow`
- adjust max extension gate
- adjust max signals/day

---

## Task 7: Add weekly self-review script

**Objective:** Produce a deterministic weekly report from artifacts.

**Files:**
- Create: `scripts/run_strategy_self_review.py`

Inputs:
- `exports/steve_algo_backtest_events.csv`
- `exports/steve_algo_backtest_trades.csv`
- optional manual journal CSV under `data/strategy_journal.csv`

Outputs:
- `exports/strategy_self_review.json`
- `exports/strategy_self_review.md`

Report sections:
- current baseline
- unresolved/open events
- winner/loser feature attribution
- bad cohorts to avoid
- promising cohorts to paper-track
- rejected rule proposals
- watch-only rule proposals
- hard verdict: `NO_PRODUCTION_CHANGE` unless approval criteria met.

---

## Task 8: Add Streamlit tab/view

**Objective:** Show the self-improvement loop in the app without implying auto-trading.

**Files:**
- Modify: `app.py`

Add view:
- `Strategy Learning Lab`

Display:
- learning status
- latest self-review report if present
- proposed rules with status
- open questions
- export buttons

UI copy must say:
> Self-improvement proposes hypotheses. It does not change live thresholds or authorize capital.

---

## Task 9: Add optional weekly cron/manual command

**Objective:** Enable periodic self-review only after deterministic script works.

Command:
```bash
python3 scripts/run_strategy_self_review.py
```

Optional cron later:
- weekly after scanner/backtest refresh
- no-agent deterministic output
- silent unless a new `WATCH` proposal appears or data breaks.

---

## Promotion Criteria

A rule can move from `WATCH` to `PAPER_ONLY` only if:
- sample size >= 100 trades or >= 50 OOS trades for narrow cohort.
- OOS expectancy > baseline OOS by at least +0.05R.
- OOS PF >= 1.20.
- max DD not worse than baseline by >20%.
- beats matched random benchmark by >= +0.05R.
- no single top-5 trades contribute >30% of total R.

A rule can move to live consideration only after:
- at least 30-50 forward paper trades.
- manual chart review of worst 10 and best 10 trades.
- Portfolio Risk Gate pass.
- Antonio explicit approval.

---

## Implementation Order

1. `strategy_learning.py` schema/stat helpers.
2. Feature attribution.
3. Rule-candidate generator.
4. Deterministic self-review script.
5. Streamlit `Strategy Learning Lab` view.
6. Weekly no-agent cron only if Antonio wants recurring review.

---

## Initial Verdict

Activate self-improvement as a **research loop**, not as autonomous learning. The system should learn what to investigate and what to reject. It should not learn to trade by itself.
