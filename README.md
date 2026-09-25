# Toxic-Flow-Aware Execution Agent — Project Introduction & Analysis

## 1. Project Overview

The goal of this project is to build an end-to-end trading system that:

1. **Estimates** the conditional probability that incoming order flow is *toxic* — i.e. likely to be followed by an adverse price move — given the current limit-order-book (LOB) state and recent flow.
2. **Uses** that probability to execute (or quote) more intelligently than a classical, probability-agnostic baseline.

It is structured as a six-phase pipeline — data, classical baseline, feature engineering + probability model, decision layer, backtest/diagnostics, and write-up — deliberately built up one layer at a time (classical → supervised → decision) rather than assembled from an existing template, so every design choice can be justified in an interview setting.

This document cross-references the original phase plan against the current notebook (`toxic_flow_execution_agent.ipynb`) to establish **what has actually been built, what it shows, and what remains**.

## 2. Data & Environment

- **Source:** LOBSTER, level-5 order book + message stream.
- **Instruments / session:** AAPL and INTC, single trading day (2012-06-21), loaded from Google Drive in a Colab runtime.
- **Loader:** `load_ticker()` merges the message and order-book files 1:1, rescales LOBSTER's fixed-point prices (`/10000`), and derives `mid_price` and `spread`. A row-count assertion guards against message/book misalignment.
- **Scale:** ~301K events for AAPL, ~581K for INTC on the single day, of which ~35K (AAPL) / ~32K (INTC) are aggressive (marketable) executions — the population the toxicity label is defined over.

This satisfies Phase 0's requirement for a clean, event-based loader, but the pipeline currently runs on **one symbol-day**, not a multi-day / multi-regime panel — worth flagging explicitly as a scope limitation rather than an oversight.

## 3. Toxicity Label Design (Phase 0 → Phase 2 boundary)

Toxicity is defined per aggressive execution as: does the mid-price move against the passive side (in the aggressor's favor) by more than `threshold_bps` within `horizon_events` events?

The notebook does this properly, not by picking numbers ad hoc:

- A **grid sweep** over `horizon_events ∈ {200,300,400,500}` × `threshold_bps ∈ {1,1.5,2,2.5,3,4}` is run and tabulated, explicitly checking that the resulting toxic rate lands in a sane 10–40% band (not ~0% or ~100%, which would signal a miscalibrated label).
- **Chosen:** `horizon_events=300`, `threshold_bps=2` → toxic rate **30%** for AAPL, **54%** for INTC (INTC trades more of the day toward adverse continuation — a first hint the two names have different microstructure regimes).
- A **leakage check** (`check_no_leakage`) confirms every label's forward window stays within bounds and doesn't reach past the data — good practice, though it only checks the *label*, not yet the *features* (see §7).

This is the strongest-executed part of the project: it directly answers the "why this label definition, how do you avoid look-ahead" questions the plan calls out.

## 4. Classical Baseline — TWAP (Phase 1)

- Parent order: **sell 1,000 shares of AAPL over 30 minutes**, sliced into 30 equal child trades (1/min), executed at the prevailing mid-price at each slice time.
- **Result:**

| Metric | Value |
|---|---|
| Arrival price | 585.51 |
| Average execution price | 586.31 |
| Implementation shortfall | **797.00** (0.136%) |

The IS calculation and its percentage form are both implemented, giving a concrete, auditable reference number. What's **not yet present**: risk metrics beyond IS (e.g. execution variance across slices), and the baseline is only ever run once — it isn't yet re-run across the toxic-flow-conditioned dataset for a like-for-like comparison against the ML-informed policy in Phase 4.

## 5. Feature Engineering (Phase 2)

Ten microstructure features are engineered, matching every bullet in the original spec:

| Category | Features |
|---|---|
| Order-flow imbalance | `order_flow_imbalance` |
| Depth / micro-price | `total_ask_depth`, `total_bid_depth`, `depth_imbalance`, `micro_price`, `spread` |
| Trade aggressiveness | `aggressiveness_volume_imbalance`, `aggressiveness_count_imbalance` |
| Volatility / arrival rate | `short_horizon_volatility`, `arrival_rate` |

All are rolling-window computations built with the same pattern (create temp buy/sell columns → rolling sum → ratio/imbalance → drop temp columns), applied to both AAPL and INTC.

## 6. Probability Model & Calibration (Phase 2)

- **Model:** `LogisticRegression(solver='liblinear')`, trained on the **AAPL dataset only** (the INTC feature set is built but never used for training or held out as a cross-symbol test — a gap, see §8).
- **Split:** `train_test_split(..., test_size=0.2, stratify=y, random_state=42)` — a **random row-wise split**, not a time-based / walk-forward split.

### 6.1 What the diagnostics actually show

| Diagnostic | Result | Interpretation |
|---|---|---|
| ROC AUC | **0.52** | Essentially no better than a coin flip — the model has almost no discriminative power yet |
| Calibration curve | Predicted probabilities top out around **~0.31**; below the diagonal at low predictions, close to it at higher ones | The model never becomes confident; it's mildly under-confident where it does predict, but its ceiling is far below 1.0 |
| Confusion matrix (test set) | TN 1671, FP 3186, FN 671, TP 1447 | Precision ≈ 0.31, Recall ≈ 0.68 at the *mean-predicted-probability* threshold used for this evaluation |

**This is the single most important finding in the notebook so far**: with AUC ≈ 0.52, the 10 engineered features barely separate toxic from non-toxic flow in the current setup. That's a modeling/feature problem to solve *before* investing further in the decision layer or backtest — see recommendations.

## 7. Decision Layer (Phase 3) — started, and currently broken

The notebook begins Phase 3 with a simple rule: flag `toxicity_decision = 1` if `P(toxic) ≥ 0.7`. Applied to the full AAPL dataset:

```
Value counts for 'toxicity_decision':
0    34875
```

**Every single row is flagged non-toxic.** This is a direct, mechanical consequence of §6.1: since the model's calibrated probabilities never exceed ~0.31, a fixed 0.7 threshold can never fire. The evaluation step earlier in the notebook worked around this by using a *dynamic* threshold (the mean predicted probability) — but the Phase 3 decision rule reverts to a hardcoded 0.7 and silently produces a no-op policy. This should be treated as a live bug, not a stylistic choice.

## 8. Status vs. the Original 6-Phase Plan

| Phase | Planned | Status |
|---|---|---|
| 0 — Setup & data | Loader, LOB reconstruction, synthetic toxic label | **Done**, single symbol-day scope only |
| 1 — Classical baseline | TWAP/AC, IS + risk metrics | **Mostly done** — TWAP + IS done; broader risk metrics and repeatable runs across scenarios not yet done |
| 2 — Features & probability | Features, label, calibrated model | **Done but underperforming** — AUC 0.52, needs iteration |
| 3 — Decision layer | Rule-based P(toxic) gate | **Started, currently non-functional** (threshold bug) |
| 4 — Backtest & diagnostics | Walk-forward eval, baseline comparison, regime breakdown, ablation | **Not started** |
| 5 — Write-up & packaging | Report, README, design-choice log | **Not started** (this document is a first step toward it) |

## 9. Recommended Next Steps

Given the current state, the highest-leverage work is **not** finishing the decision layer as-is — it's fixing what feeds it:

1. **Fix the threshold bug immediately**: derive `toxicity_threshold` from the score distribution (e.g. a target flag-rate quantile), not a hardcoded 0.7 — and unit-test that the decision layer can actually produce both classes.
2. **Diagnose the AUC 0.52 result** before building further on top of it:
   - Check whether rolling-window features actually vary meaningfully across the AAPL day (a near-random AUC can hide a bug in a feature — e.g. a window that's always full/empty, or a feature computed on the wrong side of the label's time boundary).
   - Extend the leakage check (§3) from the *label* to the *features* — confirm each feature at row *t* only uses information available at or before *t*.
   - Try a stronger model (gradient boosting) and a richer feature set (e.g. multi-scale OFI, LOB imbalance at more levels) only after the leakage/window checks pass.
3. **Switch to a time-respecting split**: replace the random `train_test_split` with a chronological or walk-forward split — a random split on time-series features with rolling windows risks train/test leakage across adjacent, overlapping windows.
4. **Use the INTC dataset you already built**: either as a second training set (regime generalization) or as a genuinely held-out cross-symbol test — right now it's computed and discarded.
5. **Only then resume Phase 3–4**: rule-based decision layer → re-run the TWAP baseline under toxic vs. non-toxic conditioning → compare implementation shortfall, adverse-selection rate, and inventory risk between the classical and toxicity-aware policies, broken out by regime.
6. **Engineering hygiene for the next iteration** (per your stated preference for clean design):
   - Pull the loader, labeler, feature builders, and TWAP simulator out of notebook cells into a small `src/` package with one function/class per pipeline stage — the notebook already follows a clean single-responsibility style per cell, so this is mostly extraction, not a rewrite.
   - Model the decision layer as a small **Strategy pattern** (`ExecutionPolicy` interface with `TWAPPolicy` and `ToxicityAwarePolicy` implementations) so Phase 4's baseline-vs-ML comparison is a policy swap, not duplicated code.
   - Wrap label/feature parameters (`horizon_events`, `threshold_bps`, `window_size`, `toxicity_threshold`) in a single config object so a design-log entry can point at one place per decision.
