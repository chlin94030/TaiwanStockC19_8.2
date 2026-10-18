# Alpha Radar V19.1 - historical research report

Status: **RESEARCH_PARTIAL**
Data: 2023-10-16 to 2026-10-15; 92 symbols.

Entry: next common trading-session open. Exit: T+h close. Costs and gap cap are in settings.
All results are research estimates, NOT actual fills or proof of optimal weights.

## Seeded cutoff dates

2024-09-11, 2024-10-16, 2024-12-19, 2025-04-15, 2025-06-09, 2025-09-04, 2025-11-18, 2026-01-19, 2026-04-14, 2026-05-25

## Equal-weight portfolio net returns

| Cutoff | Sessions | Baseline | Adaptive | 0050 | Complete |
|---|---:|---:|---:|---:|---|
| 2024-09-11 | 5 | +3.18% | +3.18% | +1.80% | True |
| 2024-09-11 | 20 | +11.10% | +11.10% | +10.99% | True |
| 2024-09-11 | 60 | +1.43% | +1.43% | +9.01% | True |
| 2024-10-16 | 5 | -3.23% | -3.23% | +0.54% | True |
| 2024-10-16 | 20 | -9.17% | -9.17% | -2.02% | True |
| 2024-10-16 | 60 | -14.85% | -14.85% | +2.08% | True |
| 2024-12-19 | 5 | -- | -- | +1.83% | False |
| 2024-12-19 | 20 | -11.52% | -11.52% | +2.22% | True |
| 2024-12-19 | 60 | -0.75% | -0.75% | -3.93% | True |
| 2025-04-15 | 5 | -0.88% | -0.88% | -5.25% | True |
| 2025-04-15 | 20 | +16.94% | +16.94% | +11.12% | True |
| 2025-04-15 | 60 | +88.08% | +88.08% | +20.08% | True |
| 2025-06-09 | 5 | -5.77% | -5.77% | -- | True |
| 2025-06-09 | 20 | -6.87% | -6.87% | +4.56% | True |
| 2025-06-09 | 60 | +12.23% | +12.23% | +12.74% | True |
| 2025-09-04 | 5 | -7.65% | -7.65% | +4.71% | True |
| 2025-09-04 | 20 | -9.65% | -9.65% | +13.40% | True |
| 2025-09-04 | 60 | -12.27% | -12.27% | +17.64% | True |
| 2025-11-18 | 5 | -2.12% | -2.12% | -0.15% | True |
| 2025-11-18 | 20 | +8.70% | +8.70% | +2.99% | True |
| 2025-11-18 | 60 | +37.69% | +37.69% | +29.86% | True |
| 2026-01-19 | 5 | -2.03% | -2.03% | +1.41% | True |
| 2026-01-19 | 20 | -6.98% | -6.98% | +13.83% | True |
| 2026-01-19 | 60 | -11.02% | -11.02% | +30.54% | True |
| 2026-04-14 | 5 | -0.44% | -0.44% | +1.76% | True |
| 2026-04-14 | 20 | +21.67% | +21.67% | +13.01% | True |
| 2026-04-14 | 60 | +2.64% | +2.64% | +25.19% | True |
| 2026-05-25 | 5 | +2.24% | +2.24% | +3.34% | True |
| 2026-05-25 | 20 | -0.08% | -0.08% | +7.84% | True |
| 2026-05-25 | 60 | -20.26% | -20.26% | +1.60% | True |

## Paired-case summary (arithmetic means, not a compounded curve)

| Sessions | Model pairs | Fixed mean | Adaptive mean | Delta | Benchmark pairs | Adaptive matched | Benchmark matched | Excess |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 9 | -1.85% | -1.85% | +0.00% | 8 | -1.36% | +1.02% | -2.39% |
| 20 | 10 | +1.41% | +1.41% | +0.00% | 10 | +1.41% | +7.79% | -6.38% |
| 60 | 10 | +8.29% | +8.29% | +0.00% | 10 | +8.29% | +14.48% | -6.19% |

## Evaluation role

DEVELOPMENT_REPLAY_NOT_NEW_HOLDOUT
These exposed historical dates are development diagnostics, not a fresh blind test.
Fixed/Adaptive are study rankings. Guarded is a separate past-evidence stock/cash policy, not an executed portfolio.
Benchmark missing values are not zeros. Interior gaps can permit endpoint returns but never invented path-risk estimates.

## Case diagnostics

| Date | Sessions | Model reason | Benchmark reason | Price bases match |
|---|---:|---|---|---|
| 2024-12-19 | 5 | SELECTED_OUTCOME_MISSING | ASSUMED_OPEN_FILL | False |
| 2025-06-09 | 5 | COMPLETE | EXIT_NO_VOLUME | False |
| 2025-06-09 | 60 | ENDPOINT_COMPLETE_PATH_INCOMPLETE | ASSUMED_OPEN_FILL | True |

## Active optional-factor coverage

{"revenue": 0.0, "quality": 0.0, "foreign": 0.0, "trust": 0.0, "joint": 0.0, "dealer": 0.0, "margin": 0.0, "holders": 0.0}

## Data classification

PROVIDER_HISTORY

## Weight fit statuses

{"BASELINE_STABILITY_NOT_PROVEN": 19, "BASELINE_INSUFFICIENT_INDEPENDENT_BLOCKS": 5, "BASELINE_VALIDATION_COHORT_INCOMPLETE": 5, "BASELINE_TRAIN_COHORT_INCOMPLETE": 1}

## Limitations
- Current/user universe is not verified historical membership; survivorship bias remains.
- Adjusted-price coverage differs by provider. Raw price returns are not total returns.
- Daily open fills and slippage are assumptions, not executed trades.
- Ten dates can overlap, especially at 60 sessions; they are not independent trials.
- No historical intraday path: daily low/open/close cannot prove a live reversal prediction.
- No official limit-price series: limit-up hit probability is not reported.

## Configuration

```json
{
  "years": 3,
  "stocks": 100,
  "dates": 10,
  "horizons": [
    5,
    20,
    60
  ],
  "trials": 24,
  "seed": 1900,
  "top_k": 5,
  "warmup": 80,
  "train_dates": 35,
  "min_universe": 10,
  "min_turnover": 5000000.0,
  "max_entry_gap": 0.04,
  "buy_fee": 0.001425,
  "sell_fee": 0.001425,
  "sell_tax": 0.003,
  "benchmark_sell_tax": 0.001,
  "slippage": 0.0005,
  "optional_data": false,
  "request_budget": 260,
  "timeout_seconds": 12,
  "benchmark": "0050.TW",
  "as_of": "2026-10-15",
  "prefer_adjusted": true,
  "validation_blocks": 4,
  "improvement_floor": 0.002,
  "weight_shrinkage": 0.5,
  "max_mae_deterioration": 0.01,
  "min_validation_fill": 0.6,
  "replay_cutoffs": [
    "2024-09-11",
    "2024-10-16",
    "2024-12-19",
    "2025-04-15",
    "2025-06-09",
    "2025-09-04",
    "2025-11-18",
    "2026-01-19",
    "2026-04-14",
    "2026-05-25"
  ],
  "evaluation_role": "DEVELOPMENT_REPLAY_NOT_NEW_HOLDOUT"
}
```