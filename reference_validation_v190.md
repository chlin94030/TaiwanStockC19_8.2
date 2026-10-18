# Alpha Radar V19.0 - historical research report

Status: **RESEARCH_PARTIAL**
Data: 2023-10-16 to 2026-10-15; 92 symbols.

Entry: next common trading-session open. Exit: T+h close. Costs and gap cap are in settings.
All results are research estimates, NOT actual fills or proof of optimal weights.

## Seeded cutoff dates

2024-09-11, 2024-10-16, 2024-12-19, 2025-04-15, 2025-06-09, 2025-09-04, 2025-11-18, 2026-01-19, 2026-04-14, 2026-05-25

## Equal-weight portfolio net returns

| Cutoff | Sessions | Baseline | Adaptive | 0050 | Complete |
|---|---:|---:|---:|---:|---|
| 2024-09-11 | 5 | +3.18% | +9.06% | +1.80% | True |
| 2024-09-11 | 20 | +11.10% | +11.10% | +10.99% | True |
| 2024-09-11 | 60 | +1.43% | +1.43% | +9.01% | True |
| 2024-10-16 | 5 | -3.23% | -3.23% | +0.54% | True |
| 2024-10-16 | 20 | -12.38% | -12.38% | -2.02% | True |
| 2024-10-16 | 60 | -17.89% | -17.89% | +2.08% | True |
| 2024-12-19 | 5 | -- | -- | +1.83% | False |
| 2024-12-19 | 20 | -- | -- | +0.83% | False |
| 2024-12-19 | 60 | -- | -- | -5.24% | False |
| 2025-04-15 | 5 | -1.49% | -1.49% | -5.25% | True |
| 2025-04-15 | 20 | +19.29% | +19.29% | +11.12% | True |
| 2025-04-15 | 60 | -- | -- | -- | False |
| 2025-06-09 | 5 | -5.77% | -5.77% | -- | True |
| 2025-06-09 | 20 | -6.95% | -9.98% | -- | True |
| 2025-06-09 | 60 | +11.85% | +11.85% | -- | True |
| 2025-09-04 | 5 | -7.65% | -7.65% | +4.71% | True |
| 2025-09-04 | 20 | -9.65% | -9.65% | +13.40% | True |
| 2025-09-04 | 60 | -12.27% | -12.27% | +17.64% | True |
| 2025-11-18 | 5 | -2.12% | -2.12% | -0.15% | True |
| 2025-11-18 | 20 | +8.70% | +8.70% | +2.99% | True |
| 2025-11-18 | 60 | +37.69% | +37.69% | +28.06% | True |
| 2026-01-19 | 5 | -2.05% | -2.05% | +0.00% | True |
| 2026-01-19 | 20 | -6.98% | -6.98% | +12.25% | True |
| 2026-01-19 | 60 | -11.38% | -11.38% | +28.72% | True |
| 2026-04-14 | 5 | -0.44% | +0.28% | +1.76% | True |
| 2026-04-14 | 20 | +21.67% | +12.87% | +13.01% | True |
| 2026-04-14 | 60 | +1.69% | +5.07% | +25.19% | True |
| 2026-05-25 | 5 | +2.24% | +2.24% | +3.34% | True |
| 2026-05-25 | 20 | -0.40% | -0.40% | +7.84% | True |
| 2026-05-25 | 60 | -21.23% | -18.32% | +0.99% | True |

## Data classification

PROVIDER_HISTORY

## Weight fit statuses

{"RESEARCH_FIT": 16, "BASELINE_VALIDATION_NOT_BETTER": 12, "BASELINE_INSUFFICIENT_TRAINING": 2}

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
  "as_of": "2026-10-15"
}
```