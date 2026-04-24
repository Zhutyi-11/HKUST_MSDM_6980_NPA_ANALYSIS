# NPA Repayment Data Schema Reference

## Required Input File Format

The skill expects an **Excel file (`.xlsx`)** containing a portfolio of non-performing accounts with historical repayment outcomes.

### Required Columns

| Column | Type | Description |
|--------|------|-------------|
| `id` | string/number | Unique account identifier |
| `data_type` | string | **Required**. Must contain values `M` (modeling/training) and `T` (test/validation) |
| `payer_3yr` | string | Target variable: `"Y"` = paid within 3 years, otherwise not paid |
| `debtor_last` | string | Debtor name (excluded from modeling for privacy) |

### Categorical Features

| Column | Description | Typical Values |
|--------|-------------|---------------|
| `loan_type` | Product type | Credit Card, Personal Loan, etc. |
| `purchased_bal_gp` | Balance group bucket | `00. <=200`, `01. <=5k`, ..., `07. 200k+` (7 levels) |
| `district` | Geographic district | TUEN MUN, KWUN TONG, etc. (will be uppercased) |
| `multiple_acct` | Multiple account flag | Y / N |
| `home_phone_flag` | Has home phone | Y / N |
| `mobile_phone_flag` | Has mobile phone | Y / N |

### Numeric Features

| Column | Unit | Missing Handling |
|--------|------|-----------------|
| `last_act_closing_m` | Months since last activity | `-1` if missing + flag column |
| `open_closing_m` | Months from account open to closing | - |
| `co_closing_m` | Months since write-off to closing | - |
| `last_pay_date_client_closing_m` | Months since last payment to original creditor | `-1` if missing (= never paid) + flag |
| `birth_yr` | Birth year | Numeric year (e.g., 1975) |

### Data Quality Rules

- **Minimum size**: ~1,000+ records recommended; ~10K+ for stable model training
- **Split requirement**: Must have both `data_type=M` and `data_type=T` rows
- **Target balance**: Expect ~8-12% positive rate (`payer_3yr=Y`) for typical NPA portfolios
- **Missing values**: Handled automatically by the pipeline

## Output Files Generated

After running the full pipeline, the output directory contains:

```
agent_outputs/baseline_comparison_run/
├── metrics.json                    # All model metrics & results
├── npa_repayment_model.joblib      # Trained model bundle (pipeline + calibrator)
├── test_scored_accounts.csv        # Test set with predictions & queue assignments
├── production_queue_summary.csv    # Queue-level aggregation
├── champion_challenger_summary.csv # Model comparison table
├── agent_vs_baseline_summary.csv   # Agent vs baseline comparison
├── feature_importance.csv          # Top features per model
├── payer_rate_by_balance.csv       # Payer rate by balance group
├── payer_rate_by_loan.csv          # Payer rate by loan type
├── collection_strategy_report.md   # Full markdown strategy report
├── production_config_used.json     # Economic assumptions used
└── dashboard.html                  # Interactive dashboard (generated separately)
```
