---
name: npa-repayment-analyzer
description: This skill should be used when the task involves unsecured debt, credit card, NPA repayment prediction, historical collection analysis, portfolio scoring, production policy optimization, or translating model outputs into ROI-driven collection strategies. Trigger especially when a user provides portfolio files such as .xlsx/.csv and asks for repayment modeling, probability calibration, collection segmentation, queue optimization, collection strategy reports, or generating an interactive analysis dashboard.
---

# NPA Repayment Analyzer

End-to-end ML pipeline for non-performing asset (NPA) repayment prediction. Given an Excel file with historical account data and 3-year repayment labels, train calibrated models, compare champion vs baseline, optimize collection queues under economic constraints, produce strategy reports, and render an interactive dashboard — all in one run.

## When to Use

Trigger this skill when any of the following apply:

- User provides an `.xlsx`/`.csv` file with debt/portfolio data and wants predictive modeling
- User asks about repayment probability, recovery prediction, or collection optimization
- User mentions NPA, bad debt, credit card collections, or unsecured loan scoring
- User wants to build a collection strategy report or queue allocation plan
- User needs model comparison (champion-challenger) for debt recovery
- User requests an interactive dashboard for portfolio analysis

## Quick Start: One Command to Run Everything

### Step 1: Run Full Pipeline

Execute `scripts/run_full_workflow.py` with the data file:

```bash
python scripts/run_full_workflow.py <path/to/data.xlsx>
```

Optional flags:
- `--output-dir <dir>`   : Custom output directory (default: `agent_outputs/baseline_comparison_run/`)
- `--config <config.json>`: Custom economic assumptions JSON

This single command:
1. Loads and preprocesses the data
2. Trains 4 models (Logistic Regression / Random Forest / XGBoost / MLP v2)
3. Calibrates probabilities via Platt scaling
4. Selects champion by expected net recovery + AUC
5. Scores test set and assigns collection queues
6. Outputs all CSV/JSON/model artifacts
7. Generates markdown strategy report

**Expected runtime**: 2–5 minutes depending on data size.

### Step 2: Generate Dashboard

After pipeline completes successfully:

```bash
python scripts/generate_dashboard.py
```

This reads all output files from Step 1 and produces `dashboard.html` — a self-contained interactive HTML dashboard with **9 tabs**: Overview, Models, Compare, Features, Stats, Queue, Tuning, Report, Suggestions.

Open `dashboard.html` in any modern browser. No server required.

## Standard Workflow (Detailed)

### Phase 1: Data Validation

Before running the pipeline, verify the input data meets requirements:

1. File format must be Excel (`.xlsx`)
2. Must contain column `data_type` with values `"M"` (training) and `"T"` (test)
3. Must contain target column `payer_3yr` with values `"Y"` / other
4. Recommended minimum: 1,000+ records; optimal: 10,000+
5. Expected positive rate: 5–20% (`payer_3yr=Y`)

For full schema details, read `references/data_schema.md`.

If validation fails, inform the user of the specific missing columns or data issues before proceeding.

### Phase 2: Model Training & Evaluation

The pipeline automatically trains these 4 candidate models:

| Model | Type | Strength |
|-------|------|----------|
| `baseline_logistic_regression` | Linear | Interpretable baseline, fast training |
| `balanced_random_forest` | Tree ensemble | Handles mixed features, robust |
| `xgboost` | Gradient boosting | Typically best accuracy, handles imbalance |
| `deep_mlp` (optional) | Neural network | Captures nonlinear interactions (requires PyTorch) |

**Training protocol**:
- Split M-set into Train (60%) / Calibration (15%) / Validation (25%) via stratified split
- T-set is held out as final test (never seen during training)
- All models use class-weighted/balanced loss for imbalanced data
- Probability calibration: Platt scaling (logistic regression on out-of-fold predictions)

**Model selection criteria** (in order):
1. Expected net recovery total on validation set
2. ROC-AUC
3. Recall for payer class

### Phase 3: Economic Optimization

After model selection, the champion model scores all accounts and assigns them to one of four action queues:

| Queue | Channel | Cost Tier | Use Case |
|-------|---------|-----------|----------|
| High Priority | Agent Call | High (~¥85/acct) | Top-tier recoverable accounts |
| Medium Priority | Auto-Dialer | Medium (~¥12/acct) | Mid-range volume coverage |
| Low Priority | SMS/Email | Low (~¥1.5/acct) | Digital-only outreach |
| Write-off / Ignore | None | Zero | Negative net recovery |

Queue assignment respects capacity constraints (configurable ratios per channel).

Default economic assumptions can be overridden via `--config config.json`. See `DEFAULT_PRODUCTION_CONFIG` in `scripts/pipeline.py`.

### Phase 4: Report Generation

The pipeline auto-generates:
- **collection_strategy_report.md** — Comprehensive markdown report (Chinese by default)
- **metrics.json** — Machine-readable all-metrics dump
- **champion_challenger_summary.csv** — Model comparison table
- **test_scored_accounts.csv** — Row-level predictions with queue assignments

### Phase 5: Dashboard Generation

Run `scripts/generate_dashboard.py` to produce the interactive dashboard. The script reads all output files from Phase 3 and renders:

| Tab | Content |
|-----|---------|
| **Overview** | KPI cards (AUC, Brier, Recall, Net Recovery, ROI), Dev Split pie chart, Confusion Matrix, Economic Assumptions |
| **Models** | Sortable model table (expandable detail rows), AUC bar chart, Economic comparison chart (Net Recovery + ROI) |
| **Compare** | Radar chart (6-dim normalized), AUC-vs-Brier scatter, Threshold sensitivity curves, Brier comparison bars, Precision-Recall scatter, Decision matrix table |
| **Features** | Feature importance bar chart (dropdown by model), Crossover tables, Payer rate drill-down, Business interpretation |
| **Stats** | Descriptive statistics table, Histogram distributions (7 numeric variables), Box plots |
| **Queue** | Strategy matrix pie chart, ROI-by-action bars, Cost vs Net Recovery scatter, Concentration metrics, Top 200 accounts table (searchable/exportable) |
| **Tuning** | Hyperparameter cards, MLP training sweep visualization |
| **Report** | Strategy report rendered as English brief |
| **Suggestions** | Actionable collection strategy recommendations |

The dashboard uses Chart.js for visualization, vanilla JS for interactivity, and requires no backend server.

## Customization

### Changing Economic Assumptions

Create a JSON file with overrides:

```json
{
  "economics": {
    "balance_recovery_rate": 0.35,
    "agent_call_cost": 85.0,
    "auto_dialer_cost": 12.0,
    "sms_email_cost": 1.5
  },
  "capacity": {
    "max_agent_ratio": 0.18,
    "max_auto_ratio": 0.42,
    "max_sms_ratio": 0.30
  }
}
```

Pass it to the pipeline: `python scripts/run_full_workflow.py data.xlsx --config my_config.json`

### Re-running Only Queue Optimization

If the model is already trained but economics changed, use the policy optimization function directly (via Python import):

```python
from pipeline import optimize_collection_policy
result = optimize_collection_policy(
    file_path='data.xlsx',
    model_path='output/npa_repayment_model.joblib',
    output_dir='output/',
    config_path='my_config.json'
)
```

### Regenerating Dashboard After Changes

Simply re-run `python scripts/generate_dashboard.py` after any output file changes. The dashboard generator reads fresh data each time.

## Output Rules

- Present results concisely, ROI-oriented, with actionable conclusions
- Distinguish clearly between raw scores, calibrated probabilities, and economic value
- Never expose personally identifiable information (`debtor_last`) in summaries
- Report grouped findings and operational recommendations
- State which economic assumptions drove queue recommendations when config was provided
- If PyTorch is unavailable, the MLP model will be skipped automatically; inform the user
