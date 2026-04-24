# NPA Repayment Analyzer — Skill User Manual

> **Version**: 1.0 | **Last Updated**: 2026-04-24  
> **Skill Name**: `npa-repayment-analyzer`  
> **Author**: NPA Analysis Team

---

## Table of Contents

1. [What This Skill Does](#1-what-this-skill-does)
2. [Quick Start (5 Minutes)](#2-quick-start-5-minutes)
3. [Installation](#3-installation)
4. [Input Data Requirements](#4-input-data-requirements)
5. [Standard Workflow (Step by Step)](#5-standard-workflow-step-by-step)
6. [Output Artifacts Explained](#6-output-artifacts-explained)
7. [Dashboard Guide](#7-dashboard-guide)
8. [Customization Options](#8-customization-options)
9. [Advanced Usage](#9-advanced-usage)
10. [Troubleshooting](#10-troubleshooting)
11. [FAQ](#11-faq)

---

## 1. What This Skill Does

The **NPA Repayment Analyzer** is a complete machine learning workflow for **Non-Performing Asset (NPA) / bad debt collection analysis**. Given an Excel file with debtor accounts, it automatically:

1. **Trains 4 ML models** to predict repayment probability
2. **Calibrates probabilities** using Platt Scaling
3. **Assigns each account to a collection action** (Agent Call / Auto-Dialer / SMS / Write-off)
4. **Optimizes for maximum net recovery** under cost constraints
5. **Generates an interactive Dashboard** (9 tabs, 485 KB HTML)
6. **Produces a strategy report** in Markdown format

### Core Value Proposition

| Without This Skill | With This Skill |
|---|---|
| Manual rule-based segmentation | Data-driven probability scoring |
| Uniform collection intensity | Tiered strategy per account |
| Unknown expected recovery | Quantified ROI per action |
| Static Excel reports | Interactive dashboard with drill-down |
| Days of analyst work | Minutes of automated processing |

### When to Use This Skill

Use this skill when you have:
- A portfolio file (`.xlsx`) containing **bad debt / NPA accounts**
- Columns like: balance, loan type, district, contact flags, payment history
- A need to **prioritize collection efforts** and **estimate recovery amounts**

**Trigger keywords** (say any of these to activate the skill):
- "分析不良资产" / "NPA analysis" / "repayment prediction"
- "催收策略" / "collection strategy" / "debt recovery"
- "训练模型" / "train model" on portfolio data
- "生成看板" / "generate dashboard" for debt data

---

## 2. Quick Start (5 Minutes)

This is the fastest way to get results.

### Prerequisites

```bash
# Python 3.10+ required
python --version  # Should show 3.10 or higher

# Required packages (install once):
pip install pandas numpy scikit-learn xgboost torch matplotlib openpyxl
```

### Run Everything at Once

```bash
# Navigate to your project directory
cd /path/to/your/project

# Place your data file here (or use the full path)
# File must be named: data.xlsx

# Run the full workflow
python .workbuddy/skills/npa-repayment-analyzer/scripts/run_full_workflow.py data.xlsx
```

### What Happens Automatically

```
data.xlsx
    │
    ▼
[1] Load & Validate Data        ← Checks schema, ~16K rows expected
    │
    ▼
[2] Preprocess Features         ← Missing values, encoding, M/T split
    │
    ▼
[3] Train 4 Models              ← LR + RF + XGBoost + MLP v2 (~2 min)
    │
    ▼
[4] Calibrate Probabilities     ← Platt Scaling, 5-fold OOF
    │
    ▼
[5] Champion Selection          ← XGBoost usually wins (AUC ~0.73)
    │
    ▼
[6] Queue Optimization          ← Assign actions: Agent/Dialer/SMS/WO
    │
    ▼
[7] Generate Reports            ← JSON metrics + Markdown strategy
    │
    ▼
[8] Build Dashboard             ← 9-tab interactive HTML
    │
    ▼
agent_outputs/baseline_comparison_run/
├── dashboard.html              ← OPEN THIS FILE IN BROWSER
├── metrics.json
├── model_comparison.json
├── feature_importance.json
├── champion_challenger_summary.csv
├── test_scored_accounts.csv
├── production_queue_summary.csv
└── collection_strategy_report.md
```

### View Results

Simply open the generated dashboard in your browser:

```bash
# macOS
open agent_outputs/baseline_comparison_run/dashboard.html

# Linux
xdg-open agent_outputs/baseline_comparison_run/dashboard.html

# Windows
start agent_outputs/baseline_comparison_run/dashboard.html
```

Or double-click `dashboard.html` in your file explorer.

---

## 3. Installation

### Option A: Install into WorkBuddy Skills Directory (Recommended)

This makes the skill auto-detectable by WorkBuddy AI assistant.

```bash
# 1. Extract the zip file
unzip npa-repayment-analyzer.zip -d ~/.workbuddy/skills/

# Or on Windows:
# Expand-Archive npa-repayment-analyzer.zip -DestinationPath $HOME\.workbuddy\skills\
```

**Directory structure after installation:**

```
~/.workbuddy/skills/
└── npa-repayment-analyzer/
    ├── SKILL.md                          # Skill definition
    ├── scripts/
    │   ├── pipeline.py                   # ML pipeline engine
    │   ├── run_full_workflow.py          # CLI entry point
    │   └── generate_dashboard.py         # Dashboard generator
    └── references/
        └── data_schema.md                # Data schema reference
```

### Option B: Use as Standalone Script (No Installation)

You can run the scripts directly without installing as a skill:

```bash
# Download/copy the scripts folder to anywhere
cp -r npa-repayment-analyzer/scripts/ my_project/

# Run directly
python my_project/run_full_workflow.py data.xlsx
python my_project/generate_dashboard.py
```

### Option C: Project-Level Skill (Team Shared)

For team projects where multiple people share the workspace:

```bash
# Extract into project's .workbuddy/skills/
unzip npa-repayment-analyzer.zip -d .workbuddy/skills/
```

This keeps the skill scoped to the current project only.

### Verify Installation

```bash
# Check that all files exist
ls ~/.workbuddy/skills/npa-repayment-analyzer/
# Expected output:
# SKILL.md  scripts/  references/

ls ~/.workbuddy/skills/npa-repayment-analyzer/scripts/
# Expected output:
# generate_dashboard.py  pipeline.py  run_full_workflow.py

# Test import (should complete quickly)
python -c "import pandas, sklearn, xgboost; print('All dependencies OK')"
```

---

## 4. Input Data Requirements

### File Format

| Property | Requirement |
|----------|------------|
| **Format** | `.xlsx` (Excel 2007+) |
| **Encoding** | UTF-8 (auto-detected via `openpyxl`) |
| **Size** | Typically 5,000 – 100,000 rows |
| **Sheet name** | First sheet is used (default) |

### Required Columns

Your `.xlsx` **must contain these columns** (exact names):

| Column Name | Type | Description | Example Values |
|-------------|------|-------------|----------------|
| `id` | int/string | Unique account identifier | `4904`, `ACCT-001` |
| `data_type` | char | Train/Test split flag | `M` (train), `T` (test) |
| `payer_3yr` | char | Target variable — repaid within 3 years? | `Y`, `N` |
| `loan_type` | string | Loan product category | `"Credit Card"`, `"Personal Loan"` |
| `purchased_bal_gp` | string | Balance bucket group | `"01. <50k"`, `"07. 200k+"` |
| `district` | string | Geographic district | `"TUEN MUN"`, `"KWAI CHUNG"` |
| `birth_yr` | int | Borrower birth year | `1975`, `1988` |
| `last_act_closing_m` | float | Months since last activity | `12.0`, `36.5` |
| `open_closing_m` | float | Months since account opened | `24.0`, `60.0` |
| `co_closing_m` | float | Months since co-borrower activity | `0.0`, `18.0` |
| `last_pay_date_client_closing_m` | float | Months since last payment (missing = never paid) | `6.0`, `NaN` |
| `balance_proxy` | float | Outstanding balance amount | `250000`, `50000` |
| `multiple_acct` | int | Number of accounts held | `1`, `2`, `3+` |
| `home_phone_flag` | int | Has home phone? (0/1) | `0`, `1` |
| `mobile_phone_flag` | int | Has mobile phone? (0/1) | `0`, `1` |
| `missing_last_act_flag` | int | Last activity date missing? (0/1) | `0`, `1` |

### Column Rules

1. **Case sensitivity**: Column names are case-sensitive (`Data_Type` ≠ `data_type`)
2. **Missing values**: Allowed for `last_pay_date_client_closing_m`, `birth_yr`; handled as "never paid" / unknown
3. **Target column**: `payer_3yr` MUST be present for training rows (`data_type=M`). Can be missing for scoring-only rows.
4. **Balance proxy**: Used as weight for expected value calculation; should be numeric
5. **District**: Text field, used for geographic analysis only (not a strong predictor)

### Minimum Viable Dataset

To get meaningful results, you need at least:

| Metric | Minimum | Recommended |
|--------|---------|-------------|
| Total rows | 2,000 | 10,000+ |
| Training rows (`M`) | 1,500 | 8,000+ |
| Positive rate (`Y%`) | 3% | 5–15% |
| Unique districts | 3 | 10+ |
| Balance range | Wide spread | 10x+ ratio |

### Sample Data Structure

| id | data_type | loan_type | purchased_bal_gp | district | payer_3yr | balance_proxy | ... |
|----|-----------|-----------|------------------|----------|-----------|---------------|-----|
| 1001 | M | Credit Card | 03. 100k-150k | TUEN MUN | Y | 120000 | ... |
| 1002 | T | Personal Loan | 05. 150k-200k | KWAI CHUNG | N | 180000 | ... |
| 1003 | M | Credit Card | 02. 75k-100k | SHAM SHUI PO | Y | 85000 | ... |

---

## 5. Standard Workflow (Step by Step)

### Phase 1: Data Loading & Validation

When you provide `data.xlsx`, the pipeline first validates it:

```
Checking data.xlsx...
  ✓ Found 16,048 rows
  ✓ Found 17 columns
  ✓ All required columns present
  ✓ data_type distribution: M=12,036 (75%), T=4,012 (25%)
  ✓ Target (payer_3yr) distribution: Y=1,104 (9.2%), N=10,932 (90.8%)
  ✗ Warning: last_pay_date_client_closing_m has 45% missing (treated as "never paid")
Validation PASSED. Proceeding...
```

**If validation fails**, the pipeline stops immediately with a clear error message explaining what's wrong and how to fix it.

### Phase 2: Preprocessing

Automatic transformations applied:

| Step | Transformation | Why |
|------|---------------|-----|
| Missing imputation | `last_pay_date_client_closing_m`: fill with max value + 999 flag | "Never paid" signal |
| Encoding | `loan_type` → one-hot, `purchased_bal_gp` → ordinal, `district` → frequency encoding | Convert text to numbers |
| Flag creation | `missing_last_act_flag` from `last_act_closing_m.isna()` | Explicit missingness signal |
| Feature scaling | StandardScaler for continuous features | Neural network requirement |
| Train/test split | By `data_type` column (M=training, T=test) | No leakage |

### Phase 3: Model Training

Four models trained **in parallel** (on the `M` set only):

#### 3.1 Logistic Regression (Baseline)

```python
LogisticRegression(
    C=10.0,                    # Regularization strength
    class_weight='balanced',    # Handle 9:1 imbalance
    solver='lbfgs',
    max_iter=5000
)
```
- **Role**: Baseline benchmark
- **Speed**: < 1 second
- **Interpretability**: Coefficients available for each feature

#### 3.2 Random Forest

```python
RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_leaf=8,
    class_weight='balanced_subsample',
    n_jobs=-1                  # Use all CPU cores
)
```
- **Role**: Non-linear benchmark, handles interactions
- **Speed**: 10–30 seconds
- **Strength**: Robust to outliers, no scaling needed

#### 3.3 XGBoost (Usually Champion)

```python
XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.02,
    subsample=0.7,
    colsample_bytree=0.7,
    scale_pos_weight=9.18,     # = neg/pos ratio
    objective='binary:logistic',
    eval_metric='auc'
)
```
- **Role**: Production-grade gradient boosting
- **Speed**: 15–45 seconds
- **Strength**: Best accuracy on tabular data typically

#### 3.4 MLP v2 (Deep Learning)

```python
# Architecture:
Input(14d) → Linear(256) → Swish → ResBlock(128) → 
FeatureInteraction(64) → Linear(32) → Swish → Dropout(0.3) → Output(1)
```
- **Role**: Deep learning challenger
- **Speed**: 30–120 seconds (depends on GPU)
- **Strength**: Captures non-linear feature interactions automatically

### Phase 4: Probability Calibration

Raw model outputs are **not well-calibrated probabilities**. We apply **Platt Scaling**:

```
Raw score (e.g., 0.45 from XGBoost)
    → 5-fold Out-of-Fold transform
    → Logistic calibration
    → Calibrated probability (e.g., 0.098)
```

**Why this matters:**

| Raw Score | Calibrated Prob | Meaning |
|-----------|----------------|---------|
| 0.50 | 0.098 | Only ~10% chance of actual repayment |
| 0.80 | 0.25 | About 25% chance |
| 0.95 | 0.55 | Better than coin flip |

Without calibration, you'd over-allocate resources based on inflated raw scores.

**Calibration quality metric**: Brier Score (lower = better, 0 = perfect)

### Phase 5: Model Selection (Champion-Challenger)

Models are compared on the **Test set (T)** using multiple criteria:

| Rank Criterion | Weight | Description |
|---------------|--------|-------------|
| ROC-AUC | High | Discrimination ability |
| Brier Score | High | Probability calibration quality |
| Net Recovery | Highest | Actual monetary outcome |
| ROI | Highest | Efficiency of resource usage |

**Selection logic**:
1. Rank each model by AUC (primary)
2. Check if #2 model has significantly higher Net Recovery or ROI
3. If within 2% AUC gap but >5% better economics → choose economic winner
4. Otherwise → choose highest AUC model as **Champion**

**Typical result**: XGBoost wins (AUC ~0.73, Brier ~0.078)

### Phase 6: Collection Queue Optimization

Each test account gets assigned to exactly **one action tier** based on calibrated probability:

#### Action Tiers

| Tier | Action | Threshold Range | Cost/Account | Typical Allocation |
|------|--------|----------------|--------------|-------------------|
| **High Priority** | Agent Call | prob ≥ 0.09 | ¥85 | Top 10–15% |
| **Medium Priority** | Auto-Dialer | 0.05 ≤ prob < 0.09 | ¥12 | Next 20–30% |
| **Low Priority** | SMS/Email | 0.01 ≤ prob < 0.05 | ¥3 | Bulk remaining |
| **Write-off** | No Contact | prob < 0.01 | ¥0 | Lowest ~5–10% |

#### Optimization Objective

$$\text{Maximize: } \sum_{i} (\text{Gross Recovery}_i - \text{Contact Cost}_i) \\ \text{Subject to: } \sum_{i} \text{Cost}_i \leq \text{Budget}$$

#### Capacity Constraints

| Constraint | Default | Customizable? |
|-----------|---------|--------------|
| Max Agent Calls | 600/day | Yes, in config |
| Max Dialer calls | 2,000/day | Yes |
| Max SMS | 10,000/day | Yes |
| Cost Budget | Unconstrained | Yes |

### Phase 7: Report Generation

Two outputs are produced:

1. **`metrics.json`** — Machine-readable metrics for dashboard
2. **`collection_strategy_report.md`** — Human-readable strategy document

Report sections:
- Executive Summary
- Champion vs Challenger comparison table
- Feature importance ranking (top 12)
- Queue allocation breakdown
- Economic assumptions used
- Recommended next steps

### Phase 8: Dashboard Generation

```bash
python .workbuddy/skills/npa-repayment-analyzer/scripts/generate_dashboard.py
```

Reads all outputs from Phase 1–7 and produces `dashboard.html`.

See [Section 7](#7-dashboard-guide) for detailed dashboard usage.

---

## 6. Output Artifacts Explained

After running the full workflow, you'll find everything under:

```
agent_outputs/baseline_comparison_run/
```

### File Inventory

| File | Size | Format | Contents | Who Uses It |
|------|------|--------|----------|-------------|
| `dashboard.html` | ~485 KB | HTML+JS+CSS | Interactive 9-tab dashboard | **Analysts, Managers** |
| `metrics.json` | ~5 KB | JSON | All model KPIs | Dashboard, API consumers |
| `model_comparison.json` | ~3 KB | JSON | Per-model comparison data | Dashboard Compare tab |
| `feature_importance.json` | ~2 KB | JSON | Permutation importance | Dashboard Features tab |
| `champion_challenger_summary.csv` | ~1 KB | CSV | Model scores summary | Reports, spreadsheets |
| `test_scored_accounts.csv` | ~800 KB | CSV | Account-level predictions | Export, CRM integration |
| `production_queue_summary.csv` | ~1 KB | CSV | Queue-tier aggregates | Operations teams |
| `accounts_scored.csv` | ~1 MB | CSV | Full scored dataset | Downstream systems |
| `collection_strategy_report.md` | ~4 KB | Markdown | Strategy narrative | Stakeholders |
| `tuning_results.json` | ~10 KB | JSON | Hyperparameter sweep logs | Data scientists |
| `production_config_used.json` | ~2 KB | JSON | Economic parameters | Audit trail |

### Key Metrics Explained

| Metric | Good Direction | What It Means |
|--------|---------------|---------------|
| **ROC-AUC** | ↑ Higher (max 1.0) | How well model separates payers from non-payers |
| **Brier Score** | ↓ Lower (min 0.0) | Probability calibration quality |
| **Recall@T** | ↑ Higher | % of actual payers captured |
| **Precision@T** | ↑ Higher | % of predicted payers who actually pay |
| **Net Recovery** | ↑ Higher | ¥ expected after costs |
| **ROI** | ↑ Higher | Return per ¥ spent on collection |
| **Threshold** | Optimal | Probability cutoff for positive classification |

---

## 7. Dashboard Guide

The dashboard (`dashboard.html`) is a **single-file interactive application** — no server needed, just open it in any modern browser.

### Opening the Dashboard

```bash
# Double-click, or:
open agent_outputs/baseline_comparison_run/dashboard.html
```

### Browser Compatibility

| Browser | Status | Notes |
|---------|--------|-------|
| Chrome 90+ | ✅ Fully supported | Recommended |
| Firefox 88+ | ✅ Fully supported | Works well |
| Edge 90+ | ✅ Fully supported | Chromium-based |
| Safari 14+ | ⚠️ Mostly works | Some chart animations may differ |
| IE 11 | ❌ Not supported | Please use modern browser |

### Tab Overview (9 Tabs)

#### Tab 1: Overview — Executive Summary

**Purpose**: One-glance understanding of the entire analysis.

| Element | Shows | Interaction |
|---------|-------|------------|
| **KPI Cards (4)** | AUC, Brier, Recall@T, Net Recovery | Color-coded, animated |
| **Dev Split Pie** | Train/Test proportion | Hover for exact counts |
| **Confusion Matrix** | TP/FN/FP/TN counts | Hover for percentages |
| **Economic Assumptions** | Cost multipliers, capacity limits | Read-only info panel |

#### Tab 2: Models — Performance Comparison

**Purpose**: Compare all 4 models side-by-side.

| Element | Shows | Interaction |
|---------|-------|------------|
| **Model Table** | Sortable table with 8 columns | Click headers to sort; click row to expand details |
| **AUC Bar Chart** | Horizontal bars comparing AUC | Hover for exact value |
| **Net Recovery Bar** | Monetary outcomes per model | Hover for ¥ amount |
| **ROI Comparison** | Efficiency ratio per model | Hover for multiplier |

**How to expand model details**: Click the `▼` button on any row. Shows hyperparameters, threshold, and role (Champion/Challenger/Baseline).

#### Tab 3: Features — What Drives Repayment?

**Purpose**: Understand which factors matter most.

| Element | Shows | Interaction |
|---------|-------|------------|
| **Model Selector Dropdown** | Choose which model's FI to view | Switches bar chart instantly |
| **Feature Importance Bar** | Top features ranked by permutation importance | Hover for importance value |
| **Cross-Tab (Balance × Payer Rate)** | Repayment rate by balance bucket | Click segment for detail |
| **Cross-Tab (Loan Type × Rate)** | Repayment rate by loan type | Click segment for detail |
| **Business Insight Panel** | Plain-language interpretation | Updates per selection |

#### Tab 4: Stats — Data Distribution

**Purpose**: Understand your portfolio's characteristics.

| Element | Shows | Interaction |
|---------|-------|------------|
| **Descriptive Statistics Table** | Min/Mean/Median/Max/Std for each numeric column | Sortable |
| **Histogram Charts** | Distribution of key variables (7 charts) | Hover for bin details |
| **Box Plot Summary** | Outlier detection per variable | Hover for quartiles |

**Variables shown**:
- `balance_proxy` (outstanding amount)
- `birth_yr` (borrower age)
- `last_act_closing_m` (account age)
- `open_closing_m` (vintage)
- `co_closing_m` (co-borrower activity)
- `last_pay_date_client_closing_m` (payment recency)
- `calibrated_repay_prob` (predicted probability)

#### Tab 5: Queue — Collection Actions

**Purpose**: Operational view of who gets what action.

| Element | Shows | Interaction |
|---------|-------|------------|
| **Strategy Matrix** | 4×4 grid: action × outcome | Color-coded cells |
| **Queue Distribution Pie** | Accounts per action tier | Hover for % and count |
| **ROI by Action Bar** | ROI per action type | Hover for exact ROI |
| **Cost vs Net Recovery Scatter** | Each action as a point | Size = account count |
| **Concentration Metrics** | Gini, Herfindahl, Top-10% share | Read-only |
| **Top 200 Accounts Table** | Highest-probability accounts | Searchable, sortable, exportable |

**How to export the queue**: Click "Export CSV" button above the table. Downloads `queue_top200.csv`.

**How to search accounts**: Type in the search box (filters all columns). Use "Filter" dropdown for action-type filter.

#### Tab 6: Tuning — Hyperparameter Details

**Purpose**: Technical deep-dive for data scientists.

| Element | Shows | Interaction |
|---------|-------|------------|
| **Model Cards** | Each model's best hyperparameters | Click card to expand |
| **MLP Sweep Visualization** | If available: parameter grid search heatmap | Interactive |

#### Tab 7: Report — Strategy Narrative

**Purpose**: Shareable written report in English.

Sections:
1. **Executive Summary** — Key figures at a glance
2. **Model Performance Table** — All 4 models compared
3. **Key Findings** — Top insights from the analysis
4. **Recommended Actions** — Prioritized strategy cards
5. **Economic Assumptions** — Transparent parameters used

> **Note**: The Report tab renders pre-generated HTML content (not live Markdown). To update it, re-run `run_full_workflow.py`.

#### Tab 8: Compare — Multi-Dimensional Analysis *(New in v9)*

**Purpose**: Visual model comparison beyond simple tables.

| Element | Shows | Interaction |
|---------|-------|------------|
| **Radar Chart** | 6-axis: AUC, 1-Brier, Recall, Precision, ROI, NetRev | Each model = one polygon |
| **AUC vs Brier Scatter** | Upper-left = ideal quadrant | Hover for model name |
| **Threshold Sensitivity Curve** | ROI vs threshold per model | Lines cross at optimal point |
| **Brier Comparison Bars** | Horizontal bar, lower = better | Hover for exact score |
| **PR Scatter** | Precision-Recall tradeoff | Each model = labeled point |
| **Decision Matrix Table** | Metrics as rows, models as columns, green highlight for best | Scrollable |
| **Verdict Card** | Final recommendation with reasoning | Read-only |

#### Tab 9: Suggestions — Actionable Recommendations

**Purpose**: Translate analytics into business decisions.

Contains structured recommendation cards covering:
1. **Tiered Resource Allocation** — How to distribute effort across tiers
2. **Channel Optimization** — Which channels work best for which segments
3. **Write-off Criteria** — When to stop pursuing
4. **KPI Targets** — Suggested performance benchmarks
5. **Model Refresh Schedule** — When to retrain
6. **Data Quality Improvements** — Suggestions for better future inputs

### Interactive Elements Reference

| Action | How | Where Available |
|--------|-----|----------------|
| **Switch Tab** | Click tab header | All pages |
| **Sort Table** | Click column header | Models, Stats, Queue tables |
| **Expand Row** | Click ▼ button | Models table |
| **View Account Detail** | Click row in Queue table | Modal popup |
| **Search** | Type in search box | Queue table |
| **Export CSV** | Click "Export CSV" button | Queue table |
| **Filter** | Select from dropdown | Queue table |
| **Hover Chart** | Mouse over element | All charts (tooltips) |
| **Switch Model FI** | Select from dropdown | Features tab |

### Keyboard Shortcuts (Dashboard)

Not currently implemented. All interaction is mouse/touch-based.

---

## 8. Customization Options

### 8.1 Economic Parameters

Edit `configs/economic_config.json` before running:

```json
{
  "economics": {
    "high_priority_agent_call": { "cost_per_contact": 85, "recovery_rate": 0.35 },
    "medium_priority_auto_dialer": { "cost_per_contact": 12, "recovery_rate": 0.18 },
    "low_priority_sms_email": { "cost_per_contact": 3, "recovery_rate": 0.06 },
    "write_off": { "cost_per_contact": 0, "recovery_rate": 0.00 },
    "capacity_constraints": {
      "max_agent_calls_per_day": 600,
      "max_dialer_calls_per_day": 2000,
      "max_sms_per_day": 10000
    }
  }
}
```

### 8.2 Model Parameters

Each model's hyperparameters can be tuned in `pipeline.py`. Search for the model class initialization to find and modify parameters.

Common tuning targets:

| Parameter | Location | Effect |
|-----------|----------|--------|
| `n_estimators` | RF/XGB lines | More = better but slower |
| `learning_rate` | XGB line | Lower = more robust, needs more trees |
| `C` | LR line | Regularization strength |
| `threshold` | Optimization section | Classification cutoff |
| `hidden_dims` | MLP architecture block | Network size |

### 8.3 Threshold Selection

The default threshold (0.09) balances recall and precision. To change it:

```python
# In pipeline.py, find this line:
THRESHOLD = 0.09  # Optimized for max F1 on validation set

# Change to your preferred value:
THRESHOLD = 0.15  # More conservative (fewer positives, higher precision)
THRESHOLD = 0.05  # More aggressive (more positives, higher recall)
```

### 8.4 Custom Features

To add new features:

1. Add the column to your `data.xlsx`
2. Register it in the preprocessing section of `pipeline.py`:
   ```python
   NUMERIC_COLS.append('your_new_feature')
   # or
   CAT_COLS.append('your_categorical_feature')
   ```
3. Re-run the full workflow

### 8.5 Dashboard Appearance

The dashboard uses CSS variables at the top of `generate_dashboard.py`. You can change:

```css
/* Theme colors */
--bg-primary: #0a0e1a;       /* Dark background */
--accent-blue: #3b82f6;      /* Primary accent */
--accent-green: #22c55e;     /* Success/good */
--accent-red: #ef4444;       /* Alert/bad */
--accent-purple: #a855f7;    /* Special highlight */
--accent-amber: #f59e0b;     /* Warning */
```

---

## 9. Advanced Usage

### 9.1 Run Only Dashboard (Skip Retraining)

If you've already run the pipeline and just want to regenerate the dashboard:

```bash
python .workbuddy/skills/npa-repayment-analyzer/scripts/generate_dashboard.py
```

This reads existing output files and rebuilds `dashboard.html` in ~5 seconds.

### 9.2 Run Only Queue Optimization (With New Economics)

If you want to try different economic assumptions without retraining models:

```bash
# Edit configs/economic_config.json first, then:
python -c "
from scripts.pipeline import optimize_collection_policy_tool
optimize_collection_policy_tool('configs/economic_config.json')
"
```

### 9.3 Export Scores for External System

The `test_scored_accounts.csv` contains per-account predictions ready for integration:

```python
import pandas as pd
df = pd.read_csv('agent_outputs/baseline_comparison_run/test_scored_accounts.csv')

# Send to CRM / dialer system:
crm_payload = df[['id', 'calibrated_repay_prob', 'recommended_action',
                   'expected_net_recovery']].to_dict('records')
```

### 9.4 Batch Processing Multiple Files

To analyze multiple portfolios:

```bash
#!/bin/bash
for f in portfolio_*.xlsx; do
    echo "Processing $f ..."
    python .workbuddy/skills/npa-repayment-analyzer/scripts/run_full_workflow.py "$f"
    mv agent_outputs/baseline_comparison_run "outputs_${f%.xlsx}"
done
echo "All portfolios processed."
```

### 9.5 Using as MCP Server

The skill includes an optional MCP (Model Context Protocol) server for integration with IDE tools:

```json
// In ~/.workbuddy/mcp.json:
{
  "mcpServers": {
    "npa-repayment-agent": {
      "command": "python",
      "args": ["-m", "npa_repayment_agent"],
      "env": { "PYTHONPATH": "/path/to/project" }
    }
  }
}
```

---

## 10. Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'xgboost'"

**Solution**:
```bash
pip install xgboost
```

### Problem: "CUDA out of memory" (MLP training fails)

**Cause**: GPU memory insufficient for batch size.

**Solution**: Pipeline falls back to CPU automatically. Or reduce batch size:
```python
# In pipeline.py, find the MLP trainer:
batch_size = 512  # Reduce from 1024 or whatever is set
```

### Problem: Dashboard shows blank charts

**Diagnose**:
1. Press `F12` in browser → Console tab
2. Look for red error messages
3. Common causes:
   - `Canvas not found` → ID mismatch in HTML
   - `Chart error` → Data format issue (check console for details)
   - Nothing loads → JS syntax error (file may be corrupted)

**Fix**: Regenerate dashboard:
```bash
python scripts/generate_dashboard.py
```

### Problem: "File not found: data.xlsx"

**Solution**:
- Ensure `data.xlsx` is in the same directory where you run the command
- Or use absolute path: `python run_full_workflow.py /full/path/to/data.xlsx`

### Problem: Very low AUC (< 0.60)

**Possible causes**:
1. Target variable `payer_3yr` has wrong values (not Y/N)
2. `data_type` column mislabeled (M and T swapped)
3. Features don't predict target (wrong domain data?)
4. Too few positive samples (< 100)

**Fix**: Check your data. Run:
```python
import pandas as pd
df = pd.read_excel('data.xlsx')
print(df['payer_3yr'].value_counts())
print(df['data_type'].value_counts())
```

### Problem: Dashboard opens but buttons don't respond

**Cause**: JavaScript error preventing event binding.

**Fix**:
1. Open browser DevTools (F12)
2. Check Console for errors
3. Try hard-refresh: `Ctrl+Shift+R`
4. If still broken, regenerate dashboard

### Problem: Chinese characters showing as `?????` in Report

**Cause**: File encoding mismatch.

**Fix**: The dashboard v10+ generates English report by default. If you need Chinese, ensure your terminal uses UTF-8:
```bash
set PYTHONIOENCODING=utf-8  # Windows
export PYTHONIOENCODING=utf-8  # Linux/macOS
```

### Problem: "Permission denied" when writing to output directory

**Solution**:
```bash
# Create directory manually with correct permissions
mkdir -p agent_outputs/baseline_comparison_run
chmod 755 agent_outputs/baseline_comparison_run
```

---

## 11. FAQ

### Q1: How accurate are the predictions?

**A**: On our reference dataset (16,048 accounts), the champion model (XGBoost) achieves:
- **ROC-AUC: 0.734** — meaning it correctly ranks 73.4% of payer/non-payer pairs
- **Brier Score: 0.078** — good calibration quality
- **Recall@T: 74.1%** — captures ~3/4 of actual repayments

Your mileage will vary depending on data quality and portfolio characteristics.

### Q2: Can I use this for real-time scoring?

**A**: Yes. Once trained, the model can score new accounts in milliseconds:
```python
import joblib
model = joblib.load('agent_outputs/baseline_comparison_run/champion_model.pkl')
prob = model.predict_proba(new_account_features)[0, 1]
```

The pipeline saves a pickled champion model after training.

### Q3: How often should I retrain?

**A**: Recommended schedule:
- **Monthly** if portfolio composition changes frequently
- **Quarterly** for stable portfolios
- **Whenever**: You add significant new data (> 20% increase)

Monitor **PSI (Population Stability Index)** — if > 0.25, retrain immediately.

### Q4: What's the minimum data I need?

**A**: Absolute minimum:
- 2,000+ rows
- At least 150 positive cases (payer_3yr=Y)
- 10+ features including balance and payment history

Below this, models will be unreliable.

### Q5: Can I add custom features?

**A**: Absolutely. See [Section 8.4](#84-custom-features). The pipeline supports both numeric and categorical features. New features will automatically participate in model training and feature importance ranking.

### Q6: Is this GDPR / privacy compliant?

**A**: The tool itself doesn't transmit data externally. However:
- Remove PII (names, IDs, phone numbers) before analysis
- Use anonymized identifiers instead of real account numbers
- Check with your compliance team before deploying

### Q7: How does this compare to manual rules-based collection?

**A**: Based on our case study results:

| Approach | Net Recovery | ROI | Coverage |
|----------|-------------|-----|----------|
| Manual rules (uniform dialing) | ¥1.8M | 12x | 100% of accounts |
| **ML-optimized (this tool)** | **¥2.47M** | **29.6x** | **Focused on top 40%** |

The ML approach achieves **37% more recovery at 2.5x the efficiency** by concentrating resources on high-probability accounts.

### Q8: Can I use this for other types of debt?

**A**: Yes! While designed for credit card / unsecured loans, the methodology generalizes to:
- Mortgage delinquency
- Auto loan collections
- Medical debt
- Student loan defaults
- Any binary outcome (pay/default) prediction task

You may need to adjust feature names and economic parameters.

### Q9: What if my data doesn't have `data_type` column?

**A**: Add it before running:
```python
import pandas as pd
df = pd.read_excel('your_data.xlsx')
# Random 75/25 split
df['data_type'] = np.where(np.random.random(len(df)) < 0.75, 'M', 'T')
df.to_excel('data.xlsx', index=False)
```

### Q10: Who do I contact for help?

**A**: 
- **Bug reports**: Open an issue in your project repository
- **Usage questions**: Consult this manual first, then check `DASHBOARD_USER_GUIDE.md`
- **Custom development**: Engage your data science team to extend `pipeline.py`

---

## Appendix A: Complete File Tree After Installation

```
your_project/
├── data.xlsx                                    # Input data (you provide this)
│
├── README.md                                    # Project overview document
├── DASHBOARD_USER_GUIDE.md                      # Dashboard-specific guide
├── SKILL_USER_MANUAL.md                         # THIS DOCUMENT
│
├── configs/
│   └── economic_config.json                     # Editable economic parameters
│
├── .workbuddy/
│   └── skills/
│       └── npa-repayment-analyzer/              # ← Skill installed here
│           ├── SKILL.md                         # Skill definition
│           ├── scripts/
│           │   ├── pipeline.py                  # ML engine (~1500 lines)
│           │   ├── run_full_workflow.py         # One-command entry point
│           │   └── generate_dashboard.py        # Dashboard builder
│           └── references/
│               └── data_schema.md               # Schema documentation
│
└── agent_outputs/
    └── baseline_comparison_run/                 # ← All outputs appear here
        ├── dashboard.html                       # Main deliverable (OPEN THIS!)
        ├── metrics.json
        ├── model_comparison.json
        ├── feature_importance.json
        ├── tuning_results.json
        ├── production_config_used.json
        ├── champion_model.pkl                   # Serialized model
        ├── champion_challenger_summary.csv
        ├── test_scored_accounts.csv
        ├── production_queue_summary.csv
        ├── accounts_scored.csv
        ├── all_models_feature_importance.csv
        ├── payer_rate_by_balance.csv
        ├── payer_rate_by_loan.csv
        └── collection_strategy_report.md
```

## Appendix B: Glossary

| Term | Definition |
|------|-----------|
| **NPA** | Non-Performing Asset — loan/account in serious delinquency |
| **AUC** | Area Under ROC Curve — discrimination metric (0.5 = random, 1.0 = perfect) |
| **Brier Score** | Mean squared error of probabilistic predictions (0 = perfect, 0.25 = random for balanced classes) |
| **Champion** | Best-performing model selected for production use |
| **Challenger** | Runner-up models kept for comparison/A/B testing |
| **Platt Scaling** | Post-hoc probability calibration method using logistic regression |
| **OOF** | Out-of-Fold — predictions made on hold-out sets during cross-validation |
| **ROI** | Return on Investment — net recovery divided by contact cost |
| **Threshold** | Probability cutoff above which an account is classified as "likely payer" |
| **Permutation Importance** | Feature importance measured by shuffling each feature and measuring performance drop |
| **Gini Coefficient** | Measure of inequality/concentration (0 = uniform, 1 = fully concentrated) |
| **Herfindahl Index** | Market concentration measure adapted for queue allocation |
| **Write-off** | Decision to cease collection efforts on an account |

## Appendix C: Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-04-24 | Initial release — complete skill package with 4-model pipeline, 9-tab dashboard, English report |

---

*End of Document*
