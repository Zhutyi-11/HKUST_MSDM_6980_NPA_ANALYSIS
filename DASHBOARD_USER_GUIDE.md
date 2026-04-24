# Dashboard User Guide

**NPA Repayment Analysis Interactive Dashboard — Version 10**

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Dashboard Structure](#2-dashboard-structure)
3. [Tab-by-Tab Guide](#3-tab-by-tab-guide)
4. [Interactive Features](#4-interactive-features)
5. [Data Sources & Refresh](#5-data-sources--refresh)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Quick Start

### How to Open

The dashboard is a **self-contained HTML file** (no server required):

```bash
# Navigate to the output directory
cd c:\Users\marcozhu\Desktop\6980\agent_outputs\baseline_comparison_run

# Double-click or open in browser
dashboard.html
```

Alternatively, use the preview URL:
```
file:///c:/Users/marcozhu/Desktop/6980/agent_outputs/baseline_comparison_run/dashboard.html
```

### Browser Compatibility

| Browser | Status | Notes |
|---------|--------|-------|
| Chrome 90+ | ✅ Full support | Recommended |
| Firefox 88+ | ✅ Full support | |
| Edge 90+ | ✅ Full support | Chromium-based |
| Safari 14+ | ⚠️ Partial | Some animations may differ |

> **Note:** JavaScript must be enabled. The dashboard uses Chart.js v4 for all visualizations.

### Performance

- File size: ~484 KB (single HTML file, no external dependencies except CDN-hosted Chart.js)
- Initial load: < 2 seconds on modern hardware
- All data embedded as JSON in the page source (no API calls needed)

---

## 2. Dashboard Structure

### Layout Overview

```
+----------------------------------------------------------+
|  HEADER: NPA Repayment Analysis | Version Badge           |
+----------------------------------------------------------+
|  TAB BAR: Overview | Models | Compare | Features |      |
|            Stats | Queue | Tuning | Report                  |
+----------------------------------------------------------+
|                                                          |
|  ACTIVE TAB CONTENT                                      |
|  - KPI Cards / Charts / Tables                           |
|                                                          |
+----------------------------------------------------------+
|  FOOTER: Timestamp + Data Source                         |
+----------------------------------------------------------+
```

### Tab Summary (9 Tabs)

| # | Tab Name | Primary Purpose | Key Visualizations |
|---|----------|----------------|--------------------|
| 1 | **Overview** | Executive KPIs at a glance | Confusion Matrix, Dev Split, Economic Assumptions |
| 2 | **Models** | Model performance comparison | Sortable table, AUC bar chart, Economic metrics |
| 3 | **Compare** | Multi-dimensional model analysis | Radar chart, Scatter plots, Decision matrix |
| 4 | **Features** | Feature importance & drill-down | Per-model FI bar chart, Crosstabs, Payer rates |
| 5 | **Stats** | Data distribution analysis | Histograms, Box plots, Descriptive stats table |
| 6 | **Queue** | Collection action queue | Strategy matrix, ROI by action, Top 200 accounts |
| 7 | **Tuning** | Hyperparameter search results | Parameter cards, MLP sweep visualization |
| 8 | **Report** | English executive summary | Model comparison, Key findings, Recommendations |
| 9 | *(Suggestions)* | Collection strategy advice | Resource allocation, Channel optimization |

---

## 3. Tab-by-Tab Guide

### 3.1 Overview Tab

**Purpose:** High-level summary for executives and stakeholders.

#### KPI Cards (Top Row)

| Card | Metric | What it tells you |
|------|--------|-------------------|
| **ROC-AUC** | Discrimination ability | How well the model separates payers vs non-payers (0.5 = random, 1.0 = perfect) |
| **Brier Score** | Calibration quality | Probability prediction accuracy (lower = better; 0.08 is good for 9% base rate) |
| **Recall @ Threshold** | Payer capture rate | % of actual payers correctly identified at decision threshold |
| **Precision @ Threshold** | Prediction accuracy | Of predicted payers, what % actually paid |
| **Net Recovery (Val)** | Expected money recovered | HKD amount after contact costs (primary business metric) |
| **ROI** | Return on investment | Net recovery divided by total contact cost |

#### Charts

| Chart | Type | Interpretation |
|-------|------|---------------|
| **Dev Split Pie** | Doughnut | Training (M) vs Test (T) set sizes (~75%/25%) |
| **Confusion Matrix** | Heatmap | TP/FP/TN/FN counts; darker = more cases |
| **Economic Assumptions** | Info cards | Cost parameters used for ROI calculation |

---

### 3.2 Models Tab

**Purpose:** Compare all candidate models side-by-side.

#### Sortable Table

Click any column header to sort:

| Column | Description |
|--------|-------------|
| **Model** | Algorithm name + role badge (Champion/Baseline/Challenger) |
| **AUC** | ROC-AUC on validation set |
| **Brier** | Brier Score (calibration) |
| **LogLoss** | Logarithmic Loss |
| **Recall** | Sensitivity at optimal threshold |
| **Precision** | Positive predictive value |
| **Net Recovery** | Expected net recovery (HKD) |
| **ROI** | Return on investment ratio |
| **Threshold** | Decision threshold used |

**Expand rows** by clicking the `+` button on each row to reveal detailed hyperparameters.

#### AUC Bar Chart

Horizontal bar chart comparing AUC across models. Higher and further right = better.

#### Net Recovery & ROI Comparison

Two bar charts showing:
- **Expected Net Recovery** (absolute HKD amounts)
- **ROI** (ratio: net recovery / contact cost)

These are the **business-critical metrics** that drive collection strategy decisions.

---

### 3.3 Compare Tab

**Purpose:** Deep multi-model comparison with advanced visualizations.

### Radar Chart

Six-axis radar comparing normalized performance:

| Axis | Meaning |
|------|---------|
| AUC | Discrimination (higher = better) |
| 1-Brier | Calibration accuracy (higher = better) |
| Recall | Coverage of actual payers (higher = better) |
| Precision | Accuracy of positive predictions (higher = better) |
| ROI | Cost-efficiency (higher = better) |
| Net Rev | Absolute recovery amount (higher = better) |

**Interpretation:** Larger polygon area = more balanced, stronger overall model.

### AUC vs Brier Trade-off (Scatter Plot)

- **X-axis:** Brier Score (lower = better calibration)
- **Y-axis:** AUC (higher = better discrimination)
- **Ideal position:** Top-left corner
- Each point = one model

### Threshold Sensitivity Analysis (Line Chart)

Shows how **ROI changes** as the classification threshold varies from 3% to 25%.

**Key insight:** Most models peak around their optimal threshold (typically 5-9%). Moving too far in either direction degrades ROI.

### Calibration (Brier Score Bar Chart)

Horizontal bars showing Brier Score per model. Lower is better.

### Precision vs Recall (Scatter Plot)

- **X-axis:** Precision (% of predicted positives who actually paid)
- **Y-axis:** Recall (% of actual positives captured)
- **Trade-off visible:** Higher precision usually means lower recall

### Model Decision Matrix

Table where each row is a metric and each column is a model. The best value in each row is **highlighted green**.

### Selection Verdict

Summary card explaining which model was selected as Champion and why.

---

### 3.4 Features Tab

**Purpose:** Understand which features drive predictions.

### Feature Importance Bar Chart

- **Dropdown selector:** Choose model (xgboost, balanced_random_forest, deep_mlp, baseline_logistic_regression)
- **Bar chart:** Shows top features ranked by importance score
- **Importance method:** Permutation importance (how much model degrades when feature is shuffled)

**Top features observed:**
1. **birth_yr** — Debtor's birth year (age proxy)
2. **purchased_bal_gp** — Purchased balance group
3. **district** — Geographic district
4. **multiple_acct** — Has multiple accounts flag
5. **last_act_closing_m** — Months since last activity

### Feature Cross-tabulation Table

For the selected top feature, shows how payer rate varies across categories.

Example for `purchased_bal_gp`:
| Balance Group | Count | Payer Rate |
|--------------|-------|------------|
| <=200 | 4 | 25.0% |
| <=5k | 1,031 | 15.5% |
| <=10k | 1,741 | 14.1% |
| <=25k | 6,633 | 11.3% |
| ... | ... | ... |

### Payer Rate Drill-down (Bar Chart)

Visual representation of the cross-tabulation data.

### Business Interpretation Panel

Text explanation of what the selected feature means for collection strategy.

---

### 3.5 Stats Tab

**Purpose:** Understand the underlying data distribution.

### Numeric Variable Distributions (Histograms)

For each numeric column, a histogram with 20 bins:

| Variable | Unit | Typical Range |
|----------|------|--------------|
| last_act_closing_m | months | -1 to 116 |
| open_closing_m | months | 88 to 162 |
| co_closing_m | months | 66 to 1338 |
| last_pay_date_client_closing_m | months | -1 to 160 |
| birth_yr | year | 1946 to 1987 |
| balance_proxy | HKD | varies widely |
| calibrated_repay_prob | probability | 0.0 to 1.0 |

Each histogram includes:
- Gradient fill (blue gradient)
- X-axis label with unit
- Y-axis showing frequency count

### Descriptive Statistics Table

| Stat | Meaning |
|------|---------|
| Min | Minimum value |
| Max | Maximum value |
| Mean | Average value |
| Median | 50th percentile (robust to outliers) |
| Std | Standard deviation (spread) |
| Count | Non-null observations |
| Nulls | Missing value count |

### Box Plots

Compact visual showing median, quartiles, and outlier range for each variable.

---

### 3.6 Queue Tab

**Purpose:** Translate model scores into actionable collection queue.

### Strategy Matrix (Pie Chart)

Breakdown of 4,012 test accounts into action buckets:

| Action | Accounts | Share | Avg Prob |
|--------|----------|-------|----------|
| **High Priority (Agent Call)** | 722 | 18.0% | 10.2% |
| **Medium Priority (Auto-Dialer)** | 1,685 | 42.0% | 9.6% |
| **Low Priority (SMS/Email)** | 1,203 | 30.0% | 8.9% |
| **Write-off / Ignore** | 402 | 10.0% | 12.2% |

### ROI by Action (Bar Chart)

Compares ROI efficiency of each channel:
- Agent Call: ~17.7x (high cost but targets highest-probability accounts)
- Auto-Dialer: ~49.1x (best balance of cost and reach)
- SMS/Email: ~105.6x (cheapest channel, lowest cost per account)
- Write-off: 0x (no investment)

### Cost vs Net Recovery (Scatter Plot)

- **X-axis:** Contact cost (HKD)
- **Y-axis:** Net recovery (HKD)
- Point size: Number of accounts
- **Ideal:** High Y, low X (upper-left quadrant)

### Concentration Metrics

| Metric | Value | Meaning |
|--------|-------|---------|
| Overall Payer Rate | 9.15% | Baseline repayment rate in test set |
| Top-20 (by prob) Payer Rate | 17.83% | Nearly 2x improvement when targeting top scores |
| Top-20 Capture Share | 38.96% | Top-20 prob-sorted accounts capture ~39% of all recoveries |

### Top 200 Accounts Table

Sortable, searchable, filterable table of the highest-probability accounts:

| Column | Description |
|--------|-------------|
| ID | Account identifier |
| Loan Type | Credit Card / Personal Loan / Overdraft |
| Balance Group | Purchased balance bracket |
| District | Hong Kong district name |
| Raw Prob | Unadjusted model probability |
| Calibrated Prob | Platt-calibrated probability (more reliable) |
| Predicted Flag | Y/N based on threshold |
| Exp. Gross Recovery | Expected gross recovery (HKD) |
| Exp. Net Recovery | Expected gross minus contact cost |
| Contact Cost | Channel-specific contact cost |
| Recommended Action | Suggested collection channel |

**Interactive features:**
- Click column headers to sort
- Use search box to filter by text
- Click account row for detail modal
- Export button downloads CSV

---

### 3.7 Tuning Tab

**Purpose:** Review hyperparameter search results for each model.

### Model Cards

One card per algorithm (LR / RF / XGB / MLP), each showing:

| Field | Example Value |
|-------|--------------|
| Best Validation AUC | 0.7305 (XGBoost) |
| Total Configurations Searched | 3,888 (XGBoost grid search) |
| Key Hyperparameters | n_estimators=200, max_depth=6, lr=0.05, etc. |

### MLP Sweep Visualization

For the deep learning model specifically, shows all 8 configurations tested with their AUC scores.

---

### 3.8 Report Tab

**Purpose:** Concise English executive summary for stakeholders who don't need interactive charts.

Sections:
1. **Executive Summary** — Portfolio size, champion model, key numbers
2. **Model Performance Table** — All models compared on core metrics
3. **Key Findings** — 4 bullet-point insights from the analysis
4. **Recommended Actions** — 4 strategy recommendations with expected impact
5. **Economic Assumptions** — All cost/recovery parameters documented

---

## 4. Interactive Features

### Tab Navigation

Click any tab in the tab bar. Active tab is highlighted with an accent underline. Content fades in smoothly.

### Sorting

All tables support click-to-sort on column headers:
- **First click:** Ascending (A→Z, 0→9)
- **Second click:** Descending (Z→A, 9→0)
- Arrow indicator shows current sort direction

### Account Detail Modal

In the Queue tab, clicking any account row opens a modal showing:
- Full account details
- Risk tier classification
- AI-generated interpretation of why this account received its recommended action

### Search & Filter (Queue Table)

- **Search box:** Real-time text filtering across all columns
- **Export CSV:** Downloads filtered/sorted view as CSV file

### Model Selector (Features Tab)

Dropdown to switch between models for feature importance visualization. Chart updates instantly.

---

## 5. Data Sources & Refresh

### Input Files

| File | Location | Contents |
|------|----------|----------|
| `data.xlsx` | Project root (`c:\Users\marcozhu\Desktop\6980\`) | Raw NPA portfolio data (16,048 records) |
| `metrics.json` | `agent_outputs/baseline_comparison_run/` | All computed metrics (model performance, confusion matrices, etc.) |
| `champion_challenger_summary.csv` | Same dir | Model comparison results |
| `test_scored_accounts.csv` | Same dir | Individual account scores and recommendations |
| `production_queue_summary.csv` | Same dir | Queue-level aggregated statistics |
| `all_models_feature_importance.csv` | Same dir | Permutation importance per model |
| `collection_strategy_report.md` | Same dir | Markdown strategy report (source for Report tab) |

### Regenerating the Dashboard

To update the dashboard with new data:

```bash
cd c:\Users\marcozhu\Desktop\6980

# Run the generation script
python .workbuddy/skills/npa-repayment-agent/scripts/generate_dashboard.py

# Output: agent_outputs/baseline_comparison_run/dashboard.html
```

### Data Pipeline

```
data.xlsx
    │
    ├── pipeline.py ──► model training & evaluation
    │                     │
    │                     ├── npa_repayment_model.joblib (trained model)
    │                     ├── metrics.json
    │                     ├── *_summary.csv files
    │                     └── test_scored_accounts.csv
    │
    └── generate_dashboard.py ──► dashboard.html (this file)
                                  (reads all .json/.csv outputs)
```

---

## 6. Troubleshooting

### Issue: Charts show blank/empty

**Symptom:** Chart canvas areas are empty, no bars or lines rendered.

**Common causes:**
1. **Browser cache:** Hard-refresh (Ctrl+F5) to clear cached version
2. **JavaScript error:** Open F12 → Console tab, look for red error messages
3. **Data issue:** Check that input CSV/JSON files exist in `agent_outputs/baseline_comparison_run/`

**Debug tip:** The dashboard logs `[Dashboard]` prefixed messages to console when charts fail to render.

### Issue: Buttons don't respond

**Symptom:** Clicking tabs, sort headers, or export buttons has no effect.

**Solution:**
1. Ensure JavaScript is enabled
2. Check F12 Console for errors like "Cannot read property of undefined"
3. Try a different browser (Chrome recommended)

### Issue: Numbers look wrong

**Check:**
1. Currency units: All monetary values in **HKD**
2. Percentages: Some shown as decimals (0.09 = 9%), others with % suffix
3. Threshold: Current default is **0.09** (9%) for logistic regression baseline

### Issue: Page loads slowly

**Normal behavior:** First load may take 1-3 seconds due to embedding ~16KB of JSON data. Subsequent loads should be instant if browser cache is active.

### Issue: Chinese characters display incorrectly

**Note:** Report tab content is now in English. If you see garbled characters elsewhere, ensure your browser encoding is set to UTF-8.

---

## Appendix: Color Legend

| Color | Usage |
|-------|-------|
| `#3b82f6` (blue) | XGBoost / primary model |
| `#22c55e` (green) | Random Forest / positive values |
| `#a855f7` (purple) | MLP / neutral |
| `#f59e0b` (amber) | Baseline Logistic Regression |
| `#ef4444` (red) | Negative values / write-offs |
| `#06b6d4` (cyan) | Secondary data series |
| `#f97316` (orange) | Tertiary data series |

---

*Last updated: April 2026 | Dashboard Version 10*
