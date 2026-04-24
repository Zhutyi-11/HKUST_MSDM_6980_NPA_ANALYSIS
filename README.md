# NPA Repayment Prediction & Collection Strategy Analysis

**Non-Performing Assets (NPA) Portfolio — Machine Learning-Driven Recovery Optimization**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Background & Business Context](#2-background--business-context)
3. [Data Description](#3-data-description)
4. [Methodology](#4-methodology)
5. [Model Architecture](#5-model-architecture)
6. [Research Process](#6-research-process)
7. [Results](#7-results)
8. [Collection Strategy](#8-collection-strategy)
9. [Key Findings](#9-key-findings)
10. [Project Structure](#10-project-structure)
11. [How to Run](#11-how-to-run)
12. [Deliverables](#12-deliverables)

---

## 1. Project Overview

| Item | Detail |
|------|--------|
| **Domain** | Non-Performing Asset (NPA) / Bad Debt Recovery |
| **Objective** | Predict repayment probability of delinquent accounts and optimize collection resource allocation |
| **Portfolio Size** | 16,048 accounts (HK NPA portfolio) |
| **Positive Rate** | ~9.65% (accounts that repaid within 3 years) |
| **Models Compared** | 4 (Logistic Regression, Random Forest, XGBoost, Deep MLP) |
| **Champion Model** | XGBoost (AUC = 0.7305 on validation) |
| **Best Economic Outcome** | Logistic Regression Baseline (Net Recovery: HKD 2.27M, ROI: 27.2x) |
| **Tech Stack** | Python, scikit-learn, XGBoost, PyTorch, Chart.js |

### One-Liner Summary

> Build a probability model to predict which NPA accounts will repay, calibrate those probabilities for business use, and translate scores into a cost-optimal collection action queue — maximizing net recovery per dollar spent.

---

## 2. Background & Business Context

### The Problem

Financial institutions hold large portfolios of non-performing assets (NPAs) — loans or credit obligations where borrowers have stopped making payments. These portfolios are typically sold at deep discounts to specialized debt collection agencies.

The core challenge: **Given limited collection resources (agent time, dialer capacity, SMS budget), which accounts should we prioritize to maximize total money recovered?**

### Why Machine Learning?

Traditional collection strategies use simple heuristics (e.g., "call everyone with balance > $50k" or "focus on recent defaults"). These approaches:

- ❌ Waste resources on accounts unlikely to repay
- ❌ Miss high-recovery opportunities buried in low-balance segments
- ❌ Cannot quantify trade-offs between coverage and precision

ML-based scoring provides:

- ✅ **Probability estimates:** Not just "will/won't pay," but *how likely* each account is to repay
- ✅ **Calibrated scores:** Probabilities that reflect real-world frequencies
- ✅ **Economic optimization:** Expected value calculations incorporating contact costs
- ✅ **Explainability:** Feature importance showing what drives predictions

### The NPA Repayment Problem

This is a **binary classification problem with severe class imbalance** (~9.65% positive rate):

| Class | Definition | Count | Share |
|-------|-----------|-------|-------|
| **Positive (Payer)** | Account holder repaid within 3 years of purchase date | ~1,550 | 9.65% |
| **Negative (Non-payer)** | No repayment within 3 years | ~14,498 | 90.35% |

The target variable is `payer_3yr` — whether the debtor made any payment to the original creditor within 3 years of account closing.

### Economic Framework

Each account's **expected net recovery** is calculated as:

```
Expected Net Recovery = P(repay) × Balance × Recovery_Rate × Channel_Multiplier - Contact_Cost
```

Where:
- **Recovery Rate:** 35% (assumption: when someone repays, they pay ~35% of balance)
- **Channel Multipliers:** Agent Call = 100%, Auto-Dialer = 72%, SMS/Email = 35%
- **Contact Costs:** Agent = HKD 85, Dialer = HKD 12, SMS/Email = HKD 1.5

The optimization objective is to **maximize total expected net recovery across the entire portfolio**, subject to channel capacity constraints.

---

## 3. Data Description

### Source Data

| Property | Value |
|----------|-------|
| **File** | `data.xlsx` (project root) |
| **Total Records** | 16,048 |
| **Data Split** | Training (M): 12,036 / Test (T): 4,012 |
| **Features** | 14 (6 categorical + 7 numeric + 1 engineered) |
| **Target** | `payer_3yr` (binary: Y/N) |
| **Missing Values** | Minimal (`last_pay_date_client_closing_m`: 1,165; `last_act_closing_m`: 31) |

### Feature Dictionary

#### Categorical Features

| Feature | Type | Unique Values | Description |
|---------|------|--------------|-------------|
| `loan_type` | Nominal | 3 | Credit Card / Personal Loan / Overdraft |
| `purchased_bal_gp` | Ordinal | 8 | Purchased balance group (<=200 to 200k+) |
| `district` | Nominal | 127 | Hong Kong district (e.g., TUEN MUN, YUEN LONG) |
| `multiple_acct` | Binary | 2 | Multiple accounts flag (Y/N) |
| `home_phone_flag` | Binary | 2 | Home phone available (Y/N) |
| `mobile_phone_flag` | Binary | 2 | Mobile phone available (Y/N) |

#### Numeric Features

| Feature | Range | Unit | Description |
|---------|-------|------|-------------|
| `birth_yr` | 1946–1987 | Year | Debtor's birth year (age proxy) |
| `last_act_closing_m` | -1–116 | Months | Time since last account activity |
| `open_closing_m` | 88–162 | Months | Account age at closing |
| `co_closing_m` | 66–1338 | Months | Write-off to closing interval |
| `last_pay_date_client_closing_m` | -1–160 | Months | Last payment to original creditor |
| `balance_proxy` | 0–250,000 | HKD | Numeric proxy for balance group |
| `calibrated_repay_prob` | 0–1 | Probability | Model output after Platt calibration |

#### Engineered Features

| Feature | Logic |
|---------|-------|
| `never_paid_to_client_flag` | `= 1 if last_pay_date_client_closing_m is NaN else 0` |
| `missing_last_act_flag` | `= 1 if last_act_closing_m is NaN else 0` |
| `balance_proxy` | Maps `purchased_bal_gp` category to numeric midpoint |

### Data Quality Notes

- **Identifier columns excluded from modeling:** `id`, `debtor_last` (privacy protection)
- **Missing `last_pay_date_client_closing_m`:** Interpreted as "never paid to original creditor" → set to -1 + engineered flag
- **Missing `last_act_closing_m`:** Set to -1 (no recent activity signal)
- **District names:** Uppercased and stripped for consistency
- **Class imbalance:** ~9.65% positive rate handled via `class_weight='balanced'`

### Target Variable Distribution

| Split | Total | Payer (Y) | Non-Payer (N) | Positive Rate |
|-------|-------|-----------|--------------|---------------|
| **Training (M)** | 12,036 | ~1,181 | ~10,855 | 9.81% |
| **Test (T)** | 4,012 | ~367 | ~3,645 | 9.15% |
| **Overall** | 16,048 | ~1,548 | ~14,500 | 9.65% |

---

## 4. Methodology

### 4.1 Overall Pipeline

```
┌─────────────┐    ┌──────────────────┐    ┌──────────────────────┐    ┌─────────────────┐
│   data.xlsx │───▶│ Data Cleaning &  │───▶│ Model Training &     │───▶│ Metrics &        │
│  (Raw Data) │    │ Feature Eng.     │    │ Hyperparameter Tuning│    │ Outputs          │
└─────────────┘    └──────────────────┘    └──────────────────────┘    └─────────────────┘
                                                                               │
                          ┌──────────────────────────────────────────────────┘
                          ▼
                   ┌──────────────────┐    ┌────────────────────┐    ┌─────────────┐
                   │ Score Test Set   │───▶│ Queue Assignment   │───▶│ Dashboard    │
                   │ (Probabilities)  │    │ & Strategy Report  │    │ (HTML)       │
                   └──────────────────┘    └────────────────────┘    └─────────────┘
```

### 4.2 Data Preprocessing Pipeline

Implemented as an scikit-learn `ColumnTransformer`:

**Categorical Features → One-Hot Encoding:**
```python
ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("num", SimpleImputer(strategy="median"), NUMERIC),
    ]
)
```

**Numeric Features:**
1. Median imputation (for rare missing values)
2. StandardScaler normalization (mean=0, std=1)

### 4.3 Model Training Protocol

For each candidate algorithm:

1. **Split M-set into Train/Calibration/Validation:**
   - Train: 60% (7,221 records) — model fitting
   - Calibration: 20% (2,407 records) — Platt scaling fit
   - Validation: 20% (2,408 records) — hyperparameter selection

2. **Hyperparameter Search** (strategies vary by model):
   - **Logistic Regression:** GridSearchCV (20 combinations)
   - **Random Forest:** GridSearchCV (72 combinations)
   - **XGBoost:** Custom grid search (3,888 combinations)
   - **MLP:** Manual sweep (8 configurations)

3. **Platt Probability Calibration:**
   - Use out-of-fold predictions from 5-fold CV on M-set
   - Fit isotonic/platt calibration on held-out calibration split
   - Apply calibrated probabilities to T-set

4. **Evaluation on T-set** (held-out test, never used in training/tuning):
   - ROC-AUC, Brier Score, Log Loss, Recall, Precision
   - Confusion Matrix at optimal threshold
   - Economic metrics: Net Recovery, ROI

### 4.4 Champion Selection Criteria

| Priority | Metric | Direction | Rationale |
|----------|--------|-----------|-----------|
| **Primary** | `expected_net_recovery_total` | Maximize | Direct business outcome |
| **Secondary** | `roc_auc` | Maximize | Discrimination quality |

> **Important note:** In this dataset, the baseline logistic regression achieved the highest *economic* outcome (Net Recovery = HKD 2.27M), while XGBoost achieved the highest *discrimination* (AUC = 0.7305). This highlights the critical distinction between statistical performance and business value — higher AUC does not always mean more money recovered.

---

## 5. Model Architecture

### 5.1 Logistic Regression (Baseline)

```
Input (14 features, one-hot encoded → ~140 dims)
    │
    ▼
StandardScaler → Linear (w·x + b) → Sigmoid → P(repay)
```

**Hyperparameters (best):**
- Regularization: L2 (C = 10.0)
- Solver: lbfgs
- Class weight: balanced
- Max iterations: 5000

**Role:** Serves as the interpretable baseline against which all complex models are compared.

### 5.2 Random Forest (Balanced)

```
Input
    │
    ▼
500 Decision Trees (max_depth=10, min_samples_leaf=8)
    │
    ▼
Average predictions → P(repay)
```

**Hyperparameters (best):**
- n_estimators: 500
- max_depth: 10
- min_samples_leaf: 8
- class_weight: balanced_subsample
- random_state: 42

**Strengths:** Handles non-linear relationships, robust to outliers, built-in feature importance.

### 5.3 XGBoost (Gradient Boosting)

```
Input
    │
    ▼
Sequential Weak Learners (200 trees, depth=6)
    │
    ▼
Weighted ensemble → P(repay)
```

**Hyperparameters (best):**
- n_estimators: 200
- max_depth: 6
- learning_rate: 0.05
- subsample: 0.85
- colsample_bytree: 0.7
- min_child_weight: 5
- reg_lambda: 0.5
- scale_pos_weight: 9.18 (auto-balanced)
- objective: binary:logistic

**Strengths:** Best AUC among all models; handles mixed feature types well; regularized to prevent overfitting.

### 5.4 Deep MLP v2 (Neural Network)

```
Input (d dimensions)
    │
    ├──▶ FeatureInteraction Layer (64-dim cross features)
    │         │
    │         ▼
    │    Concatenate: [original_features || interaction_features]
    │
    ▼
Linear(→128) → BatchNorm → Swish → Dropout(0.3)
    │
    ▼
Linear(128→64) → BatchNorm → Swish → Dropout(0.3)
    │
    ▼
[Residual Block × 2]  (each: Linear→BN→Swish→Dropout + skip connection)
    │
    ▼
Linear(64→1) → Sigmoid → P(repay)
```

**Architecture innovations over v1:**
| Improvement | Detail |
|------------|--------|
| **Residual Connections** | Skip connections mitigate vanishing gradients in deeper networks |
| **Swish Activation** | `x · sigmoid(x)` — smoother than ReLU, eliminates dead neurons |
| **Feature Interaction Layer** | Learns explicit cross-feature patterns (64 interaction dims) |
| **Batch Normalization** | Stabilizes training, allows higher learning rates |
| **Label Smoothing (ε=0.03)** | Reduces overconfidence on noisy labels |
| **OneCycleLR** | Cyclical learning rate for faster convergence |
| **Gradient Clipping (norm=3.0)** | Prevents gradient explosion |
| **AMP (Mixed Precision)** | Faster training on GPUs with lower memory usage |

**Hyperparameters (best config #6):**
- hidden_dims: [128, 64, 32]
- epochs: 200
- batch_size: 256
- lr: 0.0003
- weight_decay: 0.0005
- num_residual_blocks: 2
- label_smoothing: 0.03

### 5.5 Probability Calibration (Platt Scaling)

All models' raw outputs are calibrated using **Platt scaling** (logistic regression on out-of-fold predictions):

```
P_calibrated = 1 / (1 + exp(a × P_raw + b))
```

Parameters (a, b) are fitted on the calibration split using maximum likelihood estimation.

**Why calibration matters:** Raw model scores often poorly reflect true probabilities. For example, a raw score of 0.50 might only correspond to a 9% actual repayment rate. Calibration ensures that P=0.09 means approximately 9% of such accounts actually repay — essential for expected-value-based economic decisions.

---

## 6. Research Process

### Phase 1: Data Exploration & Understanding

- Explored all 14 features for distributions, cardinality, missing values
- Identified key patterns:
  - Strong correlation between balance group and payer rate (smaller balances → higher rates)
  - District-level variation in repayment behavior
  - Mobile phone availability as positive signal
- Established data quality baseline and preprocessing strategy

### Phase 2: Baseline Model Development

- Built logistic regression as first-pass model
- Achieved AUC ≈ 0.70, establishing a reasonable floor
- Validated feature coefficients aligned with domain intuition
- This became the permanent **baseline** for all subsequent comparisons

### Phase 3: Advanced Model Exploration

Iteratively tested increasingly sophisticated models:

| Iteration | Model | Key Finding |
|-----------|-------|-------------|
| 3a | Random Forest | Improved AUC to 0.716, better handling of categorical interactions |
| 3b | XGBoost | Best discrimination (AUC 0.731), but required extensive tuning (3,888 configs searched) |
| 3c | Deep MLP v1 | Competitive but unstable training |
| 3d | Deep MLP v2 | Added residual blocks, Swish, FeatureInteraction — stabilized at AUC 0.706 |

### Phase 4: Economic Evaluation

Critical insight emerged here: **statistical superiority ≠ business superiority**

While XGBoost won on AUC, its optimal threshold (5%) produced fewer high-confidence predictions, leading to lower total net recovery compared to the simpler logistic regression baseline operating at threshold 9%.

This led to the adoption of a **dual-metric selection framework:**
1. Primary: Expected Net Recovery (business impact)
2. Secondary: ROC-AUC (model quality)

### Phase 5: Strategy Translation

Translated model probabilities into actionable collection queue:

1. **Score all test accounts** using champion model
2. **Apply calibrated probabilities** (not raw scores)
3. **Calculate expected value** per account per channel
4. **Assign actions** based on capacity constraints and EV ranking
5. **Generate strategy report** with ROI projections

### Phase 6: Visualization & Reporting

Built interactive dashboard (v1 → v10) with iterative refinement:

| Version | Key Change |
|---------|-----------|
| v1-v6 | Initial builds, various rendering issues |
| v7 | Complete rewrite fixing JS parsing errors |
| v8 | Bug fixes for 6 categories of display issues |
| v9 | UI overhaul + Compare tab + cleanup |
| v10 | Fixed chart rendering root cause (Chart.js type parameter bug) + English Report tab |

---

## 7. Results

### 7.1 Model Performance Comparison (Validation Set)

| Model | Role | AUC ↑ | Brier ↓ | LogLoss ↓ | Recall @Thr | Precision @Thr | Net Recovery (HKD) | ROI ↑ |
|-------|------|-------|---------|----------|-------------|----------------|-------------------|------|
| **XGBoost** | Challenger | **0.7305** | 0.0910 | 0.3481 | 62.7% | **18.3%** | 1,270,096 | 25.4x |
| **Random Forest** | Challenger | 0.7161 | 0.0855 | 0.3054 | **75.4%** | 16.0% | 1,241,415 | 24.8x |
| **Deep MLP v2** | Challenger | 0.7063 | 0.0869 | 0.3211 | 74.6% | 15.8% | 1,404,720 | 28.1x |
| **Logistic Regression** | **Baseline** | 0.6994 | **0.0844** | **0.2986** | 74.2% | 15.4% | **1,467,880** | **29.3x** |

*Arrows indicate direction of desirability (higher = better for ↑, lower = better for ↓)*

### 7.2 Test Set Performance (Champion: LR Baseline)

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.6803 |
| Brier Score (calibrated) | 0.0803 |
| Log Loss (calibrated) | 0.2896 |
| Recall @ Threshold (9%) | 73.3% |
| Precision @ Threshold (9%) | 13.7% |
| **Net Recovery (Total)** | **HKD 2,268,560** |
| **ROI (Total)** | **27.2x** |
| Contact Cost (Total) | HKD 83,395 |

### 7.3 Confusion Matrix (Test Set, threshold=9%)

| | Predicted Non-Payer | Predicted Payer |
|--|--------------------|-----------------|
| **Actual Non-Payer** | TN: 1,951 | FP: 1,694 |
| **Actual Payer** | FN: 98 | TP: 269 |

- **True Positive Rate (Recall):** 269 / (269+98) = 73.3%
- **False Discovery Rate:** 1694 / (1694+269) = 86.3%

### 7.4 Collection Queue Summary (Test Set: 4,012 Accounts)

| Action Tier | Accounts | % of Queue | Avg Calib. Prob | Net Recovery (HKD) | Cost (HKD) | ROI |
|-------------|----------|-----------|-----------------|---------------------|-----------|-----|
| **High Priority (Agent Call)** | 722 | 18.0% | 10.17% | 1,085,006 | 61,370 | **17.7x** |
| **Medium Priority (Auto-Dialer)** | 1,685 | 42.0% | 9.56% | 993,085 | 20,220 | **49.1x** |
| **Low Priority (SMS/Email)** | 1,203 | 30.0% | 8.89% | 190,469 | 1,805 | **105.6x** |
| **Write-off / Ignore** | 402 | 10.0% | 12.18% | 0 | 0 | N/A |
| **TOTAL** | **4,012** | **100%** | — | **2,268,560** | **83,395** | **27.2x** |

### 7.5 Concentration Analysis

| Segment | Payer Rate | vs Baseline | Capture Share |
|---------|-----------|-------------|---------------|
| All accounts (baseline) | 9.15% | 1.0x | 100% |
| Top-20 by probability | **17.83%** | **1.95x** | 38.96% |
| Top-20 by net recovery | 8.48% | 0.93x | 18.53% |

**Interpretation:** Focusing on the top 20 highest-probability accounts nearly doubles the payer capture rate (from 9.15% to 17.83%) and captures ~39% of total recoveries from just 0.5% of accounts.

---

## 8. Collection Strategy

### 8.1 Action Assignment Logic

Accounts are assigned to action tiers based on **expected net recovery per dollar of contact cost:**

```
if P(calibrated) >= HIGH_THRESHOLD AND balance >= MIN_BALANCE_HIGH:
    → Agent Call (personalized negotiation)
elif P(calibrated) >= MED_THRESHOLD:
    → Auto-Dialer (automated outreach at scale)
elif P(calibrated) >= LOW_THRESHOLD:
    → SMS/Email (lowest-cost touchpoint)
else:
    → Write-off (below economic viability threshold)
```

### 8.2 Capacity Constraints

| Channel | Max % of Portfolio | Rationale |
|---------|-------------------|-----------|
| Agent Calls | 18% | Limited by agent headcount and call duration |
| Auto-Dialer | 42% | Dialer line capacity and regulatory limits |
| SMS/Email | 30% | Cost-effective bulk channel |

### 8.3 Economic Assumptions (Configurable)

| Parameter | Value | Source |
|-----------|-------|--------|
| Balance Recovery Rate | 35% | Industry benchmark for NPA portfolios |
| Agent Call Cost | HKD 85/call | Fully-loaded agent cost (time + overhead) |
| Auto-Dialer Cost | HKD 12/call | Automated system cost |
| SMS/Email Cost | HKD 1.5/contact | Bulk messaging platform fee |
| Agent Effectiveness Multiplier | 1.00x | Personalized = full effectiveness |
| Dialer Effectiveness Multiplier | 0.72x | Automated reach < personal touch |
| SMS Effectiveness Multiplier | 0.35x | Low-engagement channel |

> These parameters can be adjusted in `production_config_used.json` without retraining models. The `optimize_collection_policy_tool()` function recalculates the entire queue instantly.

---

## 9. Key Findings

### Finding 1: Simpler Models Can Outperform Complex Ones Economically

Despite XGBoost achieving the highest AUC (0.7305), **Logistic Regression delivered the best financial outcome** (Net Recovery: HKD 1.47M vs XGBoost's HKD 1.27M). This occurs because:
- LR's optimal threshold (9%) captures more true positives than XGB's (5%)
- Higher recall compensates for lower precision when recovery amounts are large
- **Takeaway:** Always evaluate ML models on business metrics, not just statistical ones

### Finding 2: Age (birth_yr) Is the Single Most Important Feature

Across tree-based models (RF, XGBoost), `birth_yr` consistently ranks #1 in permutation importance (~11-13%). Younger debtors show materially higher repayment rates, likely due to:
- Greater income trajectory upside
- Higher future credit motivation
- More active digital presence (easier to locate/contact)

### Finding 3: Balance Group Has Strong Inverse Relationship with Payer Rate

Smaller balance accounts have significantly higher repayment rates:

| Balance Group | Payer Rate |
|--------------|-----------|
| <= HKD 200 | 25.0% |
| <= HKD 5,000 | 15.5% |
| <= HKD 10,000 | 14.1% |
| <= HKD 25,000 | 11.3% |
| <= HKD 50,000 | 7.2% |
| <= HKD 100,000 | 4.1% |
| <= HKD 200,000 | 2.0% |
| > HKD 200,000 | 0.0% |

**Strategic implication:** Small-balance accounts are not "worthless" — they may be the most efficient targets for low-cost channels like SMS.

### Finding 4: Massive ROI Potential Across All Channels

Even the most expensive channel (Agent Call at HKD 85/account) delivers **17.7x ROI**. Cheaper channels deliver extraordinary returns:
- Auto-Dialer: **49.1x ROI**
- SMS/Email: **105.6x ROI**

This indicates the current portfolio has substantial untapped recovery potential, and even expanding collection activity would remain highly profitable.

### Finding 5: Top-Scoring Accounts Show Nearly 2x Payer Rate

The top 20 highest-scoring accounts achieve a **17.83% payer rate** vs 9.15% baseline — nearly double. This validates that the model successfully identifies the subset of accounts worth prioritizing.

### Finding 6: Geographic Clustering Matters

Top districts by account volume (TUEN MUN, YUEN LONG, KWAI CHUNG) suggest geographic clustering. District-specific collection campaigns could improve efficiency through localized agent deployment.

---

## 10. Project Structure

```
c:\Users\marcozhu\Desktop\6980\
│
├── data.xlsx                              # Raw NPA portfolio data (source of truth)
│
├── npa_repayment_agent/                  # Core analysis package
│   ├── __init__.py                       # Package init
│   ├── pipeline.py                       # Full ML pipeline (train/tune/score/calibrate)
│   ├── optimize_collection_policy_tool.py # Re-run queue assignment without retraining
│   ├── run_analysis.py                   # Main entry point (orchestrates pipeline → dashboard)
│   ├── __pycache__/                      # Compiled Python cache
│   └── *.pyc                             # Compiled modules
│
├── .workbuddy/
│   ├── skills/
│   │   └── npa-repayment-agent/          # WorkBuddy Skill definition
│   │       ├── SKILL.md                  # Skill metadata & trigger words
│   │       └── scripts/
│   │           └── generate_dashboard.py # Dashboard HTML generator (reads outputs → HTML)
│   ├── memory/                           # Working memory files
│   │   ├── MEMORY.md                     # Long-term project context
│   │   └── YYYY-MM-DD.md                 # Daily session logs
│   └── mcp.json                          # MCP server configuration
│
├── configs/
│   ├── production_config.json            # Production-ready configuration
│   └── tuning_config.json                # Hyperparameter search space definitions
│
├── agent_outputs/
│   └── baseline_comparison_run/          # Latest analysis outputs
│       ├── dashboard.html                # ★ Interactive dashboard (v10)
│       ├── metrics.json                  # All computed metrics
│       ├── production_config_used.json   # Config snapshot used for this run
│       ├── champion_challenger_summary.csv
│       ├── agent_vs_baseline_summary.csv
│       ├── all_models_feature_importance.csv
│       ├── feature_importance.csv
│       ├── test_scored_accounts.csv      # Individual account scores & recommendations
│       ├── production_queue_summary.csv  # Queue-tier aggregation
│       ├── payer_rate_by_balance.csv     # Segmented payer rates
│       ├── payer_rate_by_loan.csv
│       ├── payer_rate_by_mobile.csv
│       ├── collection_strategy_report.md  # Markdown strategy report
│       └── npa_repayment_model.joblib    # Serialized trained model
│
├── outputs/                              # Previous iteration outputs
│
├── DASHBOARD_USER_GUIDE.md               # ★ Dashboard user manual (this file)
├── README.md                             # ★ Project documentation (this file)
└── dist/                                 # Distribution archive
```

---

## 11. How to Run

### Prerequisites

- Python 3.13+ (managed runtime available)
- Required packages: pandas, numpy, scikit-learn, xgboost, torch, joblib, openpyxl

### Quick Start (Full Pipeline)

```bash
cd c:\Users\marcozhu\Desktop\6980

# Activate environment (if needed)
# source .venv/bin/activate  # Linux/Mac
# .\venv\Scripts\activate   # Windows

# Run complete analysis pipeline
python -m npa_repayment_agent.run_analysis

# Output: agent_outputs/baseline_comparison_run/dashboard.html
```

### Regenerate Dashboard Only

If models are already trained and you only need to refresh the visualization:

```bash
python .workbuddy/skills/npa-repayment-agent/scripts/generate_dashboard.py
```

### Re-run Queue Optimization

Update economic assumptions and recalculate the collection queue without retraining:

```bash
python -c "
from npa_repayment_agent.optimize_collection_policy_tool import optimize_policy
optimize_policy(
    config_path='agent_outputs/baseline_comparison_run/production_config_used.json',
    output_dir='agent_outputs/baseline_comparison_run'
)
"
```

### Using the WorkBuddy Skill

The project includes a reusable WorkBuddy Skill for NPA repayment analysis:

```bash
# Trigger via WorkBuddy agent conversation:
# "Run NPA repayment analysis on data.xlsx"
# "Generate dashboard for my NPA portfolio"
# "Optimize collection policy with new cost assumptions"
```

The skill automatically handles:
1. Data loading and validation
2. Model training and comparison
3. Probability calibration
4. Queue assignment
5. Dashboard generation
6. Strategy report creation

---

## 12. Deliverables

### Primary Deliverable

| Artifact | Location | Format |
|----------|----------|--------|
| **Interactive Dashboard** | `agent_outputs/baseline_comparison_run/dashboard.html` | HTML (self-contained, no server) |
| **Dashboard User Guide** | `DASHBOARD_USER_GUIDE.md` | Markdown |
| **Project Documentation** | `README.md` | Markdown |

### Data Artifacts

| File | Content |
|------|---------|
| `metrics.json` | Complete model evaluation results |
| `test_scored_accounts.csv` | Individual account scores, probabilities, recommended actions |
| `production_queue_summary.csv` | Aggregated queue statistics by action tier |
| `all_models_feature_importance.csv` | Permutation importance for each model |
| `champion_challenger_summary.csv` | Side-by-side model comparison |
| `collection_strategy_report.md` | Detailed strategy recommendations |

### Model Artifacts

| File | Content |
|------|---------|
| `npa_repayment_model.joblib` | Serialized trained model (scikit-learn compatible) |
| `production_config_used.json` | Exact configuration used for production scoring |

### Dashboard Contents (9 Tabs)

| Tab | Audience | Key Information |
|-----|----------|----------------|
| **Overview** | Executives | KPIs, confusion matrix, economic assumptions |
| **Models** | Data Scientists | Sortable model table, AUC/economic charts |
| **Compare** | Analysts | Radar/scatter plots, decision matrix |
| **Features** | Domain Experts | Per-model feature importance, crosstabs |
| **Stats** | Data Engineers | Distribution histograms, descriptive stats |
| **Queue** | Operations | Action assignments, ROI by channel, top accounts |
| **Tuning** | ML Engineers | Hyperparameter search details |
| **Report** | Stakeholders | English executive summary |
| *(Suggestions)* | Strategy Team | Actionable collection recommendations |

---

## References & Credits

- **Framework:** scikit-learn 1.x, XGBoost 2.x, PyTorch 2.x
- **Visualization:** Chart.js 4.x (via CDN)
- **Methodology:** Platt scaling for probability calibration, permutation importance for interpretability
- **Design Pattern:** Champion-Challenger model selection with dual-metric framework (statistical + economic)

---

*Last updated: April 2026*
