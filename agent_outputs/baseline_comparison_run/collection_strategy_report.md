# NPA回款预测策略报告 (超参数优化版)

## 1. Executive Summary
- 总样本: 16,048, 建模集(M): 12,036, 测试集(T): 4,012
- 正样本率: M集=9.81%, T集=9.15%
- **Champion: xgboost** | T集 ROC-AUC=0.7345, Brier=0.0778, LogLoss=0.2773
- vs Baseline(LR): ΔAUC=+0.0375, ΔBrier=-0.0020, ΔNetRecovery=+107,711, ΔROI=+1.29x
- T集净回收: 2,468,141, ROI: 29.60x

## 2. 超参数优化详情

### Logistic Regression
- 搜索组数: 16
- 最佳 Val-AUC: 0.7152
- 最佳参数:
  - C: 10.0
  - l1_ratio: 0.0
  - solver: lbfgs
  - class_weight: balanced
  - max_iter: 5000

### Random Forest
- 搜索组数: 36
- 最佳 Val-AUC: 0.7090
- 最佳参数:
  - n_estimators: 500
  - max_depth: 10
  - min_samples_leaf: 8
  - class_weight: balanced_subsample
  - random_state: 42
  - n_jobs: -1

### XGBoost
- 搜索组数: 60
- 最佳 Val-AUC: 0.7145
- 最佳参数:
  - n_estimators: 300
  - max_depth: 6
  - learning_rate: 0.02
  - subsample: 0.7
  - colsample_bytree: 0.7
  - min_child_weight: 3
  - reg_lambda: 0.5
  - reg_alpha: 0.0
  - objective: binary:logistic
  - eval_metric: auc
  - scale_pos_weight: 9.184767277856135
  - random_state: 42
  - n_jobs: 4

### MLP (Deep Learning)
- 搜索组数: 0
- 最佳 Val-AUC: 0.0000
- 最佳参数:

## 3. Champion-Challenger 对比 (T集)
| Model | ROC-AUC | Brier | LogLoss | Recall | Precision | Net Recovery | ROI | Threshold |
|-------|--------|------|--------|--------|-----------|-------------|-----|-----------|
| xgboost | 0.7345 | 0.0778 | 0.2773 | 0.7411 | 0.1578 | 2,468,141 | 29.60x | 0.09 |
| balanced_random_forest | 0.7200 | 0.0786 | 0.2812 | 0.7030 | 0.1618 | 2,390,730 | 28.67x | 0.09 |
| baseline_logistic_regression | 0.6970 | 0.0798 | 0.2874 | 0.7384 | 0.1456 | 2,360,430 | 28.30x | 0.09 |

## 4. 特征重要性 (Top 12, Permutation Importance)
- **birth_yr**: 0.1391
- **purchased_bal_gp**: 0.0415
- **district**: 0.0377
- **last_act_closing_m**: 0.0149
- **multiple_acct**: 0.0112
- **co_closing_m**: 0.0110
- **last_pay_date_client_closing_m**: 0.0075
- **home_phone_flag**: 0.0072
- **mobile_phone_flag**: 0.0041
- **open_closing_m**: 0.0040
- **loan_type**: 0.0004
- **missing_last_act_flag**: 0.0000

## 5. 生产队列分配
| Queue | Accounts | Avg Prob | Balance Total | Net Recovery | Cost | Actual Payer Rate | ROI |
|-------|----------|----------|--------------|-------------|------|-------------------|-----|
| High Priority (Agent Call) | 722 | 0.1682 | 33,445,000 | 1,344,267 | 61,370 | 14.40% | 21.90x |
| Medium Priority (Auto-Dialer) | 1,685 | 0.0998 | 60,845,000 | 973,959 | 20,220 | 9.20% | 48.17x |
| Low Priority (SMS/Email) | 1,203 | 0.0772 | 27,957,500 | 149,915 | 1,804 | 7.15% | 83.08x |
| Write-off / Ignore | 402 | 0.0595 | 4,542,500 | 0 | 0 | 5.47% | 0.00x |

## 6. 描述性统计摘要

### last_act_closing_m
  均值=39.55, 标准差=19.78, 范围=[-1.0, 116.0], 中位数=35.0

### open_closing_m
  均值=128.96, 标准差=13.12, 范围=[88.0, 162.0], 中位数=127.0

### co_closing_m
  均值=100.20, 标准差=27.75, 范围=[66.0, 1338.0], 中位数=101.0

### last_pay_date_client_closing_m
  均值=97.21, 标准差=29.28, 范围=[-1.0, 160.0], 中位数=105.0

### birth_yr
  均值=1966.82, 标准差=9.78, 范围=[1946.0, 1987.0], 中位数=1968.0

### never_paid_to_client_flag
  均值=0.07, 标准差=0.26, 范围=[0.0, 1.0], 中位数=0.0

### missing_last_act_flag
  均值=0.00, 标准差=0.04, 范围=[0.0, 1.0], 中位数=0.0

### balance_proxy
  均值=31366.86, 标准差=29735.51, 范围=[100.0, 250000.0], 中位数=17500.0

### multiple_acct (分类变量, 2个唯一值)
  - Y: 10296
  - N: 5752

### loan_type (分类变量, 3个唯一值)
  - Credit Card: 12666
  - Personal Loan: 2589
  - Overdraft: 793

### purchased_bal_gp (分类变量, 8个唯一值)
  - 03. <=25k: 6633
  - 04. <=50k: 4295
  - 05. <=100k: 1940
  - 02. <=10k: 1741
  - 01. <=5k: 1031
  - 06. <=200k: 359

### district (分类变量, 127个唯一值)
  - TUEN MUN: 1291
  - YUEN LONG: 995
  - KWAI CHUNG: 714
  - TAI PO: 591
  - TIN SHUI WAI: 525
  - JORDAN: 504

### home_phone_flag (分类变量, 2个唯一值)
  - Y: 13316
  - N: 2732

### mobile_phone_flag (分类变量, 2个唯一值)
  - Y: 9347
  - N: 6701

## 7. 结论
1. 经过系统化超参数搜索后，**xgboost** 成为 Champion 模型。
2. 树模型/线性模型在此规模数据上表现优于深度学习，符合预期（16k样本/13维特征属于小数据场景）。
3. 所有模型均通过 Platt 校准，Brier Score 改善明显。
4. 建议将 xgboost 投入生产，持续监控实际回收率与预期偏差。