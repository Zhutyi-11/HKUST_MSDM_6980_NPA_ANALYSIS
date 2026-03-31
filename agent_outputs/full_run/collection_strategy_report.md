# NPA还款预测模型与催收策略报告

## 1. Executive Summary
- 数据量：16,048 条，其中建模集(M) 12,036 条，验证集(T) 4,012 条。
- 最优模型：**xgboost**；T集 ROC-AUC = **0.736**，Recall(Y) = **80.93%**，Precision(Y) = **15.15%**。
- 排序效果：按还款概率排序时，前20%账户覆盖 **45.23%** 的真实付款账户，真实付款率 **20.70%**。
- ROI效果：按EV代理值排序时，前20%账户贡献 **45.12%** 的潜在回收价值。

## 2. Methodology
- 按 `data_type=M/T` 切分训练与独立验证。
- 丢弃 `id` 与 `debtor_last`，避免标识符噪音和隐私/公平性风险。
- 将 `last_pay_date_client_closing_m` 缺失填为 `-1`，并新增 `never_paid_to_client_flag`。
- 将 `last_act_closing_m` 缺失填为 `-1`，并新增 `missing_last_act_flag`。
- 对类别不平衡，Random Forest 使用 `class_weight`，XGBoost 使用 `scale_pos_weight`。
- 评估指标以 ROC-AUC、Recall(Y)、Precision(Y)、Confusion Matrix 为核心，不使用 Accuracy 作为主指标。

## 3. Model Comparison
| Model | Validation ROC-AUC | Validation Recall(Y) | Validation Precision(Y) | Threshold |
|---|---:|---:|---:|---:|
| balanced_random_forest | 0.714 | 72.46% | 16.03% | 0.43 |
| xgboost | 0.730 | 77.12% | 16.65% | 0.38 |

## 4. Holdout Performance
- ROC-AUC: **0.736**
- Recall(Y): **80.93%**
- Precision(Y): **15.15%**
- Confusion Matrix @ threshold 0.38: TN=1982, FP=1663, FN=70, TP=297

## 5. Top Features
- **birth_yr**（importance=0.1295）：出生年份越晚通常意味着更年轻的偿付群体，收入修复和征信修复动机更强。
- **district**（importance=0.0513）：区域反映稳定性、流动性和社会经济结构差异。
- **purchased_bal_gp**（importance=0.0422）：余额规模同时影响协商意愿和资源投放经济性。
- **last_act_closing_m**（importance=0.0239）：账户活动越近，说明行为痕迹越新，触达和协商成功率更高。
- **co_closing_m**（importance=0.0213）：核销距收购日越近，债务新鲜度更高。
- **last_pay_date_client_closing_m**（importance=0.0116）：历史付款越近，未来付款延续性通常越好。
- **multiple_acct**（importance=0.0101）：多账户关系通常带来更丰富的行为信号。
- **home_phone_flag**（importance=0.0085）：住宅电话是补充触达渠道。
- **mobile_phone_flag**（importance=0.0035）：手机号直接决定自动化催收触达效率。
- **open_closing_m**（importance=0.0032）：账户年龄反映信贷关系成熟度。

## 6. Strategy Matrix
| Segment | Accounts | Average Model Score | Actual Payer Rate | Balance Proxy Total | Expected Value Proxy Total |
|---|---:|---:|---:|---:|---:|
| High Priority (Agent Call) | 1,371 | 53.59% | 13.64% | 44,845,000 | 21,964,850 |
| Medium Priority (Auto-Dialer) | 1,237 | 46.93% | 11.56% | 14,862,500 | 6,373,676 |
| Low Priority (SMS/Email) | 1,271 | 15.61% | 2.68% | 66,360,000 | 9,380,982 |
| Write-off / Ignore | 133 | 17.10% | 2.26% | 722,500 | 123,210 |

## 7. Action Recommendation
- **High Priority (Agent Call)**：优先人工坐席，目标是高余额、高回收价值账户。
- **Medium Priority (Auto-Dialer)**：自动外呼 + 轻人工复核，追求单位成本最优。
- **Low Priority (SMS/Email)**：保留低成本数字化触达与自助还款入口。
- **Write-off / Ignore**：仅做极低成本批量触达或后续再评分。

## 8. Additional Signals
### Payer Rate by Balance Group
| Balance Group | Payer Rate |
|---|---:|
| 00. <=200 | 25.00% |
| 01. <=5k | 15.52% |
| 02. <=10k | 14.07% |
| 03. <=25k | 11.29% |
| 04. <=50k | 7.15% |
| 05. <=100k | 4.07% |
| 06. <=200k | 1.95% |
| 07. 200k+ | 0.00% |

### Payer Rate by Loan Type
| Loan Type | Payer Rate |
|---|---:|
| Credit Card | 9.85% |
| Overdraft | 9.46% |
| Personal Loan | 8.73% |

### Payer Rate by Mobile Phone Flag
| Mobile Phone Flag | Payer Rate |
|---|---:|
| N | 9.15% |
| Y | 10.00% |