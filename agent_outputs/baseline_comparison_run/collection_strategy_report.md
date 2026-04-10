# NPA回款预测生产版策略报告

## 1. Executive Summary
- 数据量：16,048 条，其中建模集(M) 12,036 条，独立验证集(T) 4,012 条。
- Champion 模型：**xgboost**。T集 ROC-AUC = **0.736**，Recall(Y) = **80.38%**，Precision(Y) = **15.53%**。
- 对比基线：相对 **baseline_logistic_regression**，当前 Agent 在T集 ROC-AUC 变化 **+0.086**，Brier 变化 **-0.0032**（负值更好），预期净回收代理值变化 **+316,416**，ROI 变化 **+3.79x**。
- 概率校准：采用 **Platt scaling**，Brier Score 从 **0.1827** 改善到 **0.0781**，更适合直接用于产能分配和经济测算。
- 生产策略：在默认成本假设下，T集组合的**预期净回收代理值**为 **2,575,765**，预期ROI为 **30.89x**。
- 集中度：按校准后概率排序时，前20%账户覆盖 **45.23%** 的真实付款账户；按净回收代理值排序时，前20%账户贡献 **57.35%** 的预期净回收。

## 2. Production Upgrade Highlights
- 在原有M/T分层建模基础上新增 `train / calibration / validation / holdout` 四层开发框架。
- 在模型评分之后加入概率校准，让分数更接近可执行的回收概率。
- 增加 Champion-Challenger 比较，不只看 AUC，也看校准后经济价值。
- 本版新增 Logistic Regression 基线模型，便于长期监控 Agent 是否真正跑赢简单可解释方案。
- 增加成本函数与产能约束，把模型输出直接转成坐席、自动外呼、短信三类队列。
- 通过配置文件管理经济假设，便于后续按市场/渠道/回收策略调整。

## 3. Economic Assumptions
- Base recovery rate proxy：35.00%
- Agent call cost：¥85.00 / account
- Auto-dialer cost：¥12.00 / account
- SMS/Email cost：¥1.50 / account
- Agent channel multiplier：1.00x
- Auto-dialer multiplier：0.72x
- SMS/Email multiplier：0.35x

## 4. Champion-Challenger Summary (Validation)
| Role | Model | Valid ROC-AUC | Valid Brier | Recall(Y) | Precision(Y) | Expected Net Recovery | Expected ROI | Threshold |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| agent_champion | xgboost | 0.709 | 0.0832 | 73.73% | 15.92% | 1,555,044 | 31.09x | 0.09 |
| challenger | balanced_random_forest | 0.708 | 0.0841 | 73.31% | 15.43% | 1,460,440 | 29.20x | 0.08 |
| baseline | baseline_logistic_regression | 0.678 | 0.0855 | 92.80% | 11.97% | 1,444,962 | 28.89x | 0.07 |

## 5. Agent vs Baseline on Holdout (T set)
| Role | Model | ROC-AUC | Brier | LogLoss | Recall(Y) | Precision(Y) | Expected Net Recovery | Expected ROI | Threshold |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| agent_champion | xgboost | 0.736 | 0.0781 | 0.2780 | 80.38% | 15.53% | 2,575,765 | 30.89x | 0.09 |
| baseline | baseline_logistic_regression | 0.649 | 0.0813 | 0.2952 | 85.83% | 11.23% | 2,259,348 | 27.09x | 0.07 |
- Agent 相对基线的净回收代理值差额：**+316,416**；ROI 差额：**+3.79x**。
- Agent 相对基线的 Recall(Y) 变化：**-5.45%**；Precision(Y) 变化：**+4.30%**。
- Agent 相对基线的 Brier 变化：**-0.0032**（负值更好）；LogLoss 变化：**-0.0172**（负值更好）。

## 6. Champion Detailed Holdout Metrics
- ROC-AUC(raw / calibrated): **0.736 / 0.736**
- Brier(raw / calibrated): **0.1827 / 0.0781**
- LogLoss(raw / calibrated): **0.5389 / 0.2780**
- Recall(Y): **80.38%**
- Precision(Y): **15.53%**
- Confusion Matrix @ threshold 0.09: TN=2040, FP=1605, FN=72, TP=295
- Baseline 参照：ROC-AUC **0.649**，Brier **0.0813**，LogLoss **0.2952**。

## 7. Top Predictive Features
- **birth_yr**（importance=0.1295）：出生年份越晚通常意味着更年轻的偿付群体，收入修复和征信修复动机更强。
- **district**（importance=0.0513）：区域反映稳定性、流动性和社会经济结构差异。
- **purchased_bal_gp**（importance=0.0422）：余额规模同时影响协商意愿和资源投放经济性。
- **last_act_closing_m**（importance=0.0240）：账户活动越近，说明行为痕迹越新，触达和协商成功率更高。
- **co_closing_m**（importance=0.0212）：核销距收购日越近，债务新鲜度更高。
- **last_pay_date_client_closing_m**（importance=0.0116）：历史付款越近，未来付款延续性通常越好。
- **multiple_acct**（importance=0.0101）：多账户关系通常带来更丰富的行为信号。
- **home_phone_flag**（importance=0.0085）：住宅电话是补充触达渠道。
- **mobile_phone_flag**（importance=0.0035）：手机号直接决定自动化催收触达效率。
- **open_closing_m**（importance=0.0032）：账户年龄反映信贷关系成熟度。
- **loan_type**（importance=0.0014）：该变量对区分付款人与非付款人具有明显增益。
- **missing_last_act_flag**（importance=0.0000）：该变量对区分付款人与非付款人具有明显增益。

## 8. Production Queue Summary
| Queue | Accounts | Avg Calibrated PD | Actual Payer Rate | Balance Proxy Total | Expected Gross Recovery | Expected Net Recovery | Contact Cost | ROI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| High Priority (Agent Call) | 722 | 15.44% | 12.88% | 36,587,500 | 1,464,962 | 1,403,592 | 61,370 | 22.87x |
| Medium Priority (Auto-Dialer) | 1,685 | 10.76% | 9.91% | 57,300,000 | 1,046,327 | 1,026,107 | 20,220 | 50.75x |
| Low Priority (SMS/Email) | 1,203 | 7.87% | 7.15% | 27,757,500 | 147,870 | 146,065 | 1,804 | 80.95x |
| Write-off / Ignore | 402 | 5.52% | 5.22% | 5,145,000 | 0 | 0 | 0 | 0.00x |

## 9. Action Recommendation
- **High Priority (Agent Call)**：优先给人工坐席，关注高校准概率、高余额、净回收最高的账户。
- **Medium Priority (Auto-Dialer)**：给自动外呼，承担规模化覆盖和低成本触达任务。
- **Low Priority (SMS/Email)**：仅保留极低成本数字化触达。
- **Write-off / Ignore**：若净回收为负或挤占产能，则直接放弃当前轮人工资源。

## 10. Additional Portfolio Signals
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

## 11. Deployment Notes
- 本报告中的净回收金额仍是基于余额代理值的运营口径，不是财务确认回款。
- 后续应持续用基线模型做回归测试：若复杂模型长期跑不赢基线，就该回到特征工程和经济假设，而不是继续堆模型。
- 若要真正投产，应把实际 settlement rate、通话成本、渠道转化率按市场/批次写入 config 后再跑。
- 当前模型文件已包含校准器与默认策略配置，可直接对新组合打分并给出推荐队列。