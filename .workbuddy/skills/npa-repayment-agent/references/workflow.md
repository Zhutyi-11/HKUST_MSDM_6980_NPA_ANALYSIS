# NPA Repayment Agent Workflow Reference

## 1. Intended Use Cases

- 无抵押贷款/信用卡/NPA 历史组合建模
- 新资产包回款概率评分
- 概率校准后的生产策略优化
- 产能约束下的催收队列分配
- 向管理层输出可执行的ROI报告

## 2. Required Data Assumptions

Expected columns:

- `data_type`
- `id`
- `multiple_acct`
- `loan_type`
- `purchased_bal_gp`
- `last_act_closing_m`
- `open_closing_m`
- `co_closing_m`
- `last_pay_date_client_closing_m`
- `debtor_last`
- `birth_yr`
- `district`
- `home_phone_flag`
- `mobile_phone_flag`
- `payer_3yr`

## 3. Core Business Rules

- Treat `data_type=M` as model development data.
- Treat `data_type=T` as holdout validation data.
- Remove `id` and `debtor_last` from model features.
- Use missing `last_pay_date_client_closing_m` as a “never paid original creditor” signal.
- Use balance-group representative values only as an EV proxy, not as booked financial value.
- Use calibrated probabilities, not raw model output, for queue allocation and economic simulation.

## 4. Production Assumptions

The production config JSON can control:

- `balance_recovery_rate`
- `agent_call_cost`
- `auto_dialer_cost`
- `sms_email_cost`
- `agent_call_multiplier`
- `auto_dialer_multiplier`
- `sms_email_multiplier`
- `max_agent_ratio`
- `max_auto_ratio`
- `max_sms_ratio`

Recommended practice:

- Adjust costs using actual operation budgets.
- Adjust multipliers using historical channel uplift or pilot-test evidence.
- Re-run policy optimization when capacity changes, even if the model stays unchanged.

## 5. Deliverables Expected from the MCP Service

### Preprocess deliverables

- `preprocessed_full.csv`
- `preprocessed_model.csv`
- `preprocessed_test.csv`
- `profile.json`
- `production_config_used.json`

### Training deliverables

- `npa_repayment_model.joblib`
- `metrics.json`
- `champion_challenger_summary.csv`
- `production_queue_summary.csv`
- `test_scored_accounts.csv`
- `feature_importance.csv`
- `collection_strategy_report.md`

### Policy optimization deliverables

- `policy_scored_accounts.csv`
- `recommended_queue_summary.csv`
- `policy_optimization_summary.json`

## 6. Summary Writing Checklist

When writing the final response:

1. Quote ROC-AUC, Recall(Y), Precision(Y), Brier, Log Loss, and confusion matrix.
2. Explain whether calibration improved deployment readiness.
3. Separate:
   - raw model score
   - calibrated probability
   - expected net recovery proxy
4. Recommend collector allocation by queue.
5. State which assumptions drive the current queue mix.
6. Avoid overclaiming predicted cash as guaranteed cash.

## 7. Suggested User Phrasing That Should Trigger This Skill

- “帮我给无抵押债务组合建模”
- “对这个 NPA 资产包做回款预测”
- “做一个催收策略报告”
- “把历史付款数据变成催收分层”
- “给新资产包打分并划分催收队列”
- “把这套模型升级成投产版”
- “按新的渠道成本重新优化策略”
