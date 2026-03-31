# NPA Repayment Agent Workflow Reference

## 1. Intended Use Cases

- 无抵押贷款/信用卡/NPA 历史组合建模
- 新资产包回款概率评分
- 催收策略分层
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

## 4. Deliverables Expected from the MCP Service

### Preprocess deliverables

- `preprocessed_full.csv`
- `preprocessed_model.csv`
- `preprocessed_test.csv`
- `profile.json`

### Training deliverables

- `npa_repayment_model.joblib`
- `metrics.json`
- `test_scored_accounts.csv`
- `strategy_summary.csv`
- `feature_importance.csv`
- `collection_strategy_report.md`

### Prediction deliverables

- `scored_accounts.csv`
- `prediction_summary.json`

## 5. Summary Writing Checklist

When writing the final response:

1. Quote ROC-AUC, Recall(Y), Precision(Y), and confusion matrix.
2. Explain the top predictive features in business language.
3. Separate:
   - probability-first ranking
   - EV-first ranking
4. Recommend collector allocation by queue.
5. Avoid overclaiming predicted cash as guaranteed cash.

## 6. Suggested User Phrasing That Should Trigger This Skill

- “帮我给无抵押债务组合建模”
- “对这个 NPA 资产包做回款预测”
- “做一个催收策略报告”
- “把历史付款数据变成催收分层”
- “给新资产包打分并划分催收队列”
