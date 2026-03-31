# NPA Repayment Agent

## 1. 目标
将无抵押债务/NPA回款预测流程封装成一个可复用的本地 Agent，包含：

- 本地 MCP 服务
- 项目级 Skill
- 标准化建模/评分/催收分层输出

## 2. 目录结构

```text
npa_repayment_agent/
  __init__.py
  pipeline.py
  mcp_server.py
.workbuddy/skills/npa-repayment-agent/
configs/mcp.npa-repayment-agent.example.json
```

## 3. MCP 工具能力

### preprocess_npa_data_tool
- 输入：Excel 数据文件路径
- 输出：清洗后的全量/M/T数据、profile.json
- 规则：
  - `id`、`debtor_last` 不入模
  - `last_pay_date_client_closing_m` 缺失填 `-1` 并新增 `never_paid_to_client_flag`
  - `last_act_closing_m` 缺失填 `-1` 并新增 `missing_last_act_flag`
  - `district` 清洗后统一大写

### train_repayment_model_tool
- 训练 Random Forest / XGBoost
- 自动处理类别不平衡
- 输出：
  - `npa_repayment_model.joblib`
  - `metrics.json`
  - `test_scored_accounts.csv`
  - `strategy_summary.csv`
  - `feature_importance.csv`
  - `collection_strategy_report.md`

### predict_repayment_probability_tool
- 用已训练模型对新组合打分
- 输出：
  - 回款概率
  - 预测付款标记
  - EV代理值
  - 催收分段

### build_collection_strategy_report_tool
- 一步跑完整流程
- 适合直接产出管理层报告

## 4. 接入方式
将 `configs/mcp.npa-repayment-agent.example.json` 中的 server 配置合并到 WorkBuddy 的 `~/.workbuddy/mcp.json` 的 `mcpServers` 节点即可。

## 5. 运行前提
当前实现依赖：
- pandas
- scikit-learn
- xgboost
- openpyxl
- fastmcp
- joblib

## 6. 典型使用方式

### 方式A：完整报告
让 Agent 调用 `build_collection_strategy_report_tool`，输入 `data.xlsx`，直接输出完整策略报告。

### 方式B：训练后复用
1. 调用 `train_repayment_model_tool`
2. 拿到 `npa_repayment_model.joblib`
3. 用 `predict_repayment_probability_tool` 对新增资产包打分

## 7. 业务解释约束
- 不输出 `debtor_last`
- 不用 Accuracy 作为主指标
- 核心评价口径：ROC-AUC / Recall(Y) / Precision(Y) / Confusion Matrix
- 最终必须落到催收资源分层与ROI优先级
