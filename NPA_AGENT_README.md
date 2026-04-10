# NPA Repayment Agent

## 1. 目标
将无抵押债务/NPA回款预测流程升级成可投产复用的本地 Agent，包含：

- 本地 MCP 服务
- 项目级 Skill
- 概率校准后的模型产物
- 配置化 ROI / 产能策略引擎
- Champion-Challenger 机制

## 2. 目录结构

```text
npa_repayment_agent/
  __init__.py
  pipeline.py
  mcp_server.py
.workbuddy/skills/npa-repayment-agent/
configs/
  mcp.npa-repayment-agent.example.json
  npa-production-assumptions.example.json
```

## 3. 投产级能力

### 3.1 概率校准
- 使用 Platt scaling 对模型输出进行校准
- 同时输出 raw probability 与 calibrated probability
- 额外监控：
  - Brier score
  - Log loss

### 3.2 Champion-Challenger
- 候选模型至少包括：
  - Logistic Regression（显式基线模型）
  - Random Forest
  - XGBoost
- 决策不只看 ROC-AUC，还看：
  - 校准后 Brier
  - Recall(Y)
  - Expected Net Recovery
- 训练产物会额外输出 Agent Champion vs Baseline 的 T 集对比，避免复杂模型只是在开发集上看起来更强。

### 3.3 策略优化引擎
- 在模型分数之上增加经济假设：
  - base recovery rate proxy
  - agent call / auto-dialer / sms 成本
  - 各渠道回收乘数
- 增加产能约束：
  - `max_agent_ratio`
  - `max_auto_ratio`
  - `max_sms_ratio`
- 输出推荐队列：
  - High Priority (Agent Call)
  - Medium Priority (Auto-Dialer)
  - Low Priority (SMS/Email)
  - Write-off / Ignore

## 4. MCP 工具能力

### preprocess_npa_data_tool
- 输入：`file_path`，可选 `config_path`
- 输出：
  - 清洗后的全量 / M / T 数据
  - `profile.json`
  - `production_config_used.json`

### train_repayment_model_tool
- 输入：`file_path`，可选 `config_path`
- 输出：
  - `npa_repayment_model.joblib`
  - `metrics.json`
  - `champion_challenger_summary.csv`
  - `agent_vs_baseline_summary.csv`
  - `production_queue_summary.csv`
  - `test_scored_accounts.csv`
  - `feature_importance.csv`
  - `collection_strategy_report.md`

### predict_repayment_probability_tool
- 输入：`file_path` + `model_path`，可选 `config_path`
- 输出：
  - 校准后回款概率
  - 推荐催收队列
  - 预期毛回收 / 净回收代理值

### optimize_collection_policy_tool
- 用已有模型 + 新成本假设重新优化催收策略
- 适合渠道成本、产能、回收率假设变化后的快速重跑

### build_collection_strategy_report_tool
- 一步跑完整投产版流程
- 适合直接产出管理层报告与队列文件

## 5. 经济假设配置
使用：
- `configs/npa-production-assumptions.example.json`

可调项包括：
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

## 6. 接入方式
将 `configs/mcp.npa-repayment-agent.example.json` 中的 server 配置合并到 WorkBuddy 的 `~/.workbuddy/mcp.json` 的 `mcpServers` 节点即可。

## 7. 典型使用方式

### 方式A：完整管理层报告
调用 `build_collection_strategy_report_tool`，输入历史资产包 Excel，输出：
- Champion 模型
- 概率校准表现
- 队列优化结果
- 管理层报告

### 方式B：训练后反复评分
1. 调用 `train_repayment_model_tool`
2. 保存 `model_path`
3. 对新资产包调用 `predict_repayment_probability_tool`
4. 若经济假设变化，再调用 `optimize_collection_policy_tool`

## 8. 业务解释约束
- 不输出 `debtor_last`
- 不用 Accuracy 作为主指标
- 核心评价口径：ROC-AUC / Recall(Y) / Precision(Y) / Brier / Log Loss / Confusion Matrix
- 最终必须落到催收资源分层、产能约束与 ROI 优先级

## 9. 当前版本边界
当前版本已具备“投产前一公里”的框架，但仍建议上线前补：
- 真实 settlement 数据回灌
- 渠道 uplift 的 A/B 实测
- 实际现金回款口径替代 balance proxy
- 按批次/月份做漂移监控和再训练计划
