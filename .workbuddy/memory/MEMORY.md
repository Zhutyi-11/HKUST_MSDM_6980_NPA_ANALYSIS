# MEMORY

- Workspace 当前围绕NPA/不良资产回收分析展开，核心数据文件为 `data.xlsx`。
- 用户偏好专业、直接、业务导向的结论表达，尤其关注 ROI、回收金额集中度、资源投放优先级，而不是泛泛的技术说明。
- 当前标准分析口径：优先按 `data_type=M/T` 切分训练与验证；不使用 `id`、`debtor_last` 入模；对 `last_pay_date_client_closing_m` 的缺失按“从未对原债权人付款”处理，并将结果转换成可执行的催收策略矩阵。
- 工作区现已沉淀可复用的 `npa_repayment_agent` 本地MCP服务与项目级 Skill `npa-repayment-agent`，标准交付包括预处理、训练、评分和催收策略报告。
- WorkBuddy 的 `~/.workbuddy/mcp.json` 已配置 `npa-repayment-agent` 服务，使用受管 Python venv + `PYTHONPATH=c:/Users/marcozhu/Desktop/6980` 启动本地 MCP。
- 当前投产版标准能力已包含：Platt 概率校准、Champion-Challenger 比较、配置化经济假设（成本/渠道乘数/产能约束）、以及 `optimize_collection_policy_tool` 用于在不重训模型时重跑催收策略。



