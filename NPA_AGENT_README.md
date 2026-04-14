# NPA 不良资产回款预测 — 完整项目文档

## 📌 项目概述

本项目是一个**可投产级 NPA（Non-Performing Asset）回款预测系统**，将无抵押债务/不良资产的回收分析流程升级为包含以下能力的完整 Agent：

- **本地 MCP 服务** — 可通过 WorkBuddy 直接调用
- **多模型 Champion-Challenger 框架** — LR / RF / XGBoost / Deep MLP 四模型对比
- **概率校准引擎** (Platt Scaling) — 让原始概率更接近真实回收率
- **配置化 ROI/产能策略引擎** — 自动生成催收队列
- **交互式 Dashboard** — 可视化分析面板，支持排序/筛选/下钻

---

## 📁 项目目录结构

```
6980/
├── data.xlsx                              # ★ 原始数据源（M/T 切分的资产包 Excel）
│
├── npa_repayment_agent/                   # ★ 核心建模管线（Python 包）
│   ├── __init__.py                        #   包初始化 & 公开 API 导出
│   ├── pipeline.py                        #   ★ 主建模逻辑：预处理→训练→校准→评分→策略→报告
│   └── mcp_server.py                      #   MCP 服务端定义（供 WorkBuddy 调用）
│
├── configs/                               # 配置文件目录
│   ├── npa-production-assumptions.example.json  # 经济假设配置模板（成本/乘数/产能）
│   └── mcp.npa-repayment-agent.example.json     # MCP 服务连接配置模板
│
├── agent_outputs/                         # ★ 建模产物输出目录（按运行命名子目录）
│   ├── baseline_comparison_run/           #   最新一次完整运行（含4模型对比+MLP）
│   │   ├── npa_repayment_model.joblib     #     ★ 已训练模型文件（pipeline + 校准器 + 配置）
│   │   ├── metrics.json                  #     ★ 全部指标汇总（KPI + 模型对比 + 策略结果）
│   │   ├── champion_challenger_summary.csv    #   验证集上各模型的对比表
│   │   ├── agent_vs_baseline_summary.csv      #   T 集 Agent vs Baseline 对比表
│   │   ├── production_queue_summary.csv       #   生产队列汇总（4类队列的指标）
│   │   ├── test_scored_accounts.csv            #   ★ T 集每账户评分明细（2500+字段）
│   │   ├── feature_importance.csv             #   Top 特征重要性排名
│   │   ├── payer_rate_by_balance.csv          #   按余额分组的付款率信号
│   │   ├── payer_rate_by_loan.csv             #   按贷款类型的付款率信号
│   │   ├── payer_rate_by_mobile.csv           #   按手机号标记的付款率信号
│   │   ├── production_config_used.json        #   本次运行使用的经济假设配置快照
│   │   ├── collection_strategy_report.md      #   ★ 策略报告（Markdown）
│   │   └── dashboard.html                     #   ★ 交互式可视化面板（单文件 HTML）
│   │
│   ├── full_run/                            #   早期运行（无基线对比）
│   ├── production_run/                      #   投产版运行
│   └── policy_reopt/                        #   仅策略重优化运行
│
├── outputs/                                # 早期版本遗留输出（可归档）
│
├── dist/
│   └── npa-repayment-agent.zip              # 打包分发文件
│
├── .workbuddy/skills/npa-repayment-agent/
│   └── scripts/
│       └── run_full_workflow.py            # ★ 一键运行脚本（CLI 入口）
│
├── NPA_AGENT_README.md                     # ★ 本文件（你正在阅读的）
├── npa_model_analysis.py                    # 早期独立分析脚本（已弃用）
└── generate_dashboard.py                   # Dashboard 生成脚本
```

---

## 🚀 快速开始

### 方式一：一键跑完整流程（推荐）

```bash
python .workbuddy/skills/npa-repayment-agent/scripts/run_full_workflow.py \
    data.xlsx \
    --output-dir agent_outputs/baseline_comparison_run \
    --config-path configs/npa-production-assumptions.example.json
```

### 方式二：通过 MCP 工具调用（WorkBuddy 内）

1. 将 `configs/mcp.npa-repayment-agent.example.json` 的内容合并到 `~/.workbuddy/mcp.json`
2. 重启 WorkBuddy
3. 调用 `build_collection_strategy_report_tool`，传入 `file_path` 即可

### 方式三：分步调用 Python API

```python
from npa_repayment_agent.pipeline import (
    preprocess_npa_data,
    train_repayment_model,
    optimize_collection_policy,
)

# 步骤1：预处理（可选）
profile = preprocess_npa_data("data.xlsx", output_dir="agent_outputs/preprocess")

# 步骤2：训练模型
result = train_repayment_model("data.xlsx", output_dir="agent_outputs/baseline_comparison_run")
print(result["best_model"])       # e.g., "deep_mlp"
print(result["baseline_model"])  # "baseline_logistic_regression"

# 步骤3：策略重优化（不重训模型）
policy = optimize_collection_policy(
    file_path="data.xlsx",
    model_path=result["model_path"],
    output_dir="agent_outputs/policy_reopt",
)
```

---

## 📊 数据说明

### 输入数据格式 (`data.xlsx`)

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | str | 账户唯一标识符 |
| `debtor_last` | str | 债务人姓氏（不入模） |
| `data_type` | str | **关键切分字段**：`M` = 建模集，`T` = 独立验证集 |
| `loan_type` | cat | 贷款产品类型（Credit Card / Personal Loan 等） |
| `purchased_bal_gp` | cat | 购买余额分组（<=100k / <=200k / 200k+ 等） |
| `district` | cat | 地区编码 |
| `multiple_acct` | Y/N | 是否多账户关系 |
| `home_phone_flag` | Y/N | 是否有座机号 |
| `mobile_phone_flag` | Y/N | 是否有手机号 |
| `last_act_closing_m` | num | 距最后活动月数（缺失 → -1） |
| `open_closing_m` | num | 距开户月数 |
| `co_closing_m` | num | 距关账月数 |
| `last_pay_date_client_closing_m` | num | 距原债权人最后付款月数（缺失 = 从未付款） |
| `birth_yr` | num | 出生年份 |
| `payer_3yr` | Y/N | **目标变量**：3年内是否实际付款 |

### 数据量参考（当前数据集）

```
总记录数:     16,048 条
建模集(M):    12,036 条 (正样本率 ~9.8%)
验证集(T):     4,012 条 (正样本率 ~9.15%)
特征维度:     12 个 (6 分类 + 6 数值)
```

---

## 🤖 模型体系

### 当前候选模型

| 模型 | 类型 | 角色定位 | 核心特点 |
|------|------|---------|---------|
| **Logistic Regression** | 传统线性 | **Baseline（基线）** | 可解释性强、计算快、作为复杂模型的"底线" |
| **Random Forest** | 集成树 | Challenger | 抗过拟合、支持非线性 |
| **XGBoost** | GBDT | Challenger (前Champion) | 表现通常最强、工业标准 |
| **MLP (Deep)** | 神经网络 | **Deep Champion (当前)** | 3层全连接网络(128→64→32)，BatchNorm + Dropout + EarlyStopping |

### 训练框架

```
数据
 ├─ 清洗（字符串标准化 / 缺失填充 / 衍生标志）
 ├─ M/T 切分
 └─ M 集内部分层：
     ├─ Train (60%)     ← 模型参数学习
     ├─ Calibration (20%)← Platt 校准器拟合
     └─ Validation (20%)← Champion-Challenger 选型
         ↓
     4 个候选模型并行训练 + 校准 + 阈值搜索(F2最优)
         ↓
     按 [Expected Net Recovery > ROC-AUC > Recall] 选 Champion
         ↓
     Champion 在全集 M 上重训 + OOF Platt 校准
         ↓
     在 T 集(Holdout) 上评估 + Baseline 对比
         ↓
     策略引擎: 成本函数 × 产能约束 → 4 类队列
         ↓
     输出: 模型 / 指标 / 报告 / CSV / Dashboard
```

---

## ⚙️ 经济假设配置

编辑 `configs/npa-production-assumptions.example.json`：

```json
{
  "calibration": {
    "enabled": true,
    "method": "platt",
    "oof_folds": 5
  },
  "economics": {
    "balance_recovery_rate": 0.35,    // 基础回收代理率
    "agent_call_cost": 85.0,         // 人工坐席成本/户
    "auto_dialer_cost": 12.0,        // 自动外呼成本/户
    "sms_email_cost": 1.5,           // 短信邮件成本/户
    "agent_call_multiplier": 1.0,    // 人工渠道回收乘数
    "auto_dialer_multiplier": 0.72,  // 外呼渠道回收乘数
    "sms_email_multiplier": 0.35     // 数字化渠道回收乘数
  },
  "capacity": {
    "max_agent_ratio": 0.18,         // 人工队列占比上限
    "max_auto_ratio": 0.42,          // 外呼队列占比上限
    "max_sms_ratio": 0.30            // 短信队列占比上限
  },
  "selection": {
    "primary_metric": "expected_net_recovery_total",  // 首选指标
    "secondary_metric": "roc_auc"                     // 次选指标
  }
}
```

---

## 📋 各产物文件详细说明

### `npa_repayment_model.joblib`
**作用**: 已训练的完整模型包（可直接加载对新数据打分）

**包含内容**:
- `pipeline`: sklearn Pipeline（预处理器 + 模型）
- `calibrator`: Platt 校准器
- `best_model`: 冠军模型名称
- `threshold`: 最优决策阈值
- `config`: 使用的经济假设配置
- `metadata`: 全部 metrics 汇总

**使用方式**:
```python
import joblib
bundle = joblib.load("agent_outputs/baseline_comparison_run/npa_repayment_model.joblib")
prob = bundle["pipeline"].predict_proba(new_data)[:, 1]
calibrated_prob = ...  # 使用 calibrator 转换
```

### `metrics.json`
**作用**: 全部量化指标的 JSON 汇总，是报告和 Dashboard 的数据源。

**核心字段**:
- `data_overview`: 数据量/正样本率/缺失统计
- `development_split`: 训练/校准/验证集行数
- `champion_challenger`: 验证集上每个模型的对比行
- `test_metrics`: Champion 的 T 集表现
- `baseline_test_metrics`: Baseline 的 T 集表现
- `agent_vs_baseline`: Delta 对比 + holdout 并排
- `policy_summary`: 队列级汇总（净回收/ROI）
- `top_features`: 特征重要性排名
- `concentration`: Top20 集中度指标

### `champion_challenger_summary.csv`
**作用**: 每个模型在验证集上的对比表，适合直接导入 Excel 分析。

**列**: model_role / model_name / roc_auc / brier / recall / precision / expected_net_recovery / expected_roi / threshold

### `agent_vs_baseline_summary.csv`
**作用**: Champion vs Baseline 在 T 集上的正式对比表（用于回归测试）。

**列**: model_role / model_name / roc_auc / brier / log_loss / recall / precision / expected_net_recovery / expected_roi / threshold

### `production_queue_summary.csv`
**作用**: 4 个生产队列的汇总指标。

**列**: recommended_action / accounts / avg_calibrated_prob / actual_payer_rate / balance_proxy_total / expected_gross/net_recovery / contact_cost / roi

### `test_scored_accounts.csv`
**作用**: T 集**每一条账户**的评分和推荐动作。这是最详细的操作文件。

**关键字段**:
| 字段 | 说明 |
|------|------|
| `calibrated_repay_prob` | 校准后付款概率 |
| `predicted_payer_flag` | 模型判定(Y/N) |
| `expected_net_recovery` | 预期净回收代理值 |
| `recommended_action` | 推荐催收队列 |
| `recommended_contact_cost` | 推荐触达成本 |
| `policy_rank` | 优先级排名 |

### `feature_importance.csv`
**作用**: Permutation Importance 排名的 Top 12 特征。

### `payer_rate_by_balance.csv` / `loan.csv` / `mobile.csv`
**作用**: 按不同维度切片的组合信号分析——用于发现业务洞察和异常模式。

### `collection_strategy_report.md`
**作用**: Markdown 格式的管理层策略报告。

### `dashboard.html`
**作用**: 单文件自包含的交互式可视化面板（无需服务器）。
- KPI 卡片行
- Agent vs Baseline 对比表 + 解读
- 5 个 Tab（模型比较 / 队列 / 账户明细 / 特征 / 组合信号）
- 表格排序 + 多维筛选 + 下钻模态框 + Chart.js 图表

---

## 🔧 MCP 工具 API 参考

| 工具名 | 功能 | 输入 | 关键输出 |
|--------|------|------|----------|
| `preprocess_npa_data_tool` | 数据清洗与剖析 | file_path, config_path? | profile.json, preprocessed CSVs |
| `train_repayment_model_tool` | 训练全部模型并选 Champion | file_path, config_path?, output_dir? | model.joblib, metrics.json, report.md |
| `predict_repayment_probability_tool` | 对新数据打分 + 策略分配 | file_path, model_path, config_path? | scored CSV, queue summary |
| `optimize_collection_policy_tool` | 更新经济假设后重跑策略 | file_path, model_path, config_path? | 新队列分配 |
| `build_collection_strategy_report_tool` | 一键完整流水线 | file_path, config_path?, output_dir? | 所有产物 |

---

## 📈 核心评价指标

| 指标 | 说明 | 优劣方向 |
|------|------|---------|
| **ROC-AUC** | 区分力 | ↑ 越高越好 |
| **Brier Score** | 概率校准质量 | ↓ 越低越好 |
| **Log Loss** | 预测损失 | ↓ 越低越好 |
| **Recall(Y)** | 付款人召回率 | ↑ 越高越好 |
| **Precision(Y)** | 付款人精确度 | ↑ 越高越好 |
| **F2-Score** | 偏向 Recall 的综合 | ↑ 越高越好 |
| **Expected Net Recovery** | 业务价值（代理口径） | ↑ 越高越好 |
| **ROI** | 投入产出比 | ↑ 越高越好 |

> **注意**：本项目不以 Accuracy 作为主指标（因为正样本仅 ~9%，全猜 N 就有 ~91% 准确率但毫无业务价值）。

---

## ⚠️ 重要约束

1. **不输出 `debtor_last` 字段**到任何产物中（隐私保护）
2. **必须按 `data_type=M/T` 切分**，不能随机划分
3. `last_pay_date_client_closing_m` 缺失 = **从未对原债权人付款**（强负信号）
4. 所有金额均为**运营代理口径**，不是财务确认回款
5. 最终结论必须落到**催收资源分层、产能约束与 ROI**

---

## 🔄 版本迭代历史

| 版本 | 变更内容 |
|------|---------|
| v1.0 | 基础版：LR/RF/XGB + Platt 校准 + 策略引擎 |
| v1.1 | + Logistic Regression 显式基线 + Agent vs Baseline T集对比 |
| v1.2 | + Deep MLP (PyTorch) 深度学习模型 + 交互式 Dashboard |
| v1.3 (当前) | + 详细研究报告重写 + README 完善 + Dashboard UI修复 |

---

## 🛠️ 开发与调试

```bash
# 安装依赖
pip install scikit-learn xgboost pandas openpyxl numpy joblib torch

# 运行完整流程
python .workbuddy/skills/npa-repayment-agent/scripts/run_full_workflow.py data.xlsx

# 只重新生成 Dashboard（不需要重训模型）
python generate_dashboard.py

# 查看 lint 错误
# （IDE 中自动显示）
```
