"""
NPA 回款预测 Dashboard 生成器
读取 baseline_comparison_run 的全部产物，输出一个可交互的单文件 HTML Dashboard。
"""

import json
import csv
import os
from pathlib import Path

OUTPUT_DIR = Path(r"c:\Users\marcozhu\Desktop\6980\agent_outputs\baseline_comparison_run")
OUTPUT_HTML = Path(r"c:\Users\marcozhu\Desktop\6980\agent_outputs\baseline_comparison_run\dashboard.html")


def read_csv_rows(filepath: Path) -> list[dict]:
    """读取CSV为list of dict。"""
    if not filepath.exists():
        return []
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)


def fmt_num(val, decimals=2):
    """安全格式化数字。"""
    try:
        v = float(val)
        if abs(v) >= 1_000_000:
            return f"{v:,.{decimals}f}"
        elif abs(v) >= 1000:
            return f"{v:,.{decimals}f}"
        else:
            return f"{v:.{decimals}f}"
    except (ValueError, TypeError):
        return val


def pct(val):
    try:
        return f"{float(val)*100:.2f}%"
    except (ValueError, TypeError):
        return str(val)


def generate_dashboard():
    # ===== 加载全部数据 =====
    metrics = json.loads((OUTPUT_DIR / "metrics.json").read_text(encoding="utf-8"))
    champion_csv = read_csv_rows(OUTPUT_DIR / "champion_challenger_summary.csv")
    ab_csv = read_csv_rows(OUTPUT_DIR / "agent_vs_baseline_summary.csv")
    queue_csv = read_csv_rows(OUTPUT_DIR / "production_queue_summary.csv")
    feature_csv = read_csv_rows(OUTPUT_DIR / "feature_importance.csv")
    payer_balance = read_csv_rows(OUTPUT_DIR / "payer_rate_by_balance.csv")
    payer_loan = read_csv_rows(OUTPUT_DIR / "payer_rate_by_loan.csv")
    payer_mobile = read_csv_rows(OUTPUT_DIR / "payer_rate_by_mobile.csv")

    # 账户级数据（限制前2000行，避免HTML过大）
    scored_raw = read_csv_rows(OUTPUT_DIR / "test_scored_accounts.csv")
    scored_sampled = scored_raw[:2500]

    tm = metrics["test_metrics"]
    btm = metrics["baseline_test_metrics"]
    policy = metrics["policy_summary"]
    bpolicy = metrics["baseline_policy_summary"]
    ab = metrics["agent_vs_baseline"]
    delta = ab["delta_agent_minus_baseline"]
    conc = metrics["concentration"]
    do = metrics["data_overview"]

    # 构建HTML
    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>NPA 回款预测 — 交互式分析面板</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --primary: #0ea5e9;
    --primary-dark: #0369a1;
    --success: #10b981;
    --danger: #ef4444;
    --warning: #f59e0b;
    --bg-dark: #0f172a;
    --card-bg: #1e293b;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --border: #334155;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: linear-gradient(135deg, var(--bg-dark) 0%, #1a1a2e 50%, #16213e 100%);
    color: var(--text-primary);
    min-height: 100vh;
  }}
  .glass-card {{
    background: rgba(30,41,59,0.75);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    transition: transform 0.2s, box-shadow 0.2s;
  }}
  .glass-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.3);
  }}
  .kpi-value {{ font-size: 1.75rem; font-weight: 800; letter-spacing: -0.5px; }}
  .kpi-label {{ font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1.5px; color: var(--text-secondary); }}
  .delta-up {{ color: var(--success); font-weight: 700; }}
  .delta-down {{ color: var(--danger); font-weight: 700; }}
  .delta-neutral {{ color: var(--warning); font-weight: 700; }}

  /* 表格样式 */
  .data-table {{
    width: 100%; border-collapse: separate; border-spacing: 0;
    font-size: 0.85rem;
  }}
  .data-table thead th {{
    background: rgba(15,23,42,0.9);
    color: var(--text-secondary);
    font-weight: 600;
    padding: 12px 14px;
    text-align: left;
    position: sticky; top: 0; z-index: 10;
    border-bottom: 2px solid var(--border);
    cursor: pointer;
    user-select: none;
    white-space: nowrap;
  }}
  .data-table thead th:hover {{ color: var(--primary); }}
  .data-table tbody td {{
    padding: 10px 14px;
    border-bottom: 1px solid rgba(51,65,85,0.5);
    white-space: nowrap;
    max-width: 220px;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .data-table tbody tr:hover {{ background: rgba(14,165,233,0.06); }}
  .data-table tbody tr.selected-row {{ background: rgba(14,165,233,0.15); outline: 2px solid var(--primary); }}

  /* 排序箭头 */
  .sort-icon {{ opacity: 0.3; margin-left: 4px; font-size: 0.7rem; }}
  .sort-asc .sort-icon, .sort-desc .sort-icon {{ opacity: 1; }}

  /* 标签页 */
  .tab-btn {{
    padding: 10px 22px;
    border-radius: 10px 10px 0 0;
    cursor: pointer;
    font-weight: 600;
    font-size: 0.88rem;
    border: none;
    background: transparent;
    color: var(--text-secondary);
    transition: all 0.25s;
    border-bottom: 3px solid transparent;
  }}
  .tab-btn.active {{
    color: var(--primary);
    background: rgba(14,165,233,0.08);
    border-bottom-color: var(--primary);
  }}
  .tab-content {{ display: none; animation: fadeIn 0.3s ease; }}
  .tab-content.active {{ display: block; }}
  @keyframes fadeIn {{ from{{opacity:0;transform:translateY(8px)}} to{{opacity:1;transform:translateY(0)}} }}

  /* 进度条 */
  .bar-container {{
    height: 28px; background: rgba(51,65,85,0.5); border-radius: 14px;
    overflow: hidden; position: relative;
  }}
  .bar-fill {{
    height: 100%; border-radius: 14px; display: flex; align-items: center;
    justify-content: flex-end; padding-right: 10px;
    font-size: 0.75rem; font-weight: 600; color: #fff;
    transition: width 0.8s cubic-bezier(0.25,0.46,0.45,0.94);
  }}

  /* 徽章 */
  .badge {{
    display: inline-flex; align-items: center; padding: 3px 10px;
    border-radius: 9999px; font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.5px;
  }}
  .badge-agent {{ background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }}
  .badge-baseline {{ background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }}
  .badge-challenger {{ background: rgba(99,102,241,0.15); color: #a5b4fc; border: 1px solid rgba(99,102,241,0.3); }}

  /* 筛选输入 */
  .filter-input {{
    background: rgba(15,23,42,0.7); border: 1px solid var(--border);
    border-radius: 10px; padding: 10px 16px; color: var(--text-primary);
    font-size: 0.88rem; outline: none; width: 260px; transition: border-color 0.2s;
  }}
  .filter-input:focus {{ border-color: var(--primary); box-shadow: 0 0 0 3px rgba(14,165,233,0.15); }}
  .filter-select {{
    background: rgba(15,23,42,0.7); border: 1px solid var(--border);
    border-radius: 10px; padding: 10px 16px; color: var(--text-primary);
    font-size: 0.88rem; outline: none; cursor: pointer;
  }}

  /* 解读面板 */
  .insight-panel {{
    background: linear-gradient(135deg, rgba(14,165,233,0.08), rgba(16,185,129,0.05));
    border-left: 4px solid var(--primary);
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    font-size: 0.88rem; line-height: 1.7;
  }}
  .insight-panel strong {{ color: var(--primary); }}

  /* 滚动条美化 */
  ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
  ::-webkit-scrollbar-track {{ background: transparent; }}
  ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 4px; }}
  ::-webkit-scrollbar-thumb:hover {{ background: #475569; }}

  /* 下钻详情模态框 */
  .modal-overlay {{
    display: none; position: fixed; inset: 0; z-index: 100;
    background: rgba(0,0,0,0.65); backdrop-filter: blur(4px);
    justify-content: center; align-items: center; animation: fadeIn 0.2s;
  }}
  .modal-overlay.show {{ display: flex; }}
  .modal-body {{
    background: var(--card-bg); border: 1px solid var(--border);
    border-radius: 20px; max-width: 720px; width: 92%;
    max-height: 85vh; overflow-y: auto; padding: 32px;
    box-shadow: 0 25px 60px rgba(0,0,0,0.5);
  }}

  /* 图表容器 */
  .chart-wrapper {{ position: relative; height: 280px; }}
</style>
</head>
<body>

<!-- ==================== HEADER ==================== -->
<header class="py-6 px-8 border-b border-white/5">
  <div class="max-w-[1600px] mx-auto flex items-center justify-between">
    <div class="flex items-center gap-4">
      <div class="w-11 h-11 rounded-xl bg-gradient-to-br from-sky-500 to-emerald-500 flex items-center justify-center shadow-lg shadow-sky-500/25">
        <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/></svg>
      </div>
      <div>
        <h1 class="text-xl font-bold tracking-tight">NPA 不良资产回款预测 — 交互式分析面板</h1>
        <p class="text-sm mt-0.5" style="color:var(--text-secondary)">Agent Champion vs 基线模型 · T集验证 · 可排序/筛选/下钻</p>
      </div>
    </div>
    <div class="flex items-center gap-3">
      <span id="dataTimestamp" class="text-xs px-3 py-1.5 rounded-full" style="background:rgba(255,255,255,0.06);color:var(--text-secondary)">数据源: baseline_comparison_run</span>
      <span class="badge badge-agent">Champion: {metrics['best_model']}</span>
    </div>
  </div>
</header>

<main class="max-w-[1600px] mx-auto px-8 py-8">

<!-- ==================== KPI 卡片行 ==================== -->
<section class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-4 mb-8" id="kpiRow">

  <div class="glass-card p-5">
    <div class="kpi-label">T集 ROC-AUC</div>
    <div class="kpi-value" style="color:#38bdf8">{fmt_num(tm['roc_auc'],3)}</div>
    <div class="mt-1 text-xs delta-up">+{fmt_num(delta.get('roc_auc',0),3)} vs 基线</div>
  </div>

  <div class="glass-card p-5">
    <div class="kpi-label">Brier Score ↓</div>
    <div class="kpi-value" style="color:#a78bfa">{fmt_num(tm['brier'],4)}</div>
    <div class="mt-1 text-xs {'delta-up' if float(delta.get('brier',0))<0 else 'delta-down'}">{float(delta.get('brier',0)):+.4f}</div>
  </div>

  <div class="glass-card p-5">
    <div class="kpi-label">Recall(Y)</div>
    <div class="kpi-value" style="color:#fb923c">{pct(tm['recall'])}</div>
    <div class="mt-1 text-xs {'delta-down' if float(delta.get('recall',0))<0 else 'delta-up'}">{float(delta.get('recall',0)):+.2%}</div>
  </div>

  <div class="glass-card p-5">
    <div class="kpi-label">Precision(Y)</div>
    <div class="kpi-value" style="color:#34d399">{pct(tm['precision'])}</div>
    <div class="mt-1 text-xs delta-up">+{float(delta.get('precision',0)):+.2%}</div>
  </div>

  <div class="glass-card p-5 col-span-2">
    <div class="kpi-label">预期净回收（代理值）</div>
    <div class="kpi-value" style="color:#4ade80">¥{fmt_num(policy['expected_net_recovery_total'],0)}</div>
    <div class="flex items-center gap-4 mt-1">
      <span class="text-xs delta-up">+¥{fmt_num(delta.get('expected_net_recovery_total',0),0)} vs 基线</span>
      <span class="text-xs" style="color:var(--text-secondary)">基线: ¥{fmt_num(bpolicy['expected_net_recovery_total'],0)}</span>
    </div>
  </div>

  <div class="glass-card p-5">
    <div class="kpi-label">预期 ROI</div>
    <div class="kpi-value" style="color:#f472b6">{float(policy['expected_roi']):.2f}x</div>
    <div class="mt-1 text-xs delta-up">+{float(delta.get('expected_roi',0)):+.2f}x vs 基线</div>
  </div>

  <div class="glass-card p-5">
    <div class="kpi-label">T集账户数</div>
    <div class="kpi-value" style="color:#c4b5fd">{do['test_rows']:,}</div>
    <div class="mt-1 text-xs" style="color:var(--text-secondary)">正样本率 {do['test_positive_rate_pct']:.2f}%</div>
  </div>

</section>


<!-- ==================== 核心对比区域 ==================== -->
<section class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">

  <!-- Agent vs Baseline 并排对比 -->
  <div class="lg:col-span-2 glass-card p-6">
    <h2 class="text-lg font-bold mb-5 flex items-center gap-2">
      <svg class="w-5 h-5" style="color:var(--primary)" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/></svg>
      Agent vs 基线模型 — T集核心指标对比
    </h2>
    <div class="overflow-x-auto">
      <table class="data-table" id="abCompareTable">
        <thead><tr>
          <th>指标</th><th>{metrics['best_model']}<br/><span class="badge badge-agent text-xs">Agent Champion</span></th>
          <th>{metrics['baseline_model']}<br/><span class="badge badge-baseline text-xs">Baseline</span></th>
          <th>差额 (Agent - 基线)</th><th>判断</th></tr></thead>
        <tbody>
          <tr>
            <td class="font-semibold">ROC-AUC ↑</td><td style="color:#38bdf8;font-weight:700">{fmt_num(tm['roc_auc'],3)}</td>
            <td style="color:#fbbf24;font-weight:700">{fmt_num(btm['roc_auc'],3)}</td>
            <td class="{'delta-up' if float(delta.get('roc_auc',0))>0 else 'delta-down'}">+{fmt_num(delta.get('roc_auc',0),3)}</td>
            <td><span class="text-xs px-2 py-1 rounded-full" style="background:rgba(16,185,129,0.15);color:#34d399">✓ 更优区分力</span></td>
          </tr>
          <tr>
            <td class="font-semibold">Brier Score ↓</td><td style="color:#38bdf8;font-weight:700">{fmt_num(tm['brier'],4)}</td>
            <td style="color:#fbbf24;font-weight:700">{fmt_num(btm['brier'],4)}</td>
            <td class="{'delta-up' if float(delta.get('brier',0))<0 else 'delta-down'}">{float(delta.get('brier',0)):+.4f}</td>
            <td><span class="text-xs px-2 py-1 rounded-full" style="background:rgba(16,185,129,0.15);color:#34d399">✓ 校准更准</span></td>
          </tr>
          <tr>
            <td class="font-semibold">LogLoss ↓</td><td style="color:#38bdf8;font-weight:700">{fmt_num(tm['log_loss'],4)}</td>
            <td style="color:#fbbf24;font-weight:700">{fmt_num(btm['log_loss'],4)}</td>
            <td class="{'delta-up' if float(delta.get('log_loss',0))<0 else 'delta-down'}">{float(delta.get('log_loss',0)):+.4f}</td>
            <td><span class="text-xs px-2 py-1 rounded-full" style="background:rgba(16,185,129,0.15);color:#34d399">✓ 预测损失更低</span></td>
          </tr>
          <tr>
            <td class="font-semibold">Recall(Y) ↑</td><td style="color:#38bdf8;font-weight:700">{pct(tm['recall'])}</td>
            <td style="color:#fbbf24;font-weight:700">{pct(btm['recall'])}</td>
            <td class="{'delta-down' if float(delta.get('recall',0))<0 else 'delta-up'}">{float(delta.get('recall',0)):+.2%}</td>
            <td><span class="text-xs px-2 py-1 rounded-full" style="background:rgba(245,158,11,0.15);color:#fbbf24">⚠ 少召回 5.45pp</span></td>
          </tr>
          <tr>
            <td class="font-semibold">Precision(Y) ↑</td><td style="color:#38bdf8;font-weight:700">{pct(tm['precision'])}</td>
            <td style="color:#fbbf24;font-weight:700">{pct(btm['precision'])}</td>
            <td class="{'delta-up' if float(delta.get('precision',0))>0 else 'delta-down'}">+{float(delta.get('precision',0)):+.2%}</td>
            <td><span class="text-xs px-2 py-1 rounded-full" style="background:rgba(16,185,129,0.15);color:#34d399">✓ 精度更高</span></td>
          </tr>
          <tr>
            <td class="font-semibold">预期净回收 ↑</td><td style="color:#4ade80;font-weight:700">¥{fmt_num(policy['expected_net_recovery_total'],0)}</td>
            <td style="color:#fbbf24;font-weight:700">¥{fmt_num(bpolicy['expected_net_recovery_total'],0)}</td>
            <td class="delta-up">+¥{fmt_num(delta.get('expected_net_recovery_total',0),0)}</td>
            <td><span class="text-xs px-2 py-1 rounded-full" style="background:rgba(16,185,129,0.15);color:#34d399">✓ 业务价值更高</span></td>
          </tr>
          <tr>
            <td class="font-semibold">预期 ROI ↑</td><td style="color:#f472b6;font-weight:700">{float(policy['expected_roi']):.2f}x</td>
            <td style="color:#fbbf24;font-weight:700">{float(bpolicy['expected_roi']):.2f}x</td>
            <td class="delta-up">+{float(delta.get('expected_roi',0)):+.2f}x</td>
            <td><span class="text-xs px-2 py-1 rounded-full" style="background:rgba(16,185,129,0.15);color:#34d399">✓ 投入产出更好</span></td>
          </tr>
          <tr>
            <td class="font-semibold">决策阈值</td><td style="color:#38bdf8;font-weight:700">{fmt_num(tm['threshold'],2)}</td>
            <td style="color:#fbbf24;font-weight:700">{fmt_num(btm['threshold'],2)}</td>
            <td class="delta-neutral">+{float(tm.get('threshold',0)-btm.get('threshold',0)):+.2f}</td>
            <td><span class="text-xs px-2 py-1 rounded-full" style="background:rgba(148,163,184,0.15);color:#94a3b8">策略阈值不同</span></td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- 数据解读面板 -->
  <div class="glass-card p-6 flex flex-col">
    <h2 class="text-lg font-bold mb-4 flex items-center gap-2">
      <svg class="w-5 h-5" style="color:#10b981" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
      数据解读
    </h2>
    <div class="space-y-4 flex-1">
      <div class="insight-panel">
        <strong>结论：Agent 模型值得投产。</strong>
        <p class="mt-2" style="color:var(--text-secondary)">
          XGBoost 在 T 集 ROC-AUC 达 <strong>0.736</strong>，比 Logistic Regression 基线高出 <strong>+0.086</strong>，
          Brier 和 LogLoss 也全面优于基线。虽然 Recall 低了 5.45 个百分点（基线用极低阈值换来了高召回），
          但 Precision 提升了 <strong>+4.30pp</strong>，意味着每触达 100 个"预计会付款"的账户，实际付款人从 11 人提升到 15 人。
        </p>
      </div>

      <div class="insight-panel" style="border-color:#10b981;">
        <strong>经济价值：</strong>
        在相同产能约束和成本假设下，Agent 的预期净回收比基线高 <strong>¥{fmt_num(abs(float(delta.get('expected_net_recovery_total',316416))),0)}</strong>，
        ROI 从 <strong>{float(bpolicy['expected_roi']):.1f}x</strong> 提升到 <strong>{float(policy['expected_roi']):.1f}x</strong>。
        这说明<strong>更高的区分精度直接转化为催收资源投放效率的改善</strong>。
      </div>

      <div class="insight-panel" style="border-color:#f59e0b;">
        <strong>风险提示：</strong>
        基线模型的 Recall 更高（86% vs 80%），
        如果业务场景对"漏掉任何一个可能付款的客户"极度敏感，需要结合具体容忍度调整阈值或采用混合策略。
      </div>
    </div>
  </div>
</section>


<!-- ==================== 标签页导航 ==================== -->
<div class="flex items-center gap-1 border-b border-white/5 mb-0" role="tablist">
  <button class="tab-btn active" onclick="switchTab(event,'tab-models')" role="tab">🏆 模型比较</button>
  <button class="tab-btn" onclick="switchTab(event,'tab-queue')" role="tab">📋 生产队列</button>
  <button class="tab-btn" onclick="switchTab(event,'tab-accounts')" role="tab">👤 账户明细</button>
  <button class="tab-btn" onclick="switchTab(event,'tab-features')" role="tab">🎯 特征重要性</button>
  <button class="tab-btn" onclick="switchTab(event,'tab-signals')" role="tab">📊 组合信号</button>
</div>


<!-- ==================== Tab 1: 模型比较 ==================== -->
<div class="tab-content active glass-card p-6" id="tab-models">
  <div class="flex items-center justify-between mb-5">
    <h2 class="text-lg font-bold">Champion-Challenger 对比表（验证集）</h2>
    <span class="text-xs" style="color:var(--text-secondary)">点击表头可按列排序 · 点击行可查看详情</span>
  </div>
  <div class="overflow-x-auto rounded-xl" style="border:1px solid var(--border)">
    <table class="data-table" id="championTable">
      <thead><tr>
        <th data-sort="model_role" data-type="str">角色 <span class="sort-icon">↕</span></th>
        <th data-sort="model_name" data-type="str">模型 <span class="sort-icon">↕</span></th>
        <th data-sort="roc_auc" data-type="num">Valid ROC-AUC <span class="sort-icon">↕</span></th>
        <th data-sort="brier" data-type="num">Valid Brier <span class="sort-icon">↕</span></th>
        <th data-sort="recall" data-type="num">Recall(Y) <span class="sort-icon">↕</span></th>
        <th data-sort="precision" data-type="num">Precision(Y) <span class="sort-icon">↕</span></th>
        <th data-sort="expected_net_recovery_total" data-type="num">Expected Net Recovery <span class="sort-icon">↕</span></th>
        <th data-sort="expected_roi" data-type="num">Expected ROI <span class="sort-icon">↕</span></th>
        <th data-sort="threshold" data-type="num">Threshold <span class="sort-icon">↕</span></th></tr></thead>
      <tbody>
'''

    # Champion rows
    for row in champion_csv:
        role_class = "badge-baseline" if row.get("model_role") == "baseline" else ("badge-agent" if row.get("model_role") == "agent_champion" else "badge-challenger")
        html += f'''
        <tr onclick="showModelDetail(this)" data-model='{json.dumps(row, ensure_ascii=False)}'>
          <td><span class="badge {role_class}">{row.get('model_role','-')}</span></td>
          <td class="font-semibold">{row.get('model_name','-')}</td>
          <td>{fmt_num(row.get('roc_auc'),3)}</td>
          <td>{fmt_num(row.get('brier'),4)}</td>
          <td>{pct(row.get('recall'))}</td>
          <td>{pct(row.get('precision'))}</td>
          <td style="color:#4ade80;font-weight:600">¥{fmt_num(row.get('expected_net_recovery_total'),0)}</td>
          <td style="color:#f472b6;font-weight:600">{row.get('expected_roi','-.')}x</td>
          <td>{fmt_num(row.get('threshold'),2)}</td>
        </tr>'''

    html += '''
      </tbody>
    </table>
  </div>

  <!-- 混淆矩阵对比 -->
  <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mt-8">
    <div class="rounded-xl p-5" style="background:rgba(15,23,42,0.5);border:1px solid var(--border)">
      <h3 class="font-bold mb-3 text-sm" style="color:var(--text-secondary)">Agent Champion 混淆矩阵 @ threshold=''' + f"{float(tm['threshold']):.2f}" + '''</h3>
      <div class="grid grid-cols-3 gap-2 text-center text-sm">
        <div></div>
        <div class="font-bold pb-2" style="color:#94a3b8">Pred N</div>
        <div class="font-bold pb-2" style="color:#94a3b8">Pred Y</div>
        <div class="font-bold pt-2" style="color:#94a3b8">Actual N</div>
        <div class="rounded-lg p-3" style="background:rgba(56,189,248,0.1)"><span class="font-bold text-lg" style="color:#38bdf8">''' + str(int(tm['confusion_matrix']['tn'])) + '''</span><div class="text-xs" style="color:var(--text-secondary)">TN</div></div>
        <div class="rounded-lg p-3" style="background:rgba(239,68,68,0.1)"><span class="font-bold text-lg" style="color:#f87171">''' + str(int(tm['confusion_matrix']['fp'])) + '''</span><div class="text-xs" style="color:var(--text-secondary)">FP</div></div>
        <div class="font-bold pt-2" style="color:#94a3b8">Actual Y</div>
        <div class="rounded-lg p-3" style="background:rgba(251,146,60,0.1)"><span class="font-bold text-lg" style="color:#fb923c">''' + str(int(tm['confusion_matrix']['fn'])) + '''</span><div class="text-xs" style="color:var(--text-secondary)">FN</div></div>
        <div class="rounded-lg p-3" style="background:rgba(52,211,153,0.1)"><span class="font-bold text-lg" style="color:#34d399">''' + str(int(tm['confusion_matrix']['tp'])) + '''</span><div class="text-xs" style="color:var(--text-secondary)">TP</div></div>
      </div>
    </div>
    <div class="rounded-xl p-5" style="background:rgba(15,23,42,0.5);border:1px solid var(--border)">
      <h3 class="font-bold mb-3 text-sm" style="color:var(--text-secondary)">Baseline 混淆矩阵 @ threshold=''' + f"{float(btm['threshold']):.2f}" + '''</h3>
      <div class="grid grid-cols-3 gap-2 text-center text-sm">
        <div></div>
        <div class="font-bold pb-2" style="color:#94a3b8">Pred N</div>
        <div class="font-bold pb-2" style="color:#94a3b8">Pred Y</div>
        <div class="font-bold pt-2" style="color:#94a3b8">Actual N</div>
        <div class="rounded-lg p-3" style="background:rgba(56,189,248,0.1)"><span class="font-bold text-lg" style="color:#38bdf8">''' + str(int(btm['confusion_matrix']['tn'])) + '''</span><div class="text-xs" style="color:var(--text-secondary)">TN</div></div>
        <div class="rounded-lg p-3" style="background:rgba(239,68,68,0.1)"><span class="font-bold text-lg" style="color:#f87171">''' + str(int(btm['confusion_matrix']['fp'])) + '''</span><div class="text-xs" style="color:var(--text-secondary)">FP</div></div>
        <div class="font-bold pt-2" style="color:#94a3b8">Actual Y</div>
        <div class="rounded-lg p-3" style="background:rgba(251,146,60,0.1)"><span class="font-bold text-lg" style="color:#fb923c">''' + str(int(btm['confusion_matrix']['fn'])) + '''</span><div class="text-xs" style="color:var(--text-secondary)">FN</div></div>
        <div class="rounded-lg p-3" style="background:rgba(52,211,153,0.1)"><span class="font-bold text-lg" style="color:#34d399">''' + str(int(btm['confusion_matrix']['tp'])) + '''</span><div class="text-xs" style="color:var(--text-secondary)">TP</div></div>
      </div>
    </div>
  </div>
'''

    html += '</div><!-- end tab-models -->'


    # ==================== Tab 2: 生产队列 ====================
    html += '''
<!-- ==================== Tab 2: 生产队列 ==================== -->
<div class="tab-content glass-card p-6" id="tab-queue">
  <div class="flex items-center justify-between mb-5">
    <h2 class="text-lg font-bold">生产队列分布与收益结构</h2>
  </div>

  <!-- 队列可视化条形图 -->
  <div class="space-y-5 mb-8">
    <h3 class="text-sm font-bold" style="color:var(--text-secondary)">各队列账户数占比 & 预期净回收贡献</h3>
'''

    queue_colors = {
        "High Priority (Agent Call)": {"bg": "#10b981", "light": "rgba(16,185,129,0.18)"},
        "Medium Priority (Auto-Dialer)": {"bg": "#0ea5e9", "light": "rgba(14,165,233,0.18)"},
        "Low Priority (SMS/Email)": {"bg": "#f59e0b", "light": "rgba(245,158,11,0.18)"},
        "Write-off / Ignore": {"bg": "#64748b", "light": "rgba(100,116,139,0.18)"},
    }

    total_acc = sum(int(q.get("accounts", 0)) for q in queue_csv)
    total_nr = sum(float(q.get("expected_net_recovery_total", 0)) for q in queue_csv)

    for q in queue_csv:
        name = q.get("recommended_action", "-")
        acc = int(q.get("accounts", 0))
        nr = float(q.get("expected_net_recovery_total", 0))
        roi = float(q.get("expected_roi", 0))
        apr = float(q.get("actual_payer_rate", 0))
        acc_pct = acc / max(total_acc, 1) * 100
        nr_pct = nr / max(total_nr, 1) * 100
        qc = queue_colors.get(name, {"bg": "#64748b", "light": "rgba(100,116,139,0.18)"})
        short_name = name.split("(")[0].strip()

        html += f'''
    <div>
      <div class="flex items-center justify-between mb-1.5">
        <div class="flex items-center gap-2">
          <span class="w-3 h-3 rounded-full" style="background:{qc['bg']}"></span>
          <span class="font-semibold text-sm">{short_name}</span>
        </div>
        <div class="flex items-center gap-4 text-xs" style="color:var(--text-secondary)">
          <span>{acc:,} 户 ({acc_pct:.1f}%)</span>
          <span>|</span>
          <span>净回收 ¥{fmt_num(nr,0)} ({nr_pct:.1f}%)</span>
          <span>|</span>
          <span>ROI {roi:.1f}x</span>
          <span>|</span>
          <span>实付率 {apr*100:.1f}%</span>
        </div>
      </div>
      <div class="bar-container">
        <div class="bar-fill" style="width:{nr_pct:.1f}%;background:linear-gradient(90deg,{qc['light']},{qc['bg']});">{nr_pct:.1f}% of 净回收</div>
      </div>
    </div>'''

    html += '''
  </div>

  <!-- 队列表格 -->
  <h3 class="text-sm font-bold mb-3" style="color:var(--text-secondary)">队列详细指标</h3>
  <div class="overflow-x-auto rounded-xl" style="border:1px solid var(--border)">
    <table class="data-table" id="queueTable">
      <thead><tr>
        <th data-sort="recommended_action" data-type="str">推荐动作 <span class="sort-icon">↕</span></th>
        <th data-sort="accounts" data-type="num">账户数 <span class="sort-icon">↕</span></th>
        <th data-sort="avg_calibrated_prob" data-type="num">Avg Calib PD <span class="sort-icon">↕</span></th>
        <th data-sort="actual_payer_rate" data-type="num">实付率 <span class="sort-icon">↕</span></th>
        <th data-sort="balance_proxy_total" data-type="num">余额代理总值 <span class="sort-icon">↕</span></th>
        <th data-sort="expected_gross_recovery_total" data-type="num">毛回收预期 <span class="sort-icon">↕</span></th>
        <th data-sort="expected_net_recovery_total" data-type="num">净回收预期 <span class="sort-icon">↕</span></th>
        <th data-sort="contact_cost_total" data-type="num">触达成本 <span class="sort-icon">↕</span></th>
        <th data-sort="expected_roi" data-type="num">ROI <span class="sort-icon">↕</span></th></tr></thead>
      <tbody>
'''

    for q in queue_csv:
        html += f'''
        <tr>
          <td class="font-semibold">{q.get('recommended_action','-')}</td>
          <td>{int(q.get('accounts',0)):,}</td>
          <td>{fmt_num(q.get('avg_calibrated_prob'),4)}</td>
          <td>{pct(q.get('actual_payer_rate'))}</td>
          <td>¥{fmt_num(q.get('balance_proxy_total'),0)}</td>
          <td style="color:#86efac">¥{fmt_num(q.get('expected_gross_recovery_total'),0)}</td>
          <td style="color:#4ade80;font-weight:600">¥{fmt_num(q.get('expected_net_recovery_total'),0)}</td>
          <td>¥{fmt_num(q.get('contact_cost_total'),0)}</td>
          <td style="color:#f472b6;font-weight:600">{float(q.get('expected_roi',0)):.2f}x</td>
        </tr>'''

    html += '''  </tbody></table></div>

  <div class="mt-6 insight-panel">
    <strong>队列解读：</strong>
    <ul class="list-disc ml-5 mt-2 space-y-1" style="color:var(--text-secondary)">
'''

    # 动态生成解读
    hp = next((q for q in queue_csv if "High" in q.get("recommended_action","")), None)
    mp = next((q for q in queue_csv if "Medium" in q.get("recommended_action","")), None)
    lp = next((q for q in queue_csv if "Low" in q.get("recommended_action","")), None)
    wo = next((q for q in queue_csv if "Write" in q.get("recommended_action","")), None)

    if hp:
        html += f'<li><strong>人工坐席队列（{int(hp["accounts"])}户）</strong>：平均校准概率最高({float(hp["avg_calibrated_prob"])*100:.1f}%)，单户净回收期望也最高，是催收资源的"头部战场"。但ROI仅{float(hp["expected_roi"]):.1f}x——因为人工成本高，需精选高概率、高余额账户。</li>\n'
    if mp:
        html += f'<li><strong>自动外呼队列（{int(mp["accounts"])}户）</strong>：规模最大的中间层，ROI高达{float(mp["expected_roi"]):.1f}x，是性价比最优的批量触达渠道。</li>\n'
    if lp:
        html += f'<li><strong>短信/邮件队列（{int(lp["accounts"])}户）</strong>：成本极低（¥{float(lp["contact_cost_total"]):,.0f}总成本），ROI达到{float(lp["expected_roi"]):.1f}x，适合做低成本覆盖兜底。</li>\n'
    if wo:
        apr_wo = float(wo.get("actual_payer_rate",0))*100
        html += f'<li><strong>核销/忽略队列（{int(wo["accounts"])}户）</strong>：模型判断不值得投入触达资源，实际付款率仅{apr_wo:.1f}%。若后续有零成本渠道（如APP推送），可考虑重新评估。</li>\n'

    html += '</ul></div></div>'  # end tab-queue


    # ==================== Tab 3: 账户明细 ====================
    html += f'''
<!-- ==================== Tab 3: 账户明细 ==================== -->
<div class="tab-content glass-card p-6" id="tab-accounts">
  <div class="flex items-center justify-between mb-5">
    <h2 class="text-lg font-bold">T集账户级评分明细 <span class="text-sm font-normal" style="color:var(--text-secondary)">（展示前 {len(scored_sampled)} 条 / 共 {len(scored_raw)} 条）</span></h2>
  </div>

  <!-- 筛选栏 -->
  <div class="flex flex-wrap items-center gap-3 mb-5">
    <input type="text" class="filter-input" placeholder="🔍 搜索 ID / 地区 / 类型..." oninput="filterAccounts()" id="acctSearch"/>
    <select class="filter-select" onchange="filterAccounts()" id="acctQueueFilter">
      <option value="">全部队列</option>
'''
    for q in queue_csv:
        html += f'<option value="{q["recommended_action"]}">{q["recommended_action"].split("(")[0].strip()}</option>\n'

    html += f'''    </select>
    <select class="filter-select" onchange="filterAccounts()" id="acctPayerFilter">
      <option value="">全部标记</option>
      <option value="Y">实际付款人 (Y)</option>
      <option value="N">非付款人 (N)</option>
    </select>
    <select class="filter-select" onchange="filterAccounts()" id="acctLoanTypeFilter">
      <option value="">全部贷款类型</option>
'''
    loan_types = sorted(set(r.get("loan_type","") for r in scored_sampled if r.get("loan_type")))
    for lt in loan_types:
        html += f'<option value="{lt}">{lt}</option>\n'

    html += f'''    </select>
    <span id="acctFilteredCount" class="text-sm ml-auto" style="color:var(--text-secondary)"></span>
  </div>

  <div class="overflow-x-auto rounded-xl" style="border:1px solid var(--border);max-height:600px;overflow-y:auto">
    <table class="data-table" id="accountTable">
      <thead><tr>
        <th data-sort="id" data-type="str">ID <span class="sort-icon">↕</span></th>
        <th data-sort="loan_type" data-type="str">类型 <span class="sort-icon">↕</span></th>
        <th data-sort="purchased_bal_gp" data-type="str">余额组 <span class="sort-icon">↕</span></th>
        <th data-sort="district" data-type="str">地区 <span class="sort-icon">↕</span></th>
        <th data-sort="payer_3yr" data-type="str">实付 <span class="sort-icon">↕</span></th>
        <th data-sort="balance_proxy" data-type="num">余额 <span class="sort-icon">↕</span></th>
        <th data-sort="calibrated_repay_prob" data-type="num">Calib PD <span class="sort-icon">↕</span></th>
        <th data-sort="raw_repay_prob" data-type="num">Raw PD <span class="sort-icon">↕</span></th>
        <th data-sort="predicted_payer_flag" data-type="str">预判 <span class="sort-icon">↕</span></th>
        <th data-sort="expected_net_recovery" data-type="num">净回收 <span class="sort-icon">↕</span></th>
        <th data-sort="recommended_action" data-type="str">推荐 <span class="sort-icon">↕</span></th>
        <th data-sort="recommended_contact_cost" data-type="num">成本 <span class="sort-icon">↕</span></th>
        <th data-sort="policy_rank" data-type="num">优先级 <span class="sort-icon">↕</span></th></tr></thead>
      <tbody id="accountTableBody">
'''

    for i, r in enumerate(scored_sampled):
        action = r.get("recommended_action", "")
        action_short = action.split("(")[0].strip() if action else "-"
        payer_flag = r.get("payer_3yr", "")
        pred_flag = r.get("predicted_payer_flag", "")

        # 行样式
        row_style = ""
        if payer_flag == "Y":
            row_style = "background:rgba(52,211,153,0.04)"
        elif pred_flag == "Y" and payer_flag == "N":
            row_style = "background:rgba(239,68,68,0.04)"

        html += f'''
        <tr class="account-row" data-loan="{r.get('loan_type','')}" data-payer="{r.get('payer_3yr','')}" data-action="{action}" style="{row_style}"
            onclick="showAccountDetail(this)" data-account-index="{i}">
          <td class="font-mono text-xs">{r.get('id','')}</td>
          <td><span class="text-xs px-2 py-0.5 rounded" style="background:rgba(99,102,241,0.12);color:#a5b4fc">{r.get('loan_type','-')}</span></td>
          <td>{r.get('purchased_bal_gp','-')}</td>
          <td>{r.get('district','-')}</td>
          <td><span class="font-bold {'text-emerald-400' if payer_flag=='Y' else ''}">{payer_flag}</span></td>
          <td>{fmt_num(r.get('balance_proxy'),0)}</td>
          <td style="color:#38bdf8;font-weight:600">{fmt_num(r.get('calibrated_repay_prob'),4)}</td>
          <td style="color:var(--text-secondary)">{fmt_num(r.get('raw_repay_prob'),4)}</td>
          <td><span class="font-bold {'text-emerald-400' if pred_flag=='Y' else 'text-red-400'}">{pred_flag}</span></td>
          <td style="color:#4ade80;font-weight:600">¥{fmt_num(r.get('expected_net_recovery'),0)}</td>
          <td><span class="text-xs px-2 py-0.5 rounded" style="
            {"background:rgba(16,185,129,0.12);color:#34d399" if "High" in action else
             ("background:rgba(14,165,233,0.12);color:#38bdf8" if "Medium" in action else
              ("background:rgba(245,158,11,0.12);color:#fbbf24" if "Low" in action else
               "background:rgba(100,116,139,0.12);color:#94a3b8"))}
          ">{action_short}</span></td>
          <td>¥{fmt_num(r.get('recommended_contact_cost'),0)}</td>
          <td>{r.get('policy_rank','-')}</td>
        </tr>'''

    html += '''
      </tbody>
    </table>
  </div>
</div>'''  # end tab-accounts


    # ==================== Tab 4: 特征重要性 ====================
    html += f'''
<!-- ==================== Tab 4: 特征重要性 ==================== -->
<div class="tab-content glass-card p-6" id="tab-features">
  <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
    <div>
      <h2 class="text-lg font-bold mb-5">Top 预测特征排名</h2>
      <div class="space-y-3">
'''

    max_imp = max((float(f.get("importance", 0)) for f in feature_csv), default=1)
    feature_names_map = {
        "birth_yr": "出生年份",
        "district": "所在地区",
        "purchased_bal_gp": "购买余额分组",
        "last_act_closing_m": "距最后活动月数",
        "co_closing_m": "距关账月数",
        "last_pay_date_client_closing_m": "距原债权人最后付款月数",
        "multiple_acct": "是否多账户",
        "home_phone_flag": "是否有座机",
        "mobile_phone_flag": "是否有手机号",
        "open_closing_m": "距开户月数",
        "loan_type": "贷款产品类型",
        "missing_last_act_flag": "最后活动缺失标记",
    }

    for fi, feat in enumerate(feature_csv):
        imp = float(feat.get("importance", 0))
        imp_pct = imp / max(max_imp, 0.001) * 100
        fname = feat.get("feature", "")
        cname = feature_names_map.get(fname, fname)
        bar_color = "#10b981" if fi == 0 else ("#0ea5e9" if fi < 4 else "#64748b")
        bar_light = "rgba(16,185,129,0.2)" if fi == 0 else ("rgba(14,165,233,0.2)" if fi < 4 else "rgba(100,116,139,0.2)")

        html += f'''
        <div>
          <div class="flex items-center justify-between mb-1">
            <span class="font-medium text-sm">{cname}<span class="ml-2 text-xs font-mono" style="color:var(--text-secondary)">({fname})</span></span>
            <span class="text-sm font-bold" style="color:{bar_color}">{imp:.4f}</span>
          </div>
          <div class="bar-container" style="height:20px">
            <div class="bar-fill" style="width:{imp_pct:.1f}%;background:linear-gradient(90deg,{bar_light},{bar_color});padding-right:8px;font-size:0.7rem;">#{fi+1}</div>
          </div>
        </div>'''

    html += '''  </div>
    </div>

    <div>
      <div class="chart-wrapper mb-6">
        <canvas id="featureChart"></canvas>
      </div>
      <div class="insight-panel mt-4">
        <strong>特征解读：</strong>
        <ul class="list-disc ml-5 mt-2 space-y-1" style="color:var(--text-secondary)">
'''

    # 动态特征解读
    business_insights = {
        "birth_yr": "<strong>出生年份</strong>是最强预测因子，暗示还款能力与年龄阶段高度相关（年轻群体可能收入上升期，年长群体可能有更多储蓄）。",
        "district": "<strong>所在地区</strong>排名第二，不同地区的经济活跃度和执法环境直接影响回收可行性。",
        "purchased_bal_gp": "<strong>余额分组</strong>影响显著，大额账户往往对应不同的处置方式和谈判空间。",
        "last_act_closing_m": "<strong>距最后活动时间</strong>越久，账户「沉睡」程度越高，回收难度越大。",
        "co_closing_m": "<strong>距关账时间</strong>反映不良资产的账龄，账龄越长通常回收率越低。",
        "last_pay_date_client_closing_m": "<strong>原债权人处最后付款时间</strong>缺失值本身就是一个强信号（从未付款）。",
        "multiple_acct": "<strong>多账户标记</strong>暗示债务复杂度和偿债优先级的差异。",
        "home_phone_flag": "<strong>座机号码</strong>的存在增加触达成功率。",
        "mobile_phone_flag": "<strong>手机号码</strong>同样提升联系可能性。",
        "open_closing_m": "<strong>开户时长</strong>反映客户关系深度。",
    }

    for fi, feat in enumerate(feature_csv[:6]):
        fname = feat.get("feature", "")
        insight = business_insights.get(fname, f"<strong>{fname}</strong>对区分付款人与非付款人有正向贡献。")
        html += f"<li>{insight}</li>\n"

    html += '''
        </ul>
      </div>
    </div>
  </div>
</div>'''  # end tab-features


    # ==================== Tab 5: 组合信号 ====================
    html += f'''
<!-- ==================== Tab 5: 组合信号 ==================== -->
<div class="tab-content glass-card p-6" id="tab-signals">
  <h2 class="text-lg font-bold mb-5">组合维度信号分析</h2>
  <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">

    <!-- 余额分组 -->
    <div class="rounded-xl p-5" style="background:rgba(15,23,42,0.5);border:1px solid var(--border)">
      <h3 class="font-bold mb-4 flex items-center gap-2">
        <span class="w-2 h-2 rounded-full" style="background:#10b981"></span> 按余额分组的付款率
      </h3>
      <div class="chart-wrapper" style="height:220px">
        <canvas id="balanceChart"></canvas>
      </div>
      <div class="mt-4 space-y-2">
'''

    for row in payer_balance:
        html += f'<div class="flex justify-between text-sm"><span>{row.get("purchased_bal_gp","-")}</span><span class="font-bold" style="color:{"#34d399" if float(row.get("payer_rate_pct",0)) > 10 else "#fbbf24"}">{row.get("payer_rate_pct","-.")}%</span></div>\n'

    html += '''  </div></div>

    <!-- 贷款类型 -->
    <div class="rounded-xl p-5" style="background:rgba(15,23,42,0.5);border:1px solid var(--border)">
      <h3 class="font-bold mb-4 flex items-center gap-2">
        <span class="w-2 h-2 rounded-full" style="background:#0ea5e9"></span> 按贷款类型的付款率
      </h3>
      <div class="chart-wrapper" style="height:220px">
        <canvas id="loanChart"></canvas>
      </div>
      <div class="mt-4 space-y-2">
'''
    for row in payer_loan:
        html += f'<div class="flex justify-between text-sm"><span>{row.get("loan_type","-")}</span><span class="font-bold" style="color:{"#34d399" if float(row.get("payer_rate_pct",0)) > 10 else "#fbbf24"}">{row.get("payer_rate_pct","-.")}%</span></div>\n'

    html += '''  </div></div>

    <!-- 手机标记 -->
    <div class="rounded-xl p-5" style="background:rgba(15,23,42,0.5);border:1px solid var(--border)">
      <h3 class="font-bold mb-4 flex items-center gap-2">
        <span class="w-2 h-2 rounded-full" style="background:#f59e0b"></span> 按手机号标记的付款率
      </h3>
      <div class="chart-wrapper" style="height:220px">
        <canvas id="mobileChart"></canvas>
      </div>
      <div class="mt-4 space-y-2">
'''
    for row in payer_mobile:
        flag_val = row.get("mobile_phone_flag", "?")
        pr = row.get("payer_rate_pct", 0)
        flag_label = "有手机号" if str(flag_val).lower() == "y" else ("无手机号" if str(flag_val).lower() == "n" else flag_val)
        html += f'<div class="flex justify-between text-sm"><span>{flag_label}</span><span class="font-bold" style="color:{"#34d399" if float(pr) > 10 else "#fbbf24"}">{pr}%</span></div>\n'

    html += '''  </div></div>

  </div>

  <!-- 集中度指标 -->
  <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mt-8">
    <div class="rounded-xl p-5 text-center" style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.2)">
      <div class="text-3xl font-extrabold" style="color:#34d399">{conc.get('prob_top20_actual_payer_capture_share_pct',0):.1f}%</div>
      <div class="text-xs mt-1" style="color:var(--text-secondary)">Top 20% 概率排序<br/>捕获的真实付款人占比</div>
    </div>
    <div class="rounded-xl p-5 text-center" style="background:rgba(14,165,233,0.08);border:1px solid rgba(14,165,233,0.2)">
      <div class="text-3xl font-extrabold" style="color:#38bdf8">{conc.get('net_top20_expected_net_recovery_capture_share_pct',0):.1f}%</div>
      <div class="text-xs mt-1" style="color:var(--text-secondary)">Top 20% 净回收排序<br/>贡献的净回收占比</div>
    </div>
    <div class="rounded-xl p-5 text-center" style="background:rgba(168,85,247,0.08);border:1px solid rgba(168,85,247,0.2)">
      <div class="text-3xl font-extrabold" style="color:#c084fc">{conc.get('prob_top20_actual_payer_rate_pct',0):.1f}%</div>
      <div class="text-xs mt-1" style="color:var(--text-secondary)">Top 20% 概率排序<br/>的实际付款率</div>
    </div>
    <div class="rounded-xl p-5 text-center" style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2)">
      <div class="text-3xl font-extrabold" style="color:#fbbf24">{conc.get('top20_accounts',0):,}</div>
      <div class="text-xs mt-1" style="color:var(--text-secondary)">Top 20% 对应<br/>账户绝对数量</div>
    </div>
  </div>

  <div class="mt-6 insight-panel">
    <strong>集中度解读：</strong>
    <p class="mt-2" style="color:var(--text-secondary)">
      按校准后概率排序时，前 20% 的账户就能捕获 <strong>{conc.get('prob_top20_actual_payer_capture_share_pct',0):.1f}%</strong> 的真实付款人；
      按净回收代理值排序时，前 20% 贡献了 <strong>{conc.get('net_top20_expected_net_recovery_capture_share_pct',0):.1f}%</strong> 的预期净回收。
      这意味着 <strong>集中资源打头部账户是当前最优策略</strong>，尾部账户的边际收益递减非常明显。
    </p>
  </div>
</div>'''  # end tab-signals


    # ==================== 账户详情模态框 ====================
    html += '''
<!-- ==================== 账户详情模态框 ==================== -->
<div class="modal-overlay" id="accountModal" onclick="if(event.target===this)this.classList.remove('show')">
  <div class="modal-body">
    <div class="flex items-center justify-between mb-6">
      <h2 class="text-xl font-bold" id="modalTitle">账户详情</h2>
      <button onclick="document.getElementById('accountModal').classList.remove('show')" class="w-9 h-9 rounded-full flex items-center justify-center hover:bg-red-500/20 transition" style="border:1px solid var(--border)">
        ✕
      </button>
    </div>
    <div id="modalContent" class="space-y-4"></div>
  </div>
</div>


<!-- ==================== 模型详情模态框 ==================== -->
<div class="modal-overlay" id="modelModal" onclick="if(event.target===this)this.classList.remove('show')">
  <div class="modal-body">
    <div class="flex items-center justify-between mb-6">
      <h2 class="text-xl font-bold" id="modalModelTitle">模型详情</h2>
      <button onclick="document.getElementById('modelModal').classList.remove('show')" class="w-9 h-9 rounded-full flex items-center justify-center hover:bg-red-500/20 transition" style="border:1px solid var(--border)">✕</button>
    </div>
    <div id="modalModelContent"></div>
  </div>
</div>


</main>

<footer class="text-center py-6 mt-8 text-xs" style="color:var(--text-secondary);border-top:1px solid rgba(255,255,255,0.04)">
  NPA Repayment Analysis Dashboard · Generated from baseline_comparison_run · Data is operational proxy only, not financial confirmation.
</footer>


<!-- ==================== JavaScript 交互逻辑 ==================== -->
<script>
// ========== 全局数据 ==========
const ACCOUNT_DATA = {json.dumps(scored_sampled, ensure_ascii=False)};

// ========== 标签页切换 ==========
function switchTab(e, tabId){{
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c=>c.classList.remove('active'));
  e.target.classList.add('active');
  document.getElementById(tabId).classList.add('active');

  // 切换到对应tab时渲染图表
  if(tabId === 'tab-features') renderFeatureChart();
  if(tabId === 'tab-signals') renderSignalCharts();
}}

// ========== 排序引擎 ==========
let currentSort = {{ table:null, key:null, direction:'desc' }};

function makeSortable(tableId){{
  const table = document.getElementById(tableId);
  if(!table) return;
  const headers = table.querySelectorAll('thead th[data-sort]');
  headers.forEach(th => {{
    th.addEventListener('click', () => {{
      const key = th.dataset.sort;
      const type = th.dataset.type || 'str';
      let dir = 'asc';
      if(currentSort.table === tableId && currentSort.key === key){{
        dir = currentSort.direction === 'asc' ? 'desc' : 'asc';
      }}
      currentSort = {{ table: tableId, key, direction: dir }};

      // 清除其他排序状态
      headers.forEach(h => h.classList.remove('sort-asc','sort-desc'));
      th.classList.add(dir === 'asc' ? 'sort-asc' : 'sort-desc');

      // 排序tbody rows
      const tbody = table.querySelector('tbody');
      const rows = Array.from(tbody.querySelectorAll('tr'));
      rows.sort((a,b) => {{
        let va = a.querySelector(`td:nth-child(${{Array.from(headers).indexOf(th)+1}})`)?.textContent?.trim() ?? '';
        let vb = b.querySelector(`td:nth-child(${{Array.from(headers).indexOf(th)+1}})`)?.textContent?.trim() ?? '';
        if(type === 'num'){{
          va = parseFloat(va.replace(/[^\\d.\\-]/g,'')) || 0;
          vb = parseFloat(vb.replace(/[^\\d.\\-]/g,'')) || 0;
        }}else{{
          va = va.toLowerCase(); vb = vb.toLowerCase();
        }}
        if(va < vb) return dir === 'asc' ? -1 : 1;
        if(va > vb) return dir === 'asc' ? 1 : -1;
        return 0;
      }});
      rows.forEach(r => tbody.appendChild(r));
    }});
  }});
}}

makeSortable('championTable');
makeSortable('queueTable');
makeSortable('accountTable');


// ========== 账户筛选 ==========
function filterAccounts(){{
  const search = (document.getElementById('acctSearch').value || '').toLowerCase();
  const queueF = document.getElementById('acctQueueFilter').value;
  const payerF = document.getElementById('acctPayerFilter').value;
  const loanF = document.getElementById('acctLoanTypeFilter').value;

  let count = 0;
  document.querySelectorAll('#accountTableBody tr.account-row').forEach(tr => {{
    const show =
      (!search || JSON.stringify(Object.values(tr.dataset)).toLowerCase().includes(search) ||
       tr.textContent.toLowerCase().includes(search)) &&
      (!queueF || tr.dataset.action === queueF) &&
      (!payerF || tr.dataset.payer === payerF) &&
      (!loanF || tr.dataset.loan === loanF);

    tr.style.display = show ? '' : 'none';
    if(show) count++;
  }});

  const total = ACCOUNT_DATA.length;
  document.getElementById('acctFilteredCount').textContent = `显示 ${count.toLocaleString()} / ${total.toLocaleString()}`;
}}

// ========== 账户下钻详情 ==========
function showAccountDetail(tr){{
  const idx = parseInt(tr.dataset.accountIndex);
  const d = ACCOUNT_DATA[idx];
  if(!d) return;

  document.getElementById('modalTitle').textContent = `账户 #${{d.id}} — 详情`;
  const actionColor = d.recommended_action.includes('High') ? '#10b981'
                    : d.recommended_action.includes('Medium') ? '#0ea5e9'
                    : d.recommended_action.includes('Low') ? '#f59e0b'
                    : '#64748b';

  const payerMatch = d.payer_3yr === d.predicted_payer_flag;
  const matchBadge = payerMatch
    ? '<span class="text-xs px-2 py-1 rounded-full" style="background:rgba(16,185,129,0.15);color:#34d399">✓ 判定一致</span>'
    : '<span class="text-xs px-2 py-1 rounded-full" style="background:rgba(239,68,68,0.15);color:#f87171">✗ 判定偏差</span>';

  document.getElementById('modalContent').innerHTML = `
    <div class="grid grid-cols-2 gap-4">
      <div class="rounded-lg p-4" style="background:rgba(15,23,42,0.5)">
        <div class="text-xs font-bold mb-2" style="color:var(--text-secondary)">基本信息</div>
        <div class="space-y-2 text-sm">
          <div class="flex justify-between"><span style="color:var(--text-secondary)">ID</span><span class="font-mono">${{d.id}}</span></div>
          <div class="flex justify-between"><span style="color:var(--text-secondary)">贷款类型</span><span>${{d.loan_type}}</span></div>
          <div class="flex justify-between"><span style="color:var(--text-secondary)">余额分组</span><span>${{d.purchased_bal_gp}}</span></div>
          <div class="flex justify-between"><span style="color:var(--text-secondary)">地区</span><span>${{d.district}}</span></div>
          <div class="flex justify-between"><span style="color:var(--text-secondary)">余额代理</span><span class="font-bold">¥${{Number(d.balance_proxy||0).toLocaleString()}}</span></div>
          <div class="flex justify-between"><span style="color:var(--text-secondary)">实际付款</span><span class="font-bold ${{d.payer_3yr==='Y'?'text-emerald-400':'text-red-400'}}">${{d.payer_3yr}}</span></div>
        </div>
      </div>
      <div class="rounded-lg p-4" style="background:rgba(15,23,42,0.5)">
        <div class="text-xs font-bold mb-2" style="color:var(--text-secondary)">模型评分</div>
        <div class="space-y-2 text-sm">
          <div class="flex justify-between"><span style="color:var(--text-secondary)">原始概率</span><span>${{(Number(d.raw_repay_prob)||0).toFixed(4)}}</span></div>
          <div class="flex justify-between"><span style="color:var(--text-secondary)">校准概率</span><span class="font-bold" style="color:#38bdf8">${{(Number(d.calibrated_repay_prob)||0).toFixed(4)}}</span></div>
          <div class="flex justify-between"><span style="color:var(--text-secondary)">判定标签</span><span class="font-bold ${{d.predicted_payer_flag==='Y'?'text-emerald-400':'text-red-400'}}">${{d.predicted_payer_flag}}</span></div>
          <div class="flex justify-between"><span style="color:var(--text-secondary)">判定结果</span><span>${{matchBadge}}</span></div>
          <div class="flex justify-between"><span style="color:var(--text-secondary)">优先级排名</span><span class="font-bold">#${{d.policy_rank}}</span></div>
        </div>
      </div>
    </div>

    <div class="rounded-lg p-4 mt-4" style="background:rgba(15,23,42,0.5)">
      <div class="text-xs font-bold mb-3" style="color:var(--text-secondary)">经济测算</div>
      <div class="grid grid-cols-3 gap-4 text-center">
        <div class="rounded-lg p-3" style="background:rgba(134,239,172,0.08)">
          <div class="text-xs" style="color:var(--text-secondary)">毛回收预期</div>
          <div class="font-bold text-lg" style="color:#86efac">¥${{Number(d.expected_gross_recovery||0).toLocaleString(undefined,{{maximumFractionDigits:0}})}}</div>
        </div>
        <div class="rounded-lg p-3" style="background:rgba(74,222,128,0.08)">
          <div class="text-xs" style="color:var(--text-secondary)">净回收预期</div>
          <div class="font-bold text-lg" style="color:#4ade80">¥${{Number(d.expected_net_recovery||0).toLocaleString(undefined,{{maximumFractionDigits:0}})}}</div>
        </div>
        <div class="rounded-lg p-3" style="background:rgba(248,113,113,0.08)">
          <div class="text-xs" style="color:var(--text-secondary)">触达成本</div>
          <div class="font-bold text-lg" style="color:#f87171">¥${{Number(d.recommended_contact_cost||0).toLocaleString(undefined,{{maximumFractionDigits:0}})}}</div>
        </div>
      </div>
    </div>

    <div class="rounded-lg p-4 mt-4" style="background:rgba(15,23,42,0.5)">
      <div class="text-xs font-bold mb-2" style="color:var(--text-secondary)">推荐动作</div>
      <div class="flex items-center gap-3">
        <span class="px-4 py-2 rounded-lg font-bold text-sm" style="background:${{actionColor}}22;color:${{actionColor}};border:1px solid ${{actionColor}}44">
          ${{d.recommended_action}}
        </span>
        <span class="text-xs" style="color:var(--text-secondary)">
          ${{d.recommended_action.includes('High')?'该账户被分配到人工坐席队列，建议优先处理，关注其较高的付款概率和余额。'
           :d.recommended_action.includes('Medium')?'该账户进入自动外呼队列，适合规模化触达。'
           :d.recommended_action.includes('Low')?'该账户通过短信/邮件低成本触达即可。'
           :'该账户不建议投入当前轮次的人工或外呼资源。'}}
        </span>
      </div>
    </div>

    <div class="rounded-lg p-4 mt-4" style="background:linear-gradient(135deg,rgba(14,165,233,0.06),rgba(16,185,129,0.04));border-left:3px solid var(--primary)">
      <div class="text-xs font-bold mb-1">AI 解读</div>
      <div class="text-sm leading-relaxed" style="color:var(--text-secondary)">
        ${{generateAccountInsight(d)}}
      </div>
    </div>
  `;
  document.getElementById('accountModal').classList.add('show');
}}

function generateAccountInsight(d){{
  const calibP = Number(d.calibrated_repay_prob)||0;
  const rawP = Number(d.raw_repay_prob)||0;
  const bal = Number(d.balance_proxy)||0;
  const isPayer = d.payer_3yr === 'Y';
  const isPredicted = d.predicted_payer_flag === 'Y';
  const netR = Number(d.expected_net_recovery)||0;

  let parts = [];

  if(calibP > 0.15) parts.push(`该账户的校准付款概率高达 **$${(calibP*100).toFixed(1)}%** ，远高于T集平均水平（~9%），属于高潜客户。`);
  else if(calibP > 0.08) parts.push(`校准概率 **$${(calibP*100).toFixed(1)}%** ，处于中等水平，需要结合余额和其他信号综合判断。`);
  else parts.push(`校准概率 **$${(calibP*100).toFixed(1)}%** 较低，默认情况下不建议投入高成本触达渠道。`);

  if(bal > 150000) parts.push(`余额代理值 ¥${{bal.toLocaleString()}} 属于较高水平，即便概率偏低也可能产生可观回收。`);
  else if(bal > 50000) parts.push(`余额处于中等区间（¥${{bal.toLocaleString()}}）。`);
  else parts.push(`余额较低（¥${{bal.toLocaleString()}}），回收上限有限。`);

  if(isPayer && isPredicted) parts.push(`**模型命中**：该客户确实发生了付款，模型正确识别。`);
  else if(isPayer && !isPredicted) parts.push(`**漏判案例**：该客户实际付款了但模型未预测到（False Negative），值得回溯其特征模式。`);
  else if(!isPayer && isPredicted) parts.push(`**误判案例**：模型预测会付款但实际未付（False Positive），可能是概率校准偏乐观或存在未被模型捕捉的特殊情况。`);
  else parts.push(`**正确排除**：模型正确识别该客户不会在观察期内付款。`);

  return parts.join(' ');
}}

// ========== 模型行点击详情 ==========
function showModelDetail(tr){{
  const d = JSON.parse(tr.dataset.model || '{{}}');
  if(!d || Object.keys(d).length === 0) return;

  const roleBadge = d.model_role === 'baseline' ? '<span class="badge badge-baseline">Baseline</span>'
                   : d.model_role === 'agent_champion' ? '<span class="badge badge-agent">Champion</span>'
                   : '<span class="badge badge-challenger">Challenger</span>';

  document.getElementById('modalModelTitle').textContent = `${{d.model_name}} 详情`;
  document.getElementById('modalModelContent').innerHTML = `
    <div class="text-center mb-6">
      ${roleBadge}
      <h3 class="text-2xl font-bold mt-3">${{d.model_name}}</h3>
    </div>
    <div class="grid grid-cols-2 gap-4">
      <div class="rounded-lg p-4 text-center" style="background:rgba(56,189,248,0.08);border:1px solid rgba(56,189,248,0.2)">
        <div class="text-xs" style="color:var(--text-secondary)">ROC-AUC</div>
        <div class="text-2xl font-bold" style="color:#38bdf8">${{Number(d.roc_auc).toFixed(3)}}</div>
      </div>
      <div class="rounded-lg p-4 text-center" style="background:rgba(167,139,250,0.08);border:1px solid rgba(167,139,250,0.2)">
        <div class="text-xs" style="color:var(--text-secondary)">Brier</div>
        <div class="text-2xl font-bold" style="color:#a78bfa">${{Number(d.brier).toFixed(4)}}</div>
      </div>
      <div class="rounded-lg p-4 text-center" style="background:rgba(251,146,60,0.08);border:1px solid rgba(251,146,60,0.2)">
        <div class="text-xs" style="color:var(--text-secondary)">Recall(Y)</div>
        <div class="text-2xl font-bold" style="color:#fb923c">${{(Number(d.recall)*100).toFixed(2)}}%</div>
      </div>
      <div class="rounded-lg p-4 text-center" style="background:rgba(52,211,153,0.08);border:1px solid rgba(52,211,153,0.2)">
        <div class="text-xs" style="color:var(--text-secondary)">Precision(Y)</div>
        <div class="text-2xl font-bold" style="color:#34d399">${{(Number(d.precision)*100).toFixed(2)}}%</div>
      </div>
      <div class="rounded-lg p-4 text-center" style="background:rgba(74,222,128,0.08);border:1px solid rgba(74,222,128,0.2)">
        <div class="text-xs" style="color:var(--text-secondary)">Expected Net Recovery</div>
        <div class="text-2xl font-bold" style="color:#4ade80">¥${{Number(d.expected_net_recovery_total).toLocaleString(undefined,{{maximumFractionDigits:0}})}}</div>
      </div>
      <div class="rounded-lg p-4 text-center" style="background:rgba(244,114,182,0.08);border:1px solid rgba(244,114,182,0.2)">
        <div class="text-xs" style="color:var(--text-secondary)">ROI</div>
        <div class="text-2xl font-bold" style="color:#f472b6">${{Number(d.expected_roi).toFixed(2)}}x</div>
      </div>
      <div class="col-span-2 rounded-lg p-4 text-center" style="background:rgba(148,163,184,0.08);border:1px solid rgba(148,163,184,0.2)">
        <div class="text-xs" style="color:var(--text-secondary)">Decision Threshold</div>
        <div class="text-2xl font-bold" style="color:#cbd5e1">${{Number(d.threshold).toFixed(2)}}</div>
      </div>
    </div>
  `;
  document.getElementById('modelModal').classList.add('show');
}}

// ========== 图表渲染 ==========
let featureChartRendered = false;
let signalChartsRendered = false;

function renderFeatureChart(){{
  if(featureChartRendered) return;
  featureChartRendered = true;
  const ctx = document.getElementById('featureChart');
  if(!ctx) return;
  new Chart(ctx, {{
    type: 'bar',
    data: {{
      labels: {json.dumps([f.get('feature','') for f in feature_csv], ensure_ascii=False)},
      datasets: [{{
        label: 'Importance',
        data: {[float(f.get('importance',0)) for f in feature_csv]},
        backgroundColor: [
          'rgba(16,185,129,0.7)','rgba(14,165,233,0.7)','rgba(14,165,233,0.7)',
          'rgba(14,165,233,0.7)','rgba(100,116,139,0.5)','rgba(100,116,139,0.5)',
          'rgba(100,116,139,0.5)','rgba(100,116,139,0.5)','rgba(100,116,139,0.5)',
          'rgba(100,116,139,0.5)','rgba(100,116,139,0.5)','rgba(100,116,139,0.5)',
        ],
        borderRadius: 6,
        borderSkipped: false,
      }}]
    }},
    options: {{
      indexAxis: 'y',
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{display:false}}, title: {{display:true,text:'特征重要性分布',color:'#94a3b8',font:{{size:13}}} }}},
      scales: {{
        x: {{ ticks: {{color:'#64748b'}, grid: {{color:'rgba(51,65,85,0.3)'}} }},
        y: {{ ticks: {{color:'#94a3b8',font:{{size:11}}}}, grid: {{display:false}} }}
      }}
    }}
  }});
}}

function renderSignalCharts(){{
  if(signalChartsRendered) return;
  signalChartsRendered = true;

  // Balance chart
  const bc = document.getElementById('balanceChart');
  if(bc) new Chart(bc, {{
    type: 'doughnut',
    data: {{
      labels: {json.dumps([r.get('purchased_bal_gp','') for r in payer_balance], ensure_ascii=False)},
      datasets: [{{
        data: {[float(r.get('payer_rate_pct',0)) for r in payer_balance]},
        backgroundColor: ['rgba(16,185,129,0.7)','rgba(14,165,233,0.7)','rgba(245,158,11,0.7)','rgba(168,85,247,0.7)'],
        borderWidth: 0,
      }}]
    }},
    options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{position:'bottom',labels:{{color:'#94a3b8',font:{{size:10},boxWidth:12}}}} }, title: {{display:true,text:'付款率 %',color:'#94a3b8',font:{{size:11}}}} }}, cutout:'55%' }}
  }});

  // Loan chart
  const lc = document.getElementById('loanChart');
  if(lc) new Chart(lc, {{
    type: 'doughnut',
    data: {{
      labels: {json.dumps([r.get('loan_type','') for r in payer_loan], ensure_ascii=False)},
      datasets: [{{
        data: {[float(r.get('payer_rate_pct',0)) for r in payer_loan]},
        backgroundColor: ['rgba(14,165,233,0.7)','rgba(251,146,60,0.7)','rgba(168,85,247,0.7)','rgba(236,72,153,0.7)'],
        borderWidth: 0,
      }}]
    }},
    options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{position:'bottom',labels:{{color:'#94a3b8',font:{{size:10},boxWidth:12}}}} }, title: {{display:true,text:'付款率 %',color:'#94a3b8',font:{{size:11}}}} }}, cutout:'55%' }}
  }});

  // Mobile chart
  const mc = document.getElementById('mobileChart');
  if(mc) new Chart(mc, {{
    type: 'doughnut',
    data: {{
      labels: {json.dumps([("有手机号" if str(r.get('mobile_phone_flag','')).lower()=='y' else ("无手机号" if str(r.get('mobile_phone_flag','')).lower()=='n' else r.get('mobile_phone_flag','?'))) for r in payer_mobile], ensure_ascii=False)},
      datasets: [{{
        data: {[float(r.get('payer_rate_pct',0)) for r in payer_mobile]},
        backgroundColor: ['rgba(245,158,11,0.7)','rgba(100,116,139,0.5)'],
        borderWidth: 0,
      }}]
    }},
    options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{position:'bottom',labels:{{color:'#94a3b8',font:{{size:10},boxWidth:12}}}} }, title: {{display:true,text:'付款率 %',color:'#94a3b8',font:{{size:11}}}} }}, cutout:'55%' }}
  }});
}}

// ========== 键盘快捷键 ==========
document.addEventListener('keydown', (e) => {{
  if(e.key === 'Escape'){{
    document.querySelectorAll('.modal-overlay').forEach(m=>m.classList.remove('show'));
  }}
}});
</script>
</body>
</html>'''

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"Dashboard 已生成: {OUTPUT_HTML}")
    print(f"大小: {len(html):,} 字符 | 包含 {len(scored_sampled)} 条账户记录")


if __name__ == "__main__":
    generate_dashboard()
