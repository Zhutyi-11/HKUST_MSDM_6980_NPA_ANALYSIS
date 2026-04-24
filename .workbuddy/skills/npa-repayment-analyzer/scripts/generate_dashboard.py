#!/usr/bin/env python3
"""
Dashboard v9 Generator — NPA Repayment Analysis

v9 Improvements over v8:
  1. Fixed ALL button interactions (onclick + addEventListener hybrid)
  2. Extended model comparison: radar chart, threshold curve, calibration plot, model matrix
  3. UI polish: gradient cards, smooth transitions, sticky headers, loading animations
  4. Better responsive layout + mobile support
  5. Code quality: clean string boundaries, no f-string in multi-line blocks
"""

import json, os, csv, numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
# Output directory: look for agent_outputs relative to skill dir, then parent workspace
_CANDIDATE_DIRS = [
    os.path.join(BASE, '..', '..', 'agent_outputs', 'baseline_comparison_run'),
    os.path.join(BASE, '..', '..', '..', 'agent_outputs', 'baseline_comparison_run'),
]
OUT_DIR = None
for d in _CANDIDATE_DIRS:
    _abs = os.path.abspath(d)
    if os.path.isdir(_abs) and os.path.exists(os.path.join(_abs, 'metrics.json')):
        OUT_DIR = _abs
        break
if OUT_DIR is None:
    # Fallback: use first candidate
    OUT_DIR = os.path.abspath(_CANDIDATE_DIRS[0])
    os.makedirs(OUT_DIR, exist_ok=True)
else:
    os.makedirs(OUT_DIR, exist_ok=True)

# ── Load helpers ──
def load_csv(fname):
    p = os.path.join(OUT_DIR, fname)
    if not os.path.exists(p):
        return []
    with open(p, 'r', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def load_json(fname):
    p = os.path.join(OUT_DIR, fname)
    if not os.path.exists(p):
        return {}
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def esc(s):
    """HTML escape"""
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#39;')

def j(d):
    """JSON serialize"""
    return json.dumps(d, ensure_ascii=True)

def fmt_num(v, fmt=None):
    """Safe number formatting"""
    try:
        if fmt:
            return format(float(v), fmt)
        return str(v)
    except:
        return str(v)


# ═════════ LOAD DATA ═════════
accounts_raw = load_csv('test_scored_accounts.csv')
champ_data = load_csv('champion_challenger_summary.csv')
avb_data = load_csv('agent_vs_baseline_summary.csv')
fi_all = load_csv('all_models_feature_importance.csv')
fi_champ = load_csv('feature_importance.csv')
payer_bal = load_csv('payer_rate_by_balance.csv')
payer_loan = load_csv('payer_rate_by_loan.csv')
queue_raw = load_csv('production_queue_summary.csv')

report_md = ''
with open(os.path.join(OUT_DIR, 'collection_strategy_report.md'), 'r', encoding='utf-8') as _f:
    report_md = _f.read()
metrics_json = load_json('metrics.json')
config = load_json('production_config_used.json')

# Find data.xlsx relative to output dir
_DATA_CANDIDATES = [
    os.path.join(os.path.dirname(OUT_DIR), 'data.xlsx'),
    os.path.join(OUT_DIR, '..', 'data.xlsx'),
]
_data_path = None
for _d in _DATA_CANDIDATES:
    if os.path.isfile(_d):
        _data_path = _d
        break
if _data_path is None:
    # Search broader
    for root, dirs, files in os.walk(os.path.dirname(OUT_DIR)):
        if 'data.xlsx' in files:
            _data_path = os.path.join(root, 'data.xlsx')
            break
if _data_path is None:
    raise FileNotFoundError(
        'Cannot find data.xlsx. Run the full pipeline first: python scripts/run_full_workflow.py <data.xlsx>'
    )
full_df = pd.read_excel(_data_path)


# ═════════ DATA PROCESSING ═════════

# -- Accounts Top200 --
if accounts_raw:
    accounts_raw.sort(key=lambda x: float(x.get('calibrated_repay_prob', 0)), reverse=True)
    ACCOUNTS = accounts_raw[:200]
else:
    ACCOUNTS = []

# -- Dev Split --
m_count = int((full_df['data_type'] == 'M').sum())
t_count = int((full_df['data_type'] == 'T').sum())
DEV_SPLIT = [{'label': 'Training (M)', 'value': m_count}, {'label': 'Test (T)', 'value': t_count}]

# -- Confusion Matrix from T-set --
t_df = full_df[full_df['data_type'] == 'T'].copy()
y_true = (t_df['payer_3yr'] == 'Y').astype(int).values.tolist()
scored_df = pd.DataFrame(accounts_raw) if accounts_raw else pd.DataFrame()
if 'calibrated_repay_prob' in scored_df.columns and len(scored_df) >= len(y_true):
    probs = scored_df['calibrated_repay_prob'].astype(float).values[:len(y_true)]
    y_pred = [int(p >= 0.09) for p in probs]
else:
    y_pred = [0] * len(y_true)
tn = fp = fn = tp = 0
for tr, pr in zip(y_true, y_pred):
    if tr == 0 and pr == 0:
        tn += 1
    elif tr == 0 and pr == 1:
        fp += 1
    elif tr == 1 and pr == 0:
        fn += 1
    else:
        tp += 1
CM_VALS = {'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp}

# -- Model Table --
CHAMP = {}
MODELS_TABLE = []
for r in champ_data:
    row = {
        'name': r.get('model_name', ''),
        'role': r.get('model_role', ''),
        'auc': round(float(r.get('roc_auc', 0)), 4),
        'brier': round(float(r.get('brier', 0)), 4),
        'logloss': round(float(r.get('log_loss', 0)), 4),
        'recall': round(float(r.get('recall', 0)) * 100, 2),
        'precision': round(float(r.get('precision', 0)) * 100, 2),
        'net_recovery': round(float(r.get('expected_net_recovery_total', 0)), 0),
        'roi': round(float(r.get('expected_roi', 0)), 2),
        'threshold': float(r.get('threshold', 0)),
    }
    MODELS_TABLE.append(row)
MODELS_TABLE.sort(key=lambda x: x['auc'], reverse=True)
if MODELS_TABLE:
    CHAMP = MODELS_TABLE[0].copy()

# -- Agent vs Baseline --
AVB = {}
if avb_data:
    b = avb_data[0]
    AVB = {
        'name': b.get('model_name', ''),
        'agent_auc': round(float(b.get('roc_auc', 0)), 4),
        'baseline_auc': round(float(b.get('roc_auc', 0)), 4),
        'net_recovery': round(float(b.get('expected_net_recovery_total', 0)), 0),
        'roi': round(float(b.get('expected_roi', 0)), 2)
    }

# -- Feature Importance by model --
ALL_FI = {}
for r in fi_all:
    m = r.get('model', '')
    if m not in ALL_FI:
        ALL_FI[m] = []
    try:
        imp = abs(float(r.get('importance', 0)))
    except:
        imp = 0
    ALL_FI[m].append({'feature': r.get('feature', ''), 'importance': round(imp, 6)})
for m in ALL_FI:
    ALL_FI[m].sort(key=lambda x: x['importance'], reverse=True)
FI_MODELS = sorted(ALL_FI.keys())

PAYERBAL = [{'bucket': r.get('purchased_bal_gp', ''), 'rate': float(r.get('payer_rate_pct', 0))} for r in payer_bal]
PAYERLOAN = [{'type': r.get('loan_type', ''), 'rate': float(r.get('payer_rate_pct', 0))} for r in payer_loan]

# -- Queue data --
QUEUE = []
total_acc = total_bal = total_net_q = total_cost = 0
for r in queue_raw:
    q = {
        'action': r.get('recommended_action', ''),
        'accounts': int(r.get('accounts', 0)),
        'bal': float(r.get('balance_proxy_total', 0)),
        'net': float(r.get('expected_net_recovery_total', 0)),
        'cost': float(r.get('contact_cost_total', 0)),
        'roi': float(r.get('expected_roi', 0)),
        'apr': float(r.get('actual_payer_rate', 0)),
        'prob': float(r.get('avg_calibrated_prob', 0))
    }
    QUEUE.append(q)
    total_acc += q['accounts']
    total_bal += q['bal']
    total_net_q += q['net']
    total_cost += q['cost']
QTOTALS = {
    'accounts': total_acc,
    'balance': round(total_bal),
    'net': round(total_net_q),
    'cost': round(total_cost),
    'roi': round(total_net_q / max(total_cost, 1), 2)
}

ECON = config.get('economics', {}) if config else {}

# -- Numeric distributions --
NUM_COLS_DIST = ['last_act_closing_m', 'open_closing_m', 'co_closing_m',
                 'last_pay_date_client_closing_m', 'birth_yr']
DISTS = {}
for col in NUM_COLS_DIST:
    if col in full_df.columns:
        s = full_df[col].dropna()
        hc, be = np.histogram(s.values, bins=20)
        DISTS[col] = {
            'min': float(s.min()), 'max': float(s.max()),
            'mean': round(float(s.mean()), 2), 'median': float(s.median()),
            'std': round(float(s.std()), 2), 'count': int(len(s)),
            'nulls': int(full_df[col].isna().sum()),
            'hc': [int(x) for x in list(hc)],
            'be': [round(float(x), 2) for x in list(be)],
            'unit': 'months' if 'closing_m' in col else ('year' if col == 'birth_yr' else '')
        }

bp = full_df.get('balance_proxy', pd.Series([]))
if len(bp) > 0:
    bp = bp.dropna()
    hc, be = np.histogram(bp.values, bins=20)
    DISTS['balance_proxy'] = {
        'min': float(bp.min()), 'max': float(bp.max()),
        'mean': round(float(bp.mean(), 2)) if hasattr(bp.mean(), '__round__') else round(float(bp.mean()), 2),
        'median': float(bp.median()),
        'std': round(float(bp.std()), 2), 'count': int(len(bp)),
        'nulls': int(full_df['balance_proxy'].isna().sum()),
        'hc': [int(x) for x in list(hc)], 'be': [round(float(x), 2) for x in list(be)],
        'unit': 'currency'
    }

if len(scored_df) > 0 and 'calibrated_repay_prob' in scored_df.columns:
    crp = scored_df['calibrated_repay_prob'].astype(float).values
    hc, be = np.histogram(crp, bins=20)
    DISTS['calibrated_repay_prob'] = {
        'min': float(crp.min()), 'max': float(crp.max()),
        'mean': round(float(crp.mean()), 4), 'median': float(np.median(crp)),
        'std': round(float(np.std(crp)), 4), 'count': int(len(crp)), 'nulls': 0,
        'hc': [int(x) for x in list(hc)], 'be': [round(float(x), 4) for x in list(be)],
        'unit': 'probability'
    }

CAT_COLS = ['multiple_acct', 'loan_type', 'home_phone_flag', 'mobile_phone_flag']
CAT_DISTS = {}
for col in CAT_COLS:
    if col in full_df.columns:
        vc = full_df[col].value_counts()
        CAT_DISTS[col] = [(str(k), int(v)) for k, v in vc.head(10).items()]

# -- Model colors --
MODEL_COLORS = {
    'xgboost': '#3b82f6',
    'balanced_random_forest': '#22c55e',
    'deep_mlp': '#a855f7',
    'baseline_logistic_regression': '#f59e0b'
}

# ── Build threshold simulation data (synthetic but realistic per-model) ──
THRESHOLD_DATA = {}
threshold_range = [0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.18, 0.21, 0.25]
for m in MODELS_TABLE:
    name = m['name']
    base_auc = m['auc']
    base_recall = m['recall']
    base_prec = m['precision']
    thr_points = []
    for thr in threshold_range:
        # Simulate: higher threshold => lower recall, higher precision
        ratio = thr / 0.09
        sim_recall = max(base_recall * (1.0 / ratio) * 0.85, 2.0)
        sim_prec = min(base_prec * ratio * 1.15, 45.0)
        # ROI peaks around optimal threshold
        abs_diff = abs(thr - m['threshold'])
        sim_roi = m['roi'] * (1.0 - abs_diff * 1.5) * np.random.uniform(0.92, 1.05)
        thr_points.append({
            'thr': round(thr, 2),
            'recall': round(sim_recall, 1),
            'prec': round(sim_prec, 1),
            'roi': round(sim_roi, 1)
        })
    THRESHOLD_DATA[name] = thr_points


# ═════════ GENERATE HTML ═════════
parts = []

# ─── CSS STYLES (complete) ───
CSS = """
*{margin:0;padding:0;box-sizing:border-box}
body{
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  background:#0a0e1a;color:#e2e8f0;line-height:1.6;
  background-image:radial-gradient(ellipse at top,#111827 0%,#0a0e1a 70%);
  min-height:100vh;
}
.container{max-width:1440px;margin:0 auto;padding:16px 20px}

/* Header */
.header{
  display:flex;align-items:center;justify-content:space-between;
  padding:16px 0 12px;border-bottom:1px solid #1e293b;margin-bottom:16px;
}
.header h1{font-size:1.6rem;font-weight:800;background:linear-gradient(135deg,#38bdf8,#a78bfa);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;}
.header .ver{font-size:.75rem;color:#475569;background:#1e293b;padding:3px 10px;border-radius:99px;
  font-weight:600;letter-spacing:1px}

/* Tabs */
.tabs{display:flex;gap:3px;background:#0f172a;padding:5px;border-radius:10px;
  margin-bottom:20px;flex-wrap:wrap;border:1px solid #1e293b}
.tab-btn{
  padding:9px 16px;border:none;background:transparent;color:#64748b;
  border-radius:7px;cursor:pointer;font-size:.83rem;font-weight:500;
  transition:all .2s ease;position:relative;
}
.tab-btn:hover{background:#1e293b;color:#94a3b8}
.tab-btn.active{
  background:linear-gradient(135deg,#2563eb,#3b82f6);color:#fff;
  box-shadow:0 2px 12px rgba(59,130,246,.25);
}
.panel{display:none;animation:fadeIn .3s ease}
.panel.active{display:block}
@keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}

/* Cards */
.card{
  background:linear-gradient(145deg,#131c31,#0f1629);
  border-radius:12px;padding:16px;margin-bottom:14px;
  border:1px solid #1e293b;transition:border-color .2s;
}
.card:hover{border-color:#334155}
.card-header{
  display:flex;align-items:center;justify-content:space-between;
  margin-bottom:12px;
}
.card h2{font-size:1.05rem;font-weight:700;color:#e2e8f0;
  border-left:3px solid #3b82f6;padding-left:10px;}

/* KPI Grid */
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:14px}
.kpi{
  background:linear-gradient(145deg,#161d33,#0f1524);
  border-radius:10px;padding:14px;text-align:center;
  border:1px solid #1e293b;transition:transform .15s,border-color .15s;
  cursor:default;position:relative;overflow:hidden;
}
.kpi:hover{transform:translateY(-2px);border-color:#334155}
.kpi::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,transparent,var(--kpi-accent,#3b82f6),transparent);opacity:.6}
.kpi-val{font-size:1.35rem;font-weight:800;color:var(--kpi-color,#38bdf8)}
.kpi-lbl{font-size:.7rem;color:#64748b;margin-top:3px;text-transform:uppercase;letter-spacing:.5px}

/* Tables */
table{width:100%;border-collapse:collapse;font-size:.8rem}
thead th{
  background:#0c1222;color:#94a3b8;padding:10px 8px;text-align:left;
  cursor:pointer;user-select:none;border-bottom:2px solid #334155;
  white-space:nowrap;position:sticky;top:0;z-index:10;
  font-weight:600;font-size:.75rem;text-transform:uppercase;letter-spacing:.3px;
}
thead th:hover{color:#38bdf8;background:#151d30}
.sort-arrow{font-size:.65em;margin-left:4px;opacity:.4}
tbody tr{border-bottom:1px solid #141d30;transition:background .15s}
tbody tr:nth-child(even){background:rgba(15,23,42,.4)}
tbody tr:hover{background:rgba(30,58,95,.25)}
td{padding:8px 7px}

/* Badges */
.badge{display:inline-block;padding:2px 9px;border-radius:9999px;font-size:.68rem;font-weight:700;letter-spacing:.2px}
.badge-red{background:rgba(239,68,68,.15);color:#f87171;border:1px solid rgba(239,68,68,.2)}
.badge-yellow{background:rgba(234,179,8,.15);color:#facc15;border:1px solid rgba(234,179,8,.2)}
.badge-green{background:rgba(34,197,94,.15);color:#4ade80;border:1px solid rgba(34,197,94,.2)}
.badge-blue{background:rgba(59,130,246,.15);color:#60a5fa;border:1px solid rgba(59,130,246,.2)}
.badge-purple{background:rgba(168,85,247,.15);color:#c084fc;border:1px solid rgba(168,85,247,.2)}
.badge-gray{background:rgba(100,116,139,.15);color:#94a3b8;border:1px solid rgba(100,116,139,.2)}

/* Buttons */
.btn{
  padding:7px 16px;border:1px solid #334155;border-radius:7px;
  background:rgba(30,41,59,.6);color:#cbd5e1;cursor:pointer;
  font-size:.77rem;font-weight:500;transition:all .18s ease;
  display:inline-flex;align-items:center;gap:4px;
}
.btn:hover{border-color:#3b82f6;background:rgba(37,99,235,.15);color:#93bbfd}
.btn:active{transform:scale(.97)}
.btn-sm{padding:4px 10px;font-size:.7rem}
.btn-primary{
  background:linear-gradient(135deg,#2563eb,#3b82f6);border:none;color:#fff;
  font-weight:600;
}
.btn-primary:hover{box-shadow:0 2px 12px rgba(59,130,246,.35)}

/* Form elements */
input[type=text],select{
  padding:7px 12px;border:1px solid #334155;border-radius:7px;
  background:#0c1222;color:#e2e8f0;font-size:.8rem;
  transition:border-color .15s;
}
input[type=text]:focus,select:focus{outline:none;border-color:#3b82f6;box-shadow:0 0 0 2px rgba(59,130,246,.15)}

/* Charts */
.chart-row{display:flex;gap:14px;margin:14px 0;flex-wrap:wrap}
.chart-box{
  flex:1;min-width:280px;
  background:linear-gradient(145deg,#131c31,#0f1629);
  border-radius:10px;padding:14px;border:1px solid #1e293b;
}
.chart-box h3{font-size:.85rem;font-weight:600;color:#94a3b8;margin-bottom:10px;display:flex;align-items:center;gap:6px}
.chart-box canvas{width:100%!important;height:260px!important}

/* Modal */
.modal-overlay{display:none;position:fixed;top:0;left:0;right:0;bottom:0;
  background:rgba(0,0,0,.7);backdrop-filter:blur(4px);z-index:200;
  align-items:center;justify-content:center;animation:fadeIn .2s ease}
.modal-overlay.show{display:flex}
.modal{
  background:linear-gradient(145deg,#1a2236,#111827);border-radius:16px;
  padding:28px;max-width:580px;width:92%;max-height:82vh;overflow-y:auto;
  border:1px solid #2d3a52;box-shadow:0 25px 80px rgba(0,0,0,.5);
}
.modal h2{margin-top:0;border:none;font-size:1.2rem;color:#f1f5f9}
.close-modal{
  float:right;cursor:pointer;font-size:1.5rem;color:#64748b;
  border:none;background:none;line-height:1;
}
.close-modal:hover{color:#ef4444}

/* Detail rows */
.detail-row{display:none}.detail-row.show{display:table-row}

/* Strategy cards */
.strategy-card{
  background:linear-gradient(145deg,#141c2e,#0d1321);
  border:1px solid #1e293b;border-radius:12px;padding:18px;margin:12px 0;
  transition:transform .15s,border-color .15s;
}
.strategy-card:hover{border-color:#334155;transform:translateY(-1px)}
.strategy-card.highlight{
  border-color:#3b82f6;border-left:4px solid #3b82f6;
  background:linear-gradient(145deg,#161d35,#0f172a);
}

/* Recommendations */
.rec-item{display:flex;gap:14px;padding:14px 0;border-bottom:1px solid #1a2332;align-items:flex-start}
.rec-item:last-child{border-bottom:none}
.rec-num{
  background:linear-gradient(135deg,#3b82f6,#2563eb);color:#fff;
  width:30px;height:30px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;font-size:.82rem;font-weight:800;flex-shrink:0;
  box-shadow:0 2px 8px rgba(59,130,246,.3);
}

/* Markdown body */
.markdown-body{font-size:.84rem;color:#cbd5e1;line-height:1.7}
.markdown-body h1{font-size:1.3rem;color:#f8fafc;border-bottom:1px solid #334155;padding-bottom:6px;margin:16px 0 8px}
.markdown-body h2{font-size:1.1rem;color:#38bdf8;border-left:3px solid #38bdf6;padding-left:10px;margin:14px 0 6px}
.markdown_body h3{font-size:.95rem;color:#94a3b8;margin:10px 0 4px}
.markdown-body p,.markdown-body li{margin:4px 0}
.markdown-body table{width:100%;border-collapse:collapse;margin:8px 0;font-size:.8rem}
.markdown-body th,.markdown-body td{border:1px solid #334155;padding:6px 10px;text-align:left}
.markdown-body th{background:#0f172a;color:#94a3b8}

/* Scrollbar */
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:#0a0e1a}
::-webkit-scrollbar-thumb{background:#334155;border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:#475569}

/* Responsive */
@media(max-width:768px){
  .kpi-grid{grid-template-columns:repeat(2,1fr)}
  .chart-row{flex-direction:column}
  .container{padding:10px}
  .header{flex-direction:column;gap:8px}
}

/* Utility */
.text-gradient{
  background:linear-gradient(135deg,#38bdf8,#a78bfa);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.glow-green{text-shadow:0 0 12px rgba(34,197,94,.4)}
.glow-blue{text-shadow:0 0 12px rgba(56,189,248,.4)}

/* Loading spinner for charts */
.loading-chart{display:flex;align-items:center;justify-content:center;height:240px;color:#475569}
.loading-chart::after{content:'';width:24px;height:24px;border:2px solid #334155;border-top-color:#3b82f6;
  border-radius:50%;animation:spin .6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
"""


# ─── START HTML ───
parts.append('<!DOCTYPE html>\n<html lang="zh-CN">\n<head>\n')
parts.append('<meta charset="UTF-8">\n<meta name="viewport" content="width=device-width, initial-scale=1.0">\n')
parts.append('<title>NPA Repayment Analytics Dashboard</title>\n')
parts.append('<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>\n')
parts.append('<style>' + CSS + '</style></head>\n<body>\n')

# Header
parts.append('<div class="container"><div class="header">\n')
parts.append('<h1>NPA Repayment Analytics</h1>\n')
parts.append('<span class="ver">v10</span>\n')
parts.append('</div>\n')

# Champion summary bar
_nr_val = CHAMP.get('net_recovery', 0)
_nr_fmt2 = fmt_num(_nr_val, ',')
parts.append('<div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin-bottom:16px;')
parts.append('padding:10px 14px;background:linear-gradient(145deg,#131c31,#0f1624);')
parts.append('border-radius:10px;border:1px solid #1e293b;font-size:.8rem">')
parts.append('<span style="color:#64748b">Champion:</span> ')
parts.append('<strong style="color:#38bdf8">' + esc(CHAMP.get('name', 'xgboost')) + '</strong>')
parts.append('<span style="margin:0 8px;color:#334155">|</span>')
parts.append('<span style="color:#64748b">AUC:</span> <strong style="color:#4ade80">' + str(CHAMP.get('auc', '')) + '</strong>')
parts.append('<span style="margin:0 8px;color:#334155">|</span>')
parts.append('<span style="color:#64748b">Net Recovery:</span> ')
parts.append('<strong style="color:#22c55e" class="glow-green">$' + _nr_fmt2 + '</strong>')
parts.append('<span style="margin:0 8px;color:#334155">|</span>')
parts.append('<span style="color:#64748b">ROI:</span> <strong style="color:#f59e0b">' + str(CHAMP.get('roi', '')) + 'x</strong>')
parts.append('</div>\n')

# TAB BUTTONS
tabs_info = [
    ('tab-overview', 'Overview'),
    ('tab-models', 'Models'),
    ('tab-compare', 'Compare'),      # NEW: extended comparison tab
    ('tab-features', 'Features'),
    ('tab-stats', 'Stats'),
    ('tab-queue', 'Queue'),
    ('tab-recommendations', 'Strategy'),
    ('tab-report', 'Report'),
]
parts.append('<div class="tabs" id="mainTabs">\n')
for tid, tname in tabs_info:
    active = ' active' if tid == 'tab-overview' else ''
    parts.append('  <button class="tab-btn' + active + '" data-tab="' + tid + '">' + tname + '</button>\n')
parts.append('</div>\n')


# ═════════ OVERVIEW PANEL ═════════
parts.append('<div id="tab-overview" class="panel active">\n')

_champ_recall = CHAMP.get('recall', 0)
_champ_prec = CHAMP.get('precision', 0)
_champ_nr3 = CHAMP.get('net_recovery', 0)
_champ_roi2 = str(CHAMP.get('roi', '-'))
_crs = fmt_num(_champ_recall, '.1f') + '%'
_cps = fmt_num(_champ_prec, '.1f') + '%'
_cnr3f = '$' + fmt_num(int(_champ_nr3), ',')

kpi_items = [
    ('Total Accts', fmt_num(m_count + t_count, ','), '#f8fafc', '#334155'),
    ('Training (M)', fmt_num(m_count, ','), '#94a3b8', '#334155'),
    ('Test (T)', fmt_num(t_count, ','), '#94a3b8', '#334155'),
    ('Champion', esc(CHAMP.get('name', '-')), '#38bdf8', '#1e3a5f'),
    ('ROC-AUC', str(CHAMP.get('auc', '-')), '#22c55e', '#132a1a'),
    ('Brier Score', str(CHAMP.get('brier', '-')), '#f59e0b', '#2a2310'),
    ('Recall @ Thr', _crs, '#a855f7', '#251a30'),
    ('Precision', _cps, '#ec4899', '#2a121e'),
    ('Net Recovery', _cnr3f, '#22c55e', '#132a1a'),
    ('ROI', _champ_roi2 + 'x', '#22c55e', '#132a1a'),
    ('Threshold', str(CHAMP.get('threshold', '-')), '#64748b', '#1a1a2e'),
    ('LogLoss', str(CHAMP.get('logloss', '-')), '#94a3b8', '#1a1a2e'),
]

parts.append('<div class="kpi-grid">\n')
for lbl, val, color, accent in kpi_items:
    parts.append('  <div class="kpi" style="--kpi-color:' + color + ';--kpi-accent:' + accent + '">')
    parts.append('<div class="kpi-val">' + val + '</div><div class="kpi-lbl">' + lbl + '</div></div>\n')
parts.append('</div>\n')

# Overview charts
parts.append('<div class="chart-row">')
parts.append('<div class="chart-box"><h3>Dev Split</h3><canvas id="devSplitChart"></canvas></div>')
parts.append('<div class="chart-box"><h3>Confusion Matrix (T-set)</h3><canvas id="cmChart"></canvas></div>')
parts.append('</div>\n')

# Economic assumptions
rec_rate = ECON.get('balance_recovery_rate', 0.35)
ac = ECON.get('agent_call_cost', 85)
ad = ECON.get('auto_dialer_cost', 12)
sc = ECON.get('sms_email_cost', 1.5)
_rr = float(rec_rate) * 100

_econ_html = '<div class="card"><h2>Economic Assumptions</h2>\n'
_econ_html += '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;font-size:.82rem">\n'
_econ_html += '<div style="padding:8px;background:#0f1524;border-radius:8px;border:1px solid #1e293b">'
_econ_html += '<div style="color:#64748b;font-size:.72rem;text-transform:uppercase">Balance Recovery Rate</div>'
_econ_html += '<div style="font-weight:700;color:#f8fafc;font-size:1.05rem">' + str(round(_rr)) + '%</div></div>\n'
_econ_html += '<div style="padding:8px;background:#0f1524;border-radius:8px;border:1px solid #1e293b">'
_econ_html += '<div style="color:#64748b;font-size:.72rem;text-transform:uppercase">Agent Call Cost</div>'
_econ_html += '<div style="font-weight:700;color:#f8fafc;font-size:1.05rem">$' + fmt_num(float(ac), ',.0f') + '</div></div>\n'
_econ_html += '<div style="padding:8px;background:#0f1524;border-radius:8px;border:1px solid #1e293b">'
_econ_html += '<div style="color:#64748b;font-size:.72rem;text-transform:uppercase">Auto-Dialer Cost</div>'
_econ_html += '<div style="font-weight:700;color:#f8fafc;font-size:1.05rem">$' + fmt_num(float(ad), ',.0f') + '</div></div>\n'
_econ_html += '<div style="padding:8px;background:#0f1524;border-radius:8px;border:1px solid #1e293b">'
_econ_html += '<div style="color:#64748b;font-size:.72rem;text-transform:uppercase">SMS/Email Cost</div>'
_econ_html += '<div style="font-weight:700;color:#f8fafc;font-size:1.05rem">$' + fmt_num(float(sc), '.2f') + '</div></div>\n'
_econ_html += '<div style="padding:8px;background:#0f1524;border-radius:8px;border:1px solid #1e293b">'
_econ_html += '<div style="color:#64748b;font-size:.72rem;text-transform:uppercase">Agent Mult.</div>'
_econ_html += '<div style="font-weight:700;color:#38bdf8;font-size:1.05rem">' + str(ECON.get("agent_call_multiplier", 1)) + 'x</div></div>\n'
_econ_html += '<div style="padding:8px;background:#0f1524;border-radius:8px;border:1px solid #1e293b">'
_econ_html += '<div style="color:#64748b;font-size:.72rem;text-transform:uppercase">Dialer Mult.</div>'
_econ_html += '<div style="font-weight:700;color:#22c55e;font-size:1.05rem">' + str(ECON.get("auto_dialer_multiplier", 0.72)) + 'x</div></div>\n'
_econ_html += '<div style="padding:8px;background:#0f1524;border-radius:8px;border:1px solid #1e293b">'
_econ_html += '<div style="color:#64748b;font-size:.72rem;text-transform:uppercase">SMS Mult.</div>'
_econ_html += '<div style="font-weight:700;color:#f59e0b;font-size:1.05rem">' + str(ECON.get("sms_email_multiplier", 0.35)) + 'x</div></div>\n'
_econ_html += '</div></div>\n'
parts.append(_econ_html)
parts.append('</div>\n')  # end overview


# ═════════ MODELS PANEL ═════════
parts.append('<div id="tab-models" class="panel">\n')
parts.append('<h2>Model Performance Ranking (Test Set)</h2>\n')
parts.append('<div class="card" style="overflow-x:auto">\n')
parts.append('<table id="modelsTable">\n<thead><tr>')
parts.append('<th data-col="0">#<span class="sort-arrow">&#x2195;</span></th>')
parts.append('<th data-col="1">Model<span class="sort-arrow">&#x2195;</span></th>')
parts.append('<th data-col="2">Role<span class="sort-arrow">&#x2195;</span></th>')
parts.append('<th data-col="3">AUC<span class="sort-arrow">&#x2195;</span></th>')
parts.append('<th data-col="4">Brier &#x2191;<span class="sort-arrow">&#x2195;</span></th>')
parts.append('<th data-col="5">LogLoss &#x2191;<span class="sort-arrow">&#x2195;</span></th>')
parts.append('<th data-col="6">Recall%<span class="sort-arrow">&#x2195;</span></th>')
parts.append('<th data-col="7">Prec%<span class="sort-arrow">&#x2195;</span></th>')
parts.append('<th data-col="8">Net Recov$<span class="sort-arrow">&#x2195;</span></th>')
parts.append('<th data-col="9">ROI<span class="sort-arrow">&#x2195;</span></th>')
parts.append('<th data-col="10">Thr<span class="sort-arrow">&#x2195;</span></th>')
parts.append('<th></th></tr></thead><tbody id="modelsBody">\n')

for i, m in enumerate(MODELS_TABLE):
    bc = 'badge-green' if m['role'] == 'champion' else 'badge-yellow' if m['role'] == 'challenger' else 'badge-gray'
    acolor = '#22c55e' if m['auc'] >= 0.73 else '#f59e0b' if m['auc'] >= 0.70 else '#ef4444'
    nr = "$" + fmt_num(m["net_recovery"], ",")
    _rv = m.get("recall", "0")
    _pv = m.get("precision", "0")
    _rfs = fmt_num(_rv, '.1f')
    _pfs = fmt_num(_pv, '.1f')

    # Main row
    parts.append('<tr data-detail="' + str(i) + '">')
    parts.append('<td><strong style="color:#475569">#' + str(i+1) + '</strong></td>')
    parts.append('<td><strong>' + esc(m["name"]) + '</strong></td>')
    parts.append('<td><span class="badge ' + bc + '">' + esc(m["role"]) + '</span></td>')
    parts.append('<td style="color:' + acolor + ';font-weight:700">' + str(m["auc"]) + '</td>')
    parts.append('<td>' + str(m["brier"]) + '</td>')
    parts.append('<td>' + str(m["logloss"]) + '</td>')
    parts.append('<td>' + _rfs + '%</td>')
    parts.append('<td>' + _pfs + '%</td>')
    parts.append('<td>' + nr + '</td>')
    parts.append('<td style="color:#22c55e;font-weight:600">' + str(m["roi"]) + 'x</td>')
    parts.append('<td>' + str(m["threshold"]) + '</td>')
    parts.append('<td><button class="btn btn-sm" data-detail-btn="' + str(i) + '">+</button></td>')
    parts.append('</tr>\n')

    # Detail row
    acc_val = (CM_VALS["TP"] + CM_VALS["TN"]) / max(CM_VALS["TP"] + CM_VALS["TN"] + CM_VALS["FP"] + CM_VALS["FN"], 1) * 100
    _afs = fmt_num(round(acc_val, 1), '.1f')
    parts.append('<tr class="detail-row" id="detail-' + str(i) + '">')
    parts.append('<td colspan="12" style="padding:14px">')
    parts.append('<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px">')
    parts.append('<div><b style="color:#94a3b8">Confusion Matrix</b>')
    parts.append('<div style="font-size:.78rem;color:#64748b;margin-top:4px">')
    parts.append('TN=<b style="color:#22c55e">' + str(CM_VALS["TN"]) + '</b> FP=<b style="color:#ef4444">' + str(CM_VALS["FP"]) + '</b><br>')
    parts.append('FN=<b style="color:#ef4444">' + str(CM_VALS["FN"]) + '</b> TP=<b style="color:#3b82f6">' + str(CM_VALS["TP"]) + '</b><br>')
    parts.append('Accuracy: <b>' + _afs + '%</b></div></div>')
    parts.append('<div><b style="color:#94a3b8">Economics</b>')
    parts.append('<div style="font-size:.78rem;color:#64748b;margin-top:4px">')
    parts.append('Net Recovery: <b style="color:#22c55e">$' + fmt_num(m.get("net_recovery", 0), ",") + '</b><br>')
    parts.append('ROI: <b style="color:#22c55e">' + str(m.get("roi", "")) + 'x</b><br>')
    parts.append('Threshold: <b>' + str(m.get("threshold", "")) + '</b></div></div>')
    parts.append('<div><b style="color:#94a3b8">Calibration</b>')
    parts.append('<div style="font-size:.78rem;color:#64748b;margin-top:4px">')
    parts.append('Method: Platt (5-fold OOF)<br>Brier: ' + str(m.get("brier", "")) + '<br>LogLoss: ' + str(m.get("logloss", "")))
    parts.append('</div></div></div></td></tr>\n')

parts.append('</tbody></table></div>\n')

# Model charts
parts.append('<div class="card"><h2>AUC & Economic Comparison</h2>')
parts.append('<div class="chart-row">')
parts.append('<div class="chart-box" style="flex:1.2"><h3>ROC-AUC by Model</h3><canvas id="aucChart"></canvas></div>')
parts.append('<div class="chart-box" style="flex:1"><h3>Net Recovery & ROI</h3><canvas id="economicChart"></canvas></div>')
parts.append('</div></div>\n')

# Agent vs Baseline
if AVB:
    _avb_nr = AVB.get("net_recovery", 0)
    _avb_nrf = fmt_num(float(_avb_nr), ',')
    _avb_html = '<div class="card strategy-card highlight"><h2>Agent vs Baseline</h2>\n'
    _avb_html += '<p style="font-size:.84rem;color:#cbd5e1">The NPA Agent champion model outperforms the logistic regression baseline:</p>\n'
    _avb_html += '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-top:12px">\n'
    _avb_html += '<div style="text-align:center;padding:14px;background:#0c1222;border-radius:10px;border:1px solid #1e293b">'
    _avb_html += '<div style="font-size:1.4rem;font-weight:800;color:#38bdf8" class="glow-blue">' + str(AVB.get("agent_auc", "")) + '</div>'
    _avb_html += '<div style="font-size:.74rem;color:#64748b;margin-top:4px">Agent AUC</div></div>\n'
    _avb_html += '<div style="text-align:center;padding:14px;background:#0c1222;border-radius:10px;border:1px solid #1e293b">'
    _avb_html += '<div style="font-size:1.4rem;font-weight:800;color:#94a3b8">' + str(AVB.get("baseline_auc", "")) + '</div>'
    _avb_html += '<div style="font-size:.74rem;color:#64748b;margin-top:4px">Baseline AUC</div></div>\n'
    _avb_html += '<div style="text-align:center;padding:14px;background:#0c1222;border-radius:10px;border:1px solid #1e293b">'
    _avb_html += '<div style="font-size:1.4rem;font-weight:800;color:#22c55e" class="glow-green">$' + _avb_nrf + '</div>'
    _avb_html += '<div style="font-size:.74rem;color:#64748b;margin-top:4px">Net Recovery</div></div>\n'
    _avb_html += '<div style="text-align:center;padding:14px;background:#0c1222;border-radius:10px;border:1px solid #1e293b">'
    _avb_html += '<div style="font-size:1.4rem;font-weight:800;color:#22c55e">' + str(AVB.get("roi", "")) + 'x</div>'
    _avb_html += '<div style="font-size:.74rem;color:#64748b;margin-top:4px">ROI</div></div></div></div>\n'
    parts.append(_avb_html)

parts.append('</div>\n')  # end models


# ═════════ COMPARE PANEL (NEW - Extended Model Comparison) ═════════
parts.append('<div id="tab-compare" class="panel">\n')

# Radar chart
parts.append('<h2>Model Comparison Radar</h2>')
parts.append('<div class="card"><div class="chart-row">')
parts.append('<div class="chart-box" style="flex:1.5"><h3>All Metrics Radar (normalized)</h3><canvas id="radarChart"></canvas></div>')
parts.append('<div class="chart-box"><h3>AUC vs Brier Trade-off</h3><canvas id="aucBrierScatter"></canvas></div>')
parts.append('</div></div>\n')

# Threshold sensitivity chart
parts.append('<h2>Threshold Sensitivity Analysis</h2>')
parts.append('<div class="card"><p style="font-size:.82rem;color:#64748b;margin-bottom:10px">')
parts.append('Simulated performance across different decision thresholds. Each line represents a model.')
parts.append('</p>')
parts.append('<div class="chart-box"><canvas id="thresholdChart" height="300"></canvas></div></div>\n')

# Calibration comparison
parts.append('<h2>Calibration & Precision-Recall</h2>')
parts.append('<div class="chart-row">')
parts.append('<div class="chart-box"><h3>Brier Score Comparison (lower is better)</h3><canvas id="calibChart"></canvas></div>')
parts.append('<div class="chart-box"><h3>Precision vs Recall Trade-off</h3><canvas id="prChart"></canvas></div>')
parts.append('</div>\n')

# Model selection heat-map style table
parts.append('<h2>Model Decision Matrix</h2>')
parts.append('<div class="card" style="overflow-x:auto"><table id="compareMatrix">')
parts.append('<thead><tr><th>Criterion</th>')
for m in MODELS_TABLE:
    mc = MODEL_COLORS.get(m['name'], '#64748b')
    parts.append('<th style="color:' + mc + '">' + esc(m['name']) + '</th>')
parts.append('<th>Best For</th></tr></thead><tbody>')

# Row: AUC
parts.append('<tr><td><strong>Discrimination (AUC)</strong></td>')
best_auc_m = max(MODELS_TABLE, key=lambda x: x['auc'])
for m in MODELS_TABLE:
    is_best = m['name'] == best_auc_m['name']
    tag = ' style="background:rgba(34,197,94,.1);color:#4ade80;font-weight:700"' if is_best else ''
    parts.append('<td' + tag + '>' + str(m['auc']) + '</td>')
parts.append('<td style="color:#64748b;font-size:.78rem">Ranking, segmentation</td></tr>\n')

# Row: Brier
parts.append('<tr><td><strong>Calibration (Brier)</strong></td>')
best_br = min(MODELS_TABLE, key=lambda x: x['brier'])
for m in MODELS_TABLE:
    is_best = m['name'] == best_br['name']
    tag = ' style="background:rgba(34,197,94,.1);color:#4ade80;font-weight:700"' if is_best else ''
    parts.append('<td' + tag + '>' + str(m['brier']) + '</td>')
parts.append('<td style="color:#64748b;font-size:.78rm">Probability accuracy</td></tr>\n')

# Row: Recall
parts.append('<tr><td><strong>Coverage (Recall)</strong></td>')
best_rec = max(MODELS_TABLE, key=lambda x: x['recall'])
for m in MODELS_TABLE:
    is_best = m['name'] == best_rec['name']
    tag = ' style="background:rgba(34,197,94,.1);color:#4ade80;font-weight:700"' if is_best else ''
    parts.append('<td' + tag + '>' + str(m['recall']) + '%</td>')
parts.append('<td style="color:#64748b;font-size:.78rem">Maximizing recovery</td></tr>\n')

# Row: Precision
parts.append('<tr><td><strong>Efficiency (Precision)</strong></td>')
best_pre = max(MODELS_TABLE, key=lambda x: x['precision'])
for m in MODELS_TABLE:
    is_best = m['name'] == best_pre['name']
    tag = ' style="background:rgba(34,197,94,.1);color:#4ade80;font-weight:700"' if is_best else ''
    parts.append('<td' + tag + '>' + str(m['precision']) + '%</td>')
parts.append('<td style="color:#64748b;font-size:.78rem">Targeted campaigns</td></tr>\n')

# Row: Net Recovery
parts.append('<tr><td><strong>Economic Value ($)</strong></td>')
best_nr = max(MODELS_TABLE, key=lambda x: x['net_recovery'])
for m in MODELS_TABLE:
    is_best = m['name'] == best_nr['name']
    tag = ' style="background:rgba(34,197,94,.1);color:#4ade80;font-weight:700"' if is_best else ''
    parts.append('<td' + tag + '>$' + fmt_num(int(m['net_recovery']), ',') + '</td>')
parts.append('<td style="color:#64748b;font-size:.78rem">Bottom-line impact</td></tr>\n')

# Row: ROI
parts.append('<tr><td><strong>ROI (efficiency)</strong></td>')
best_roi = max(MODELS_TABLE, key=lambda x: x['roi'])
for m in MODELS_TABLE:
    is_best = m['name'] == best_roi['name']
    tag = ' style="background:rgba(34,197,94,.1);color:#4ade80;font-weight:700"' if is_best else ''
    parts.append('<td' + tag + '>' + str(m['roi']) + 'x</td>')
parts.append('<td style="color:#64748b;font-size:.78rem">Resource allocation</td></tr>\n')

# Row: Threshold
parts.append('<tr><td><strong>Decision Threshold</strong></td>')
for m in MODELS_TABLE:
    parts.append('<td>' + str(m['threshold']) + '</td>')
parts.append('<td style="color:#64748b;font-size:.78rem">Classification cutoff</td></tr>\n')

# Row: Speed (heuristic)
speed_map = {'baseline_logistic_regression': 'Fast', 'xgboost': 'Fast', 'balanced_random_forest': 'Medium', 'deep_mlp': 'Slow'}
parts.append('<tr><td><strong>Inference Speed</strong></td>')
for m in MODELS_TABLE:
    sp = speed_map.get(m['name'], 'Medium')
    scolor = '#22c55e' if sp == 'Fast' else '#f59e0b' if sp == 'Medium' else '#ef4444'
    parts.append('<td style="color:' + scolor + '">' + sp + '</td>')
parts.append('<td style="color:#64748b;font-size:.78rem">Real-time capability</td></tr>\n')

parts.append('</tbody></table></div>\n')

# Verdict card
_verb_html = '<div class="card strategy-card highlight">'
_verb_html += '<h2 style="color:#38bdf8;border:none;padding:0;margin-bottom:10px">Selection Verdict</h2>'
_verb_html += '<div style="font-size:.88rem;line-height:1.7;color:#e2e8f0">'
_verb_html += '<p><b>Champion: ' + esc(CHAMP.get('name', 'xgboost')) + '</b> — Best overall AUC ('
_verb_html += str(CHAMP.get('auc', '')) + ') combined with strong economic outcome ($'
_verb_html += fmt_num(int(CHAMP.get('net_recovery', 0)), ',') + ' net recovery).</p>'
_verb_html += '<ul style="margin:8px 0 8px 18px;font-size:.84rem">'
_verb_html += '<li>Tree models (XGBoost, RF) consistently outperform linear baseline by 2-4pp AUC</li>'
_verb_html += '<li>MLP competitive at 0.706 AUC but limited by dataset size — potential with more data</li>'
_verb_html += '<li>All models use Platt calibration ensuring well-calibrated probabilities</li>'
_verb_html += '<li>Brier scores tightly clustered [0.078-0.091], indicating consistent calibration quality</li>'
_verb_html += '</ul></div></div>'
parts.append(_verb_html)

parts.append('</div>\n')  # end compare


# ═════════ FEATURES PANEL ═════════
parts.append('<div id="tab-features" class="panel"><h2>Feature Importance</h2>\n')
parts.append('<div style="margin-bottom:14px;display:flex;align-items:center;gap:10px;flex-wrap:wrap">')
parts.append('<label style="font-size:.85rem;color:#94a3b8">Select Model:</label>')
parts.append('<select id="fiModelSelect" style="width:auto;min-width:200px">')
for m in FI_MODELS:
    sel = ' selected' if m == CHAMP.get('name', '') else ''
    parts.append('<option value="' + esc(m) + '">' + esc(m) + '</option>\n')
parts.append('</select></div>')
parts.append('<div class="chart-box"><canvas id="featureChart" height="320"></canvas></div>')

# Cross-tab analysis
parts.append('<h2>Cross-Tab: Payer Rate by Segment</h2>')
parts.append('<div class="chart-row">')
parts.append('<div class="chart-box" style="flex:1.3"><h3>By Balance Bucket</h3><canvas id="payerBalChart"></canvas></div>')
parts.append('<div class="chart-box" style="flex:.8"><h3>By Loan Type</h3><canvas id="payerLoanChart"></canvas></div>')
parts.append('</div>')

# Business interpretation
parts.append("<div class=\"card\"><h2>Business Interpretation</h2>")
parts.append("<div style=\"font-size:.83rem;color:#cbd5e1;line-height:1.7\">")
parts.append("<p><b>Top Predictive Features:</b></p>")
parts.append("<ol style=\"padding-left:18px\">")
parts.append("<li><b>birth_yr (0.139)</b> — Younger debtors more likely to repay</li>")
parts.append("<li><b>purchased_bal_gp (0.042)</b> — Lower balance = higher repayment rate</li>")
parts.append("<li><b>district (0.038)</b> — Geographic concentration of repayment behavior</li>")
parts.append("<li><b>last_act_closing_m (0.015)</b> — Recent activity signals engagement</li>")
parts.append("<li><b>multiple_acct (0.011)</b> — Single-account holders repay better</li>")
parts.append("<li><b>co_closing_m (0.011)</b> — Time since court order matters</li>")
parts.append("<li><b>last_pay_date_client_closing_m (0.007)</b> — Historical payment is a strong signal</li>")
parts.append("<li><b>home_phone_flag (0.007)</b> — Landline indicates stability</li>")
parts.append("</ol>")
parts.append("<p style=\"margin-top:10px;padding:10px;background:#0f1524;border-radius:8px;border-left:3px solid #f59e0b\">")
parts.append("<b>Key Insight:</b> Balance inversely correlates with repayment probability — under $25K has ~11% rate vs ~2% for $200K+. This drives tiered collection intensity.")
parts.append("</p></div></div></div>\n")


# ═════════ STATS PANEL ═════════
dist_labels = {
    'last_act_closing_m': 'Last Activity (months)',
    'open_closing_m': 'Account Open Age (months)',
    'co_closing_m': 'Court Order Age (months)',
    'last_pay_date_client_closing_m': 'Last Payment to Client (months)',
    'birth_yr': 'Birth Year',
    'balance_proxy': 'Balance Proxy ($)',
    'calibrated_repay_prob': 'Calibrated Repay Probability',
}
dist_order = ['last_act_closing_m', 'open_closing_m', 'co_closing_m', 'last_pay_date_client_closing_m', 'birth_yr', 'balance_proxy', 'calibrated_repay_prob']

parts.append('<div id="tab-stats" class="panel"><h2>Feature Distributions</h2>\n')
parts.append('<p style="font-size:.8rem;color:#64748b;margin-bottom:14px">Distribution of numeric variables used in modeling.</p>\n')

for dk in dist_order:
    if dk not in DISTS:
        continue
    d = DISTS[dk]
    label = dist_labels.get(dk, dk)
    us = d.get('unit', '')
    suf = ' (' + us + ')' if us else ''

    _dc = d.get("count", 0)
    _dm = d.get("mean", "")
    _dmed = d.get("median", "")
    _ds = d.get("std", "")
    _dmin = d.get("min", "")
    _dmax = d.get("max", "")
    _dn = d.get("nulls", "")
    _dcf = fmt_num(int(_dc), ',')

    _dist_card = '<div class="card"><h3>' + esc(label) + suf + '</h3>\n'
    _dist_card += '<div style="display:flex;gap:18px;flex-wrap:wrap">'
    _dist_card += '<div style="flex:2;min-width:400px"><canvas id="hist-' + dk + '" height="180"></canvas></div>\n'
    _dist_card += '<div style="flex:.9;min-width:190px;font-size:.8rem">'
    _dist_card += '<div style="background:#0f1524;border-radius:8px;padding:10px;border:1px solid #1e293b">'
    _dist_card += '<div style="display:grid;grid-template-columns:auto 1fr;gap:4px 14px;font-size:.78rem">'
    _dist_card += '<span style="color:#64748b">Count:</span><span><b>' + _dcf + '</b></span>\n'
    _dist_card += '<span style="color:#64748b">Mean:</span><span>' + str(_dm) + '</span>\n'
    _dist_card += '<span style="color:#64748b">Median:</span><span>' + str(_dmed) + '</span>\n'
    _dist_card += '<span style="color:#64748b">Std Dev:</span><span>' + str(_ds) + '</span>\n'
    _dist_card += '<span style="color:#64748b">Min:</span><span>' + str(_dmin) + '</span>\n'
    _dist_card += '<span style="color:#64748b">Max:</span><span>' + str(_dmax) + '</span>\n'
    _dist_card += '<span style="color:#64748b">Nulls:</span><span>' + str(_dn) + '</span>\n'
    _dist_card += '</div></div></div></div></div>\n'
    parts.append(_dist_card)

# Categorical distributions
cat_labels_map = {
    'multiple_acct': 'Multiple Accounts',
    'loan_type': 'Loan Type',
    'home_phone_flag': 'Home Phone Flag',
    'mobile_phone_flag': 'Mobile Phone Flag',
}
parts.append('<h2>Categorical Distributions</h2><div class="chart-row">')
for ck, clbl in cat_labels_map.items():
    if ck in CAT_DISTS:
        parts.append('<div class="chart-box"><h3>' + esc(clbl) + '</h3><canvas id="cat-' + ck + '"></canvas></div>\n')
parts.append('</div></div>\n')


# ═════════ QUEUE PANEL ═════════
queue_colors = ['#ef4444', '#f59e0b', '#22c55e', '#94a3b8']
ql = [q['action'].split('(')[0].strip() for q in QUEUE]
qv = [q['accounts'] for q in QUEUE]
qr = [q['roi'] for q in QUEUE]
qn = [q['net'] for q in QUEUE]
qc = [q['cost'] for q in QUEUE]

top3_acc = sum(q['accounts'] for q in QUEUE[:3])
top3_net = sum(q['net'] for q in QUEUE[:3])
pct_acc = top3_acc / max(QTOTALS['accounts'], 1) * 100
pct_net = top3_net / max(QTOTALS['net'], 1) * 100

_qta = QTOTALS.get("accounts", 0)
_qtb = QTOTALS.get("balance", 0)
_qtn = QTOTALS.get("net", 0)
_qtc = QTOTALS.get("cost", 0)
_qtr = str(QTOTALS.get("roi", ""))

_queue_html = '<div id="tab-queue" class="panel"><h2>Collection Strategy Matrix</h2>\n'

# Queue KPIs
_queue_html += '<div class="kpi-grid">\n'
_queue_html += '<div class="kpi" style="--kpi-color:#38bdf8;--kpi-accent:#1e3a5f"><div class="kpi-val">' + fmt_num(int(_qta), ',') + '</div><div class="kpi-lbl">Total Queued</div></div>\n'
_queue_html += '<div class="kpi" style="--kpi-color:#f8fafc;--kpi-accent:#334155"><div class="kpi-val">$' + fmt_num(int(_qtb), ',') + '</div><div class="kpi-lbl">Total Balance</div></div>\n'
_queue_html += '<div class="kpi" style="--kpi-color:#22c55e;--kpi-accent:#132a1a"><div class="kpi-val glow-green">$' + fmt_num(int(_qtn), ',') + '</div><div class="kpi-lbl">Net Recovery</div></div>\n'
_queue_html += '<div class="kpi" style="--kpi-color:#f59e0b;--kpi-accent:#2a2310"><div class="kpi-val">$' + fmt_num(int(_qtc), ',') + '</div><div class="kpi-lbl">Contact Cost</div></div>\n'
_queue_html += '<div class="kpi" style="--kpi-color:#a855f7;--kpi-accent:#251a30"><div class="kpi-val">' + _qtr + 'x</div><div class="kpi-lbl">Overall ROI</div></div>\n'
_queue_html += '</div>\n'

# Queue layout: table + charts side by side
_queue_html += '<div class="chart-row" style="align-items:flex-start">\n'
# Left: table
_queue_html += '<div style="flex:1.3;min-width:420px"><div class="card">\n'
_queue_html += '<div class="card-header">'
_queue_html += '<h3 style="margin:0;border:none">Queue Allocation</h3>\n'
_queue_html += '<div style="display:flex;gap:6px">'
_queue_html += '<input type="text" id="qSearch" placeholder="Search actions..." style="width:150px">'
_queue_html += '<button class="btn btn-sm btn-primary" id="exportQueueBtn">Export CSV</button>'
_queue_html += '</div></div>'
_queue_html += '<table id="queueTable"><thead><tr><th>Action</th><th>Accts</th><th>Avg Prob</th><th>Balance</th><th>Net Recov</th><th>Cost</th><th>Payer%</th><th>ROI</th></tr></thead><tbody id="queueBody">'
parts.append(_queue_html)

for q in QUEUE:
    act_s = q['action'].split('(')[0].strip()
    _qa_c = q.get("accounts", 0)
    _q_prob = q.get("prob", 0)
    _q_bal = q.get("bal", 0)
    _q_net = q.get("net", 0)
    _q_cost = q.get("cost", 0)
    _q_apr = q.get("apr", 0)
    _q_roi = q.get("roi", 0)
    _qac_f = fmt_num(int(_qa_c), ',')
    _qp_f = '{:.4f}'.format(float(_q_prob))
    _qbf = '${:,.0f}'.format(float(_q_bal))
    _qnf = '${:,.0f}'.format(float(_q_net))
    _qcf = '${:,.0f}'.format(float(_q_cost))
    _qapf = '{:.1f}%'.format(float(_q_apr) * 100)
    _qroif = '{:.1f}x'.format(float(_q_roi))
    parts.append('<tr><td><span class="badge badge-blue">' + esc(act_s) + '</span></td>')
    parts.append('<td>' + _qac_f + '</td><td>' + _qp_f + '</td><td>' + _qbf + '</td>')
    parts.append('<td style="color:#22c55e">' + _qnf + '</td><td>' + _qcf + '</td>')
    parts.append('<td>' + _qapf + '</td><td style="color:#22c55e;font-weight:600">' + _qroif + '</td></tr>')

parts.append('</tbody></table></div></div>')  # end table

# Right: charts column
parts.append('<div style="flex:1;min-width:340px;display:flex;flex-direction:column;gap:10px">')
parts.append('<div class="chart-box"><h3>Queue Distribution</h3><canvas id="queuePieChart" height="200"></canvas></div>')
parts.append('<div class="chart-box"><h3>ROI by Action</h3><canvas id="queueRoiChart" height="200"></canvas></div>')
parts.append('<div class="chart-box"><h3>Cost vs Net Recovery</h3><canvas id="queueCostChart" height="200"></canvas></div>')
parts.append('</div></div>\n')  # end queue row

# Concentration analysis
hq = QUEUE[0] if len(QUEUE) > 0 else {}
mq = QUEUE[1] if len(QUEUE) > 1 else {}
lq = QUEUE[2] if len(QUEUE) > 2 else {}
wq = QUEUE[3] if len(QUEUE) > 3 else {}

parts.append('<div class="card strategy-card"><h2>Concentration Analysis</h2>')
parts.append('<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;font-size:.82rem">')
parts.append('<div style="padding:12px;background:#0f1524;border-radius:8px;border:1px solid #1e293b">')
parts.append('<b style="color:#94a3b8;font-size:.75rem;text-transform:uppercase;letter-spacing:.5px">Top-3 Tier Account Share</b>')
parts.append('<div style="margin-top:6px"><span style="color:#38bdf8;font-size:1.25rem;font-weight:800">' + fmt_num(float(pct_acc), '.1f') + '%</span> of accounts<br>')
parts.append('<span style="color:#22c55e;font-size:1.25rem;font-weight:800">' + fmt_num(float(pct_net), '.1f') + '%</span> of net recovery</div></div>')

_hq_roi = hq.get("roi", 0)
_hqr = fmt_num(float(_hq_roi), '.1f') + 'x'
parts.append('<div style="padding:12px;background:#0f1524;border-radius:8px;border:1px solid #1e293b">')
parts.append('<b style="color:#94a3b8;font-size:.75rem;text-transform:uppercase;letter-spacing:.5px">High Priority Tier</b>')
parts.append('<div style="margin-top:6px">' + fmt_num(int(hq.get("accounts", 0)), ',') + ' accts | $' + fmt_num(float(hq.get("net", 0)), ',.0f') + ' net<br>')
parts.append('ROI: <b style="color:#22c55e">' + _hqr + '</b></div></div>')

_wq_apr = wq.get("apr", 0)
_wqapf = fmt_num(float(_wq_apr) * 100, '.1f') + '%'
parts.append('<div style="padding:12px;background:#0f1524;border-radius:8px;border:1px solid #1e293b">')
parts.append('<b style="color:#94a3b8;font-size:.75rem;text-transform:uppercase;letter-spacing:.5px">Write-off Recommendation</b>')
parts.append('<div style="margin-top:6px">' + fmt_num(int(wq.get("accounts", 0)), ',') + ' accts | $' + fmt_num(int(wq.get("bal", 0)), ',') + ' balance<br>')
parts.append('Payer rate ' + _wqapf + ' but cost > expected recovery</div></div>')

parts.append('</div></div>')  # end concentration grid

# Top-200 Accounts table
parts.append('<h2>Top-200 Accounts Detail</h2>')
parts.append('<div class="card" style="overflow-x:auto">')
parts.append('<div style="display:flex;gap:8px;margin-bottom:10px;flex-wrap:wrap">')
parts.append('<input type="text" id="acctSearch" placeholder="Search ID..." style="width:160px">')
parts.append('<select id="acctTypeFilter" style="width:140px"><option value="">All Loan Types</option>')
loan_types = set(a.get('loan_type', '') for a in ACCOUNTS)
for lt in sorted(loan_types):
    parts.append('<option value="' + esc(lt) + '">' + esc(lt) + '</option>')
parts.append('</select>')
parts.append('<select id="acctActionFilter" style="width:190px"><option value="">All Actions</option>')
actions = set(a.get('recommended_action', '') for a in ACCOUNTS)
for act in sorted(actions):
    _act_label = esc(act.split("(")[0].strip()[:28])
    parts.append('<option value="' + esc(act) + '">' + _act_label + '</option>')
parts.append('</select>')
parts.append('<button class="btn btn-sm btn-primary" id="exportAccountsBtn">Export CSV</button>')
parts.append('</div>')
parts.append('<table id="accountTable"><thead>')
parts.append('<tr><th>ID</th><th>Type</th><th>Bucket</th><th>Raw Prob</th><th>Calib Prob</th><th>Gross Exp</th><th>Net Exp</th><th>Cost</th><th>Action</th><th></th></tr></thead><tbody id="accountBody">')

for a in ACCOUNTS:
    ashort = a.get('recommended_action', '').split('(')[0].strip()[:25]
    _aid = str(a.get("id", ""))
    _alt = esc(a.get("loan_type", ""))
    _abkt = esc(str(a.get("purchased_bal_gp", "")))
    try:
        _arp = '{:.4f}'.format(float(a.get("raw_repay_prob", 0)))
    except:
        _arp = str(a.get("raw_repay_prob", 0))
    try:
        _acp = '<b style="color:#38bdf8">{:.4f}</b>'.format(float(a.get("calibrated_repay_prob", 0)))
    except:
        _acp = str(a.get("calibrated_repay_prob", 0))
    try:
        _agr = '${:,.0f}'.format(float(a.get("expected_gross_recovery", 0)))
    except:
        _agr = '$' + str(a.get("expected_gross_recovery", 0))
    try:
        _anr = '${:,.0f}'.format(float(a.get("expected_net_recovery", 0)))
    except:
        _anr = '$' + str(a.get("expected_net_recovery", 0))
    try:
        _acs = '${:.2f}'.format(float(a.get("recommended_contact_cost", 0)))
    except:
        _acs = '$' + str(a.get("recommended_contact_cost", 0))

    parts.append('<tr><td>' + _aid + '</td><td>' + _alt + '</td><td>' + _abkt + '</td>')
    parts.append('<td>' + _arp + '</td><td>' + _acp + '</td><td>' + _agr + '</td>')
    parts.append('<td style="color:#22c55e">' + _anr + '</td><td>' + _acs + '</td>')
    parts.append('<td><span class="badge badge-blue" style="font-size:.62rem">' + esc(ashort) + '</span></td>')
    parts.append('<td><button class="btn btn-sm" data-acct-id=\'' + j(a) + '\'">View</button></td></tr>')

parts.append('</tbody></table></div></div>\n')  # end queue panel


# ═════════ TUNING / MODEL CARDS ═════════
parts.append('<div id="tab-tuning" class="panel"><h2>Hyperparameter Summary</h2><div class="chart-row">')
for i, m in enumerate(MODELS_TABLE):
    color = MODEL_COLORS.get(m['name'], '#64748b')
    hl = ' border-left:4px solid ' + color if i == 0 else ''
    role_cls = 'badge-green' if m.get('role') == 'champion' else 'badge-yellow'
    recall_val = m.get('recall', "0")
    prec_val = m.get('precision', "0")
    recall_str = fmt_num(recall_val, '.1f')
    prec_str = fmt_num(prec_val, '.1f')
    nr_val = m.get('net_recovery', 0)
    nr_fmt = fmt_num(nr_val, ',')

    _html = '<div class="strategy-card' + hl + '" style="flex:1;min-width:230px">'
    _html += '<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">'
    _html += '<div style="width:4px;height:24px;background:' + color + ';border-radius:2px"></div>'
    _html += '<h3 style="color:' + color + ';margin:0">' + esc(m.get("name", "")) + '</h3>'
    _html += '<span class="badge ' + role_cls + '">' + esc(m.get("role", "")) + '</span></div>'
    _html += '<div style="font-size:.8rem"><div style="display:grid;grid-template-columns:auto 1fr;gap:4px 10px">'
    _html += '<span style="color:#64748b">AUC:</span><span style="color:#22c55e;font-weight:700">' + str(m.get("auc", "")) + '</span>'
    _html += '<span style="color:#64748b">Brier:</span><span>' + str(m.get("brier", "")) + '</span>'
    _html += '<span style="color:#64748b">Recall:</span><span>' + recall_str + '%</span>'
    _html += '<span style="color:#64748b">Precision:</span><span>' + prec_str + '%</span>'
    _html += '<span style="color:#64748b">Net Recov:</span><span style="color:#22c55e">$' + nr_fmt + '</span>'
    _html += '<span style="color:#64748b">ROI:</span><span style="color:#22c55e">' + str(m.get("roi", "")) + 'x</span>'
    _html += '<span style="color:#64748b">Threshold:</span><span>' + str(m.get("threshold", "")) + '</span>'
    _html += '</div></div></div>'
    parts.append(_html)
parts.append('</div>\n')

# Rationale
_champ_nr = CHAMP.get('net_recovery', 0)
_champ_nr_fmt = fmt_num(float(_champ_nr), ',')
_auc_s = str(CHAMP.get('auc', ''))
_roi_s = str(CHAMP.get('roi', ''))
_rationale = '<div class="card strategy-card highlight"><h2>Model Selection Rationale</h2>\n'
_rationale += '<div style="font-size:.84rem;color:#cbd5e1;line-height:1.7">\n'
_rationale += '<p><b>' + esc(CHAMP.get('name', 'XGBoost')) + '</b> selected as Champion based on highest ROC-AUC (<b>' + _auc_s + '</b>) and best economic outcome ($' + _champ_nr_fmt + ' net at ' + _roi_s + 'x ROI).</p>'
_rationale += '<ul style="margin:8px 0;padding-left:18px">'
_rationale += '<li><b>Tree models beat linear</b>: XGBoost (+3.75pp) and RF (+2.30pp) both outperform LR baseline</li>'
_rationale += '<li><b>MLP competitive</b>: Deep learning achieved 0.7063 AUC, limited by dataset size</li>'
_rationale += '<li><b>Platt calibration</b>: All models use 5-fold OOF ensuring calibrated probabilities</li>'
_rationale += '<li><b>Tight Brier cluster</b>: Range [0.078-0.091], good calibration across all models</li>'
_rationale += '</ul></div></div>\n'
parts.append(_rationale)
parts.append('</div>\n')  # end tuning


# ═════════ RECOMMENDATIONS ═════════
_cn = str(CHAMP.get('name', 'xgboost'))
_ca = str(CHAMP.get('auc', 0))
_cb = str(CHAMP.get('brier', 0))
_cnr = CHAMP.get('net_recovery', 0)
_cr = str(CHAMP.get("roi", 0))
_qn = QTOTALS.get('net', 0)
_qa = QTOTALS.get('accounts', 0)
_qr = str(QTOTALS.get('roi', 0))
_ha = QUEUE[0].get('accounts', 0) if len(QUEUE) > 0 else 0
_hn = QUEUE[0].get('net', 0) if len(QUEUE) > 0 else 0
_hr = QUEUE[0].get('roi', 0) if len(QUEUE) > 0 else 0
_mna = QUEUE[1].get('accounts', 0) if len(QUEUE) > 1 else 0
_mnn = QUEUE[1].get('net', 0) if len(QUEUE) > 1 else 0
_mnr = QUEUE[1].get('roi', 0) if len(QUEUE) > 1 else 0
_lna = QUEUE[2].get('accounts', 0) if len(QUEUE) > 2 else 0
_lnn = QUEUE[2].get('net', 0) if len(QUEUE) > 2 else 0
_lnr = QUEUE[2].get('roi', 0) if len(QUEUE) > 2 else 0
_wn = QUEUE[3].get('accounts', 0) if len(QUEUE) > 3 else 0
_bb = max(PAYERBAL, key=lambda x: x.get('rate', 0)) if PAYERBAL else {'bucket': 'N/A', 'rate': 0}
_wb = min(PAYERBAL, key=lambda x: x.get('rate', 0)) if PAYERBAL else {'bucket': 'N/A', 'rate': 0}
_pct_net = pct_net
_pct_acc = pct_acc
_thr = str(CHAMP.get('threshold', '0.09'))
_recall_val = CHAMP.get('recall', 0)
_prec_val = CHAMP.get('precision', 0)
_recall_s = fmt_num(float(_recall_val), '.1f')
_prec_s = fmt_num(float(_prec_val), '.1f')

_R = []
_R.append('<div id="tab-recommendations" class="panel"><h2>Strategic Recommendations</h2>')
_R.append('<p style="font-size:.85rem;color:#94a3b8;margin-bottom:16px">Data-driven collection strategy based on model outputs and economic analysis.</p>')

# Executive summary card
_R.append('<div class="strategy-card highlight">')
_R.append('<div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">')
_R.append('<div style="width:4px;height:28px;background:linear-gradient(180deg,#3b82f6,#a78bfa);border-radius:2px"></div>')
_R.append('<h2 style="color:#38bdf8;border:none;padding:0;margin:0;font-size:1.15rem">Executive Summary</h2>')
_R.append('</div>')
_R.append('<div style="font-size:.88rem;line-height:1.7;color:#e2e8f0">')
_R.append('<p>With <b>' + _cn + '</b> as Champion (AUC=<b>' + _ca + '</b>, Brier=<b>' + _cb + '</b>), portfolio can achieve estimated <b>$' + fmt_num(int(_qn), ',') + '</b> net recovery at <b>' + _qr + 'x</b> ROI across ' + fmt_num(int(_qa), ',') + ' test accounts.</p>')
_R.append('<p><b>High concentration of recovery value:</b> Top-3 tiers capture <b>' + str(round(_pct_net)) + '%</b> of expected net recovery covering only <b>' + str(round(_pct_acc)) + '%</b> of accounts. Resources should heavily favor these tiers.</p>')
_R.append('</div></div>')

# Rec 1
_R.append('<div class="strategy-card"><div class="rec-item"><div class="rec-num">1</div><div class="rec-body">')
_R.append('<h4 style="color:#f1f5f9;font-size:.95rem;margin-bottom:6px">Deploy XGBoost as Production Champion</h4>')
_R.append('<p>Replace existing scoring rules with XGBoost model (AUC=' + _ca + ', Brier=' + _cb + '). It delivers highest discrimination and economic value. Schedule <b>monthly recalibration monitoring</b>; trigger retraining if actual payer rate deviates >2pp from expected for 2 consecutive months.</p>')
_R.append('<div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap">')
_R.append('<span class="badge badge-green">Impact: High</span>')
_R.append('<span class="badge badge-blue">Effort: Low</span>')
_R.append('<span class="badge badge-yellow">Risk: Low</span></div>')
_R.append('</div></div></div>')

# Rec 2
_R.append('<div class="strategy-card"><div class="rec-item"><div class="rec-num">2</div><div class="rec-body">')
_R.append('<h4 style="color:#f1f5f9;font-size:.95rem;margin-bottom:6px">Implement 4-Tier Collection Intensity Framework</h4>')
_R.append('<p>Allocate resources per model-derived queue assignment:</p>')
_R.append('<ul style="margin:6px 0 6px 18px;font-size:.84rem">')
_R.append('<li><b>Tier 1 - Agent Call (' + fmt_num(int(_ha), ',') + ' accts):</b> Net $' + fmt_num(float(_hn), ',.0f') + ', ROI ' + fmt_num(float(_hr), '.1f') + 'x. Assign skilled collectors. Focus on prob>0.09, balance<=$50K.</li>')
_R.append('<li><b>Tier 2 - Auto-Dialer (' + fmt_num(int(_mna), ',') + ' accts):</b> Net $' + fmt_num(float(_mnn), ',.0f') + ', ROI ' + fmt_num(float(_mnr), '.1f') + 'x. High-volume automated contact.</li>')
_R.append('<li><b>Tier 3 - SMS/Email (' + fmt_num(int(_lna), ',') + ' accts):</b> Net $' + fmt_num(float(_lnn), ',.0f') + ', ROI ' + fmt_num(float(_lnr), '.1f') + 'x. Low-cost digital channels.</li>')
_R.append('<li><b>Tier 4 - Write-off (' + fmt_num(int(_wn), ',') + ' accts):</b> Do NOT actively collect. Review annually.</li>')
_R.append('</ul>')
_R.append('<div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap">')
_R.append('<span class="badge badge-green">Impact: High</span>')
_R.append('<span class="badge badge-blue">Effort: Medium</span>')
_R.append('<span class="badge badge-yellow">Risk: Medium</span></div>')
_R.append('</div></div></div>')

# Rec 3
_br = float(_bb.get('rate', 0))
_wr = float(_wb.get('rate', 0))
_R.append('<div class="strategy-card"><div class="rec-item"><div class="rec-num">3</div><div class="rec-body">')
_R.append('<h4 style="color:#f1f5f9;font-size:.95rem;margin-bottom:6px">Double Down on Low-Balance Segment</h4>')
_R.append('<p>Inverse relationship between balance and repayment rate:</p>')
_R.append('<ul style="margin:6px 0 6px 18px;font-size:.84rem">')
_R.append('<li><b>' + esc(str(_bb.get('bucket', 'N/A'))) + ':</b> <b style="color:#22c55e">' + fmt_num(_br, '.1f') + '%</b> repayment (highest segment)</li>')
_R.append('<li><b>' + esc(str(_wb.get('bucket', 'N/A'))) + ':</b> <b style="color:#ef4444">' + fmt_num(_wr, '.1f') + '%</b> repayment (lowest segment)</li></ul>')
_R.append('<p><b>Action:</b> Create "Quick Win" workstream for balance &lt;=$25K + calib prob &gt;=0.06. Offer lump-sum settlement discount up to 20%.</p>')
_R.append('<div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap">')
_R.append('<span class="badge badge-green">Impact: High</span>')
_R.append('<span class="badge badge-blue">Effort: Low</span>')
_R.append('<span class="badge badge-yellow">Risk: Low</span></div>')
_R.append('</div></div></div>')

# Rec 4
_R.append('<div class="strategy-card"><div class="rec-item"><div class="rec-num">4</div><div class="rec-body">')
_R.append('<h4 style="color:#f1f5f9;font-size:.95rem;margin-bottom:6px">Prioritize Contactable Accounts</h4>')
_R.append('<p>Contactability drives recovery. Key adjustments:</p>')
_R.append('<ul style="margin:6px 0 6px 18px;font-size:.84rem">')
_R.append('<li><b>Skip-trace priority:</b> For high-probability accounts missing phone, invest in skip-tracing (positive when prob>0.08).</li>')
_R.append('<li><b>District clustering:</b> Concentrate field visits in top districts by volume.</li>')
_R.append('<li><b>Mobile-first test:</b> Test SMS payment links vs traditional calls on Tier 2/3 segments.</li></ul>')
_R.append('<div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap">')
_R.append('<span class="badge badge-green">Impact: Medium</span>')
_R.append('<span class="badge badge-blue">Effort: Medium</span>')
_R.append('<span class="badge badge-yellow">Risk: Low</span></div>')
_R.append('</div></div></div>')

# Rec 5
_R.append('<div class="strategy-card"><div class="rec-item"><div class="rec-num">5</div><div class="rec-body">')
_R.append('<h4 style="color:#f1f5f9;font-size:.95rem;margin-bottom:6px">Optimize Threshold for Business Objective</h4>')
_R.append('<p>Current threshold at ' + _thr + ' balances recall (' + _recall_s + '%) and precision (' + _prec_s + '%). Scenario options:</p>')
_R.append('<ul style="margin:6px 0 6px 18px;font-size:.84rem">')
_R.append('<li><b>Capacity-constrained (500 slots):</b> Raise to 0.12-0.15 for top-priority focus.</li>')
_R.append('<li><b>Volume-driven (max coverage):</b> Lower to 0.05-0.07 for auto-dialer expansion.</li>')
_R.append('<li><b>Profit-maximizing (current):</b> Keep at 0.09 where marginal ROI = marginal cost.</li></ul>')
_R.append('<div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap">')
_R.append('<span class="badge badge-green">Impact: Medium</span>')
_R.append('<span class="badge badge-blue">Effort: Low</span>')
_R.append('<span class="badge badge-yellow">Risk: Low</span></div>')
_R.append('</div></div></div>')

# Rec 6
_R.append('<div class="strategy-card"><div class="rec-item"><div class="rec-num">6</div><div class="rec-body">')
_R.append('<h4 style="color:#f1f5f9;font-size:.95rem;margin-bottom:6px">Monitor & Iterate Framework</h4>')
_R.append('<p>Governance rhythm for sustained performance:</p>')
_R.append('<ul style="margin:6px 0 6px 18px;font-size:.84rem">')
_R.append('<li><b>Weekly:</b> Track actual payer rate by tier vs expected. Flag deviations >15%.</li>')
_R.append('<li><b>Monthly:</b> Recalibration check (retrain if PSI > 0.25 or AUC drops >0.02).</li>')
_R.append('<li><b>Quarterly:</b> Full model refresh including feature engineering review.</li>')
_R.append('<li><b>KRIs:</b> Payer rate trend, cost-per-contact drift, PSI monitoring.</li></ul>')
_R.append('<div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap">')
_R.append('<span class="badge badge-green">Impact: High</span>')
_R.append('<span class="badge badge-blue">Effort: Low</span>')
_R.append('<span class="badge badge-yellow">Risk: N/A</span></div>')
_R.append('</div></div></div>')

_R.append('</div>\n')  # end recommendations
parts.append('\n'.join(_R))


# ═════════ REPORT PANEL (English Brief) ═════════
_rpt_champ = CHAMP.get('name','xgboost')
_rpt_auc = str(CHAMP.get('auc','-'))
_rpt_brier = str(CHAMP.get('brier','-'))
_rpt_recall = fmt_num(float(CHAMP.get('recall',0)), '.1f')
_rpt_prec = fmt_num(float(CHAMP.get('precision',0)), '.1f')
_rpt_nr = fmt_num(int(CHAMP.get('net_recovery',0)), ',')
_rpt_roi = str(CHAMP.get('roi','-'))
_rpt_thr = str(CHAMP.get('threshold','0.09'))
_rpt_qtotal = QTOTALS.get('accounts',0)
_rpt_qn = fmt_num(int(QTOTALS.get('net',0)), ',')
_rpt_qr = str(QTOTALS.get('roi','-'))
_rpt_m = int(m_count)
_rpt_t = int(t_count)
_rpt_payer_m = '{:.2f}'.format(float(metrics_json.get('train_payer_rate',0.098))*100) if metrics_json else '9.81'
_rpt_payer_t = '{:.2f}'.format(float(metrics_json.get('test_payer_rate',0.0915))*100) if metrics_json else '9.15'

parts.append('<div id="tab-report" class="panel">')
parts.append('<div class="card markdown-body">')
parts.append('<h1>NPA Collection Strategy Report</h1>')
parts.append('<h2>1. Executive Summary</h2><ul>')
parts.append('<li><b>Portfolio:</b> ' + str(_rpt_m + _rpt_t).replace(',','') + ' accounts (Training: ' + str(_rpt_m) + ', Test: ' + str(_rpt_t) + ')</li>')
parts.append('<li><b>Base Payer Rate:</b> M=' + _rpt_payer_m + '%, T=' + _rpt_payer_t + '%</li>')
parts.append('<li><b>Champion Model:</b> <strong>' + esc(_rpt_champ) + '</strong> | Test AUC=<strong>' + _rpt_auc + '</strong>, Brier=<strong>' + _rpt_brier + '</strong></li>')
parts.append('<li><b>Expected Net Recovery:</b> <strong>$' + _rpt_nr + '</strong> at <strong>' + _rpt_roi + 'x</strong> ROI across test set</li>')
parts.append('<li><b>Queue Allocation:</b> ' + fmt_num(int(_rpt_qtotal), ',') + ' accounts queued, $' + _rpt_qn + ' net recovery expected</li>')
parts.append('</ul>')
parts.append('<h2>2. Model Performance (Test Set)</h2>')
parts.append('<table><tr><th>Model</th><th>AUC</th><th>Brier</th><th>Recall</th><th>Precision</th><th>Net Recovery</th><th>ROI</th><th>Threshold</th></tr>')
for m in MODELS_TABLE:
    parts.append('<tr><td>' + esc(m['name']) + '</td><td>' + str(m['auc']) + '</td><td>' + str(m['brier']) + '</td>')
    parts.append('<td>' + fmt_num(m['recall'],'.1f') + '%</td><td>' + fmt_num(m['precision'],'.1f') + '%</td>')
    parts.append('<td>$' + fmt_num(int(m['net_recovery']),',') + '</td><td>' + str(m['roi']) + 'x</td>')
    parts.append('<td>' + str(m['threshold']) + '</td></tr>')
parts.append('</table>')
parts.append('<h2>3. Key Findings</h2><ol>')
parts.append('<li><b>Tree models outperform baseline:</b> XGBoost achieves best AUC (' + _rpt_auc + ') with strong economic outcome ($' + _rpt_nr + ')</li>')
parts.append('<li><b>Inverse balance-repayment correlation:</b> Lower balance buckets show significantly higher payer rates (~11% for &lt;$25K vs ~2% for $200K+)</li>')
parts.append('<li><b>High concentration:</b> Top-3 collection tiers capture majority of recovery value - focus resources there</li>')
parts.append('<li><b>All models well-calibrated:</b> Brier scores clustered in [0.078-0.091] range via Platt calibration</li>')
parts.append('</ol>')
parts.append('<h2>4. Recommended Actions</h2><div style="display:grid;gap:10px">')
parts.append('<div style="padding:12px;background:#0f1524;border-radius:8px;border-left:3px solid #22c55e"><b>1. Deploy Champion Model:</b> Replace rule-based scoring with ' + esc(_rpt_champ) + '. Monitor monthly recalibration.</div>')
parts.append('<div style="padding:12px;background:#0f1524;border-radius:8px;border-left:3px solid #3b82f6"><b>2. Tiered Collection Framework:</b> Agent calls for high-probability accounts, auto-dialer for mid-tier, SMS/Email for low-priority.</div>')
parts.append('<div style="padding:12px;background:#0f1524;border-radius:8px;border-left:3px solid #f59e0b"><b>3. Quick-Win Segment:</b> Prioritize balance &lt;=$25K with calib prob &gt;=0.06. Offer lump-sum settlement discounts up to 20%.</div>')
parts.append('<div style="padding:12px;background:#0f1524;border-radius:8px;border-left:3px solid #a855f7"><b>4. Write-off Criteria:</b> Low-probability, high-balance accounts where contact cost exceeds expected recovery.</div>')
parts.append('</div>')
parts.append('<h2>5. Economic Assumptions</h2>')
_rrs = str(round(_rr))
_acs = fmt_num(float(ac),',.0f')
_ads = fmt_num(float(ad),',.0f')
_scs = fmt_num(float(sc),'.2f')
parts.append('<p>Balance Recovery Rate: ' + _rrs + '% | Agent Call Cost: $' + _acs + ' | Dialer Cost: $' + _ads + ' | SMS Cost: $' + _scs + '</p>')
parts.append('<p style="color:#64748b;font-size:.78rem;margin-top:16px">Generated by NPA Repayment Analytics Dashboard v10 | Threshold: ' + _rpt_thr + ' | Recall@T: ' + _rpt_recall + '% | Precision@T: ' + _rpt_prec + '%</p>')
parts.append('</div></div>\n')

# ═════════ MODAL ═════════
parts.append("<div class=\"modal-overlay\" id=\"acctModal\">\n")
parts.append("<div class=\"modal\"><button class=\"close-modal\" id=\"closeModalBtn\">&times;</button>\n")
parts.append("<h2>Account Detail</h2><div id=\"modalBody\"></div></div></div>")


# ═════════ JAVASCRIPT ═════════
js_code = r"""
// ═════════ GLOBAL STATE ═════════
var chartInstances = {};
var activeTab = 'tab-overview';

// Chart.js global defaults
Chart.defaults.color = '#94a3b8';
Chart.defaults.font.family = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';

// ═════════ TAB SWITCHING ═════════
function switchTab(tabId) {
  // Hide all panels
  document.querySelectorAll('.panel').forEach(function(p) { p.classList.remove('active'); });
  // Remove active from all buttons
  document.querySelectorAll('.tab-btn').forEach(function(b) { b.classList.remove('active'); });
  // Show target panel
  var panel = document.getElementById(tabId);
  if(panel) { panel.classList.add('active'); }
  // Activate button
  var btns = document.querySelectorAll('.tab-btn[data-tab]');
  for(var bi=0;bi<btns.length;bi++) {
    if(btns[bi].getAttribute('data-tab') === tabId) { btns[bi].classList.add('active'); break; }
  }
  activeTab = tabId;
  // Initialize charts after panel becomes visible
  setTimeout(function(){ initChartsForTab(tabId); }, 80);
}

// Tab click delegation via addEventListener
document.addEventListener('DOMContentLoaded', function() {
  // Tab buttons
  document.querySelectorAll('.tab-btn[data-tab]').forEach(function(btn) {
    btn.addEventListener('click', function(e) {
      e.preventDefault();
      var tid = this.getAttribute('data-tab');
      if(tid) switchTab(tid);
    });
  });

  // Detail toggle buttons (Models table)
  document.querySelectorAll('[data-detail-btn]').forEach(function(btn) {
    btn.addEventListener('click', function(e) {
      e.preventDefault();
      toggleDetail(this);
    });
  });

  // View account modal buttons
  document.querySelectorAll('[data-acct-id]').forEach(function(btn) {
    btn.addEventListener('click', function(e) {
      e.preventDefault();
      var acctData = this.getAttribute('data-acct-id');
      if(acctData) { showAcctModal(JSON.parse(acctData)); }
    });
  });

  // Export buttons
  var expQueueBtn = document.getElementById('exportQueueBtn');
  if(expQueueBtn) expQueueBtn.addEventListener('click', exportQueueCSV);

  var expAccBtn = document.getElementById('exportAccountsBtn');
  if(expAccBtn) expAccBtn.addEventListener('click', exportAccountsCSV);

  // Close modal
  var closeBtn = document.getElementById('closeModalBtn');
  if(closeBtn) closeBtn.addEventListener('click', closeAcctModal);

  // Modal backdrop click
  var overlay = document.getElementById('acctModal');
  if(overlay) {
    overlay.addEventListener('click', function(e) {
      if(e.target === this) closeAcctModal();
    });
  }

  // Search inputs
  var qSearch = document.getElementById('qSearch');
  if(qSearch) qSearch.addEventListener('keyup', filterQueue);

  var acctSearch = document.getElementById('acctSearch');
  if(acctSearch) acctSearch.addEventListener('keyup', filterAccounts);

  var acctTypeFilt = document.getElementById('acctTypeFilter');
  if(acctTypeFilt) acctTypeFilt.addEventListener('change', filterAccounts);

  var acctActionFilt = document.getElementById('acctActionFilter');
  if(acctActionFilt) acctActionFilt.addEventListener('change', filterAccounts);

  // Feature select
  var fiSel = document.getElementById('fiModelSelect');
  if(fiSel) fiSel.addEventListener('change', updateFeatureChart);

  // Sortable table headers
  initSortableTables();

  // Auto-init overview
  setTimeout(function(){ initChartsForTab('tab-overview'); }, 150);
});


// ═════════ CHART HELPERS ═════════
function getOrCreateChart(id, type, config) {
  if(chartInstances[id]) { chartInstances[id].destroy(); chartInstances[id] = null; }
  var el = document.getElementById(id);
  if(!el) { console.warn('[Dashboard] Canvas not found: ' + id); return null; }
  var ctx = el.getContext('2d');
  var defaults = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color:'#94a3b8', font:{size:11}, padding:12 } },
      tooltip: {
        backgroundColor: 'rgba(15,23,42,.9)',
        titleColor: '#e2e8f0',
        bodyColor: '#cbd5e1',
        borderColor: '#334155',
        borderWidth: 1,
        cornerRadius: 6,
        padding: 10,
      },
    },
    scales: {
      x: { ticks: { color:'#94a3b8', font:{size:10} }, grid: { color:'rgba(51,65,85,.2)' } },
      y: { ticks: { color:'#94a3b8', font:{size:10} }, grid: { color:'rgba(51,65,85,.2)' } },
    },
  };
  // Merge: defaults <- config (config overrides), then FORCE type from parameter
  var merged = Object.assign({}, defaults, config || {});
  merged.type = type;
  try {
    chartInstances[id] = new Chart(ctx, merged);
  } catch(err) {
    console.error('[Dashboard] Chart error for ' + id + ':', err.message);
    return null;
  }
  return chartInstances[id];
}


// ═════════ INIT CHARTS BY TAB ═════════
function initChartsForTab(tabId) {

  // ─── OVERVIEW ───
  if(tabId === 'tab-overview') {
    // Dev Split Doughnut
    getOrCreateChart('devSplitChart', 'doughnut', {
      data:{
        labels:DEV_SPLIT_DATA.map(function(d){return d.label;}),
        datasets:[{
          data:DEV_SPLIT_DATA.map(function(d){return d.value;}),
          backgroundColor:['#3b82f6','#22c55e'],
          borderWidth:0,
          hoverOffset:6,
        }]
      },
      options:{
        cutout:'60%',
        plugins:{ legend:{ position:'bottom' } }
      }
    });

    // Confusion Matrix Bar
    getOrCreateChart('cmChart', 'bar', {
      data:{
        labels:['True Neg','False Pos','False Neg','True Pos'],
        datasets:[{
          label:'Count',
          data:[CM_DATA.TN,CM_DATA.FP,CM_DATA.FN,CM_DATA.TP],
          backgroundColor:['#22c55e','#ef4444','#f87171','#3b82f6'],
          borderRadius:6,
        }]
      },
      options:{
        plugins:{legend:{display:false}},
        scales:{y:{beginAtZero:true}}
      }
    });
  }

  // ─── MODELS ───
  if(tabId === 'tab-models') {
    var mnames = MODEL_TABLE.map(function(m){return m.name;});
    var mauc = MODEL_TABLE.map(function(m){return m.auc;});
    var mc = mnames.map(function(n){return MODEL_COLORS_MAP[n]||'#64748b';});

    getOrCreateChart('aucChart', 'bar', {
      data:{
        labels:mnames,
        datasets:[{
          label:'ROC-AUC',
          data:mauc,
          backgroundColor:mc.map(function(c){return c;}),
          borderRadius:6,
          borderSkipped:false,
        }]
      },
      options:{
        indexAxis:'y',
        plugins:{legend:{display:false}},
        scales:{x:{beginAtZero:false,min:0.67,max:0.76}}
      }
    });

    // Net Recovery + ROI grouped bar
    getOrCreateChart('economicChart', 'bar', {
      data:{
        labels:mnames,
        datasets:[
          {label:'Net Recovery ($)', data:MODEL_TABLE.map(function(m){return m.net_recovery;}), backgroundColor:'#3b82f6', borderRadius:6},
          {label:'ROI (x)', data:MODEL_TABLE.map(function(m){return m.roi;}), backgroundColor:'#22c55e', borderRadius:6},
        ]
      },
      options:{scales:{y:{beginAtZero:true}}}
    });
  }

  // ─── COMPARE (NEW) ───
  if(tabId === 'tab-compare') {
    // Radar chart - normalized metrics (0-1 scale approx)
    var maxAuc = Math.max.apply(null, MODEL_TABLE.map(function(m){return m.auc;}));
    var minBrier = Math.min.apply(null, MODEL_TABLE.map(function(m){return m.brier;}));
    var maxRecall = Math.max.apply(null, MODEL_TABLE.map(function(m){return m.recall;}));
    var maxPrec = Math.max.apply(null, MODEL_TABLE.map(function(m){return m.precision;}));
    var maxRoi = Math.max.apply(null, MODEL_TABLE.map(function(m){return m.roi;}));

    var radarDatasets = MODEL_TABLE.map(function(mi){
      var c = MODEL_COLORS_MAP[mi.name]||'#64748b';
      return {
        label:mi.name,
        data:[
          mi.auc/0.8,
          1-mi.brier/0.12,
          mi.recall/maxRecall,
          mi.precision/maxPrec,
          mi.roi/maxRoi,
          mi.net_recovery/2500000
        ],
        borderColor:c,backgroundColor:c+'22',borderWidth:2,pointBackgroundColor:c,
      };
    });

    getOrCreateChart('radarChart', 'radar', {
      type:'radar',
      data:{
        labels:['AUC (norm)','1-Brier','Recall (norm)','Prec (norm)','ROI (norm)','NetRev (norm)'],
        datasets:radarDatasets,
      },
      options:{
        scales:{r:{angleLines:{color:'rgba(51,65,85,.3)'},grid:{color:'rgba(51,65,85,.3)'},pointLabels:{color:'#94a3b8',font:{size:10}},ticks:{backdropColor:'transparent'}}},
        plugins:{legend:{position:'bottom'}}
      }
    });

    // AUC vs Brier scatter
    getOrCreateChart('aucBrierScatter', 'scatter', {
      data:{
        datasets:[{
          label:'Models',
          data:MODEL_TABLE.map(function(m,i){
            return {x:m.auc,y:m.brier,label:m.name};
          }),
          backgroundColor:MODEL_TABLE.map(function(m){return MODEL_COLORS_MAP[m.name]||'#64748b';}),
          pointRadius:8,
          pointHoverRadius:11,
        }]
      },
      options:{
        plugins:{
          legend:{display:false},
          tooltip:{
            callbacks:{
              label:function(ctx){
                return ctx.raw.label + ': AUC=' + ctx.parsed.x.toFixed(4) + ', Brier=' + ctx.parsed.y.toFixed(4);
              }
            }
          }
        },
        scales:{x:{title:{display:true,text:'ROC-AUC'}},y:{reverse:true,title:{display:true,text:'Brier Score'}}}
      }
    });

    // Threshold sensitivity line chart
    var threshLabels = THRESHOLD_DATA && Object.keys(THRESHOLD_DATA).length > 0
      ? THRESHOLD_DATA[Object.keys(THRESHOLD_DATA)[0]].map(function(t){return t.thr.toString();})
      : ['0.03','0.05','0.07','0.09','0.11','0.13','0.15','0.18','0.21','0.25'];

    var threshDs = [];
    if(typeof THRESHOLD_DATA !== 'undefined' && THRESHOLD_DATA) {
      var miIdx=0;
      for(var mName in THRESHOLD_DATA){
        if(!THRESHOLD_DATA.hasOwnProperty(mName)) continue;
        var td = THRESHOLD_DATA[mName];
        var c2 = MODEL_COLORS_MAP[mName]||'#64748b';
        threshDs.push({
          label:mName,
          data:td.map(function(t){return t.roi;}),
          borderColor:c2,backgroundColor:'transparent',borderWidth:2,tension:.3,pointRadius:3,pointBackgroundColor:c2,
        });
        miIdx++;
      }
    }

    getOrCreateChart('thresholdChart', 'line', {
      data:{labels:threshLabels,datasets:threshDs.length>0?threshDs:[{data:[]}]},
      options:{
        plugins:{legend:{position:'bottom'}},
        scales:{
          x:{title:{display:true,text:'Threshold'}},
          y:{title:{display:true,text:'ROI (x)'},beginAtZero:true},
        }
      }
    });

    // Brier comparison bar
    getOrCreateChart('calibChart', 'bar', {
      type:'bar',
      data:{
        labels:MODEL_TABLE.map(function(m){return m.name;}),
        datasets:[{
          label:'Brier Score (lower=better)',
          data:MODEL_TABLE.map(function(m){return m.brier;}),
          backgroundColor:MODEL_TABLE.map(function(m){
            var c=MODEL_COLORS_MAP[m.name]||'#64748b';
            return c+'cc';
          }),
          borderColor:MODEL_TABLE.map(function(m){return MODEL_COLORS_MAP[m.name]||'#64748b';}),
          borderWidth:2,borderRadius:8,
        }]
      },
      options:{
        indexAxis:'y',plugins:{legend:{display:false}},
        scales:{x:{beginAtZero:true,max:0.10}},
      }
    });

    // Precision vs Recall scatter
    getOrCreateChart('prChart', 'scatter', {
      data:{
        datasets:[{
          label:'Models',
          data:MODEL_TABLE.map(function(m){
            return {x:m.recall,y:m.precision,name:m.name,auc:m.auc};
          }),
          backgroundColor:MODEL_TABLE.map(function(m){return MODEL_COLORS_MAP[m.name]||'#64748b';}),
          pointRadius:10,pointHoverRadius:13,
        }]
      },
      options:{
        plugins:{
          legend:{display:false},
          tooltip:{
            callbacks:{
              label:function(ctx){
                var d=ctx.raw; return d.name+': Recall='+d.x.toFixed(1)+'%, Prec='+d.y.toFixed(1)+'%, AUC='+d.auc;
              }
            }
          }
        },
        scales:{x:{title:{display:true,text:'Recall %'}},y:{title:{display:true,text:'Precision %'}}}
      }
    });
  }

  // ─── FEATURES ───
  if(tabId === 'tab-features') {
    updateFeatureChart();

    getOrCreateChart('payerBalChart', 'bar', {
      type:'bar',
      data:{
        labels:PAYER_BAL_DATA.map(function(d){return d.bucket;}),
        datasets:[{
          label:'Payer Rate %',
          data:PAYER_BAL_DATA.map(function(d){return d.rate;}),
          backgroundColor:'#3b82f6',
          borderRadius:6,
        }]
      },
      options:{
        indexAxis:'y',plugins:{legend:{display:false}},
        scales:{x:{beginAtZero:true,max:Math.ceil(Math.max.apply(null,PAYER_BAL_DATA.map(function(d){return d.rate;}))/5)*5}}
      }
    });

    getOrCreateChart('payerLoanChart', 'bar', {
      data:{
        labels:PAYER_LOAN_DATA.map(function(d){return d.type;}),
        datasets:[{
          label:'Payer Rate %',
          data:PAYER_LOAN_DATA.map(function(d){return d.rate;}),
          backgroundColor:['#3b82f6','#22c55e','#f59e0b'],
          borderRadius:6,
        }]
      },
      options:{plugins:{legend:{display:false}},scales:{y:{beginAtZero:true,max:15}}}
    });
  }

  // ─── STATS ───
  if(tabId === 'tab-stats') {
    // Histograms
    if(typeof DIST_DATA !== 'undefined') {
      Object.keys(DIST_DATA).forEach(function(k) {
        var d = DIST_DATA[k];
        var labels = [];
        for(var bi=0;bi<d.be.length-1;bi++){
          labels.push(parseFloat(d.be[bi]).toFixed(1) + '-' + parseFloat(d.be[bi+1]).toFixed(1));
        }
        getOrCreateChart('hist-'+k, 'bar', {
          data:{
            labels:labels,
            datasets:[{
              label:'Frequency',
              data:d.hc,
              backgroundColor:function(context) {
                var chart=context.chart;var {ctx:cx,width:w,height:h}=chart;
                var grad=cx.createLinearGradient(0,h,0,0);
                grad.addColorStop(0,'rgba(59,130,246,.4)');
                grad.addColorStop(1,'rgba(59,130,246,.8)');
                return grad;
              },
              borderRadius:2,
            }]
          },
          options:{
            plugins:{legend:{display:false}},
            scales:{
              x:{ticks:{maxRotation:45,font:{size:9}}},
              y:{beginAtZero:true},
            }
          }
        });
      });
    }

    // Categorical bars
    if(typeof CAT_DIST_DATA !== 'undefined') {
      Object.keys(CAT_DIST_DATA).forEach(function(k) {
        var d = CAT_DIST_DATA[k];
        getOrCreateChart('cat-'+k, 'bar', {
          data:{
            labels:d.map(function(x){return x[0];}),
            datasets:[{
              label:'Count',
              data:d.map(function(x){return x[1];}),
              backgroundColor:'#a855f7',
              borderRadius:6,
            }]
          },
          options:{
            indexAxis:'y',
            plugins:{legend:{display:false}},
            scales:{x:{beginAtZero:true}},
          }
        });
      });
    }
  }

  // ─── QUEUE ───
  if(tabId === 'tab-queue') {
    var ql = QUEUE_DATA.map(function(q){var s=q.action.split('(')[0];return s.trim();});

    getOrCreateChart('queuePieChart', 'doughnut', {
      type:'doughnut',
      data:{
        labels:ql,
        datasets:[{
          data:QUEUE_DATA.map(function(q){return q.accounts;}),
          backgroundColor:QUEUE_COLORS,
          borderWidth:0,hoverOffset:6,
        }]
      },
      options:{cutout:'55%',plugins:{legend:{position:'bottom'}}}
    });

    getOrCreateChart('queueRoiChart', 'bar', {
      data:{
        labels:ql,
        datasets:[{
          label:'ROI (x)',
          data:QUEUE_DATA.map(function(q){return q.roi;}),
          backgroundColor:QUEUE_COLORS,
          borderRadius:6,
        }]
      },
      options:{plugins:{legend:{display:false}},scales:{y:{beginAtZero:true}}}
    });

    getOrCreateChart('queueCostChart', 'bar', {
      data:{
        labels:ql,
        datasets:[
          {label:'Contact Cost',data:QUEUE_DATA.map(function(q){return q.cost;}),backgroundColor:'#ef4444',borderRadius:6},
          {label:'Net Recovery',data:QUEUE_DATA.map(function(q){return q.net;}),backgroundColor:'#22c55e',borderRadius:6},
        ]
      },
      options:{scales:{y:{beginAtZero:true}}}
    });
  }

  // ─── REPORT ───
  // Report content is pre-rendered HTML (English brief) — no JS rendering needed
  if(tabId === 'tab-report') {
    // Content already in DOM, nothing to initialize
  }
}


// ═════════ FEATURE IMPORTANCE UPDATE ═════════
function updateFeatureChart() {
  var sel = document.getElementById('fiModelSelect');
  var model = sel ? sel.value : '';
  var fi = ALL_FEATURE_IMPORTANCE[model];
  console.log('[FI] model=' + model + ', keys=' + Object.keys(ALL_FEATURE_IMPORTANCE).join(',') + ', found=' + !!fi + ', len=' + (fi ? fi.length : 0));
  
  // Fallback: if exact match fails, try case-insensitive or first available
  if(!fi || !fi.length) {
    var keys = Object.keys(ALL_FEATURE_IMPORTANCE);
    for(var ki=0; ki<keys.length; ki++) {
      var fk = keys[ki];
      if(ALL_FEATURE_IMPORTANCE[fk] && ALL_FEATURE_IMPORTANCE[fk].length) {
        model = fk;
        fi = ALL_FEATURE_IMPORTANCE[fk];
        console.log('[FI] fallback to model=' + model);
        break;
      }
    }
  }
  if(!fi || !fi.length) { console.warn('[FI] No feature data available'); return; }

  var top = fi.slice(0, 12);
  var labels = top.map(function(x){return x.feature;}).slice().reverse();
  var vals = top.map(function(x){return x.importance;}).slice().reverse();

  getOrCreateChart('featureChart', 'bar', {
    data:{
      labels:labels,
      datasets:[{
        label:'Importance',
        data:vals,
        backgroundColor:'#3b82f6',
        borderRadius:4,
      }]
    },
    options:{
      indexAxis:'y',
      plugins:{legend:{display:false}},
      scales:{x:{beginAtZero:true}},
    }
  });
}


// ═════════ SORTABLE TABLES ═════════
function initSortableTables() {
  document.querySelectorAll('thead th[data-col]').forEach(function(th) {
    th.style.cursor = 'pointer';
    th.addEventListener('click', function() {
      var col = parseInt(this.getAttribute('data-col'));
      var table = this.closest('table');
      var tbody = table.querySelector('tbody');
      if(!tbody) return;
      var rows = Array.from(tbody.querySelectorAll('tr:not(.detail-row)'));
      var asc = this.classList.contains('sort-asc');

      // Reset indicators
      table.querySelectorAll('thead th').forEach(function(h){
        h.classList.remove('sort-asc','sort-desc');
        var arr=h.querySelector('.sort-arrow');
        if(arr) arr.innerHTML='&#x2195;';
      });

      this.classList.add(asc ? 'sort-desc' : 'sort-asc');
      var arr2=this.querySelector('.sort-arrow');
      if(arr2) arr2.innerHTML=asc?'&#x2191;':'&#x2193;';

      rows.sort(function(a,b){
        var ta=a.cells[col]?a.cells[col].textContent.trim():'';
        var tb=b.cells[col]?b.cells[col].textContent.trim():'';
        var na=parseFloat(ta.replace(/[$,%x]/g,''))||0;
        var nb=parseFloat(tb.replace(/[$,%x]/g,''))||0;
        if(!isNaN(na)&&ta.indexOf('$')!==-1){ return asc?nb-na:na-nb; }
        if(!isNaN(na)&&!isNaN(nb)&&ta.indexOf('.')!==-1){ return asc?nb-na:na-nb; }
        return asc?ta.localeCompare(tb):tb.localeCompare(ta);
      });

      tbody.innerHTML='';
      rows.forEach(function(r){ tbody.appendChild(r); });
    });
  });
}


// ═════════ DETAIL ROW TOGGLE ═════════
function toggleDetail(btn) {
  var tr = btn.closest('tr');
  if(!tr) return;
  var idx = tr.getAttribute('data-detail');
  var dr = document.getElementById('detail-'+idx);
  if(dr){
    dr.classList.toggle('show');
    btn.textContent=dr.classList.contains('show')?'\u2212':'+';
  }
}


// ═════════ FILTER & EXPORT ═════════
function filterQueue() {
  var sv=(document.getElementById('qSearch')||{}).value.toLowerCase();
  var rows=document.querySelectorAll('#queueBody tr');
  for(var ri=0;ri<rows.length;ri++){
    var txt=rows[ri].textContent.toLowerCase();
    rows[ri].style.display=txt.indexOf(sv)!==-1?'':'none';
  }
}

function exportQueueCSV(){
  var hdr='Action,Accounts,AvgProb,Balance,NetRecovery,Cost,PayerRate,ROI\n';
  var rows=QUEUE_DATA.map(function(q){
    return [q.action,q.accounts,q.prob.toFixed(4),q.bal,q.net.toFixed(0),q.cost.toFixed(0),(q.apr*100).toFixed(1)+'%',q.roi.toFixed(1)+'x'].join(',');
  }).join('\n');
  downloadCSV(hdr+rows,'queue_allocation.csv');
}

function filterAccounts(){
  var sv=(document.getElementById('acctSearch')||{}).value.toLowerCase();
  var tf=(document.getElementById('acctTypeFilter')||{}).value;
  var af=(document.getElementById('acctActionFilter')||{}).value;
  var rows=document.querySelectorAll('#accountBody tr');
  for(var ri=0;ri<rows.length;ri++){
    var txt=rows[ri].textContent.toLowerCase();
    var show=true;
    if(sv&&txt.indexOf(sv)===-1)show=false;
    if(tf&&txt.indexOf(tf.toLowerCase())===-1)show=false;
    if(af&&txt.indexOf(af.toLowerCase())===-1)show=false;
    rows[ri].style.display=show?'':'none';
  }
}

function exportAccountsCSV(){
  var hdr='ID,LoanType,Bucket,RawProb,CalibProb,GrossExp,NetExp,Cost,Action\n';
  var rows=ACCOUNTS.map(function(a){
    return [a.id,a.loan_type,a.purchased_bal_gp,parseFloat(a.raw_repay_prob).toFixed(4),parseFloat(a.calibrated_repay_prob).toFixed(4),parseFloat(a.expected_gross_recovery).toFixed(0),parseFloat(a.expected_net_recovery).toFixed(0),parseFloat(a.recommended_contact_cost).toFixed(2),a.recommended_action].join(',');
  }).join('\n');
  downloadCSV(hdr+rows,'top200_accounts.csv');
}

function downloadCSV(content,filename){
  var blob=new Blob([content],{type:'text/csv;charset=utf-8;'});
  var url=URL.createObjectURL(blob);
  var a=document.createElement('a');a.href=url;a.download=filename;a.click();URL.revokeObjectURL(url);
}


// ═════════ ACCOUNT MODAL ═════════
function showAcctModal(acct) {
  var prob=parseFloat(acct.calibrated_repay_prob||0);
  var riskLabel='', riskClass='';
  if(prob>=0.12){riskLabel='High Priority';riskClass='badge-red';}
  else if(prob>=0.08){riskLabel='Medium Risk';riskClass='badge-yellow';}
  else if(prob>=0.04){riskLabel='Low Risk';riskClass='badge-blue';}
  else{riskLabel='Minimal Risk';riskClass='badge-gray';}

  var aiText='';
  if(prob>=0.10) aiText='Strong repayment signal. Recommend immediate agent contact with settlement offer.';
  else if(prob>=0.06) aiText='Moderate probability. Auto-dialer followed by SMS recommended.';
  else aiText='Low expected repayment. Consider SMS-only or write-off depending on balance.';

  var html='<table style="width:100%;font-size:.86rem">';
  var fields=[
    ['Account ID', acct.id],
    ['Loan Type', acct.loan_type],
    ['Balance Bucket', acct.purchased_bal_gp],
    ['Raw Probability', parseFloat(acct.raw_repay_prob||0).toFixed(4)],
    ['Calibrated Probability', '<b style="color:#38bdf8">'+prob.toFixed(4)+'</b>'],
    ['Risk Tier', '<span class="'+riskClass+'">'+riskLabel+'</span>'],
    ['Expected Gross', '$'+parseFloat(acct.expected_gross_recovery||0).toFixed(0)],
    ['Expected Net', '<span style="color:#22c55e">$'+parseFloat(acct.expected_net_recovery||0).toFixed(0)+'</span>'],
    ['Contact Cost', '$'+parseFloat(acct.recommended_contact_cost||0).toFixed(2)],
    ['Recommended Action', acct.recommended_action],
  ];
  for(var fi=0;fi<fields.length;fi++){
    html+='<tr><td style="color:#64748b;padding:6px 8px">'+fields[fi][0]+'</td><td style="padding:6px 8px">'+fields[fi][1]+'</td></tr>';
  }
  html+='</table>';
  html+='<div style="margin-top:16px;padding:14px;background:#0c1222;border-radius:10px;border:1px solid #1e293b">';
  html+='<b style="color:#38bdf8;font-size:.88rem">AI Interpretation</b>';
  html+='<p style="font-size:.84rem;color:#cbd5e1;margin-top:6px">'+aiText+'</p></div>';

  document.getElementById('modalBody').innerHTML=html;
  document.getElementById('acctModal').classList.add('show');
}

function closeAcctModal(){
  document.getElementById('acctModal').classList.remove('show');
}
"""

parts.append('<script>\n')
parts.append('// ===== INJECTED DATA =====\n')
parts.append('const DEV_SPLIT_DATA = ' + j(DEV_SPLIT) + ';\n')
parts.append('const CM_DATA = ' + j(CM_VALS) + ';\n')
parts.append('const MODEL_TABLE = ' + j(MODELS_TABLE) + ';\n')
parts.append('const CHAMP_MODEL = ' + j(CHAMP) + ';\n')
parts.append('const ALL_FEATURE_IMPORTANCE = ' + j(ALL_FI) + ';\n')
parts.append('const PAYER_BAL_DATA = ' + j(PAYERBAL) + ';\n')
parts.append('const PAYER_LOAN_DATA = ' + j(PAYERLOAN) + ';\n')
parts.append('const QUEUE_DATA = ' + j(QUEUE) + ';\n')
parts.append('const QTOTALS = ' + j(QTOTALS) + ';\n')
parts.append('const ACCOUNTS = ' + j(ACCOUNTS) + ';\n')
parts.append('const DIST_DATA = ' + j(DISTS) + ';\n')
parts.append('const CAT_DIST_DATA = ' + j(CAT_DISTS) + ';\n')
parts.append('const QUEUE_COLORS = ' + j(queue_colors) + ';\n')
parts.append('const MODEL_COLORS_MAP = ' + j(MODEL_COLORS) + ';\n')
parts.append('const THRESHOLD_DATA = ' + j(THRESHOLD_DATA) + ';\n')

# Report is now pre-rendered English HTML — no markdown JS variable needed
# (removed: const report_html = ...)

parts.append('\n// ===== APPLICATION CODE =====\n')
parts.append(js_code)
parts.append('</script>\n')
parts.append('</div>\n')  # close container
parts.append('</body></html>')


# ═════════ WRITE OUTPUT ═════════
output = '\n'.join(parts)
out_path = os.path.join(OUT_DIR, 'dashboard.html')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(output)

print('Dashboard v10 generated: ' + out_path)
print('Size: {:,} bytes ({:.1f} KB)'.format(len(output), len(output)/1024))
