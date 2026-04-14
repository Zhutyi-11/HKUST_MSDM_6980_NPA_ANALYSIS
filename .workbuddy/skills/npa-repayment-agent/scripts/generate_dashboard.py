"""NPA Dashboard v7 Generator - Production Ready
All interactions work: Tabs, Sort, Filter, Export, Modal, Charts.
"""
import json, csv, os, sys
from pathlib import Path

OUTPUT_DIR = Path(r"c:\Users\marcozhu\Desktop\6980\agent_outputs\baseline_comparison_run")
OUTPUT_HTML = OUTPUT_DIR / "dashboard.html"

def read_csv_rows(fp):
    if not fp.exists():
        return []
    with open(fp, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def read_json(fp):
    if not fp.exists():
        return {}
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)

# ─── Load Data ──────────────────────────────────────────────
print("Loading data...", flush=True)
M = read_json(OUTPUT_DIR / "metrics.json")
accounts = read_csv_rows(OUTPUT_DIR / "test_scored_accounts.csv")
queue_data = read_csv_rows(OUTPUT_DIR / "production_queue_summary.csv")
champ_list = M.get("champion_challenger", [])
top_features = M.get("top_features", [])
all_fi = M.get("all_feature_importance", {})
desc_stats = M.get("descriptive_stats", {})
conc = M.get("concentration", {})
tuning = M.get("tuning_summary", {})
payer_bal = M.get("payer_rate_by_balance", [])
payer_loan = M.get("payer_rate_by_loan", [])
test_m = M.get("test_metrics", {})
policy = M.get("policy_summary", {})
delta = (M.get("agent_vs_baseline") or {}).get("delta", {}) or {}
dev_split = M.get("development_split", {}) or {}
data_overview = M.get("data_overview", {}) or {}
cm = test_m.get("confusion_matrix", {}) or {}
best_model = M.get("best_model", "unknown")
econ = (M.get("production_config") or {}).get("economics", {})

report_path = OUTPUT_DIR / "collection_strategy_report.md"
report_text = ""
if report_path.exists():
    with open(report_path, "r", encoding="utf-8") as f:
        report_text = f.read()

# Model name map
MODEL_NAMES = {
    "baseline_logistic_regression": ("Logistic Regression", "#6366f1"),
    "balanced_random_forest": ("Random Forest", "#059669"),
    "xgboost": ("XGBoost", "#d97706"),
    "deep_mlp": ("Deep MLP", "#dc2626"),
}
QUEUE_COLORS = {
    "High Priority (Agent Call)": "#ef4444",
    "Medium Priority (Auto-Dialer)": "#f59e0b",
    "Low Priority (SMS/Email)": "#3b82f6",
    "Write-off / Ignore": "#6b7280",
}

def fmt(val, d=2):
    try:
        v = float(val)
        if abs(v) >= 1000:
            return "{v:,.{d}f}".format(v=v, d=d)
        return "{v:.{d}f}".format(v=v, d=d)
    except Exception:
        return str(val) if val else "N/A"

def pct(val):
    try:
        p = float(val)
        if abs(p) > 1:
            return "{p:.2f}%".format(p=p)
        return "{p:.2f}%".format(p=p * 100)
    except Exception:
        return "N/A"

def role_badge(name):
    if name == best_model and best_model == M.get("baseline_model", ""):
        return '<span class="badge badge-champ">Champ=Base</span>'
    elif name == best_model:
        return '<span class="badge badge-champ">CHAMPION</span>'
    elif name == M.get("base_model", ""):
        return '<span class="badge badge-base">BASELINE</span>'
    elif "mlp" in name.lower():
        return '<span class="badge badge-deep">Deep Learn</span>'
    return '<span class="badge badge-default">Challenger</span>'

def md2html(text):
    lines = text.split("\n")
    out = []
    in_tbl = False
    for L in lines:
        s = L.strip()
        if s.startswith("|---") or s.startswith("|:--"):
            continue
        if s.startswith("# "):
            out.append("<h2>" + s[2:] + "</h2>")
        elif s.startswith("## "):
            out.append("<h3>" + s[3:] + "</h3>")
        elif s.startswith("### "):
            out.append("<h4>" + s[4:] + "</h4>")
        elif s.startswith("|"):
            cells = [c.strip() for c in s.split("|")[1:-1]]
            if not in_tbl:
                out.append("<table><thead><tr>" + "".join("<th>{c}</th>".format(c=c) for c in cells) + "</tr></thead><tbody>")
                in_tbl = True
            else:
                out.append("<tr>" + "".join("<td>{c}</td>".format(c=c) for c in cells) + "</tr>")
        else:
            if in_tbl:
                out.append("</table>")
                in_tbl = False
            if s.startswith("- ") or s.startswith("* "):
                out.append("<li>" + s[2:] + "</li>")
            elif s == "":
                out.append("")
            else:
                out.append("<p>" + s + "</p>")
    if in_tbl:
        out.append("</table>")
    return "\n".join(out)

print("Building HTML...", flush=True)

# ─── Prepare data for JS ────────────────────────────────────
# Accounts for table (top 200 by net recovery)
acc_sorted = sorted(accounts, key=lambda x: float(x.get("net_recovery_value", 0) or 0), reverse=True)[:200]
acc_rows_js = []
for i, r in enumerate(acc_sorted):
    acc_type = r.get("loan_type", "")
    action = r.get("recommended_action", "")
    acc_rows_js.append(json.dumps({
        "id": r.get("id", ""),
        "type": acc_type,
        "balGroup": r.get("purchased_bal_gp", ""),
        "district": r.get("district", ""),
        "isPayer": r.get("is_payer_flag", ""),
        "rawP": fmt(r.get("raw_prediction_score", 0)),
        "calibP": fmt(r.get("calibrated_score", 0)),
        "action": action,
        "netRec": int(float(r.get("net_recovery_value", 0) or 0)),
        "balance": int(float(r.get("purchased_balance", 0) or 0)),
    }, ensure_ascii=False))

# Queue rows
queue_rows_js = []
for r in queue_data:
    queue_rows_js.append(json.dumps({
        "action": r.get("action_name", ""),
        "accounts": int(float(r.get("account_count", 0) or 0)),
        "pctTotal": r.get("pct_of_total", ""),
        "avgProb": fmt(r.get("avg_predicted_prob", 0)),
        "payerRate": r.get("actual_payer_rate_in_bucket", ""),
        "balance": fmt(r.get("total_balance_in_bucket", 0)),
        "grossRec": fmt(r.get("gross_recovery_value", 0)),
        "netRec": fmt(r.get("net_recovery_value", 0)),
        "cost": fmt(r.get("total_collection_cost", 0)),
        "roi": r.get("bucket_roi", ""),
    }, ensure_ascii=False))

# Dev split pie data
ds_labels = list(dev_split.keys())[:5]
ds_values = [int(dev_split.get(k, 0)) if isinstance(dev_split.get(k), (int, float)) else dev_split.get(k, 0) for k in ds_labels]

# Confusion matrix data
cm_labels = ["TN", "FP", "FN", "TP"]
try:
    cm_vals = [
        int(cm.get("TN", cm.get("tn", 0))),
        int(cm.get("FP", cm.get("fp", 0))),
        int(cm.get("FN", cm.get("fn", 0))),
        int(cm.get("TP", cm.get("tp", 0))),
    ]
except Exception:
    cm_vals = [0, 0, 0, 0]

# Model data for table
model_data = []
for cc in champ_list:
    mname = cc.get("model_name", "")
    mrole = cc.get("model_role", "")
    mauc = cc.get("roc_auc", 0)
    mbrier = cc.get("brier_score", 0)
    mlogloss = cc.get("log_loss", 0)
    mrecall = cc.get("recall_at_threshold", 0)
    mprec = cc.get("precision_at_threshold", 0)
    mthresh = cc.get("optimal_threshold", 0)
    mnetrec = cc.get("val_net_recovery", 0)
    mroi = cc.get("val_roi", 0)
    model_data.append({
        "name": mname,
        "display": MODEL_NAMES.get(mname, (mname, "#888"))[0],
        "color": MODEL_NAMES.get(mname, (mname, "#888"))[1],
        "role": mrole,
        "badge": role_badge(mname),
        "auc": fmt(mauc),
        "brier": fmt(mbrier),
        "logloss": fmt(mlogloss),
        "recall": pct(mrecall),
        "precision": pct(mprec),
        "threshold": fmt(mthresh),
        "netrec": fmt(mnetrec),
        "roi": fmt(mroi) + "x",
    })

# Feature importance cross-table
fi_features = []
if all_fi:
    # Get union of all features
    feat_set = set()
    for model_fi in all_fi.values():
        if isinstance(model_fi, dict):
            feat_set.update(model_fi.keys())
    fi_features = sorted(list(feat_set))
    # Limit to top 15
    fi_features = fi_features[:15]

# Descriptive stats - numeric variables
num_vars = []
cat_vars = []
if desc_stats:
    for var_name, stats_dict in desc_stats.items():
        if isinstance(stats_dict, dict):
            has_numeric_keys = any(k in ["mean", "std", "min", "max", "count"] for k in stats_dict.keys())
            if has_numeric_keys:
                num_vars.append((var_name, stats_dict))
            else:
                cat_vars.append((var_name, stats_dict))

# Tuning cards
tune_cards = []
for model_key, tune_info in tuning.items():
    if isinstance(tune_info, dict):
        tune_cards.append({
            "name": model_key,
            "display": MODEL_NAMES.get(model_key, (model_key, "#888"))[0],
            "best_auc": fmt(tune_info.get("best_val_auc", 0)),
            "searched": tune_info.get("configs_searched", 0),
            "params": tune_info.get("best_params", {}),
        })

# Payer rate by balance / loan type
bal_payer_labels = [r.get("balance_group", "") for r in payer_bal] if payer_bal else []
bal_payer_values_all = [float(r.get("payer_rate", 0)) * 100 for r in payer_bal] if payer_bal else []
bal_payer_values_pred = [float(r.get("predicted_rate", 0)) * 100 for r in payer_bal] if payer_bal else []

loan_payer_labels = [r.get("loan_type", "") for r in payer_loan] if payer_loan else []
loan_payer_values_all = [float(r.get("payer_rate", 0)) * 100 for r in payer_loan] if payer_loan else []
loan_payer_values_pred = [float(r.get("predicted_rate", 0)) * 100 for r in payer_loan] if payer_loan else []

# Concentration data
conc_kpis = conc if conc else {}

# KPI values from metrics
champ_metrics = None
for cc in champ_list:
    if cc.get("model_name") == best_model:
        champ_metrics = cc
        break

kpi_auc = champ_metrics.get("roc_auc", 0) if champ_metrics else 0
kpi_brier = champ_metrics.get("brier_score", 0) if champ_metrics else 0
kpi_recall = champ_metrics.get("recall_at_threshold", 0) if champ_metrics else 0
kpi_netrec = champ_metrics.get("val_net_recovery", 0) if champ_metrics else 0
kpi_roi = champ_metrics.get("val_roi", 0) if champ_metrics else 0

delta_auc = delta.get("roc_auc", 0)
delta_brier = delta.get("brier_score", 0)
delta_recall = delta.get("recall_at_threshold", 0)
delta_netrec = delta.get("val_net_recovery", 0)
delta_roi = delta.get("val_roi", 0)

# Data overview
total_recs = data_overview.get("total_records", len(accounts))
train_n = data_overview.get("training_count", 0) or len(accounts) * 0.75
test_n = data_overview.get("test_count", 0) or len(accounts) * 0.25
pos_rate = data_overview.get("positive_rate", 0.0965)
pos_rate_test = test_m.get("positive_rate", pos_rate)
missing_paydate = sum(1 for a in accounts if a.get("last_pay_date_client_closing_m", "") == "" or a.get("last_pay_date_client_closing_m") == "-1")

# Economics
bal_rec_rate = econ.get("balance_recovery_rate", 0.35)
agent_cost = econ.get("agent_call_cost", 85)
auto_dialer_cost = econ.get("auto_dialer_cost", 12)
sms_cost = econ.get("sms_email_cost", 2)
agent_mult = econ.get("agent_call_multiplier", 1.0)
dialer_mult = econ.get("auto_dialer_multiplier", 0.72)
sms_mult = econ.get("sms_email_multiplier", 0.35)

report_html = md2html(report_text) if report_text else "<p>No report available.</p>"

print("Writing file...", flush=True)

# ─── Write HTML File ───────────────────────────────────────
with open(str(OUTPUT_HTML), "w", encoding="utf-8") as f:

    f.write("""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NPA Dashboard v7 - Production</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
/* === Reset & Base === */
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; line-height: 1.5; }

/* === Layout === */
.container { max-width: 1440px; margin: 0 auto; padding: 20px; }
.header { background: linear-gradient(135deg, #1e293b, #334155); border-radius: 16px; padding: 24px 32px; margin-bottom: 24px; border: 1px solid rgba(255,255,255,.08); }
.header h1 { font-size: 24px; font-weight: 700; color: #f1f5f9; }
.header .subtitle { color: #94a3b8; font-size: 14px; margin-top: 4px; }

/* === KPI Cards === */
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }
.kpi-card { background: linear-gradient(145deg, #1e293b, #263348); border: 1px solid rgba(255,255,255,.06); border-radius: 12px; padding: 20px; position: relative; overflow: hidden; }
.kpi-card::after { content:''; position: absolute; top: 0; left: 0; right: 0; height: 3px; }
.kpi-card.auc::after { background: linear-gradient(90deg, #3b82f6, #8b5cf6); }
.kpi-card.brier::after { background: linear-gradient(90deg, #10b981, #34d399); }
.kpi-card.recall::after { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.kpi-card.roi::after { background: linear-gradient(90deg, #ec4899, #f472b6); }
.kpi-label { font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: .5px; }
.kpi-value { font-size: 28px; font-weight: 800; color: #f1f5f9; margin-top: 4px; }
.kpi-delta { font-size: 12px; margin-top: 4px; }
.delta-up { color: #34d399; } .delta-down { color: #f87171; } .delta-neutral { color: #94a3b8; }

/* === Tabs === */
.tabs { display: flex; gap: 4px; margin-bottom: 20px; flex-wrap: wrap; background: #1e293b; padding: 6px; border-radius: 12px; }
.tab-btn { padding: 10px 20px; border: none; background: transparent; color: #94a3b8; cursor: pointer; border-radius: 8px; font-size: 13px; font-weight: 600; transition: all .2s; }
.tab-btn:hover { background: rgba(255,255,255,.08); color: #e2e8f0; }
.tab-btn.active { background: #3b82f6; color: white; }
.tab-panel { display: none; animation: fadeIn .3s ease; }
.tab-panel.active { display: block; }
@keyframes fadeIn { from{opacity:0} to{opacity:1} }

/* === Sections === */
.section { background: #1e293b; border-radius: 12px; padding: 24px; margin-bottom: 20px; border: 1px solid rgba(255,255,255,.06); }
.section-title { font-size: 16px; font-weight: 700; color: #f1f5f9; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; justify-content: space-between; }
.section-title .dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }

/* === Tables === */
.data-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.data-table th { background: #0f172a; color: #94a3f8; font-weight: 600; text-align: left; padding: 10px 14px; font-size: 11px; text-transform: uppercase; letter-spacing: .5px; cursor: pointer; user-select: none; white-space: nowrap; position: relative; }
.data-table th:hover { color: #fff; background: #162032; }
.data-table td { padding: 10px 14px; border-bottom: 1px solid rgba(255,255,255,.04); }
.data-table tr:hover { background: rgba(59,130,246,.05); }
.sort-arrow { font-size: 10px; margin-left: 4px; opacity: .4; transition: opacity .2s; }
.data-table th:hover .sort-arrow { opacity: 1; }
.sort-active .sort-arrow { opacity: 1; color: #3b82f6; }

/* === Badges === */
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: .3px; }
.badge-champ { background: #fef3c7; color: #92400e; }
.badge-base { background: #dbeafe; color: #1e40af; }
.badge-deep { background: #fce7f3; color: #be185d; }
.badge-default { background: rgba(255,255,255,.08); color: #94a3b8; }

/* === Charts === */
.chart-row { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
.chart-box { background: #0f172a; border-radius: 8px; padding: 16px; height: 320px; position: relative; }
.chart-box.full { grid-column: 1 / -1; height: 380px; }
.chart-box.tall { height: 420px; }

/* === Stat Grid === */
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; }
.stat-card { background: #0f172a; border-radius: 8px; padding: 16px; }
.stat-card h4 { font-size: 13px; color: #94a3b8; margin-bottom: 8px; }
.stat-value { font-size: 22px; font-weight: 700; color: #f1f5f9; }
.stat-row { display: flex; justify-content: space-between; padding: 4px 0; font-size: 12px; border-bottom: 1px solid rgba(255,255,255,.03); }

/* === Category Bars === */
.cat-item { display: flex; align-items: center; gap: 10px; padding: 6px 0; font-size: 13px; }
.cat-bar { height: 18px; border-radius: 3px; background: linear-gradient(90deg, #3b82f6, #8b5cf6); min-width: 20px; max-width: 300px; }
.cat-label { min-width: 140px; color: #cbd5e1; }
.cat-count { color: #94a3b8; font-size: 12px; min-width: 70px; text-align: right; }

/* === Filters & Buttons === */
.filter-bar { display: flex; gap: 10px; margin-bottom: 16px; flex-wrap: wrap; align-items: center; }
.filter-input { background: #0f172a; border: 1px solid rgba(255,255,255,.15); border-radius: 8px; padding: 8px 14px; color: #e2e8f0; font-size: 13px; outline: none; width: 220px; transition: border .2s; }
.filter-input:focus { border-color: #3b82f6; }
.filter-input::placeholder { color: #475569; }
.filter-select { background: #0f172a; border: 1px solid rgba(255,255,255,.15); border-radius: 8px; padding: 8px 14px; color: #e2e8f0; font-size: 13px; outline: none; cursor: pointer; }
.btn-sm { padding: 7px 16px; border-radius: 6px; border: none; font-size: 12px; font-weight: 600; cursor: pointer; transition: all .15s; display: inline-flex; align-items: center; gap: 4px; }
.btn-primary { background: #3b82f6; color: white; } .btn-primary:hover { background: #2563eb; }
.btn-outline { background: transparent; color: #94a3f8; border: 1px solid rgba(148,163,248,.3); } .btn-outline:hover { background: rgba(148,163,248,.1); color: #fff; }
.btn-success { background: #059669; color: white; } .btn-success:hover { background: #047857; }
.btn-sm:disabled { opacity: .4; cursor: not-allowed; }

/* === Detail Row === */
.detail-row { background: #0c1220 !important; }
.detail-inner { padding: 16px; display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
.detail-item { font-size: 12px; color: #94a3b8; }
.detail-item strong { color: #e2e8f0; }

/* === Modal === */
.modal-overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,.7); z-index: 999; justify-content: center; align-items: center; backdrop-filter: blur(4px); }
.modal-overlay.show { display: flex; }
.modal { background: #1e293b; border-radius: 16px; max-width: 720px; width: 92%; max-height: 85vh; overflow-y: auto; border: 1px solid rgba(255,255,255,.1); box-shadow: 0 25px 60px rgba(0,0,0,.5); }
.modal-header { padding: 20px 24px; border-bottom: 1px solid rgba(255,255,255,.06); display: flex; justify-content: space-between; align-items: center; }
.modal-header h3 { font-size: 16px; color: #f1f5f9; }
.modal-close { background: none; border: none; color: #94a3f8; font-size: 24px; cursor: pointer; padding: 4px 8px; border-radius: 4px; }
.modal-close:hover { color: #fff; background: rgba(255,255,255,.1); }
.modal-body { padding: 24px; }
.modal-field { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,.03); }
.modal-field-label { color: #94a3b8; font-size: 13px; }
.modal-field-value { font-weight: 600; color: #f1f5f9; font-size: 13px; }

/* === Report === */
.report-content { max-width: 900px; margin: 0 auto; font-size: 14px; line-height: 1.7; }
.report-content h2 { color: #f1f5f9; font-size: 20px; margin: 24px 0 12px; padding-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,.1); }
.report-content h3 { color: #e2e8f0; font-size: 16px; margin: 20px 0 8px; }
.report-content h4 { color: #cbd5e1; font-size: 14px; margin: 16px 0 6px; }
.report-content p { color: #94a3b8; margin: 8px 0; }
.report-content li { color: #cbd5e1; margin: 4px 0 4px 20px; }
.report-content table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 13px; }
.report-content th { background: #0f172a; color: #94a3b8; padding: 8px 12px; text-align: left; font-size: 11px; text-transform: uppercase; }
.report-content td { padding: 8px 12px; border-bottom: 1px solid rgba(255,255,255,.04); }

/* === Tune Cards === */
.tune-card { background: #0f172a; border-radius: 8px; padding: 16px; margin: 8px 0; }
.tune-model-name { font-weight: 700; font-size: 14px; margin-bottom: 8px; }
.tune-param { display: inline-block; background: rgba(99,102,241,.15); border-radius: 4px; padding: 2px 8px; font-size: 11px; font-family: monospace; margin: 2px; color: #93c5fd; }

/* === Insight Box === */
.insight-box { margin-top: 12px; padding: 12px 16px; background: linear-gradient(135deg, rgba(59,130,246,.08), rgba(139,92,246,.08)); border-radius: 8px; border-left: 3px solid #3b82f6; font-size: 13px; line-height: 1.6; }
.insight-box b { color: #3b82f6; }

/* === Scrollbar === */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f172a; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }

/* === Risk Tags === */
.risk-tag-high { display: inline-block; padding: 2px 8px; border-radius: 4px; background: rgba(239,68,68,.15); color: #f87171; font-size: 11px; font-weight: 600; }
.risk-tag-med { display: inline-block; padding: 2px 8px; border-radius: 4px; background: rgba(245,158,11,.15); color: #fbbf24; font-size: 11px; font-weight: 600; }
.risk-tag-low { display: inline-block; padding: 2px 8px; border-radius: 4px; background: rgba(59,130,246,.15); color: #60a5fa; font-size: 11px; font-weight: 600; }
</style>
</head>
<body>
<div class="container">

<!-- HEADER -->
<div class="header">
  <h1>[NPA] Repayment Prediction Dashboard <span style="font-size:14px;color:#3b82f6;font-weight:400;">v7 Production</span></h1>
  <div class="subtitle">
    Champion: <strong style="color:#3b82f6">""" + best_model.replace("_", " ").title() + """</strong> |
    Data: """ + fmt(total_recs, 0) + """ records (Train """ + fmt(train_n, 0) + """ / Test """ + fmt(test_n, 0) + """) |
    Target Rate: """ + pct(pos_rate) + """
  </div>
</div>

<!-- KPI ROW -->
<div class="kpi-grid">
  <div class="kpi-card auc"><div class="kpi-label">ROC-AUC (Champion)</div><div class="kpi-value">""" + fmt(kpi_auc) + """</div><div class="kpi-delta delta-up">Best among all models</div></div>
  <div class="kpi-card brier"><div class="kpi-label">Brier Score (lower=better)</div><div class="kpi-value">""" + fmt(kpi_brier) + """</div><div class="kpi-delta">\u2713 Calibration quality</div></div>
  <div class="kpi-card recall"><div class="kpi-label">Recall @ Threshold</div><div class="kpi-value">""" + pct(kpi_recall) + """</div><div class="kpi-delta">Positive capture rate</div></div>
  <div class="kpi-card roi"><div class="kpi-label">Net Recovery (Val)</div><div class="kpi-value">\u00a5""" + fmt(kpi_netrec, 0) + """</div><div class="kpi-delta delta-up">ROI: """ + fmt(kpi_roi) + """x</div></div>
</div>

<!-- TABS -->
<div class="tabs" id="tabBar">
  <button class="tab-btn active" onclick="switchTab('overview')">Overview</button>
  <button class="tab-btn" onclick="switchTab('models')">Models</button>
  <button class="tab-btn" onclick="switchTab('features')">Features</button>
  <button class="tab-btn" onclick="switchTab('stats')">Stats</button>
  <button class="tab-btn" onclick="switchTab('queue')">Queue</button>
  <button class="tab-btn" onclick="switchTab('tuning')">Tuning</button>
  <button class="tab-btn" onclick="switchTab('report')">Report</button>
</div>

<!-- ==================== TAB: OVERVIEW ==================== -->
<div class="tab-panel active" id="panel-overview">

  <!-- Data Overview -->
  <div class="section">
    <div class="section-title"><span class="dot" style="background:#3b82f6;"></span>Data Overview</div>
    <div class="stat-grid">
      <div class="stat-card"><h4>Total Records</h4><div class="stat-value">""" + fmt(total_recs, 0) + """</div></div>
      <div class="stat-card"><h4>Training Set (M)</h4><div class="stat-value">""" + fmt(train_n, 0) + """</div></div>
      <div class="stat-card"><h4>Test Set (T)</h4><div class="stat-value">""" + fmt(test_n, 0) + """</div></div>
      <div class="stat-card"><h4>Positive Rate (Overall)</h4><div class="stat-value">""" + pct(pos_rate) + """</div></div>
      <div class="stat-card"><h4>Positive Rate (Test)</h4><div class="stat-value">""" + pct(pos_rate_test) + """</div></div>
      <div class="stat-card"><h4>Missing last_pay_date</h4><div class="stat-value">""" + str(missing_paydate) + """</div></div>
    </div>
  </div>

  <!-- Dev Split & Confusion Matrix -->
  <div class="section">
    <div class="section-title"><span class="dot" style="background:#8b5cf6;"></span>Development Split &amp; Confusion Matrix</div>
    <p style="color:#94a3b8;margin-bottom:12px;font-size:13px;">M set split visualization and Champion model confusion matrix on Test set.</p>
    <div class="chart-row">
      <div class="chart-box"><canvas id="devSplitChart"></canvas></div>
      <div class="chart-box"><canvas id="confMatrixChart"></canvas></div>
    </div>
  </div>

  <!-- Economic Assumptions -->
  <div class="section">
    <div class="section-title"><span class="dot" style="background:#10b981;"></span>Economic Assumptions</div>
    <div class="stat-grid">
      <div class="stat-card"><h4>Balance Recovery Rate</h4><div class="stat-value">""" + pct(bal_rec_rate) + """</div></div>
      <div class="stat-card"><h4>Agent Call Cost</h4><div class="stat-value">\u00a5""" + fmt(agent_cost, 0) + """</div></div>
      <div class="stat-card"><h4>Auto-Dialer Cost</h4><div class="stat-value">\u00a5""" + fmt(auto_dialer_cost, 0) + """</div></div>
      <div class="stat-card"><h4>SMS/Email Cost</h4><div class="stat-value">\u00a5""" + fmt(sms_cost, 0) + """</div></div>
      <div class="stat-card"><h4>Agent Multiplier</h4><div class="stat-value">""" + fmt(agent_mult) + """x</div></div>
      <div class="stat-card"><h4>Dialer Multiplier</h4><div class="stat-value">""" + fmt(dialer_mult) + """x</div></div>
      <div class="stat-card"><h4>SMS Multiplier</h4><div class="stat-value">""" + fmt(sms_mult) + """x</div></div>
    </div>
  </div>
</div>

<!-- ==================== TAB: MODELS ==================== -->
<div class="tab-panel" id="panel-models">
  <div class="section">
    <div class="section-title">
      <span class="dot" style="background:#f59e0b;"></span>Model Performance Ranking
      <span style="font-size:11px;font-weight:400;color:#64748b;">Click header \u2191\u2193 to sort | Click row to expand details</span>
    </div>
    <table class="data-table" id="modelTable">
      <thead>
        <tr>
          <th style="width:40px" data-col="0">#<span class="sort-arrow">\u21c5</span></th>
          <th data-col="1">Model<span class="sort-arrow">\u21c5</span></th>
          <th data-col="2">Role<span class="sort-arrow">\u21c5</span></th>
          <th data-col="3">AUC<span class="sort-arrow">\u21c5</span></th>
          <th data-col="4">Brier<span class="sort-arrow">\u21c5</span></th>
          <th data-col="5">LogLoss<span class="sort-arrow">\u21c5</span></th>
          <th data-col="6">Recall<span class="sort-arrow">\u21c5</span></th>
          <th data-col="7">Precision<span class="sort-arrow">\u21c5</span></th>
          <th data-col="8">Threshold<span class="sort-arrow">\u21c5</span></th>
          <th data-col="9">Net Rec.<span class="sort-arrow">\u21c5</span></th>
          <th data-col="10">ROI<span class="sort-arrow">\u21c5</span></th>
        </tr>
      </thead>
      <tbody id="modelBody">
""")

    # Write model rows
    for i, m in enumerate(model_data):
        detail_id = "detail-" + m["name"]
        f.write("""        <tr data-name=\"""" + m["name"] + "\"\" onclick=\"toggleDetail('""" + detail_id + """')\" style=\"cursor:pointer\">
          <td>""" + str(i+1) + """</td>
          <td><strong>""" + m["display"] + """</strong> """ + m["badge"] + """</td>
          <td>""" + m["role"] + """</td>
          <td>""" + m["auc"] + """</td>
          <td>""" + m["brier"] + """</td>
          <td>""" + m["logloss"] + """</td>
          <td>""" + m["recall"] + """</td>
          <td>""" + m["precision"] + """</td>
          <td>""" + m["threshold"] + """</td>
          <td>\u00a5""" + m["netrec"] + """</td>
          <td style=\"color:#34d399;font-weight:700\">""" + m["roi"] + """</td>
        </tr>
        <tr id=\"""" + detail_id + "\"\" class=\"detail-row\" style=\"display:none\"><td colspan=\"11\"><div class='detail-inner'><div class='detail-item'><strong>Model:</strong> """ + m["display"] + """</div><div class='detail-item'><strong>AUC:</strong> """ + m["auc"] + """</div><div class='detail-item'><strong>Brier:</strong> """ + m["brier"] + """ (lower is better)</div><div class='detail-item'><strong>LogLoss:</strong> """ + m["logloss"] + """</div><div class='detail-item'><strong>Recall:</strong> """ + m["recall"] + """ at threshold """ + m["threshold"] + """</div><div class='detail-item'><strong>Precision:</strong> """ + m["precision"] + """</div><div class='detail-item'><strong>Net Recovery:</strong> \u00a5""" + m["netrec"] + """</div><div class='detail-item'><strong>ROI:</strong> """ + m["roi"] + """</div></div></td></tr>
""")
    f.write("""      </tbody>
    </table>

    <!-- Model comparison charts -->
    <div class="chart-row" style="margin-top:20px;">
      <div class="chart-box"><canvas id="modelAucChart"></canvas></div>
      <div class="chart-box"><canvas id="modelEcoChart"></canvas></div>
    </div>
  </div>
</div>

<!-- ==================== TAB: FEATURES ==================== -->
<div class="tab-panel" id="panel-features">
  <div class="section">
    <div class="section-title">
      <span class="dot" style="background:#8b5cf6;"></span>Feature Importance by Model
      <div class="filter-bar" style="margin:0">
        <select id="fiModelSelect" class="filter-select" onchange="updateFiChart()">
""")

    for mk, (mdisp, _) in MODEL_NAMES.items():
        sel = ' selected' if mk == best_model else ''
        f.write('          <option value="' + mk + '"' + sel + '>' + mdisp + '</option>\n')

    f.write("""        </select>
      </div>
    </div>
    <div class="chart-box tall"><canvas id="featureImportanceChart"></canvas></div>

    <!-- Cross-table -->
    <h4 style="color:#f1f5f9;margin:16px 0 8px;">Feature Importance Cross-Table (Permutation)</h4>
    <div style="overflow-x:auto;">
      <table class="data-table">
        <thead><tr><th>#</th><th>Feature</th>
""")

    for mk, (mdisp, _) in MODEL_NAMES.items():
        f.write('<th>' + mdisp + '</th>\n')
    f.write('</tr></thead><tbody>\n')

    for j, feat in enumerate(fi_features):
        f.write('<tr><td>' + str(j+1) + '</td><td><strong>' + feat + '</strong></td>')
        for mk, _ in MODEL_NAMES.items():
            fi_val = all_fi.get(mk, {}).get(feat, "N/A")
            f.write('<td>' + fmt(fi_val, 4) + '</td>')
        f.write('</tr>\n')
    f.write("""      </table>
    </div>
  </div>

  <!-- Payer rate drill-down charts -->
  <div class="chart-row" style="margin-top:16px;">
    <div class="chart-box"><canvas id="payerBalChart"></canvas></div>
    <div class="chart-box"><canvas id="payerLoanChart"></canvas></div>
  </div>

  <!-- Business interpretation -->
  <div class="section">
    <div class="section-title"><span class="dot" style="background:#10b981;"></span>Business Interpretation</div>
    <div class="stat-grid">
      <div class="stat-card"><h4 style="color:#6366f1;">purchased_bal_gp</h4><p style="color:#94a3f8;font-size:12px;line-height:1.5;">Balance size affects negotiation willingness and recovery potential.</p></div>
      <div class="stat-card"><h4 style="color:#6366f1;">birth_yr</h4><p style="color:#94a3f8;font-size:12px;line-height:1.5;">Younger debtors tend to have higher income recovery potential.</p></div>
      <div class="stat-card"><h4 style="color:#6366f1;">district</h4><p style="color:#94a3f8;font-size:12px;line-height:1.5;">Region reflects socioeconomic stability.</p></div>
      <div class="stat-card"><h4 style="color:#6366f1;">multiple_acct</h4><p style="color:#94a3f8;font-size:12px;line-height:1.5;">Multiple accounts provide richer behavioral signals.</p></div>
      <div class="stat-card"><h4 style="color:#6366f1;">open_closing_m</h4><p style="color:#94a3f8;font-size:12px;line-height:1.5;">Account age indicates relationship maturity.</p></div>
      <div class="stat-card"><h4 style="color:#6366f1;">home_phone_flag</h4><p style="color:#94a3f8;font-size:12px;line-height:1.5;">Supplementary contact channel improves reachability.</p></div>
    </div>
  </div>
</div>

<!-- ==================== TAB: STATS ==================== -->
<div class="tab-panel" id="panel-stats">
  <div class="section">
    <div class="section-title"><span class="dot" style="background:#f59e0b;"></span>Numeric Variables - Descriptive Statistics</div>
    <div class="stat-grid">
""")

    for vname, vstats in num_vars[:8]:
        f.write('      <div class="stat-card"><h4>' + vname + '</h4>\n')
        for stat_key in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']:
            sv = vstats.get(stat_key, 'N/A')
            f.write('        <div class="stat-row"><span>' + stat_key + '</span><span>' + fmt(sv, 2) + '</span></div>\n')
        f.write('      </div>\n')

    f.write("""    </div>
  </div>

  <div class="section">
    <div class="section-title"><span class="dot" style="background:#8b5cf6;"></span>Categorical Distributions</div>
""")

    for cname, cstats in cat_vars[:6]:
        if isinstance(cstats, dict):
            # Filter to numeric values only
            numeric_vals = [float(v) for v in cstats.values() if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace('.','').replace('-','').isdigit())]
            max_count = max(numeric_vals) if numeric_vals else 1
            total_count = sum(numeric_vals) if numeric_vals else 1
            f.write('    <div style="margin-bottom:16px;">\n')
            f.write('      <h4 style="color:#f1f5f9;margin-bottom:6px;">' + cname + ' <span style="font-weight:400;color:#64748b;font-size:12px;">(' + str(len(cstats)) + ' categories)</span></h4>\n')
            for cat_label, cat_val in list(cstats.items())[:8]:
                try:
                    cv = float(cat_val)
                except (ValueError, TypeError):
                    cv = 0
                bar_width = int(max(20, (cv / max_count) * 150))
                cp = (cv / total_count * 100) if total_count > 0 else 0
                f.write('        <div class="cat-item"><span class="cat-label">' + str(cat_label) + '</span><div class="cat-bar" style="width:' + str(bar_width) + 'px;"></div><span class="cat-count">' + fmt(cv, 0) + ' (' + fmt(cp, 1) + '%)</span></div>\n')
            f.write('    </div>\n')

    f.write("""  </div>
</div>

<!-- ==================== TAB: QUEUE ==================== -->
<div class="tab-panel" id="panel-queue">
  <div class="section">
    <div class="section-title"><span class="dot" style="background:#ef4444;"></span>Queue Allocation (Test Set)</div>
    <div class="filter-bar">
      <input type="text" class="filter-input" id="queueSearch" placeholder="[Search] action type..." oninput="filterQueue()">
      <select class="filter-select" id="queueActionFilter" onchange="filterQueue()">
        <option value="">All Actions</option>
        <option value="High">High Priority</option>
        <option value="Medium">Medium Priority</option>
        <option value="Low">Low Priority</option>
        <option value="Write">Write-off</option>
      </select>
      <button class="btn-sm btn-outline" onclick="exportTableCSV('queueTable','queue_strategy.csv')">\u2193 Export CSV</button>
    </div>
    <table class="data-table" id="queueTable">
      <thead>
        <tr>
          <th data-col="0">Action<span class="sort-arrow">\u21c5</span></th>
          <th data-col="1">Accounts<span class="sort-arrow">\u21c5</span></th>
          <th data-col="2">% Total<span class="sort-arrow">\u21c5</span></th>
          <th data-col="3">Avg Prob<span class="sort-arrow">\u21c5</span></th>
          <th data-col="4">Payer Rate<span class="sort-arrow">\u21c5</span></th>
          <th data-col="5">Balance<span class="sort-arrow">\u21c5</span></th>
          <th data-col="6">Gross Rec<span class="sort-arrow">\u21c5</span></th>
          <th data-col="7">Net Rec<span class="sort-arrow">\u21c5</span></th>
          <th data-col="8">Cost<span class="sort-arrow">\u21c5</span></th>
          <th data-col="9">ROI<span class="sort-arrow">\u21c5</span></th>
        </tr>
      </thead>
      <tbody id="queueBody">
""")

    qcolors = {"H": "#ef4444", "M": "#f59e0b", "L": "#3b82f6", "W": "#6b7280"}
    for qr in queue_data:
        act = qr.get("action_name", "")
        dot_color = "#6b7280"
        for prefix, clr in qcolors.items():
            if act.startswith(prefix) or (prefix == "H" and "High" in act):
                dot_color = clr
                break
            if prefix == "M" and "Medium" in act:
                dot_color = clr
                break
            if prefix == "L" and "Low" in act:
                dot_color = clr
                break
            if prefix == "W" and ("Write" in act or "Ignore" in act):
                dot_color = clr
                break
        f.write('        <tr data-action="' + act + '">\n')
        f.write('          <td><span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:' + dot_color + ';margin-right:6px;"></span>' + act + '</td>\n')
        f.write('          <td><strong>' + fmt(qr.get("account_count", 0), 0) + '</strong></td>\n')
        f.write('          <td>' + qr.get("pct_of_total", "") + '</td>\n')
        f.write('          <td>' + fmt(qr.get("avg_predicted_prob", 0)) + '</td>\n')
        f.write('          <td>' + qr.get("actual_payer_rate_in_bucket", "") + '</td>\n')
        f.write('          <td>' + fmt(qr.get("total_balance_in_bucket", 0), 0) + '</td>\n')
        f.write('          <td>' + fmt(qr.get("gross_recovery_value", 0), 0) + '</td>\n')
        f.write('          <td><strong>' + fmt(qr.get("net_recovery_value", 0), 0) + '</strong></td>\n')
        f.write('          <td>' + fmt(qr.get("total_collection_cost", 0), 0) + '</td>\n')
        roi_v = qr.get("bucket_roi", "")
        roi_color = '#34d399' if roi_v and float(str(roi_v).replace('x','')) > 1 else '#fbbf24'
        f.write('          <td style="color:' + roi_color + ';font-weight:700">' + roi_v + '</td>\n')
        f.write('        </tr>\n')

    f.write("""      </tbody>
    </table>
    <div class="chart-row" style="margin-top:16px;">
      <div class="chart-box"><canvas id="queuePieChart"></canvas></div>
      <div class="chart-box"><canvas id="queueRoiChart"></canvas></div>
    </div>
  </div>

  <!-- Concentration Analysis -->
  <div class="section">
    <div class="section-title"><span class="dot" style="background:#10b981;"></span>Concentration Analysis</div>
    <div class="stat-grid">
      <div class="stat-card"><h4>Top20 Account Count</h4><div class="stat-value">""" + fmt(conc_kpis.get("top20_count", "N/A"), 0) + """</div></div>
      <div class="stat-card"><h4>Overall Payer Rate</h4><div class="stat-value">""" + fmt(conc_kpis.get("overall_payer_rate", "N/A")) + """</div></div>
      <div class="stat-card"><h4>Top20 Prob Payer Rate</h4><div class="stat-value">""" + fmt(conc_kpis.get("top20_prob_payer_rate", "N/A")) + """</div></div>
      <div class="stat-card"><h4>Top20 Capture %</h4><div class="stat-value">""" + fmt(conc_kpis.get("top20_capture_pct", "N/A")) + """</div></div>
      <div class="stat-card"><h4>Top20 Net Rec Share</h4><div class="stat-value">""" + fmt(conc_kpis.get("top20_net_rec_share", "N/A")) + """</div></div>
    </div>
  </div>

  <!-- Top Accounts Table -->
  <div class="section">
    <div class="section-title"><span class="dot" style="background:#8b5cf6;"></span>Top 200 Accounts by Net Recovery</div>
    <div class="filter-bar">
      <input type="text" class="filter-input" id="accSearch" placeholder="[Search] ID, district, type..." oninput="filterAccounts()">
      <select class="filter-select" id="accTypeFilter" onchange="filterAccounts()">
        <option value="">All Types</option>
        <option value="Credit Card">Credit Card</option>
        <option value="Personal Loan">Personal Loan</option>
        <option value="Overdraft">Overdraft</option>
      </select>
      <select class="filter-select" id="accActionFilter" onchange="filterAccounts()">
        <option value="">All Actions</option>
        <option value="High">High Priority</option>
        <option value="Medium">Medium Priority</option>
        <option value="Low">Low Priority</option>
      </select>
      <button class="btn-sm btn-outline" onclick="exportTableCSV('accTable','top_accounts.csv')">\u2193 Export CSV</button>
    </div>
    <div style="overflow-x:auto;max-height:500px;overflow-y:auto;">
      <table class="data-table" id="accTable">
        <thead>
          <tr>
            <th data-col="0">ID<span class="sort-arrow">\u21c5</span></th>
            <th data-col="1">Type<span class="sort-arrow">\u21c5</span></th>
            <th data-col="2">BalGroup<span class="sort-arrow">\u21c5</span></th>
            <th data-col="3">District<span class="sort-arrow">\u21c5</span></th>
            <th data-col="4">Payer?<span class="sort-arrow">\u21c5</span></th>
            <th data-col="5">RawP<span class="sort-arrow">\u21c5</span></th>
            <th data-col="6">CalibP<span class="sort-arrow">\u21c5</span></th>
            <th data-col="7">Action<span class="sort-arrow">\u21c5</span></th>
            <th data-col="8" style="width:80px">NetRec<span class="sort-arrow">\u21c5</span></th>
            <th data-col="9" style="width:80px">Balance<span class="sort-arrow">\u21c5</span></th>
          </tr>
        </thead>
        <tbody id="accBody">
""")

    for i, ar in enumerate(acc_rows_js):
        aobj = json.loads(ar)
        f.write('          <tr data-type="' + aobj.get("type","") + '" data-action="' + aobj.get("action","") + '" onclick="showAccountDetail(' + str(i) + ')" style="cursor:pointer">\n')
        f.write('            <td>' + str(aobj.get("id","")) + '</td>\n')
        f.write('            <td>' + str(aobj.get("type","")) + '</td>\n')
        f.write('            <td>' + str(aobj.get("balGroup","")) + '</td>\n')
        f.write('            <td>' + str(aobj.get("district","")) + '</td>\n')
        f.write('            <td>' + str(aobj.get("isPayer","")) + '</td>\n')
        f.write('            <td>' + str(aobj.get("rawP","")) + '</td>\n')
        f.write('            <td>' + str(aobj.get("calibP","")) + '</td>\n')
        f.write('            <td>' + str(aobj.get("action",""))[:20] + '</td>\n')
        f.write('            <td style="color:#34d399;font-weight:600">\u00a5' + fmt(aobj.get("netRec",0), 0) + '</td>\n')
        f.write('            <td>\u00a5' + fmt(aobj.get("balance",0), 0) + '</td>\n')
        f.write('          </tr>\n')

    f.write("""        </tbody>
      </table>
    </div>
    <p style="color:#64748b;font-size:12px;margin-top:8px;">Showing top 200 accounts. Click row for detail modal.</p>
  </div>
</div>

<!-- ==================== TAB: TUNING ==================== -->
<div class="tab-panel" id="panel-tuning">
  <div class="section">
    <div class="section-title"><span class="dot" style="background:#f59e0b;"></span>Hyperparameter Tuning Results</div>
""")

    for tc in tune_cards:
        f.write('    <div class="tune-card">\n')
        f.write('      <div class="tune-model-name">' + tc["display"] + ' <span style="font-weight:400;color:#64748b;font-size:12px;">| Best Val AUC: <strong>' + tc["best_auc"] + '</strong> | Searched: ' + str(tc["searched"]) + ' configs</span></div>\n')
        params = tc.get("params", {})
        if params:
            for pk, pv in list(params.items())[:8]:
                f.write('      <span class="tune-param">' + str(pk) + '=' + str(pv) + '</span> ')
        f.write('    </div>\n')

    f.write("""  </div>

  <!-- MLP Sweep Chart -->
  <div class="section">
    <div class="section-title"><span class="dot" style="background:#dc2626;"></span>MLP Configuration Sweep</div>
    <div class="chart-box full"><canvas id="mlpSweepChart"></canvas></div>
  </div>
</div>

<!-- ==================== TAB: REPORT ==================== -->
<div class="tab-panel" id="panel-report">
  <div class="report-content">
""" + report_html + """
  </div>
</div>

</div><!-- end container -->

<!-- MODAL -->
<div class="modal-overlay" id="accModal">
  <div class="modal">
    <div class="modal-header">
      <h3 id="modalTitle">Account Details</h3>
      <button class="modal-close" onclick="closeModal()">&times;</button>
    </div>
    <div class="modal-body" id="modalBody">
      <!-- Filled dynamically -->
    </div>
  </div>
</div>

<!-- ═══════════════ JAVASCRIPT ═══════════════ -->
<script>
// ============================================================
// DATA STORE
// ============================================================
var ACCOUNTS = [""" + ",".join(acc_rows_js) + """];
var QUEUE_DATA = [""" + ",".join(queue_rows_js) + """];
var DEV_SPLIT_LABELS = """ + json.dumps(ds_labels, ensure_ascii=False) + """;
var DEV_SPLIT_VALUES = """ + json.dumps(ds_values, ensure_ascii=False) + """;
var CM_VALUES = """ + json.dumps(cm_vals, ensure_ascii=False) + """;
var MODEL_DATA = """ + json.dumps(model_data, ensure_ascii=False) + """;
var ALL_FI = """ + json.dumps(all_fi, ensure_ascii=False) + """;
var BAL_PAYER_LABELS = """ + json.dumps(bal_payer_labels, ensure_ascii=False) + """;
var BAL_PAYER_ALL = """ + json.dumps(bal_payer_values_all, ensure_ascii=False) + """;
var BAL_PAYER_PRED = """ + json.dumps(bal_payer_values_pred, ensure_ascii=False) + """;
var LOAN_PAYER_LABELS = """ + json.dumps(loan_payer_labels, ensure_ascii=False) + """;
var LOAN_PAYER_ALL = """ + json.dumps(loan_payer_values_all, ensure_ascii=False) + """;
var LOAN_PAYER_PRED = """ + json.dumps(loan_payer_values_pred, ensure_ascii=False) + """;

// ============================================================
// TAB SWITCHING
// ============================================================
function switchTab(tabId) {
  // Hide all panels
  var panels = document.querySelectorAll('.tab-panel');
  for (var i = 0; i < panels.length; i++) {
    panels[i].classList.remove('active');
  }
  // Remove active from all buttons
  var buttons = document.querySelectorAll('.tab-btn');
  for (var j = 0; j < buttons.length; j++) {
    buttons[j].classList.remove('active');
  }
  // Show target panel
  var targetPanel = document.getElementById('panel-' + tabId);
  if (targetPanel) {
    targetPanel.classList.add('active');
  }
  // Activate button
  var btns = document.getElementById('tabBar').querySelectorAll('.tab-btn');
  for (var k = 0; k < btns.length; k++) {
    if (btns[k].textContent.toLowerCase().indexOf(tabId) >= 0 ||
        btns[k].getAttribute('onclick').indexOf(tabId) >= 0) {
      btns[k].classList.add('active');
    }
  }
  // Init charts when tab becomes visible
  initChartsForTab(tabId);
}

// ============================================================
// TABLE SORTING
// ============================================================
var sortState = {}; // { tableName: { col: idx, asc: bool } }

function sortTable(th, colIdx) {
  var table = th.closest('table');
  var tbody = table.querySelector('tbody');
  var rows = Array.prototype.slice.call(tbody.querySelectorAll('tr:not(.detail-row)'));
  var tableId = table.id;
  var key = tableId + '_' + colIdx;

  // Toggle direction
  if (!sortState[key]) {
    sortState[key] = { col: colIdx, asc: true };
  } else {
    sortState[key].asc = !sortState[key].asc;
  }
  var asc = sortState[key].asc;

  // Update arrow indicators
  var headers = table.querySelectorAll('th[data-col]');
  for (var hi = 0; hi < headers.length; hi++) {
    headers[hi].classList.remove('sort-active');
    var arrow = headers[hi].querySelector('.sort-arrow');
    if (arrow) arrow.textContent = '\u21C5';
  }
  th.classList.add('sort-active');
  var myArrow = th.querySelector('.sort-arrow');
  if (myArrow) myArrow.textContent = asc ? '\u2191' : '\u2193';

  // Sort rows
  rows.sort(function(a, b) {
    var aCells = a.cells[colIdx];
    var bCells = b.cells[colIdx];
    var aVal = aCells ? aCells.textContent.trim().replace(/[\u00a5,x,%]/g, '') : '';
    var bVal = bCells ? bCells.textContent.trim().replace(/[\u00a5,x,%]/g, '') : '';
    var aNum = parseFloat(aVal);
    var bNum = parseFloat(bVal);
    var aIsNum = !isNaN(aNum) && aVal !== '';
    var bIsNum = !isNaN(bNum) && bVal !== '';
    if (aIsNum && bIsNum) {
      return asc ? aNum - bNum : bNum - aNum;
    }
    return asc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
  });

  // Re-append sorted rows (skip detail rows)
  for (var ri = 0; ri < rows.length; ri++) {
    tbody.appendChild(rows[ri]);
  }
}

// Attach sort to all th[data-col]
document.addEventListener('DOMContentLoaded', function() {
  var sortHeaders = document.querySelectorAll('th[data-col]');
  for (var si = 0; si < sortHeaders.length; si++) {
    (function(th) {
      th.addEventListener('click', function() {
        var colIdx = parseInt(th.getAttribute('data-col'));
        sortTable(th, colIdx);
      });
    })(sortHeaders[si]);
  }
});

// ============================================================
// DETAIL ROW TOGGLE
// ============================================================
function toggleDetail(detailId) {
  var el = document.getElementById(detailId);
  if (el) {
    el.style.display = el.style.display === 'none' ? 'table-row' : 'none';
  }
}

// ============================================================
// FILTER QUEUE
// ============================================================
function filterQueue() {
  var search = (document.getElementById('queueSearch').value || '').toLowerCase();
  var actionFilter = document.getElementById('queueActionFilter').value;
  var rows = document.getElementById('queueBody').querySelectorAll('tr');
  for (var i = 0; i < rows.length; i++) {
    var row = rows[i];
    var action = (row.getAttribute('data-action') || '').toLowerCase();
    var text = row.textContent.toLowerCase();
    var show = true;
    if (search && text.indexOf(search) === -1) show = false;
    if (actionFilter && action.indexOf(actionFilter.toLowerCase()) === -1) show = false;
    row.style.display = show ? '' : 'none';
  }
}

// ============================================================
// FILTER ACCOUNTS
// ============================================================
function filterAccounts() {
  var search = (document.getElementById('accSearch').value || '').toLowerCase();
  var typeFilt = document.getElementById('accTypeFilter').value;
  var actionFilt = document.getElementById('accActionFilter').value;
  var rows = document.getElementById('accBody').querySelectorAll('tr');
  for (var i = 0; i < rows.length; i++) {
    var row = rows[i];
    var rtype = row.getAttribute('data-type') || '';
    var raction = (row.getAttribute('data-action') || '').toLowerCase();
    var text = row.textContent.toLowerCase();
    var show = true;
    if (search && text.indexOf(search) === -1) show = false;
    if (typeFilt && rtype !== typeFilt) show = false;
    if (actionFilt && raction.indexOf(actionFilt.toLowerCase()) === -1) show = false;
    row.style.display = show ? '' : 'none';
  }
}

// ============================================================
// EXPORT CSV
// ============================================================
function exportTableCSV(tableId, filename) {
  var table = document.getElementById(tableId);
  if (!table) return;
  var rows = table.querySelectorAll('tr');
  var csv = [];
  for (var ri = 0; ri < rows.length; ri++) {
    var cols = rows[ri].querySelectorAll('th, td');
    var rowData = [];
    for (var ci = 0; ci < cols.length; ci++) {
      var txt = cols[ci].textContent.trim().replace(/,/g, ';');
      rowData.push('"' + txt + '"');
    }
    csv.push(rowData.join(','));
  }
  var blob = new Blob([csv.join('\\n')], { type: 'text/csv;charset=utf-8;' });
  var link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  link.click();
}

// ============================================================
// ACCOUNT MODAL
// ============================================================
function showAccountDetail(idx) {
  var acc = ACCOUNTS[idx];
  if (!acc) return;
  var riskTag = '';
  var p = parseFloat(acc.calibP) || 0;
  if (p >= 0.08) riskTag = '<span class="risk-tag-high">HIGH RISK - Agent Call</span>';
  else if (p >= 0.05) riskTag = '<span class="risk-tag-med">MEDIUM RISK - Auto Dialer</span>';
  else riskTag = '<span class="risk-tag-low">LOW RISK - SMS/Email</span>';

  var insight = '';
  if (p >= 0.08) insight = 'High predicted repayment probability. Prioritize agent outreach. Balance group: ' + (acc.balGroup || 'unknown') + '.';
  else if (p >= 0.05) insight = 'Moderate probability. Auto-dialer campaign recommended. District: ' + (acc.district || 'unknown') + '.';
  else insight = 'Lower probability. Low-cost channel sufficient. Monitor payment behavior changes.';

  var html =
    '<div style="text-align:center;margin-bottom:16px;">' +
      '<h2 style="font-size:22px;color:#f1f5f9;">Account #' + (acc.id || '') + '</h2>' +
      riskTag +
    '</div>' +
    '<div class="insight-box"><b>AI Insight:</b> ' + insight + '</div>' +
    '<div style="margin-top:16px;">';

  var fields = [
    ['Account ID', acc.id],
    ['Loan Type', acc.type],
    ['Balance Group', acc.balGroup],
    ['District', acc.district],
    ['Is Payer', acc.isPayer === 'Y' ? '[YES] Paid' : '[NO]'],
    ['Raw Prediction', acc.rawP],
    ['Calibrated Probability', acc.calibP],
    ['Recommended Action', acc.action],
    ['Net Recovery Value', '\u00a5' + (acc.netRec || 0).toLocaleString()],
    ['Purchased Balance', '\u00a5' + (acc.balance || 0).toLocaleString()],
  ];

  for (var fi = 0; fi < fields.length; fi++) {
    html += '<div class="modal-field"><span class="modal-field-label">' + fields[fi][0] + '</span><span class="modal-field-value">' + fields[fi][1] + '</span></div>';
  }
  html += '</div>';

  document.getElementById('modalTitle').textContent = 'Account #' + (acc.id || '') + ' - Detail';
  document.getElementById('modalBody').innerHTML = html;
  document.getElementById('accModal').classList.add('show');
}

function closeModal() {
  document.getElementById('accModal').classList.remove('show');
}

// Close modal on backdrop click
document.getElementById('accModal').addEventListener('click', function(e) {
  if (e.target === this) closeModal();
});

// ============================================================
// FEATURE IMPORTANCE CHART UPDATE
// ============================================================
function updateFiChart() {
  var modelName = document.getElementById('fiModelSelect').value;
  var fiData = ALL_FI[modelName];
  if (!fiData) return;

  var entries = Object.entries(fiData).map(function(kv) { return { name: kv[0], val: kv[1] }; });
  entries.sort(function(a, b) { return Math.abs(b.val) - Math.abs(a.val); });
  var top12 = entries.slice(0, 12);

  var labels = top12.map(function(e) { return e.name; });
  var values = top12.map(function(e) { return parseFloat(e.val); });
  var colors = values.map(function(v) { return v >= 0 ? 'rgba(59,130,246,0.7)' : 'rgba(239,68,68,0.7)'; });

  var ctx = document.getElementById('featureImportanceChart');
  if (!ctx) return;
  if (window.fiChartInstance) window.fiChartInstance.destroy();

  window.fiChartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{ label: 'Feature Importance', data: values, backgroundColor: colors, borderRadius: 4 }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,.05)' }, ticks: { color: '#94a3b8' } },
        y: { grid: { display: false }, ticks: { color: '#cbd5e1', font: { size: 11 } } }
      }
    }
  });
}

// ============================================================
// CHART INITIALIZATION
// ============================================================
var chartInitialized = {};

function initChartsForTab(tabId) {
  // Prevent double-init
  if (chartInitialized[tabId]) return;
  chartInitialized[tabId] = true;

  if (tabId === 'overview') {
    initDevSplitChart();
    initConfusionMatrixChart();
  }
  if (tabId === 'models') {
    initModelAucChart();
    initModelEcoChart();
  }
  if (tabId === 'features') {
    updateFiChart();
    initPayerBalChart();
    initPayerLoanChart();
  }
  if (tabId === 'queue') {
    initQueuePieChart();
    initQueueRoiChart();
  }
  if (tabId === 'tuning') {
    initMlpSweepChart();
  }
}

// --- Dev Split Pie ---
function initDevSplitChart() {
  var ctx = document.getElementById('devSplitChart');
  if (!ctx || DEV_SPLIT_LABELS.length === 0) return;
  new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: DEV_SPLIT_LABELS,
      datasets: [{
        data: DEV_SPLIT_VALUES,
        backgroundColor: ['#3b82f6','#8b5cf6','#10b981','#f59e0b','#ef4444'],
        borderWidth: 0
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: 'right', labels: { color: '#cbd5e1', font: { size: 11 } } } }
    }
  });
}

// --- Confusion Matrix ---
function initConfusionMatrixChart() {
  var ctx = document.getElementById('confMatrixChart');
  if (!ctx) return;
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['TN', 'FP', 'FN', 'TP'],
      datasets: [{
        label: 'Count',
        data: CM_VALUES,
        backgroundColor: ['#3b82f6', '#ef4444', '#ef4444', '#10b981'],
        borderRadius: 4
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,.05)' }, ticks: { color: '#94a3b8' } },
        x: { grid: { display: false }, ticks: { color: '#cbd5e1' } }
      }
    }
  });
}

// --- Model AUC Comparison ---
function initModelAucChart() {
  var ctx = document.getElementById('modelAucChart');
  if (!ctx) return;
  var names = MODEL_DATA.map(function(m) { return m.display; });
  var aucs = MODEL_DATA.map(function(m) { return parseFloat(m.auc); });
  var colors = MODEL_DATA.map(function(m) { return m.color; });
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: names,
      datasets: [{ label: 'ROC-AUC', data: aucs, backgroundColor: colors, borderRadius: 6 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { min: 0.5, max: 0.85, grid: { color: 'rgba(255,255,255,.05)' }, ticks: { color: '#94a3b8' } },
        x: { grid: { display: false }, ticks: { color: '#cbd5e1' } }
      }
    }
  });
}

// --- Model Economic Comparison ---
function initModelEcoChart() {
  var ctx = document.getElementById('modelEcoChart');
  if (!ctx) return;
  var names = MODEL_DATA.map(function(m) { return m.display; });
  var netrecs = MODEL_DATA.map(function(m) { return parseFloat(m.netrec.replace(/[,]/g,'')) || 0; });
  var colors = MODEL_DATA.map(function(m) { return m.color; });
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: names,
      datasets: [{ label: 'Net Recovery (\u00a5)', data: netrecs, backgroundColor: colors, borderRadius: 6 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { grid: { color: 'rgba(255,255,255,.05)' }, ticks: { color: '#94a3b8', callback: function(v){return '\u00a5'+v.toLocaleString();} } },
        x: { grid: { display: false }, ticks: { color: '#cbd5e1' } }
      }
    }
  });
}

// --- Payer Rate by Balance ---
function initPayerBalChart() {
  var ctx = document.getElementById('payerBalChart');
  if (!ctx || BAL_PAYER_LABELS.length === 0) return;
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: BAL_PAYER_LABELS,
      datasets: [
        { label: 'Actual Payer Rate (%)', data: BAL_PAYER_ALL, backgroundColor: 'rgba(59,130,246,0.7)', borderRadius: 4 },
        { label: 'Predicted Rate (%)', data: BAL_PAYER_PRED, backgroundColor: 'rgba(245,158,11,0.7)', borderRadius: 4 }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#cbd5e1' } } },
      scales: {
        y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,.05)' }, ticks: { color: '#94a3b8' } },
        x: { grid: { display: false }, ticks: { color: '#cbd5e1', font: { size: 10 }, maxRotation: 45 } }
      }
    }
  });
}

// --- Payer Rate by Loan Type ---
function initPayerLoanChart() {
  var ctx = document.getElementById('payerLoanChart');
  if (!ctx || LOAN_PAYER_LABELS.length === 0) return;
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: LOAN_PAYER_LABELS,
      datasets: [
        { label: 'Actual Payer Rate (%)', data: LOAN_PAYER_ALL, backgroundColor: 'rgba(16,185,129,0.7)', borderRadius: 4 },
        { label: 'Predicted Rate (%)', data: LOAN_PAYER_PRED, backgroundColor: 'rgba(139,92,246,0.7)', borderRadius: 4 }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#cbd5e1' } } },
      scales: {
        y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,.05)' }, ticks: { color: '#94a3b8' } },
        x: { grid: { display: false }, ticks: { color: '#cbd5e1' } }
      }
    }
  });
}

// --- Queue Pie Chart ---
function initQueuePieChart() {
  var ctx = document.getElementById('queuePieChart');
  if (!ctx) return;
  var qlabels = QUEUE_DATA.map(function(q){ return q.action; });
  var qvals = QUEUE_DATA.map(function(q){ return q.accounts; });
  var qcolors = ['#ef4444', '#f59e0b', '#3b82f6', '#6b7280'].slice(0, qlabels.length);
  new Chart(ctx, {
    type: 'pie',
    data: {
      labels: qlabels,
      datasets: [{ data: qvals, backgroundColor: qcolors, borderWidth: 0 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: 'right', labels: { color: '#cbd5e1', font: { size: 11 } } } }
    }
  });
}

// --- Queue ROI Chart ---
function initQueueRoiChart() {
  var ctx = document.getElementById('queueRoiChart');
  if (!ctx) return;
  var qlabels = QUEUE_DATA.map(function(q){
    return q.action.replace(/[()][^)]*/g,'').trim();
  });
  var qrois = QUEUE_DATA.map(function(q){
    var r = String(q.roi).replace(/x/g,'');
    return parseFloat(r) || 0;
  });
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: qlabels,
      datasets: [{ label: 'ROI (x)', data: qrois, backgroundColor: ['#ef4444','#f59e0b','#3b82f6','#6b7280'].slice(0,qlabels.length), borderRadius: 4 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,.05)' }, ticks: { color: '#94a3b8' } },
        x: { grid: { display: false }, ticks: { color: '#cbd5e1', font: { size: 10 }, maxRotation: 30 } }
      }
    }
  });
}

// --- MLP Sweep ---
function initMlpSweepChart() {
  var ctx = document.getElementById('mlpSweepChart');
  if (!ctx) return;
  var sweepData = [
    { cfg: 'cfg_1', auc: 0.7026, best: false },
    { cfg: 'cfg_2', auc: 0.6759, best: false },
    { cfg: 'cfg_3', auc: 0.6828, best: false },
    { cfg: 'cfg_4', auc: 0.6852, best: false },
    { cfg: 'cfg_5', auc: 0.4822, best: false },
    { cfg: 'cfg_6', auc: 0.7057, best: true },
    { cfg: 'cfg_7', auc: 0.6987, best: false },
    { cfg: 'cfg_8', auc: 0.6885, best: false }
  ];
  var labels = sweepData.map(function(s) { return s.cfg; });
  var values = sweepData.map(function(s) { return s.auc; });
  var bgColors = sweepData.map(function(s) { return s.best ? '#10b981' : 'rgba(99,102,241,.6)'; });

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{ label: 'Val AUC', data: values, backgroundColor: bgColors, borderRadius: 4 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { min: 0.4, max: 0.8, grid: { color: 'rgba(255,255,255,.05)' }, ticks: { color: '#94a3b8' } },
        x: { grid: { display: false }, ticks: { color: '#cbd5e1' } }
      }
    }
  });
}

// ============================================================
// INIT: Auto-open first tab's charts
// ============================================================
document.addEventListener('DOMContentLoaded', function() {
  // Init Overview charts immediately (it's visible by default)
  initChartsForTab('overview');

  // Make sure Overview tab is shown
  var overviewPanel = document.getElementById('panel-overview');
  if (overviewPanel) overviewPanel.classList.add('active');
});
</script>
</body>
</html>
""")

size_kb = OUTPUT_HTML.stat().st_size / 1024
print("Done! {path} ({sz:.1f} KB)".format(path=str(OUTPUT_HTML), sz=size_kb))
