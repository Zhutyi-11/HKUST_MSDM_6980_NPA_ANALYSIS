"""
Full Model Tuning & Retraining Script
=====================================
对4个模型(LR/RF/XGB/MLP)执行网格搜索，用最优参数重跑完整pipeline。
输出到 baseline_comparison_run 目录。
"""
import sys, json, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# --- Add project to path ---
sys.path.insert(0, r"c:\Users\marcozhu\Desktop\6980")

from npa_repayment_agent.pipeline import (
    _load_excel, _clean_frame, _prepare_model_frames,
    _make_preprocessor, _split_dev_sets,
    _fit_platt_calibrator, _apply_platt_calibrator,
    _find_best_threshold, _evaluate_probabilities,
    _feature_importance, _score_frame, _apply_policy_engine,
    _fit_production_calibrator, _to_builtin, _safe_pct,
    _core_profile, _build_model_pipelines,
    DROP_FOR_MODEL, TARGET_COLUMN, DATA_TYPE_COLUMN,
    BALANCE_PROXY, DEFAULT_PRODUCTION_CONFIG,
    CATEGORICAL, NUMERIC,
)
# BASELINE_MODEL_NAME is a constant string from pipeline.py
BASELINE_MODEL_NAME = "baseline_logistic_regression"

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, make_scorer

OUT_DIR = Path(r"c:\Users\marcozhu\Desktop\6980\agent_outputs\baseline_comparison_run")
DATA_PATH = Path(r"c:\Users\marcozhu\Desktop\6980\data.xlsx")

# ============================================================
# Step 1: Load & prepare data
# ============================================================
print("=" * 60)
print("STEP 1: Loading data...")
print("=" * 60)

raw_df = _load_excel(DATA_PATH)
clean_df = _clean_frame(raw_df)
model_df, test_df = _prepare_model_frames(clean_df)

x_model = model_df.drop(columns=DROP_FOR_MODEL + ["target", "balance_proxy"])
y_model = model_df["target"]
x_test = test_df.drop(columns=DROP_FOR_MODEL + ["target", "balance_proxy"])
y_test = test_df["target"]

x_train, x_calib, x_valid, y_train, y_calib, y_valid = _split_dev_sets(x_model, y_model)

print(f"  Total: {len(clean_df)}, Train: {len(x_train)}, Calib: {len(x_calib)}, Valid: {len(x_valid)}, Test: {len(x_test)}")
print(f"  Pos rate: train={_safe_pct(y_train.mean())}%, valid={_safe_pct(y_valid.mean())}%, test={_safe_pct(y_test.mean())}%")

preprocessor = _make_preprocessor()
pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
print(f"  Scale pos weight: {pos_weight:.2f}")

auc_scorer = make_scorer(roc_auc_score, response_method="predict_proba")
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


# ============================================================
# Step 2: Logistic Regression Tuning
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Logistic Regression Grid Search...")
print("=" * 60)

lr_pipe = Pipeline([
    ("prep", preprocessor),
    ("model", LogisticRegression(random_state=42, max_iter=5000)),
])
lr_param_grid = {
    "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
    "model__solver": ["lbfgs", "liblinear"],
    "model__class_weight": ["balanced", None],
    "model__penalty": ["l2"],
}
lr_gs = GridSearchCV(lr_pipe, lr_param_grid, cv=cv, scoring=auc_scorer, n_jobs=-1, verbose=0)
lr_gs.fit(x_model, y_model)

lr_best_params = {k.replace("model__", ""): v for k, v in lr_gs.best_params_.items()}
print(f"  Best AUC: {lr_gs.best_score_:.4f}")
print(f"  Best params: {lr_best_params}")


# ============================================================
# Step 3: Random Forest Tuning
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Random Forest Grid Search...")
print("=" * 60)

rf_pipe = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestClassifier(random_state=42, n_jobs=-1)),
])
rf_param_grid = {
    "model__n_estimators": [300, 500, 700],
    "model__max_depth": [6, 8, 10, 12],
    "model__min_samples_leaf": [4, 8, 16],
    "model__class_weight": ["balanced", "balanced_subsample"],
}
rf_gs = GridSearchCV(rf_pipe, rf_param_grid, cv=cv, scoring=auc_scorer, n_jobs=-1, verbose=0)
rf_gs.fit(x_model, y_model)

rf_best_params = {k.replace("model__", ""): v for k, v in rf_gs.best_params_.items()}
print(f"  Best AUC: {rf_gs.best_score_:.4f}")
print(f"  Best params: {rf_best_params}")


# ============================================================
# Step 4: XGBoost Tuning
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: XGBoost Grid Search...")
print("=" * 60)

xgb_pipe = Pipeline([
    ("prep", preprocessor),
    ("model", XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=pos_weight,
        random_state=42,
        n_jobs=4,
    )),
])
xgb_param_grid = {
    "model__n_estimators": [200, 350, 500],
    "model__max_depth": [3, 4, 5, 6],
    "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "model__subsample": [0.7, 0.85, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 1.0],
    "model__min_child_weight": [2, 5, 10],
    "model__reg_lambda": [0.5, 1.0, 2.0],
}
xgb_gs = GridSearchCV(xgb_pipe, xgb_param_grid, cv=cv, scoring=auc_scorer, n_jobs=-1, verbose=0)
xgb_gs.fit(x_model, y_model)

xgb_best_params = {k.replace("model__", ""): v for k, v in xgb_gs.best_params_.items()}
print(f"  Best AUC: {xgb_gs.best_score_:.4f}")
print(f"  Best params: {xgb_best_params}")


# ============================================================
# Step 5: MLP Deep Learning Tuning (manual sweep)
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: MLP Deep Learning Hyperparameter Sweep...")
print("=" * 60)

try:
    import torch
    torch.manual_seed(42)
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("  PyTorch not available! Skipping MLP.")

mlp_results = []
best_mlp_auc = 0
best_mlp_params = None

if TORCH_OK:
    from npa_repayment_agent.pipeline import SklearnMLPWrapper

    # Preprocess once
    x_train_proc = preprocessor.fit_transform(x_train)
    x_valid_proc = preprocessor.transform(x_valid)

    # Convert sparse to dense if needed
    if hasattr(x_train_proc, "toarray"):
        x_train_np = x_train_proc.toarray()
        x_valid_np = x_valid_proc.toarray()
    else:
        x_train_np = np.array(x_train_proc)
        x_valid_np = np.array(x_valid_proc)

    input_dim = x_train_np.shape[1]
    print(f"  Input dimension after preprocessing: {input_dim}")

    # MLP parameter grid
    mlp_param_grid = [
        # Config 1: small fast network
        {"hidden_dims": [64, 32], "epochs": 100, "batch_size": 256, "lr": 0.001,
         "weight_decay": 1e-4, "num_residual_blocks": 0, "label_smoothing": 0.0},
        # Config 2: medium network with residual
        {"hidden_dims": [128, 64], "epochs": 150, "batch_size": 256, "lr": 0.0005,
         "weight_decay": 5e-4, "num_residual_blocks": 1, "label_smoothing": 0.03},
        # Config 3: medium network, higher lr
        {"hidden_dims": [128, 64], "epochs": 150, "batch_size": 256, "lr": 0.001,
         "weight_decay": 1e-4, "num_residual_blocks": 1, "label_smoothing": 0.05},
        # Config 4: larger network
        {"hidden_dims": [256, 128, 64], "epochs": 200, "batch_size": 512, "lr": 0.0005,
         "weight_decay": 5e-4, "num_residual_blocks": 1, "label_smoothing": 0.03},
        # Config 5: larger network, more residuals
        {"hidden_dims": [256, 128], "epochs": 200, "batch_size": 512, "lr": 0.0003,
         "weight_decay": 1e-3, "num_residual_blocks": 2, "label_smoothing": 0.05},
        # Config 6: small but deep residual
        {"hidden_dims": [128, 64, 32], "epochs": 200, "batch_size": 256, "lr": 0.0003,
         "weight_decay": 5e-4, "num_residual_blocks": 2, "label_smoothing": 0.03},
        # Config 7: wide shallow
        {"hidden_dims": [256, 64], "epochs": 150, "batch_size": 512, "lr": 0.001,
         "weight_decay": 1e-4, "num_residual_blocks": 0, "label_smoothing": 0.0},
        # Config 8: v1-style simple (no feature interaction)
        {"hidden_dims": [128, 64, 32], "epochs": 120, "batch_size": 256, "lr": 0.001,
         "weight_decay": 1e-4, "use_v2_architecture": False, "num_residual_blocks": 0,
         "label_smoothing": 0.0},
    ]

    for i, params in enumerate(mlp_param_grid):
        cfg_name = f"cfg_{i+1}"
        try:
            t0 = time.time()
            # Wrap MLP in Pipeline so preprocessing happens automatically (same as other models)
            mlp_pipe = Pipeline([
                ("prep", _make_preprocessor()),
                ("model", SklearnMLPWrapper(
                    random_state=42,
                    patience=20,
                    grad_clip_norm=3.0,
                    use_amp=False,
                    **params,
                )),
            ])
            mlp_pipe.fit(x_train, y_train)

            prob = mlp_pipe.predict_proba(x_valid)[:, 1]
            auc = roc_auc_score(y_valid, prob)
            elapsed = time.time() - t0

            result = {"config": cfg_name, "auc": auc, "params": dict(params), "time": elapsed}
            mlp_results.append(result)
            print(f"  {cfg_name}: AUC={auc:.4f} ({elapsed:.1f}s) dims={params.get('hidden_dims')} "
                  f"res={params.get('num_residual_blocks',0)} smooth={params.get('label_smoothing',0)}")

            if auc > best_mlp_auc:
                best_mlp_auc = auc
                best_mlp_params = dict(params)

        except Exception as e:
            print(f"  {cfg_name}: FAILED - {e}")
            mlp_results.append({"config": cfg_name, "auc": 0, "error": str(e)})

    print(f"\n  MLP Best AUC: {best_mlp_auc:.4f}")
    if best_mlp_params:
        print(f"  MLP Best Params: hidden_dims={best_mlp_params.get('hidden_dims')}, "
              f"lr={best_mlp_params.get('lr')}, res_blocks={best_mlp_params.get('num_residual_blocks')}, "
              f"smooth={best_mlp_params.get('label_smoothing')}")


# ============================================================
# Step 6: Build pipelines with BEST params & Evaluate on Validation
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Validation evaluation with best params...")
print("=" * 60)

config = json.loads(json.dumps(DEFAULT_PRODUCTION_CONFIG))

def build_and_eval(name, pipe, label):
    """Fit on train+calib, evaluate on validation set."""
    t0 = time.time()
    # Combine train+calib for final fitting (like production would do for dev eval)
    x_tc = pd.concat([x_train, x_calib])
    y_tc = pd.concat([y_train, y_calib])

    pipe.fit(x_tc, y_tc)

    raw_calib = pipe.predict_proba(x_calib)[:, 1]
    calibrator = _fit_platt_calibrator(raw_calib, y_calib)

    raw_valid = pipe.predict_proba(x_valid)[:, 1]
    calibrated_valid = _apply_platt_calibrator(calibrator, raw_valid)

    threshold_result = _find_best_threshold(y_valid, calibrated_valid)
    valid_metrics = _evaluate_probabilities(y_valid, raw_valid, calibrated_valid, threshold_result["threshold"])

    # Policy totals on validation split
    _, _, valid_policy_totals = _score_frame(
        frame=model_df.loc[x_valid.index],
        raw_prob=raw_valid,
        calibrated_prob=calibrated_valid,
        threshold=threshold_result["threshold"],
        config=config,
    )

    elapsed = time.time() - t0
    print(f"\n  [{label}] {name}:")
    print(f"    ROC-AUC:     {valid_metrics['roc_auc']:.4f}")
    print(f"    Brier:       {valid_metrics['brier']:.4f}")
    print(f"    LogLoss:     {valid_metrics['log_loss']:.4f}")
    print(f"    Recall:      {_safe_pct(valid_metrics['recall']):.2f}%")
    print(f"    Precision:   {_safe_pct(valid_metrics['precision']):.2f}%")
    print(f"    Threshold:   {valid_metrics['threshold']:.2f}")
    print(f"    Net Recovery: CNY {valid_policy_totals['expected_net_recovery_total']:,.0f}")
    print(f"    ROI:         {valid_policy_totals['expected_roi']:.2f}x")
    print(f"    Time:        {elapsed:.1f}s")

    return {
        "pipeline": pipe,
        "calibrator": calibrator,
        "metrics": valid_metrics,
        "policy_totals": valid_policy_totals,
        "threshold": threshold_result["threshold"],
    }


# Build best pipelines
results = {}

# 1. LR with best params
lr_best_pipe = Pipeline([
    ("prep", _make_preprocessor()),
    ("model", LogisticRegression(
        random_state=42,
        max_iter=5000,
        **{k: v for k, v in lr_best_params.items() if k != "penalty"},
    )),
])
results["baseline_logistic_regression"] = build_and_eval(
    "baseline_logistic_regression", lr_best_pipe, "BASELINE"
)

# 2. RF with best params
rf_best_pipe = Pipeline([
    ("prep", _make_preprocessor()),
    ("model", RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        **{k: v for k, v in rf_best_params.items()},
    )),
])
results["balanced_random_forest"] = build_and_eval(
    "balanced_random_forest", rf_best_pipe, "RF"
)

# 3. XGB with best params
xgb_best_pipe = Pipeline([
    ("prep", _make_preprocessor()),
    ("model", XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=pos_weight,
        random_state=42,
        n_jobs=4,
        **{k: v for k, v in xgb_best_params.items()},
    )),
])
results["xgboost"] = build_and_eval("xgboost", xgb_best_pipe, "XGB")

# 4. MLP with best params (if available)
if TORCH_OK and best_mlp_params:
    from npa_repayment_agent.pipeline import SklearnMLPWrapper

    mlp_best_pipe = Pipeline([
        ("prep", _make_preprocessor()),
        ("model", SklearnMLPWrapper(
            random_state=42,
            patience=20,
            grad_clip_norm=3.0,
            use_amp=False,
            **best_mlp_params,
        )),
    ])
    try:
        results["deep_mlp"] = build_and_eval("deep_mlp", mlp_best_pipe, "MLP")
    except Exception as e:
        print(f"\n  [MLP] Final evaluation FAILED: {e}")


# ============================================================
# Step 7: Select Champion & Run Full Test Evaluation
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Champion selection & Holdout evaluation...")
print("=" * 60)

# Sort by net recovery (primary), then AUC (secondary)
ranked = sorted(results.items(), key=lambda x: (
    x[1]["policy_totals"]["expected_net_recovery_total"],
    x[1]["metrics"]["roc_auc"],
), reverse=True)

champion_name = ranked[0][0]
champion_result = ranked[0][1]

print(f"\n  === VALIDATION RANKING (by Net Recovery → AUC) ===")
for i, (name, r) in enumerate(ranked):
    marker = " ★ CHAMPION" if name == champion_name else ""
    print(f"  {i+1}. {name}: NetRec=CNY {r['policy_totals']['expected_net_recovery_total']:,.0f}  "
          f"AUC={r['metrics']['roc_auc']:.4f}  Recall={_safe_pct(r['metrics']['recall']):.2f}%{marker}")

print(f"\n  Champion: {champion_name}")

# --- Full training on all M data, evaluate on T ---
final_pipeline = results[champion_name]["pipeline"]
final_calibrator = results[champion_name]["calibrator"]
best_threshold = champion_result["threshold"]

# Refit champion on full M data
print(f"\n  Refitting {champion_name} on full M data ({len(x_model)} samples)...")
final_pipeline.fit(x_model, y_model)

# Production calibrator (OOF)
print("  Fitting OOF Platt calibrator (5-fold)...")
prod_calibrator = _fit_production_calibrator(final_pipeline, x_model, y_model, folds=5)

# T-set predictions
raw_test = final_pipeline.predict_proba(x_test)[:, 1]
calibrated_test = _apply_platt_calibrator(prod_calibrator, raw_test)
test_metrics = _evaluate_probabilities(y_test, raw_test, calibrated_test, best_threshold)

print(f"\n  === CHAMPION TEST (T-set) METRICS ===")
print(f"  ROC-AUC (raw/cal):  {test_metrics['roc_auc_raw']:.4f} / {test_metrics['roc_auc']:.4f}")
print(f"  Brier (raw/cal):    {test_metrics['brier_raw']:.4f} / {test_metrics['brier']:.4f}")
print(f"  LogLoss (raw/cal): {test_metrics['log_loss_raw']:.4f} / {test_metrics['log_loss']:.4f}")
print(f"  Recall:            {_safe_pct(test_metrics['recall']):.2f}%")
print(f"  Precision:         {_safe_pct(test_metrics['precision']):.2f}%")

# Score T-set
scored_test, queue_summary, policy_totals = _score_frame(
    frame=test_df,
    raw_prob=raw_test,
    calibrated_prob=calibrated_test,
    threshold=best_threshold,
    config=config,
)
print(f"  Net Recovery:       CNY {policy_totals['expected_net_recovery_total']:,.0f}")
print(f"  ROI:               {policy_totals['expected_roi']:.2f}x")

# Baseline comparison on T-set
baseline_name = "baseline_logistic_regression"
bl_result = results[baseline_name]
bl_pipeline = bl_result["pipeline"]

print(f"\n  Refitting baseline on full M data...")
bl_pipeline_full = Pipeline([
    ("prep", _make_preprocessor()),
    ("model", LogisticRegression(
        random_state=42,
        max_iter=5000,
        **{k: v for k, v in lr_best_params.items() if k != "penalty"},
    )),
])
bl_pipeline_full.fit(x_model, y_model)
bl_calibrator = _fit_production_calibrator(bl_pipeline_full, x_model, y_model, folds=5)
bl_raw_test = bl_pipeline_full.predict_proba(x_test)[:, 1]
bl_calibrated_test = _apply_platt_calibrator(bl_calibrator, bl_raw_test)
bl_threshold = bl_result["threshold"]
baseline_test_metrics = _evaluate_probabilities(y_test, bl_raw_test, bl_calibrated_test, bl_threshold)
_, _, baseline_policy_totals = _score_frame(
    frame=test_df, raw_prob=bl_raw_test, calibrated_prob=bl_calibrated_test,
    threshold=bl_threshold, config=config,
)

print(f"\n  === BASELINE TEST (T-set) METRICS ===")
print(f"  ROC-AUC:  {baseline_test_metrics['roc_auc']:.4f}")
print(f"  Brier:    {baseline_test_metrics['brier']:.4f}")
print(f"  LogLoss:  {baseline_test_metrics['log_loss']:.4f}")
print(f"  Recall:   {_safe_pct(baseline_test_metrics['recall']):.2f}%")
print(f"  Net Rec:  CNY {baseline_policy_totals['expected_net_recovery_total']:,.0f}")
print(f"  ROI:      {baseline_policy_totals['expected_roi']:.2f}x")

delta = {
    "roc_auc": test_metrics["roc_auc"] - baseline_test_metrics["roc_auc"],
    "brier": test_metrics["brier"] - baseline_test_metrics["brier"],
    "log_loss": test_metrics["log_loss"] - baseline_test_metrics["log_loss"],
    "recall": test_metrics["recall"] - baseline_test_metrics["recall"],
    "precision": test_metrics["precision"] - baseline_test_metrics["precision"],
    "expected_net_recovery_total": policy_totals["expected_net_recovery_total"] - baseline_policy_totals["expected_net_recovery_total"],
    "expected_roi": policy_totals["expected_roi"] - baseline_policy_totals["expected_roi"],
}

print(f"\n  === DELTA (Champ vs Baseline) ===")
print(f"  AUC Δ:           {delta['roc_auc']:+.4f}")
print(f"  Brier Δ:         {delta['brier']:+.4f} ({'better' if delta['brier'] < 0 else 'worse'})")
print(f"  LogLoss Δ:       {delta['log_loss']:+.4f}")
print(f"  Net Recovery Δ:  CNY {delta['expected_net_recovery_total']:+,.0f}")
print(f"  ROI Δ:           {delta['expected_roi']:+.2f}x")


# ============================================================
# Step 8: Feature Importance (all models)
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: Computing feature importance for all models...")
print("=" * 60)

all_feature_importance = {}
for name, r in results.items():
    print(f"  Computing importance for {name}...")
    try:
        fi = _feature_importance(r["pipeline"], x_test, y_test)
        all_feature_importance[name] = fi.to_dict(orient="records")
        top3 = fi.head(3)
        for _, row in top3.iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")
    except Exception as e:
        print(f"    Error: {e}")
        all_feature_importance[name] = []

# Champion feature importance
champion_fi = _feature_importance(final_pipeline, x_test, y_test)


# ============================================================
# Step 9: Descriptive Statistics
# ============================================================
print("\n  Computing descriptive statistics...")

desc_stats = {}
for col in NUMERIC:
    desc_stats[col] = clean_df[col].describe().to_dict()

for col in CATEGORICAL:
    vc = clean_df[col].value_counts()
    desc_stats[col] = {
        "type": "categorical",
        "unique": len(vc),
        "top_categories": vc.head(8).to_dict(),
    }


# ============================================================
# Step 10: Payer Rate by Segment
# ============================================================
payer_rate_by_balance = (
    clean_df.groupby("purchased_bal_gp")["target"].mean().mul(100).round(2)
    .reset_index().rename(columns={"purchased_bal_gp": "group", "target": "payer_rate_pct"})
)
payer_rate_by_loan = (
    clean_df.groupby("loan_type")["target"].mean().mul(100).round(2)
    .reset_index().rename(columns={"loan_type": "group", "target": "payer_rate_pct"})
    .sort_values("payer_rate_pct", ascending=False).reset_index(drop=True)
)
payer_rate_by_mobile = (
    clean_df.groupby("mobile_phone_flag")["target"].mean().mul(100).round(2)
    .reset_index().rename(columns={"mobile_phone_flag": "group", "target": "payer_rate_pct"})
)


# ============================================================
# Step 11: Concentration Metrics
# ============================================================
top20_n = max(int(len(scored_test) * 0.2), 1)
top20_prob = scored_test.sort_values("calibrated_repay_prob", ascending=False).head(top20_n)
top20_net = scored_test.sort_values("expected_net_recovery", ascending=False).head(top20_n)
total_payers = max(int((scored_test[TARGET_COLUMN] == "Y").sum()), 1)

concentration = {
    "top20_accounts": int(top20_n),
    "overall_actual_payer_rate_pct": _safe_pct((scored_test[TARGET_COLUMN] == "Y").mean()),
    "prob_top20_actual_payer_rate_pct": _safe_pct((top20_prob[TARGET_COLUMN] == "Y").mean()),
    "prob_top20_capture_share_pct": round(float((top20_prob[TARGET_COLUMN] == "Y").sum() / total_payers) * 100, 2),
    "net_top20_actual_payer_rate_pct": _safe_pct((top20_net[TARGET_COLUMN] == "Y").mean()),
    "net_top20_capture_share_pct": round(float((top20_net[TARGET_COLUMN] == "Y").sum() / total_payers) * 100, 2),
    "net_top20_expected_net_recovery_share_pct": round(float(top20_net["expected_net_recovery"].sum() / max(scored_test["expected_net_recovery"].sum(), 1)) * 100, 2),
}


# ============================================================
# Step 12: Champion-Challenger Summary Table
# ============================================================
champion_rows = []
for name, r in results.items():
    m = r["metrics"]
    p = r["policy_totals"]
    role = "baseline" if name == baseline_name else ("agent_champion" if name == champion_name else "challenger")
    champion_rows.append({
        "model_name": name,
        "model_role": role,
        "roc_auc": m["roc_auc"],
        "brier": m["brier"],
        "log_loss": m["log_loss"],
        "recall": m["recall"],
        "precision": m["precision"],
        "expected_net_recovery_total": p["expected_net_recovery_total"],
        "expected_roi": p["expected_roi"],
        "threshold": m["threshold"],
    })

champion_df = pd.DataFrame(champion_rows).sort_values(
    ["expected_net_recovery_total", "roc_auc"], ascending=[False, False]
).reset_index(drop=True)

# Fix roles after sorting
champion_df["model_role"] = champion_df["model_name"].apply(
    lambda n: "baseline" if n == baseline_name else
              ("agent_champion" if n == champion_name else "challenger")
)


# ============================================================
# Step 13: Assemble Metrics JSON
# ============================================================
holdout_comparison_rows = [
    {
        "model_name": champion_name,
        "model_role": "agent_champion" if champion_name != baseline_name else "baseline",
        "roc_auc": test_metrics["roc_auc"],
        "brier": test_metrics["brier"],
        "log_loss": test_metrics["log_loss"],
        "recall": test_metrics["recall"],
        "precision": test_metrics["precision"],
        "expected_net_recovery_total": policy_totals["expected_net_recovery_total"],
        "expected_roi": policy_totals["expected_roi"],
        "threshold": test_metrics["threshold"],
    }
]
if champion_name != baseline_name:
    holdout_comparison_rows.append({
        "model_name": baseline_name,
        "model_role": "baseline",
        "roc_auc": baseline_test_metrics["roc_auc"],
        "brier": baseline_test_metrics["brier"],
        "log_loss": baseline_test_metrics["log_loss"],
        "recall": baseline_test_metrics["recall"],
        "precision": baseline_test_metrics["precision"],
        "expected_net_recovery_total": baseline_policy_totals["expected_net_recovery_total"],
        "expected_roi": baseline_policy_totals["expected_roi"],
        "threshold": baseline_test_metrics["threshold"],
    })

agent_vs_baseline = {
    "agent_model": champion_name,
    "baseline_model": baseline_name,
    "agent_matches_baseline": champion_name == baseline_name,
    "holdout_comparison": holdout_comparison_rows,
    "delta": delta,
}

tuning_summary = {
    "lr": {"best_params": lr_best_params, "best_val_auc": float(lr_gs.best_score_), "total_searched": len(lr_gs.cv_results_['mean_test_score'])},
    "rf": {"best_params": rf_best_params, "best_val_auc": float(rf_gs.best_score_), "total_searched": len(rf_gs.cv_results_['mean_test_score'])},
    "xgb": {"best_params": xgb_best_params, "best_val_auc": float(xgb_gs.best_score_), "total_searched": len(xgb_gs.cv_results_['mean_test_score'])},
    "mlp": {
        "best_params": {k: v for k, v in (best_mlp_params or {}).items()
                       if not callable(v) and k not in ('use_v2_architecture',)},
        "best_val_auc": float(best_mlp_auc),
        "total_searched": len(mlp_results),
        "all_configs": [{"config": r["config"], "auc": r["auc"]} for r in mlp_results if "auc" in r],
    },
}

metrics = {
    "production_config": _to_builtin(config),
    "data_overview": {
        "rows": int(len(clean_df)),
        "model_rows": int(len(model_df)),
        "test_rows": int(len(test_df)),
        "overall_positive_rate_pct": _safe_pct(clean_df["target"].mean()),
        "model_positive_rate_pct": _safe_pct(model_df["target"].mean()),
        "test_positive_rate_pct": _safe_pct(test_df["target"].mean()),
        "missing_last_pay_date_client_closing_m": int(raw_df["last_pay_date_client_closing_m"].isna().sum()),
        "missing_last_act_closing_m": int(raw_df["last_act_closing_m"].isna().sum()),
    },
    "development_split": {
        "train_rows": int(len(x_train)),
        "calibration_rows": int(len(x_calib)),
        "validation_rows": int(len(x_valid)),
    },
    "champion_challenger": _to_builtin(champion_df.to_dict(orient="records")),
    "model_selection": _to_builtin({n: r["metrics"] for n, r in results.items()}),
    "best_model": champion_name,
    "baseline_model": baseline_name,
    "test_metrics": _to_builtin(test_metrics),
    "baseline_test_metrics": _to_builtin(baseline_test_metrics),
    "policy_summary": _to_builtin(policy_totals),
    "baseline_policy_summary": _to_builtin(baseline_policy_totals),
    "agent_vs_baseline": _to_builtin(agent_vs_baseline),
    "queue_summary": _to_builtin(queue_summary.to_dict(orient="records")),
    "top_features": _to_builtin(champion_fi.head(12).to_dict(orient="records")),
    "all_feature_importance": {k: _to_builtin(v) for k, v in all_feature_importance.items()},
    "descriptive_stats": _to_builtin(desc_stats),
    "concentration": concentration,
    "payer_rate_by_balance": payer_rate_by_balance.to_dict(orient="records"),
    "payer_rate_by_loan": payer_rate_by_loan.to_dict(orient="records"),
    "payer_rate_by_mobile": payer_rate_by_mobile.to_dict(orient="records"),
    "tuning_summary": _to_builtin(tuning_summary),
}


# ============================================================
# Step 14: Save All Outputs
# ============================================================
print("\n" + "=" * 60)
print("STEP 14: Saving outputs...")
print("=" * 60)

OUT_DIR.mkdir(parents=True, exist_ok=True)

import joblib
model_bundle = {
    "pipeline": final_pipeline,
    "calibrator": prod_calibrator,
    "best_model": champion_name,
    "threshold": best_threshold,
    "categorical_features": CATEGORICAL,
    "numeric_features": NUMERIC,
    "drop_for_model": DROP_FOR_MODEL,
    "balance_proxy": BALANCE_PROXY,
    "config": config,
    "metadata": metrics,
}

joblib.dump(model_bundle, OUT_DIR / "npa_repayment_model.joblib")

(OUT_DIR / "metrics.json").write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
)

scored_test.to_csv(OUT_DIR / "test_scored_accounts.csv", index=False, encoding="utf-8-sig")
queue_summary.to_csv(OUT_DIR / "production_queue_summary.csv", index=False, encoding="utf-8-sig")
champion_df.to_csv(OUT_DIR / "champion_challenger_summary.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(agent_vs_baseline["holdout_comparison"]).to_csv(
    OUT_DIR / "agent_vs_baseline_summary.csv", index=False, encoding="utf-8-sig"
)
champion_fi.to_csv(OUT_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")

print(f"  Done! Outputs saved to: {OUT_DIR}")
print(f"\n{'='*60}")
print(f"FINAL RESULTS SUMMARY")
print(f"{'='*60}")
print(f"Champion:          {champion_name}")
print(f"T-set ROC-AUC:     {test_metrics['roc_auc']:.4f}")
print(f"T-set Brier:       {test_metrics['brier']:.4f}")
print(f"T-set LogLoss:     {test_metrics['log_loss']:.4f}")
print(f"T-set Recall:      {_safe_pct(test_metrics['recall']):.2f}%")
print(f"T-set Precision:   {_safe_pct(test_metrics['precision']):.2f}%")
print(f"T-set Net Recovery:CNY {policy_totals['expected_net_recovery_total']:,.0f}")
print(f"T-set ROI:         {policy_totals['expected_roi']:.2f}x")
print(f"vs Baseline ΔRec:  CNY {delta['expected_net_recovery_total']:+,.0f}")
