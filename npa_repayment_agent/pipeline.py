from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, fbeta_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

BALANCE_PROXY = {
    "00. <=200": 100,
    "01. <=5k": 2500,
    "02. <=10k": 7500,
    "03. <=25k": 17500,
    "04. <=50k": 37500,
    "05. <=100k": 75000,
    "06. <=200k": 150000,
    "07. 200k+": 250000,
}

CATEGORICAL = [
    "multiple_acct",
    "loan_type",
    "purchased_bal_gp",
    "district",
    "home_phone_flag",
    "mobile_phone_flag",
]

NUMERIC = [
    "last_act_closing_m",
    "open_closing_m",
    "co_closing_m",
    "last_pay_date_client_closing_m",
    "birth_yr",
    "never_paid_to_client_flag",
    "missing_last_act_flag",
]

IDENTIFIER_COLUMNS = ["id", "debtor_last"]
DROP_FOR_MODEL = ["id", "debtor_last", "payer_3yr", "data_type"]
TARGET_COLUMN = "payer_3yr"
DATA_TYPE_COLUMN = "data_type"


def _to_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _ensure_dir(path: str | Path) -> Path:
    p = _to_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_pct(value: float) -> float:
    return round(float(value) * 100, 2)


def _load_excel(file_path: str | Path) -> pd.DataFrame:
    return pd.read_excel(_to_path(file_path))


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    string_cols = [
        DATA_TYPE_COLUMN,
        "multiple_acct",
        "loan_type",
        "purchased_bal_gp",
        "district",
        "home_phone_flag",
        "mobile_phone_flag",
    ]
    if TARGET_COLUMN in x.columns:
        string_cols.append(TARGET_COLUMN)
    for col in string_cols:
        if col in x.columns:
            x[col] = x[col].astype(str).str.strip()
    if "district" in x.columns:
        x["district"] = x["district"].str.upper()
    for col in ["multiple_acct", "home_phone_flag", "mobile_phone_flag"]:
        if col in x.columns:
            x[col] = x[col].str.upper()

    x["never_paid_to_client_flag"] = x["last_pay_date_client_closing_m"].isna().astype(int)
    x["missing_last_act_flag"] = x["last_act_closing_m"].isna().astype(int)
    x["last_pay_date_client_closing_m"] = x["last_pay_date_client_closing_m"].fillna(-1)
    x["last_act_closing_m"] = x["last_act_closing_m"].fillna(-1)
    x["balance_proxy"] = x["purchased_bal_gp"].map(BALANCE_PROXY).fillna(0)
    if TARGET_COLUMN in x.columns:
        x["target"] = (x[TARGET_COLUMN] == "Y").astype(int)
    return x


def _make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist",
                                min_frequency=50,
                                sparse_output=True,
                            ),
                        ),
                    ]
                ),
                CATEGORICAL,
            ),
            (
                "num",
                Pipeline(
                    steps=[("imputer", SimpleImputer(strategy="constant", fill_value=-1))]
                ),
                NUMERIC,
            ),
        ]
    )


def _find_best_threshold(y_true: pd.Series, prob: np.ndarray) -> dict[str, float]:
    best = {"threshold": 0.5, "f2": -1.0, "recall": 0.0, "precision": 0.0}
    for threshold in np.linspace(0.05, 0.8, 76):
        pred = (prob >= threshold).astype(int)
        recall = recall_score(y_true, pred, zero_division=0)
        precision = precision_score(y_true, pred, zero_division=0)
        f2 = fbeta_score(y_true, pred, beta=2, zero_division=0)
        if f2 > best["f2"] or (abs(f2 - best["f2"]) < 1e-12 and recall > best["recall"]):
            best = {
                "threshold": float(threshold),
                "f2": float(f2),
                "recall": float(recall),
                "precision": float(precision),
            }
    return best


def _evaluate(y_true: pd.Series, prob: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def _prepare_model_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if DATA_TYPE_COLUMN not in df.columns:
        raise ValueError("数据缺少 data_type 字段，无法按 M/T 切分。")
    model_df = df[df[DATA_TYPE_COLUMN] == "M"].copy()
    test_df = df[df[DATA_TYPE_COLUMN] == "T"].copy()
    if model_df.empty or test_df.empty:
        raise ValueError("数据必须同时包含 data_type=M 和 data_type=T 的记录。")
    return model_df, test_df


def _build_model_pipelines(y_train: pd.Series) -> dict[str, Pipeline]:
    positive = int(y_train.sum())
    negative = int(len(y_train) - positive)
    scale_pos_weight = negative / max(positive, 1)
    return {
        "balanced_random_forest": Pipeline(
            steps=[
                ("prep", _make_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=700,
                        max_depth=10,
                        min_samples_leaf=8,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            steps=[
                ("prep", _make_preprocessor()),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=450,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.85,
                        colsample_bytree=0.8,
                        min_child_weight=5,
                        reg_lambda=1.0,
                        objective="binary:logistic",
                        eval_metric="auc",
                        scale_pos_weight=scale_pos_weight,
                        random_state=42,
                        n_jobs=4,
                    ),
                ),
            ]
        ),
    }


def _feature_importance(model: Pipeline, x_ref: pd.DataFrame, y_ref: pd.Series) -> pd.DataFrame:
    importance = permutation_importance(
        model,
        x_ref,
        y_ref,
        scoring="roc_auc",
        n_repeats=5,
        random_state=42,
        n_jobs=1,
    )
    return (
        pd.DataFrame({"feature": x_ref.columns, "importance": importance.importances_mean})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _label_strategy(scored: pd.DataFrame) -> pd.DataFrame:
    x = scored.copy()
    prob_mid = x["pred_repay_prob"].quantile(0.35)
    prob_high = x["pred_repay_prob"].quantile(0.70)
    bal_mid = x["balance_proxy"].median()
    bal_high = x["balance_proxy"].quantile(0.75)

    segments: list[str] = []
    for row in x.itertuples(index=False):
        p = float(row.pred_repay_prob)
        bal = float(row.balance_proxy)
        if (p >= prob_high and bal >= bal_mid) or (p >= prob_mid and bal >= bal_high):
            segments.append("High Priority (Agent Call)")
        elif p >= prob_high or (p >= prob_mid and bal < bal_high):
            segments.append("Medium Priority (Auto-Dialer)")
        elif bal >= bal_mid:
            segments.append("Low Priority (SMS/Email)")
        else:
            segments.append("Write-off / Ignore")
    x["strategy_segment"] = segments
    return x


def _core_profile(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> dict[str, Any]:
    target_counts = {}
    if TARGET_COLUMN in raw_df.columns:
        target_counts = raw_df[TARGET_COLUMN].astype(str).value_counts(dropna=False).to_dict()
    return {
        "rows": int(len(raw_df)),
        "columns": raw_df.columns.tolist(),
        "data_type_counts": raw_df[DATA_TYPE_COLUMN].astype(str).value_counts(dropna=False).to_dict(),
        "target_counts": target_counts,
        "overall_positive_rate_pct": _safe_pct(clean_df["target"].mean()) if "target" in clean_df.columns else None,
        "missing": {k: int(v) for k, v in raw_df.isna().sum().to_dict().items()},
        "preprocess_rules": {
            "drop_from_model": IDENTIFIER_COLUMNS,
            "last_pay_date_client_closing_m": "缺失填 -1，并新增 never_paid_to_client_flag",
            "last_act_closing_m": "缺失填 -1，并新增 missing_last_act_flag",
            "district": "去首尾空格并统一转大写",
            "balance_proxy": "按 purchased_bal_gp 映射代表余额，仅用于ROI/分层代理值",
        },
    }


def preprocess_npa_data(file_path: str, output_dir: str | None = None) -> dict[str, Any]:
    src = _to_path(file_path)
    out_dir = _ensure_dir(output_dir or (src.parent / "agent_outputs" / "preprocess"))
    raw_df = _load_excel(src)
    clean_df = _clean_frame(raw_df)
    model_df, test_df = _prepare_model_frames(clean_df)

    profile = _core_profile(raw_df, clean_df)
    profile.update(
        {
            "model_rows": int(len(model_df)),
            "test_rows": int(len(test_df)),
            "model_positive_rate_pct": _safe_pct(model_df["target"].mean()),
            "test_positive_rate_pct": _safe_pct(test_df["target"].mean()),
        }
    )

    preprocessed_full_path = out_dir / "preprocessed_full.csv"
    preprocessed_model_path = out_dir / "preprocessed_model.csv"
    preprocessed_test_path = out_dir / "preprocessed_test.csv"
    profile_path = out_dir / "profile.json"

    clean_df.to_csv(preprocessed_full_path, index=False, encoding="utf-8-sig")
    model_df.to_csv(preprocessed_model_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(preprocessed_test_path, index=False, encoding="utf-8-sig")
    profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "file_path": str(src),
        "output_dir": str(out_dir),
        "profile_path": str(profile_path),
        "preprocessed_full_path": str(preprocessed_full_path),
        "preprocessed_model_path": str(preprocessed_model_path),
        "preprocessed_test_path": str(preprocessed_test_path),
        "profile": profile,
    }


def _train_and_score(file_path: str, output_dir: str | None = None) -> dict[str, Any]:
    src = _to_path(file_path)
    out_dir = _ensure_dir(output_dir or (src.parent / "agent_outputs" / "training"))

    raw_df = _load_excel(src)
    clean_df = _clean_frame(raw_df)
    model_df, test_df = _prepare_model_frames(clean_df)

    x_model = model_df.drop(columns=DROP_FOR_MODEL + ["target", "balance_proxy"])
    y_model = model_df["target"]
    x_test = test_df.drop(columns=DROP_FOR_MODEL + ["target", "balance_proxy"])
    y_test = test_df["target"]

    x_train, x_valid, y_train, y_valid = train_test_split(
        x_model,
        y_model,
        test_size=0.2,
        random_state=42,
        stratify=y_model,
    )

    pipelines = _build_model_pipelines(y_train)
    model_selection: dict[str, Any] = {}
    fitted_models: dict[str, Pipeline] = {}

    for name, pipeline in pipelines.items():
        pipeline.fit(x_train, y_train)
        valid_prob = pipeline.predict_proba(x_valid)[:, 1]
        threshold_result = _find_best_threshold(y_valid, valid_prob)
        metrics = _evaluate(y_valid, valid_prob, threshold_result["threshold"])
        metrics["valid_threshold_search"] = threshold_result
        model_selection[name] = metrics
        fitted_models[name] = pipeline

    best_name = sorted(
        model_selection,
        key=lambda k: (model_selection[k]["roc_auc"], model_selection[k]["recall"]),
        reverse=True,
    )[0]
    best_threshold = float(model_selection[best_name]["threshold"])

    final_pipeline = fitted_models[best_name]
    final_pipeline.fit(x_model, y_model)
    test_prob = final_pipeline.predict_proba(x_test)[:, 1]
    test_metrics = _evaluate(y_test, test_prob, best_threshold)

    scored_test = test_df[["id", "data_type", "loan_type", "purchased_bal_gp", "district"]].copy()
    if TARGET_COLUMN in test_df.columns:
        scored_test[TARGET_COLUMN] = test_df[TARGET_COLUMN].values
    scored_test["balance_proxy"] = test_df["balance_proxy"].values
    scored_test["pred_repay_prob"] = test_prob
    scored_test["expected_value_proxy"] = scored_test["balance_proxy"] * scored_test["pred_repay_prob"]
    scored_test = _label_strategy(scored_test)

    feature_importance = _feature_importance(final_pipeline, x_test, y_test).head(10)

    top20_n = max(int(len(scored_test) * 0.2), 1)
    top20_prob = scored_test.sort_values("pred_repay_prob", ascending=False).head(top20_n)
    top20_ev = scored_test.sort_values("expected_value_proxy", ascending=False).head(top20_n)
    total_actual_payers = max(int((scored_test[TARGET_COLUMN] == "Y").sum()), 1)

    strategy_summary = (
        scored_test.groupby("strategy_segment", as_index=False)
        .agg(
            accounts=("id", "count"),
            avg_model_score=("pred_repay_prob", "mean"),
            actual_payer_rate=(TARGET_COLUMN, lambda s: (s == "Y").mean()),
            balance_proxy_total=("balance_proxy", "sum"),
            expected_value_proxy_total=("expected_value_proxy", "sum"),
        )
        .sort_values(
            "strategy_segment",
            key=lambda s: s.map(
                {
                    "High Priority (Agent Call)": 1,
                    "Medium Priority (Auto-Dialer)": 2,
                    "Low Priority (SMS/Email)": 3,
                    "Write-off / Ignore": 4,
                }
            ),
        )
        .reset_index(drop=True)
    )

    payer_rate_by_balance = (
        clean_df.groupby("purchased_bal_gp")["target"]
        .mean()
        .mul(100)
        .round(2)
        .reset_index(name="payer_rate_pct")
    )
    payer_rate_by_loan = (
        clean_df.groupby("loan_type")["target"]
        .mean()
        .mul(100)
        .round(2)
        .reset_index(name="payer_rate_pct")
        .sort_values("payer_rate_pct", ascending=False)
        .reset_index(drop=True)
    )

    payer_rate_by_mobile = (
        clean_df.groupby("mobile_phone_flag")["target"]
        .mean()
        .mul(100)
        .round(2)
        .reset_index(name="payer_rate_pct")
    )

    metrics = {
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
        "model_selection": model_selection,
        "best_model": best_name,
        "test_metrics": test_metrics,
        "top_features": feature_importance.to_dict(orient="records"),
        "concentration": {
            "top20_accounts": int(top20_n),
            "overall_actual_payer_rate_pct": _safe_pct((scored_test[TARGET_COLUMN] == "Y").mean()),
            "prob_top20_actual_payer_rate_pct": _safe_pct((top20_prob[TARGET_COLUMN] == "Y").mean()),
            "prob_top20_actual_payer_capture_share_pct": round(float((top20_prob[TARGET_COLUMN] == "Y").sum() / total_actual_payers) * 100, 2),
            "ev_top20_actual_payer_rate_pct": _safe_pct((top20_ev[TARGET_COLUMN] == "Y").mean()),
            "ev_top20_actual_payer_capture_share_pct": round(float((top20_ev[TARGET_COLUMN] == "Y").sum() / total_actual_payers) * 100, 2),
            "ev_top20_expected_value_capture_share_pct": round(float(top20_ev["expected_value_proxy"].sum() / max(scored_test["expected_value_proxy"].sum(), 1)) * 100, 2),
        },
    }

    model_bundle = {
        "pipeline": final_pipeline,
        "best_model": best_name,
        "threshold": best_threshold,
        "categorical_features": CATEGORICAL,
        "numeric_features": NUMERIC,
        "drop_for_model": DROP_FOR_MODEL,
        "balance_proxy": BALANCE_PROXY,
        "metadata": metrics,
    }

    model_path = out_dir / "npa_repayment_model.joblib"
    metrics_path = out_dir / "metrics.json"
    scored_test_path = out_dir / "test_scored_accounts.csv"
    strategy_summary_path = out_dir / "strategy_summary.csv"
    feature_importance_path = out_dir / "feature_importance.csv"
    payer_rate_by_balance_path = out_dir / "payer_rate_by_balance.csv"
    payer_rate_by_loan_path = out_dir / "payer_rate_by_loan.csv"
    payer_rate_by_mobile_path = out_dir / "payer_rate_by_mobile.csv"
    report_path = out_dir / "collection_strategy_report.md"

    joblib.dump(model_bundle, model_path)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    scored_test.drop(columns=[TARGET_COLUMN]).to_csv(scored_test_path, index=False, encoding="utf-8-sig")
    strategy_summary.to_csv(strategy_summary_path, index=False, encoding="utf-8-sig")
    feature_importance.to_csv(feature_importance_path, index=False, encoding="utf-8-sig")
    payer_rate_by_balance.to_csv(payer_rate_by_balance_path, index=False, encoding="utf-8-sig")
    payer_rate_by_loan.to_csv(payer_rate_by_loan_path, index=False, encoding="utf-8-sig")
    payer_rate_by_mobile.to_csv(payer_rate_by_mobile_path, index=False, encoding="utf-8-sig")

    report_content = _build_report_markdown(metrics, strategy_summary, payer_rate_by_balance, payer_rate_by_loan, payer_rate_by_mobile)
    report_path.write_text(report_content, encoding="utf-8")

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "scored_test_path": str(scored_test_path),
        "strategy_summary_path": str(strategy_summary_path),
        "feature_importance_path": str(feature_importance_path),
        "report_path": str(report_path),
        "metrics": metrics,
    }


def _build_report_markdown(
    metrics: dict[str, Any],
    strategy_summary: pd.DataFrame,
    payer_rate_by_balance: pd.DataFrame,
    payer_rate_by_loan: pd.DataFrame,
    payer_rate_by_mobile: pd.DataFrame,
) -> str:
    m = metrics
    tm = m["test_metrics"]
    cm = tm["confusion_matrix"]
    c = m["concentration"]
    report: list[str] = []
    report.append("# NPA还款预测模型与催收策略报告")
    report.append("")
    report.append("## 1. Executive Summary")
    report.append(f"- 数据量：{m['data_overview']['rows']:,} 条，其中建模集(M) {m['data_overview']['model_rows']:,} 条，验证集(T) {m['data_overview']['test_rows']:,} 条。")
    report.append(f"- 最优模型：**{m['best_model']}**；T集 ROC-AUC = **{tm['roc_auc']:.3f}**，Recall(Y) = **{_safe_pct(tm['recall']):.2f}%**，Precision(Y) = **{_safe_pct(tm['precision']):.2f}%**。")
    report.append(f"- 排序效果：按还款概率排序时，前20%账户覆盖 **{c['prob_top20_actual_payer_capture_share_pct']:.2f}%** 的真实付款账户，真实付款率 **{c['prob_top20_actual_payer_rate_pct']:.2f}%**。")
    report.append(f"- ROI效果：按EV代理值排序时，前20%账户贡献 **{c['ev_top20_expected_value_capture_share_pct']:.2f}%** 的潜在回收价值。")
    report.append("")
    report.append("## 2. Methodology")
    report.append("- 按 `data_type=M/T` 切分训练与独立验证。")
    report.append("- 丢弃 `id` 与 `debtor_last`，避免标识符噪音和隐私/公平性风险。")
    report.append("- 将 `last_pay_date_client_closing_m` 缺失填为 `-1`，并新增 `never_paid_to_client_flag`。")
    report.append("- 将 `last_act_closing_m` 缺失填为 `-1`，并新增 `missing_last_act_flag`。")
    report.append("- 对类别不平衡，Random Forest 使用 `class_weight`，XGBoost 使用 `scale_pos_weight`。")
    report.append("- 评估指标以 ROC-AUC、Recall(Y)、Precision(Y)、Confusion Matrix 为核心，不使用 Accuracy 作为主指标。")
    report.append("")
    report.append("## 3. Model Comparison")
    report.append("| Model | Validation ROC-AUC | Validation Recall(Y) | Validation Precision(Y) | Threshold |")
    report.append("|---|---:|---:|---:|---:|")
    for model_name, model_metric in m["model_selection"].items():
        report.append(
            f"| {model_name} | {model_metric['roc_auc']:.3f} | {_safe_pct(model_metric['recall']):.2f}% | {_safe_pct(model_metric['precision']):.2f}% | {model_metric['threshold']:.2f} |"
        )
    report.append("")
    report.append("## 4. Holdout Performance")
    report.append(f"- ROC-AUC: **{tm['roc_auc']:.3f}**")
    report.append(f"- Recall(Y): **{_safe_pct(tm['recall']):.2f}%**")
    report.append(f"- Precision(Y): **{_safe_pct(tm['precision']):.2f}%**")
    report.append(f"- Confusion Matrix @ threshold {tm['threshold']:.2f}: TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}")
    report.append("")
    report.append("## 5. Top Features")
    business_map = {
        "birth_yr": "出生年份越晚通常意味着更年轻的偿付群体，收入修复和征信修复动机更强。",
        "district": "区域反映稳定性、流动性和社会经济结构差异。",
        "purchased_bal_gp": "余额规模同时影响协商意愿和资源投放经济性。",
        "last_act_closing_m": "账户活动越近，说明行为痕迹越新，触达和协商成功率更高。",
        "co_closing_m": "核销距收购日越近，债务新鲜度更高。",
        "last_pay_date_client_closing_m": "历史付款越近，未来付款延续性通常越好。",
        "multiple_acct": "多账户关系通常带来更丰富的行为信号。",
        "home_phone_flag": "住宅电话是补充触达渠道。",
        "mobile_phone_flag": "手机号直接决定自动化催收触达效率。",
        "open_closing_m": "账户年龄反映信贷关系成熟度。",
    }
    for row in m["top_features"]:
        feature = row["feature"]
        report.append(f"- **{feature}**（importance={row['importance']:.4f}）：{business_map.get(feature, '该变量对区分付款人与非付款人具有明显增益。')}")
    report.append("")
    report.append("## 6. Strategy Matrix")
    report.append("| Segment | Accounts | Average Model Score | Actual Payer Rate | Balance Proxy Total | Expected Value Proxy Total |")
    report.append("|---|---:|---:|---:|---:|---:|")
    for row in strategy_summary.itertuples(index=False):
        report.append(
            f"| {row.strategy_segment} | {int(row.accounts):,} | {_safe_pct(row.avg_model_score):.2f}% | {_safe_pct(row.actual_payer_rate):.2f}% | {row.balance_proxy_total:,.0f} | {row.expected_value_proxy_total:,.0f} |"
        )
    report.append("")
    report.append("## 7. Action Recommendation")
    report.append("- **High Priority (Agent Call)**：优先人工坐席，目标是高余额、高回收价值账户。")
    report.append("- **Medium Priority (Auto-Dialer)**：自动外呼 + 轻人工复核，追求单位成本最优。")
    report.append("- **Low Priority (SMS/Email)**：保留低成本数字化触达与自助还款入口。")
    report.append("- **Write-off / Ignore**：仅做极低成本批量触达或后续再评分。")
    report.append("")
    report.append("## 8. Additional Signals")
    report.append("### Payer Rate by Balance Group")
    report.append("| Balance Group | Payer Rate |")
    report.append("|---|---:|")
    for row in payer_rate_by_balance.itertuples(index=False):
        report.append(f"| {row.purchased_bal_gp} | {row.payer_rate_pct:.2f}% |")
    report.append("")
    report.append("### Payer Rate by Loan Type")
    report.append("| Loan Type | Payer Rate |")
    report.append("|---|---:|")
    for row in payer_rate_by_loan.itertuples(index=False):
        report.append(f"| {row.loan_type} | {row.payer_rate_pct:.2f}% |")
    report.append("")
    report.append("### Payer Rate by Mobile Phone Flag")
    report.append("| Mobile Phone Flag | Payer Rate |")
    report.append("|---|---:|")
    for row in payer_rate_by_mobile.itertuples(index=False):
        report.append(f"| {row.mobile_phone_flag} | {row.payer_rate_pct:.2f}% |")
    return "\n".join(report)


def train_repayment_model(file_path: str, output_dir: str | None = None) -> dict[str, Any]:
    return _train_and_score(file_path=file_path, output_dir=output_dir)


def predict_repayment_probability(file_path: str, model_path: str, output_dir: str | None = None) -> dict[str, Any]:
    src = _to_path(file_path)
    bundle_path = _to_path(model_path)
    out_dir = _ensure_dir(output_dir or (src.parent / "agent_outputs" / "prediction"))

    bundle = joblib.load(bundle_path)
    pipeline: Pipeline = bundle["pipeline"]
    threshold = float(bundle["threshold"])

    raw_df = _load_excel(src)
    clean_df = _clean_frame(raw_df)
    x = clean_df.drop(columns=[c for c in DROP_FOR_MODEL + ["target", "balance_proxy"] if c in clean_df.columns])

    prob = pipeline.predict_proba(x)[:, 1]
    scored = clean_df[[c for c in ["id", "data_type", "loan_type", "purchased_bal_gp", "district"] if c in clean_df.columns]].copy()
    scored["balance_proxy"] = clean_df["balance_proxy"].values
    scored["pred_repay_prob"] = prob
    scored["predicted_payer_flag"] = np.where(prob >= threshold, "Y", "N")
    scored["expected_value_proxy"] = scored["balance_proxy"] * scored["pred_repay_prob"]
    scored = _label_strategy(scored)
    scored_path = out_dir / "scored_accounts.csv"
    summary_path = out_dir / "prediction_summary.json"
    scored.to_csv(scored_path, index=False, encoding="utf-8-sig")

    summary = {
        "source_file": str(src),
        "model_path": str(bundle_path),
        "output_dir": str(out_dir),
        "rows": int(len(scored)),
        "threshold": threshold,
        "avg_score": round(float(scored["pred_repay_prob"].mean()), 6),
        "strategy_mix": scored["strategy_segment"].value_counts(dropna=False).to_dict(),
        "top_expected_value_proxy": round(float(scored["expected_value_proxy"].max()), 2),
        "scored_path": str(scored_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def build_collection_strategy_report(file_path: str, output_dir: str | None = None) -> dict[str, Any]:
    result = _train_and_score(file_path=file_path, output_dir=output_dir)
    return {
        "report_path": result["report_path"],
        "metrics_path": result["metrics_path"],
        "strategy_summary_path": result["strategy_summary_path"],
        "feature_importance_path": result["feature_importance_path"],
        "best_model": result["metrics"]["best_model"],
        "test_metrics": result["metrics"]["test_metrics"],
    }
