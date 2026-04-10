from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, fbeta_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
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
ACTION_ORDER = [
    "High Priority (Agent Call)",
    "Medium Priority (Auto-Dialer)",
    "Low Priority (SMS/Email)",
    "Write-off / Ignore",
]
BASELINE_MODEL_NAME = "baseline_logistic_regression"


DEFAULT_PRODUCTION_CONFIG: dict[str, Any] = {
    "calibration": {
        "enabled": True,
        "method": "platt",
        "oof_folds": 5,
    },
    "economics": {
        "balance_recovery_rate": 0.35,
        "agent_call_cost": 85.0,
        "auto_dialer_cost": 12.0,
        "sms_email_cost": 1.5,
        "agent_call_multiplier": 1.0,
        "auto_dialer_multiplier": 0.72,
        "sms_email_multiplier": 0.35,
    },
    "capacity": {
        "max_agent_ratio": 0.18,
        "max_auto_ratio": 0.42,
        "max_sms_ratio": 0.30,
    },
    "selection": {
        "primary_metric": "expected_net_recovery_total",
        "secondary_metric": "roc_auc",
    },
}


BUSINESS_MAP = {
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
    "never_paid_to_client_flag": "从未向原债权人付款通常是意愿和能力双弱的负信号。",
}


def _to_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _ensure_dir(path: str | Path) -> Path:
    p = _to_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_pct(value: float) -> float:
    return round(float(value) * 100, 2)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [_to_builtin(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _load_config(config_path: str | None = None) -> dict[str, Any]:
    config = json.loads(json.dumps(DEFAULT_PRODUCTION_CONFIG))
    if config_path:
        user_config = json.loads(_to_path(config_path).read_text(encoding="utf-8"))
        for section, section_value in user_config.items():
            if isinstance(section_value, dict) and isinstance(config.get(section), dict):
                config[section].update(section_value)
            else:
                config[section] = section_value
    return config


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
                Pipeline(steps=[("imputer", SimpleImputer(strategy="constant", fill_value=-1))]),
                NUMERIC,
            ),
        ]
    )


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
        BASELINE_MODEL_NAME: Pipeline(
            steps=[
                ("prep", _make_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        solver="liblinear",
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        ),
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


def _model_role(model_name: str, champion_name: str | None = None) -> str:
    if model_name == BASELINE_MODEL_NAME:
        return "baseline"
    if champion_name and model_name == champion_name:
        return "agent_champion"
    return "challenger"



def _fit_platt_calibrator(raw_prob: np.ndarray, y_true: pd.Series) -> LogisticRegression:
    clipped = np.clip(raw_prob.astype(float), 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibrator = LogisticRegression(random_state=42, max_iter=1000)
    calibrator.fit(logits, y_true)
    return calibrator


def _apply_platt_calibrator(calibrator: LogisticRegression | None, raw_prob: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return raw_prob.astype(float)
    clipped = np.clip(raw_prob.astype(float), 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


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


def _evaluate_probabilities(y_true: pd.Series, raw_prob: np.ndarray, calibrated_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (calibrated_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "roc_auc_raw": float(roc_auc_score(y_true, raw_prob)),
        "roc_auc": float(roc_auc_score(y_true, calibrated_prob)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "brier_raw": float(np.mean((raw_prob - y_true) ** 2)),
        "brier": float(np.mean((calibrated_prob - y_true) ** 2)),
        "log_loss_raw": float(log_loss(y_true, np.clip(raw_prob, 1e-6, 1 - 1e-6))),
        "log_loss": float(log_loss(y_true, np.clip(calibrated_prob, 1e-6, 1 - 1e-6))),
        "threshold": float(threshold),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
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


def _capacity_limit(total_rows: int, ratio: float | None, absolute: int | None) -> int:
    candidates = []
    if ratio is not None:
        candidates.append(int(np.floor(total_rows * float(ratio))))
    if absolute is not None:
        candidates.append(int(absolute))
    if not candidates:
        return total_rows
    return max(0, min(max(candidates), total_rows))


def _apply_policy_engine(scored: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    x = scored.copy()
    economics = config["economics"]
    capacity = config["capacity"]
    base_rate = float(economics["balance_recovery_rate"])

    action_specs = {
        "High Priority (Agent Call)": {
            "multiplier": float(economics["agent_call_multiplier"]),
            "cost": float(economics["agent_call_cost"]),
            "max_count": _capacity_limit(len(x), capacity.get("max_agent_ratio"), capacity.get("max_agent_accounts")),
        },
        "Medium Priority (Auto-Dialer)": {
            "multiplier": float(economics["auto_dialer_multiplier"]),
            "cost": float(economics["auto_dialer_cost"]),
            "max_count": _capacity_limit(len(x), capacity.get("max_auto_ratio"), capacity.get("max_auto_accounts")),
        },
        "Low Priority (SMS/Email)": {
            "multiplier": float(economics["sms_email_multiplier"]),
            "cost": float(economics["sms_email_cost"]),
            "max_count": _capacity_limit(len(x), capacity.get("max_sms_ratio"), capacity.get("max_sms_accounts")),
        },
    }

    for action, spec in action_specs.items():
        action_key = action.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("-", "_")
        gross_col = f"gross_{action_key}"
        net_col = f"net_{action_key}"
        x[gross_col] = x["calibrated_repay_prob"] * x["balance_proxy"] * base_rate * spec["multiplier"]
        x[net_col] = x[gross_col] - spec["cost"]
        spec["gross_col"] = gross_col
        spec["net_col"] = net_col

    x["recommended_action"] = "Write-off / Ignore"
    x["expected_gross_recovery"] = 0.0
    x["expected_net_recovery"] = 0.0
    x["recommended_contact_cost"] = 0.0
    x["policy_rank"] = 0

    remaining = pd.Index(x.index)
    for action in ["High Priority (Agent Call)", "Medium Priority (Auto-Dialer)", "Low Priority (SMS/Email)"]:
        spec = action_specs[action]
        candidates = x.loc[remaining].sort_values(spec["net_col"], ascending=False)
        candidates = candidates[candidates[spec["net_col"]] > 0]
        assigned_idx = candidates.head(spec["max_count"]).index
        x.loc[assigned_idx, "recommended_action"] = action
        x.loc[assigned_idx, "expected_gross_recovery"] = x.loc[assigned_idx, spec["gross_col"]]
        x.loc[assigned_idx, "expected_net_recovery"] = x.loc[assigned_idx, spec["net_col"]]
        x.loc[assigned_idx, "recommended_contact_cost"] = spec["cost"]
        x.loc[assigned_idx, "policy_rank"] = np.arange(1, len(assigned_idx) + 1)
        remaining = remaining.difference(assigned_idx)

    x["recommended_action"] = pd.Categorical(x["recommended_action"], categories=ACTION_ORDER, ordered=True)
    x = x.sort_values(["recommended_action", "expected_net_recovery", "calibrated_repay_prob"], ascending=[True, False, False]).reset_index(drop=True)

    has_actual = TARGET_COLUMN in x.columns
    agg_map: dict[str, Any] = {
        "accounts": ("balance_proxy", "size"),
        "avg_calibrated_prob": ("calibrated_repay_prob", "mean"),
        "avg_raw_prob": ("raw_repay_prob", "mean"),
        "balance_proxy_total": ("balance_proxy", "sum"),
        "expected_gross_recovery_total": ("expected_gross_recovery", "sum"),
        "expected_net_recovery_total": ("expected_net_recovery", "sum"),
        "contact_cost_total": ("recommended_contact_cost", "sum"),
    }
    if has_actual:
        agg_map["actual_payer_rate"] = (TARGET_COLUMN, lambda s: (s == "Y").mean())
    queue_summary = x.groupby("recommended_action", as_index=False).agg(**agg_map)
    queue_summary["expected_roi"] = queue_summary.apply(
        lambda row: _safe_ratio(row["expected_net_recovery_total"], row["contact_cost_total"]),
        axis=1,
    )
    queue_summary = queue_summary.sort_values("recommended_action").reset_index(drop=True)

    totals = {
        "accounts": int(len(x)),
        "expected_gross_recovery_total": round(float(x["expected_gross_recovery"].sum()), 2),
        "expected_net_recovery_total": round(float(x["expected_net_recovery"].sum()), 2),
        "contact_cost_total": round(float(x["recommended_contact_cost"].sum()), 2),
        "expected_roi": round(_safe_ratio(x["expected_net_recovery"].sum(), x["recommended_contact_cost"].sum()), 4),
        "queue_mix": x["recommended_action"].astype(str).value_counts(dropna=False).to_dict(),
    }
    return x, queue_summary, totals


def _split_dev_sets(x_model: pd.DataFrame, y_model: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    x_train_full, x_valid, y_train_full, y_valid = train_test_split(
        x_model,
        y_model,
        test_size=0.20,
        random_state=42,
        stratify=y_model,
    )
    x_train, x_calib, y_train, y_calib = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.25,
        random_state=42,
        stratify=y_train_full,
    )
    return x_train, x_calib, x_valid, y_train, y_calib, y_valid


def _score_frame(
    frame: pd.DataFrame,
    raw_prob: np.ndarray,
    calibrated_prob: np.ndarray,
    threshold: float,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cols = [c for c in ["id", "data_type", "loan_type", "purchased_bal_gp", "district", TARGET_COLUMN] if c in frame.columns]
    scored = frame[cols].copy()
    scored["balance_proxy"] = frame["balance_proxy"].values
    scored["raw_repay_prob"] = raw_prob
    scored["calibrated_repay_prob"] = calibrated_prob
    scored["predicted_payer_flag"] = np.where(calibrated_prob >= threshold, "Y", "N")
    scored["expected_value_proxy"] = scored["calibrated_repay_prob"] * scored["balance_proxy"]
    scored, queue_summary, policy_totals = _apply_policy_engine(scored, config)
    return scored, queue_summary, policy_totals


def _fit_production_calibrator(base_pipeline: Pipeline, x_model: pd.DataFrame, y_model: pd.Series, folds: int) -> LogisticRegression:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    oof_prob = np.zeros(len(x_model), dtype=float)
    for train_idx, valid_idx in splitter.split(x_model, y_model):
        model = clone(base_pipeline)
        model.fit(x_model.iloc[train_idx], y_model.iloc[train_idx])
        oof_prob[valid_idx] = model.predict_proba(x_model.iloc[valid_idx])[:, 1]
    return _fit_platt_calibrator(oof_prob, y_model)


def preprocess_npa_data(file_path: str, output_dir: str | None = None, config_path: str | None = None) -> dict[str, Any]:
    src = _to_path(file_path)
    out_dir = _ensure_dir(output_dir or (src.parent / "agent_outputs" / "preprocess"))
    config = _load_config(config_path)
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
            "production_config": _to_builtin(config),
        }
    )

    preprocessed_full_path = out_dir / "preprocessed_full.csv"
    preprocessed_model_path = out_dir / "preprocessed_model.csv"
    preprocessed_test_path = out_dir / "preprocessed_test.csv"
    profile_path = out_dir / "profile.json"
    config_used_path = out_dir / "production_config_used.json"

    clean_df.to_csv(preprocessed_full_path, index=False, encoding="utf-8-sig")
    model_df.to_csv(preprocessed_model_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(preprocessed_test_path, index=False, encoding="utf-8-sig")
    profile_path.write_text(json.dumps(_to_builtin(profile), ensure_ascii=False, indent=2), encoding="utf-8")
    config_used_path.write_text(json.dumps(_to_builtin(config), ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "file_path": str(src),
        "output_dir": str(out_dir),
        "profile_path": str(profile_path),
        "production_config_path": str(config_used_path),
        "preprocessed_full_path": str(preprocessed_full_path),
        "preprocessed_model_path": str(preprocessed_model_path),
        "preprocessed_test_path": str(preprocessed_test_path),
        "profile": _to_builtin(profile),
    }


def _train_and_score(file_path: str, output_dir: str | None = None, config_path: str | None = None) -> dict[str, Any]:
    src = _to_path(file_path)
    out_dir = _ensure_dir(output_dir or (src.parent / "agent_outputs" / "training"))
    config = _load_config(config_path)

    raw_df = _load_excel(src)
    clean_df = _clean_frame(raw_df)
    model_df, test_df = _prepare_model_frames(clean_df)

    x_model = model_df.drop(columns=DROP_FOR_MODEL + ["target", "balance_proxy"])
    y_model = model_df["target"]
    x_test = test_df.drop(columns=DROP_FOR_MODEL + ["target", "balance_proxy"])
    y_test = test_df["target"]

    x_train, x_calib, x_valid, y_train, y_calib, y_valid = _split_dev_sets(x_model, y_model)
    candidate_pipelines = _build_model_pipelines(y_train)

    champion_rows: list[dict[str, Any]] = []
    candidate_metrics: dict[str, Any] = {}

    for name, pipeline in candidate_pipelines.items():
        pipeline.fit(x_train, y_train)
        raw_calib = pipeline.predict_proba(x_calib)[:, 1]
        calibrator = _fit_platt_calibrator(raw_calib, y_calib)
        raw_valid = pipeline.predict_proba(x_valid)[:, 1]
        calibrated_valid = _apply_platt_calibrator(calibrator, raw_valid)
        threshold_result = _find_best_threshold(y_valid, calibrated_valid)
        valid_metrics = _evaluate_probabilities(y_valid, raw_valid, calibrated_valid, threshold_result["threshold"])
        _, _, valid_policy_totals = _score_frame(
            frame=model_df.loc[x_valid.index],
            raw_prob=raw_valid,
            calibrated_prob=calibrated_valid,
            threshold=threshold_result["threshold"],
            config=config,
        )
        valid_metrics["valid_threshold_search"] = threshold_result
        valid_metrics["validation_policy"] = valid_policy_totals
        candidate_metrics[name] = valid_metrics
        champion_rows.append(
            {
                "model_name": name,
                "model_role": _model_role(name),
                "roc_auc": valid_metrics["roc_auc"],
                "brier": valid_metrics["brier"],
                "recall": valid_metrics["recall"],
                "precision": valid_metrics["precision"],
                "expected_net_recovery_total": valid_policy_totals["expected_net_recovery_total"],
                "expected_roi": valid_policy_totals["expected_roi"],
                "threshold": valid_metrics["threshold"],
            }
        )

    champion_df = pd.DataFrame(champion_rows).sort_values(
        ["expected_net_recovery_total", "roc_auc", "recall"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    best_name = str(champion_df.iloc[0]["model_name"])
    champion_df["model_role"] = champion_df["model_name"].map(lambda name: _model_role(str(name), best_name))
    best_threshold = float(candidate_metrics[best_name]["threshold"])

    final_pipeline = clone(_build_model_pipelines(y_model)[best_name])
    final_pipeline.fit(x_model, y_model)
    folds = int(config["calibration"].get("oof_folds", 5))
    final_calibrator = _fit_production_calibrator(final_pipeline, x_model, y_model, folds=folds)

    raw_test = final_pipeline.predict_proba(x_test)[:, 1]
    calibrated_test = _apply_platt_calibrator(final_calibrator, raw_test)
    test_metrics = _evaluate_probabilities(y_test, raw_test, calibrated_test, best_threshold)
    scored_test, queue_summary, policy_totals = _score_frame(
        frame=test_df,
        raw_prob=raw_test,
        calibrated_prob=calibrated_test,
        threshold=best_threshold,
        config=config,
    )

    baseline_threshold = float(candidate_metrics[BASELINE_MODEL_NAME]["threshold"])
    if best_name == BASELINE_MODEL_NAME:
        baseline_test_metrics = dict(test_metrics)
        baseline_policy_totals = dict(policy_totals)
    else:
        baseline_pipeline = clone(_build_model_pipelines(y_model)[BASELINE_MODEL_NAME])
        baseline_pipeline.fit(x_model, y_model)
        baseline_calibrator = _fit_production_calibrator(baseline_pipeline, x_model, y_model, folds=folds)
        baseline_raw_test = baseline_pipeline.predict_proba(x_test)[:, 1]
        baseline_calibrated_test = _apply_platt_calibrator(baseline_calibrator, baseline_raw_test)
        baseline_test_metrics = _evaluate_probabilities(y_test, baseline_raw_test, baseline_calibrated_test, baseline_threshold)
        _, _, baseline_policy_totals = _score_frame(
            frame=test_df,
            raw_prob=baseline_raw_test,
            calibrated_prob=baseline_calibrated_test,
            threshold=baseline_threshold,
            config=config,
        )

    holdout_comparison_rows = [
        {
            "model_name": best_name,
            "model_role": _model_role(best_name, best_name),
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
    if best_name != BASELINE_MODEL_NAME:
        holdout_comparison_rows.append(
            {
                "model_name": BASELINE_MODEL_NAME,
                "model_role": _model_role(BASELINE_MODEL_NAME, best_name),
                "roc_auc": baseline_test_metrics["roc_auc"],
                "brier": baseline_test_metrics["brier"],
                "log_loss": baseline_test_metrics["log_loss"],
                "recall": baseline_test_metrics["recall"],
                "precision": baseline_test_metrics["precision"],
                "expected_net_recovery_total": baseline_policy_totals["expected_net_recovery_total"],
                "expected_roi": baseline_policy_totals["expected_roi"],
                "threshold": baseline_test_metrics["threshold"],
            }
        )

    agent_vs_baseline = {
        "agent_model": best_name,
        "baseline_model": BASELINE_MODEL_NAME,
        "agent_matches_baseline": best_name == BASELINE_MODEL_NAME,
        "holdout_comparison": holdout_comparison_rows,
        "delta_agent_minus_baseline": {
            "roc_auc": float(test_metrics["roc_auc"] - baseline_test_metrics["roc_auc"]),
            "brier": float(test_metrics["brier"] - baseline_test_metrics["brier"]),
            "log_loss": float(test_metrics["log_loss"] - baseline_test_metrics["log_loss"]),
            "recall": float(test_metrics["recall"] - baseline_test_metrics["recall"]),
            "precision": float(test_metrics["precision"] - baseline_test_metrics["precision"]),
            "expected_net_recovery_total": float(policy_totals["expected_net_recovery_total"] - baseline_policy_totals["expected_net_recovery_total"]),
            "expected_roi": float(policy_totals["expected_roi"] - baseline_policy_totals["expected_roi"]),
        },
    }


    feature_importance = _feature_importance(final_pipeline, x_test, y_test).head(12)

    top20_n = max(int(len(scored_test) * 0.2), 1)
    top20_prob = scored_test.sort_values("calibrated_repay_prob", ascending=False).head(top20_n)
    top20_net = scored_test.sort_values("expected_net_recovery", ascending=False).head(top20_n)
    total_actual_payers = max(int((scored_test[TARGET_COLUMN] == "Y").sum()), 1)

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
        "champion_challenger": champion_df.to_dict(orient="records"),
        "model_selection": _to_builtin(candidate_metrics),
        "best_model": best_name,
        "baseline_model": BASELINE_MODEL_NAME,
        "test_metrics": _to_builtin(test_metrics),
        "baseline_test_metrics": _to_builtin(baseline_test_metrics),
        "policy_summary": _to_builtin(policy_totals),
        "baseline_policy_summary": _to_builtin(baseline_policy_totals),
        "agent_vs_baseline": _to_builtin(agent_vs_baseline),
        "queue_summary": _to_builtin(queue_summary.to_dict(orient="records")),
        "top_features": _to_builtin(feature_importance.to_dict(orient="records")),
        "concentration": {
            "top20_accounts": int(top20_n),
            "overall_actual_payer_rate_pct": _safe_pct((scored_test[TARGET_COLUMN] == "Y").mean()),
            "prob_top20_actual_payer_rate_pct": _safe_pct((top20_prob[TARGET_COLUMN] == "Y").mean()),
            "prob_top20_actual_payer_capture_share_pct": round(float((top20_prob[TARGET_COLUMN] == "Y").sum() / total_actual_payers) * 100, 2),
            "net_top20_actual_payer_rate_pct": _safe_pct((top20_net[TARGET_COLUMN] == "Y").mean()),
            "net_top20_actual_payer_capture_share_pct": round(float((top20_net[TARGET_COLUMN] == "Y").sum() / total_actual_payers) * 100, 2),
            "net_top20_expected_net_recovery_capture_share_pct": round(float(top20_net["expected_net_recovery"].sum() / max(scored_test["expected_net_recovery"].sum(), 1)) * 100, 2),
        },
    }


    model_bundle = {
        "pipeline": final_pipeline,
        "calibrator": final_calibrator,
        "best_model": best_name,
        "threshold": best_threshold,
        "categorical_features": CATEGORICAL,
        "numeric_features": NUMERIC,
        "drop_for_model": DROP_FOR_MODEL,
        "balance_proxy": BALANCE_PROXY,
        "config": _to_builtin(config),
        "metadata": _to_builtin(metrics),
    }

    holdout_comparison_df = pd.DataFrame(agent_vs_baseline["holdout_comparison"])

    model_path = out_dir / "npa_repayment_model.joblib"
    metrics_path = out_dir / "metrics.json"
    scored_test_path = out_dir / "test_scored_accounts.csv"
    queue_summary_path = out_dir / "production_queue_summary.csv"
    champion_summary_path = out_dir / "champion_challenger_summary.csv"
    holdout_comparison_path = out_dir / "agent_vs_baseline_summary.csv"
    feature_importance_path = out_dir / "feature_importance.csv"
    payer_rate_by_balance_path = out_dir / "payer_rate_by_balance.csv"
    payer_rate_by_loan_path = out_dir / "payer_rate_by_loan.csv"
    payer_rate_by_mobile_path = out_dir / "payer_rate_by_mobile.csv"
    config_used_path = out_dir / "production_config_used.json"
    report_path = out_dir / "collection_strategy_report.md"

    joblib.dump(model_bundle, model_path)
    metrics_path.write_text(json.dumps(_to_builtin(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    config_used_path.write_text(json.dumps(_to_builtin(config), ensure_ascii=False, indent=2), encoding="utf-8")
    scored_test.to_csv(scored_test_path, index=False, encoding="utf-8-sig")
    queue_summary.to_csv(queue_summary_path, index=False, encoding="utf-8-sig")
    champion_df.to_csv(champion_summary_path, index=False, encoding="utf-8-sig")
    holdout_comparison_df.to_csv(holdout_comparison_path, index=False, encoding="utf-8-sig")
    feature_importance.to_csv(feature_importance_path, index=False, encoding="utf-8-sig")
    payer_rate_by_balance.to_csv(payer_rate_by_balance_path, index=False, encoding="utf-8-sig")
    payer_rate_by_loan.to_csv(payer_rate_by_loan_path, index=False, encoding="utf-8-sig")
    payer_rate_by_mobile.to_csv(payer_rate_by_mobile_path, index=False, encoding="utf-8-sig")


    report_content = _build_report_markdown(
        metrics=metrics,
        queue_summary=queue_summary,
        champion_summary=champion_df,
        payer_rate_by_balance=payer_rate_by_balance,
        payer_rate_by_loan=payer_rate_by_loan,
        payer_rate_by_mobile=payer_rate_by_mobile,
    )
    report_path.write_text(report_content, encoding="utf-8")

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "queue_summary_path": str(queue_summary_path),
        "champion_summary_path": str(champion_summary_path),
        "holdout_comparison_path": str(holdout_comparison_path),
        "feature_importance_path": str(feature_importance_path),
        "config_used_path": str(config_used_path),
        "scored_test_path": str(scored_test_path),
        "report_path": str(report_path),
        "metrics": _to_builtin(metrics),
    }



def _build_report_markdown(
    metrics: dict[str, Any],
    queue_summary: pd.DataFrame,
    champion_summary: pd.DataFrame,
    payer_rate_by_balance: pd.DataFrame,
    payer_rate_by_loan: pd.DataFrame,
    payer_rate_by_mobile: pd.DataFrame,
) -> str:
    m = metrics
    tm = m["test_metrics"]
    baseline_tm = m["baseline_test_metrics"]
    cm = tm["confusion_matrix"]
    c = m["concentration"]
    config = m["production_config"]
    economics = config["economics"]
    ab = m["agent_vs_baseline"]
    delta = ab["delta_agent_minus_baseline"]
    holdout_comparison = pd.DataFrame(ab["holdout_comparison"])
    report: list[str] = []
    report.append("# NPA回款预测生产版策略报告")
    report.append("")
    report.append("## 1. Executive Summary")
    report.append(f"- 数据量：{m['data_overview']['rows']:,} 条，其中建模集(M) {m['data_overview']['model_rows']:,} 条，独立验证集(T) {m['data_overview']['test_rows']:,} 条。")
    report.append(f"- Champion 模型：**{m['best_model']}**。T集 ROC-AUC = **{tm['roc_auc']:.3f}**，Recall(Y) = **{_safe_pct(tm['recall']):.2f}%**，Precision(Y) = **{_safe_pct(tm['precision']):.2f}%**。")
    if ab["agent_matches_baseline"]:
        report.append(f"- 基线结论：**{m['baseline_model']}** 已成为当前 champion，说明更复杂的候选模型暂未在T集上打出更高业务价值。")
    else:
        report.append(
            f"- 对比基线：相对 **{m['baseline_model']}**，当前 Agent 在T集 ROC-AUC 变化 **{delta['roc_auc']:+.3f}**，Brier 变化 **{delta['brier']:+.4f}**（负值更好），预期净回收代理值变化 **{delta['expected_net_recovery_total']:+,.0f}**，ROI 变化 **{delta['expected_roi']:+.2f}x**。"
        )
    report.append(f"- 概率校准：采用 **Platt scaling**，Brier Score 从 **{tm['brier_raw']:.4f}** 改善到 **{tm['brier']:.4f}**，更适合直接用于产能分配和经济测算。")
    report.append(f"- 生产策略：在默认成本假设下，T集组合的**预期净回收代理值**为 **{m['policy_summary']['expected_net_recovery_total']:,.0f}**，预期ROI为 **{m['policy_summary']['expected_roi']:.2f}x**。")
    report.append(f"- 集中度：按校准后概率排序时，前20%账户覆盖 **{c['prob_top20_actual_payer_capture_share_pct']:.2f}%** 的真实付款账户；按净回收代理值排序时，前20%账户贡献 **{c['net_top20_expected_net_recovery_capture_share_pct']:.2f}%** 的预期净回收。")
    report.append("")
    report.append("## 2. Production Upgrade Highlights")
    report.append("- 在原有M/T分层建模基础上新增 `train / calibration / validation / holdout` 四层开发框架。")
    report.append("- 在模型评分之后加入概率校准，让分数更接近可执行的回收概率。")
    report.append("- 增加 Champion-Challenger 比较，不只看 AUC，也看校准后经济价值。")
    report.append("- 本版新增 Logistic Regression 基线模型，便于长期监控 Agent 是否真正跑赢简单可解释方案。")
    report.append("- 增加成本函数与产能约束，把模型输出直接转成坐席、自动外呼、短信三类队列。")
    report.append("- 通过配置文件管理经济假设，便于后续按市场/渠道/回收策略调整。")
    report.append("")
    report.append("## 3. Economic Assumptions")
    report.append(f"- Base recovery rate proxy：{_safe_pct(economics['balance_recovery_rate']):.2f}%")
    report.append(f"- Agent call cost：¥{economics['agent_call_cost']:.2f} / account")
    report.append(f"- Auto-dialer cost：¥{economics['auto_dialer_cost']:.2f} / account")
    report.append(f"- SMS/Email cost：¥{economics['sms_email_cost']:.2f} / account")
    report.append(f"- Agent channel multiplier：{economics['agent_call_multiplier']:.2f}x")
    report.append(f"- Auto-dialer multiplier：{economics['auto_dialer_multiplier']:.2f}x")
    report.append(f"- SMS/Email multiplier：{economics['sms_email_multiplier']:.2f}x")
    report.append("")
    report.append("## 4. Champion-Challenger Summary (Validation)")
    report.append("| Role | Model | Valid ROC-AUC | Valid Brier | Recall(Y) | Precision(Y) | Expected Net Recovery | Expected ROI | Threshold |")
    report.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in champion_summary.itertuples(index=False):
        report.append(
            f"| {row.model_role} | {row.model_name} | {row.roc_auc:.3f} | {row.brier:.4f} | {_safe_pct(row.recall):.2f}% | {_safe_pct(row.precision):.2f}% | {row.expected_net_recovery_total:,.0f} | {row.expected_roi:.2f}x | {row.threshold:.2f} |"
        )
    report.append("")
    report.append("## 5. Agent vs Baseline on Holdout (T set)")
    report.append("| Role | Model | ROC-AUC | Brier | LogLoss | Recall(Y) | Precision(Y) | Expected Net Recovery | Expected ROI | Threshold |")
    report.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in holdout_comparison.itertuples(index=False):
        report.append(
            f"| {row.model_role} | {row.model_name} | {row.roc_auc:.3f} | {row.brier:.4f} | {row.log_loss:.4f} | {_safe_pct(row.recall):.2f}% | {_safe_pct(row.precision):.2f}% | {row.expected_net_recovery_total:,.0f} | {row.expected_roi:.2f}x | {row.threshold:.2f} |"
        )
    if ab["agent_matches_baseline"]:
        report.append("- 当前基线已成为最终champion，因此后续重点不是追求更复杂模型，而是继续提升特征、经济假设和渠道执行质量。")
    else:
        report.append(f"- Agent 相对基线的净回收代理值差额：**{delta['expected_net_recovery_total']:+,.0f}**；ROI 差额：**{delta['expected_roi']:+.2f}x**。")
        report.append(f"- Agent 相对基线的 Recall(Y) 变化：**{delta['recall']:+.2%}**；Precision(Y) 变化：**{delta['precision']:+.2%}**。")
        report.append(f"- Agent 相对基线的 Brier 变化：**{delta['brier']:+.4f}**（负值更好）；LogLoss 变化：**{delta['log_loss']:+.4f}**（负值更好）。")
    report.append("")
    report.append("## 6. Champion Detailed Holdout Metrics")
    report.append(f"- ROC-AUC(raw / calibrated): **{tm['roc_auc_raw']:.3f} / {tm['roc_auc']:.3f}**")
    report.append(f"- Brier(raw / calibrated): **{tm['brier_raw']:.4f} / {tm['brier']:.4f}**")
    report.append(f"- LogLoss(raw / calibrated): **{tm['log_loss_raw']:.4f} / {tm['log_loss']:.4f}**")
    report.append(f"- Recall(Y): **{_safe_pct(tm['recall']):.2f}%**")
    report.append(f"- Precision(Y): **{_safe_pct(tm['precision']):.2f}%**")
    report.append(f"- Confusion Matrix @ threshold {tm['threshold']:.2f}: TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}")
    report.append(f"- Baseline 参照：ROC-AUC **{baseline_tm['roc_auc']:.3f}**，Brier **{baseline_tm['brier']:.4f}**，LogLoss **{baseline_tm['log_loss']:.4f}**。")
    report.append("")
    report.append("## 7. Top Predictive Features")
    for row in m["top_features"]:
        feature = row["feature"]
        report.append(f"- **{feature}**（importance={row['importance']:.4f}）：{BUSINESS_MAP.get(feature, '该变量对区分付款人与非付款人具有明显增益。')}")
    report.append("")
    report.append("## 8. Production Queue Summary")
    report.append("| Queue | Accounts | Avg Calibrated PD | Actual Payer Rate | Balance Proxy Total | Expected Gross Recovery | Expected Net Recovery | Contact Cost | ROI |")
    report.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in queue_summary.itertuples(index=False):
        actual_payer_rate = getattr(row, "actual_payer_rate", 0.0)
        report.append(
            f"| {row.recommended_action} | {int(row.accounts):,} | {_safe_pct(row.avg_calibrated_prob):.2f}% | {_safe_pct(actual_payer_rate):.2f}% | {row.balance_proxy_total:,.0f} | {row.expected_gross_recovery_total:,.0f} | {row.expected_net_recovery_total:,.0f} | {row.contact_cost_total:,.0f} | {row.expected_roi:.2f}x |"
        )
    report.append("")
    report.append("## 9. Action Recommendation")
    report.append("- **High Priority (Agent Call)**：优先给人工坐席，关注高校准概率、高余额、净回收最高的账户。")
    report.append("- **Medium Priority (Auto-Dialer)**：给自动外呼，承担规模化覆盖和低成本触达任务。")
    report.append("- **Low Priority (SMS/Email)**：仅保留极低成本数字化触达。")
    report.append("- **Write-off / Ignore**：若净回收为负或挤占产能，则直接放弃当前轮人工资源。")

    report.append("")
    report.append("## 10. Additional Portfolio Signals")
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
    report.append("")
    report.append("## 11. Deployment Notes")
    report.append("- 本报告中的净回收金额仍是基于余额代理值的运营口径，不是财务确认回款。")
    report.append("- 后续应持续用基线模型做回归测试：若复杂模型长期跑不赢基线，就该回到特征工程和经济假设，而不是继续堆模型。")
    report.append("- 若要真正投产，应把实际 settlement rate、通话成本、渠道转化率按市场/批次写入 config 后再跑。")
    report.append("- 当前模型文件已包含校准器与默认策略配置，可直接对新组合打分并给出推荐队列。")
    return "\n".join(report)



def train_repayment_model(file_path: str, output_dir: str | None = None, config_path: str | None = None) -> dict[str, Any]:
    return _train_and_score(file_path=file_path, output_dir=output_dir, config_path=config_path)


def optimize_collection_policy(file_path: str, model_path: str, output_dir: str | None = None, config_path: str | None = None) -> dict[str, Any]:
    src = _to_path(file_path)
    bundle_path = _to_path(model_path)
    out_dir = _ensure_dir(output_dir or (src.parent / "agent_outputs" / "policy_optimization"))

    bundle = joblib.load(bundle_path)
    pipeline: Pipeline = bundle["pipeline"]
    calibrator: LogisticRegression | None = bundle.get("calibrator")
    threshold = float(bundle["threshold"])
    config = bundle.get("config") or _load_config(config_path)
    if config_path:
        config = _load_config(config_path)

    raw_df = _load_excel(src)
    clean_df = _clean_frame(raw_df)
    x = clean_df.drop(columns=[c for c in DROP_FOR_MODEL + ["target", "balance_proxy"] if c in clean_df.columns])
    raw_prob = pipeline.predict_proba(x)[:, 1]
    calibrated_prob = _apply_platt_calibrator(calibrator, raw_prob)

    scored, queue_summary, policy_totals = _score_frame(
        frame=clean_df,
        raw_prob=raw_prob,
        calibrated_prob=calibrated_prob,
        threshold=threshold,
        config=config,
    )

    scored_path = out_dir / "policy_scored_accounts.csv"
    queue_summary_path = out_dir / "recommended_queue_summary.csv"
    config_used_path = out_dir / "production_config_used.json"
    summary_path = out_dir / "policy_optimization_summary.json"

    scored.to_csv(scored_path, index=False, encoding="utf-8-sig")
    queue_summary.to_csv(queue_summary_path, index=False, encoding="utf-8-sig")
    config_used_path.write_text(json.dumps(_to_builtin(config), ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "source_file": str(src),
        "model_path": str(bundle_path),
        "output_dir": str(out_dir),
        "threshold": threshold,
        "rows": int(len(scored)),
        "policy_summary": _to_builtin(policy_totals),
        "queue_summary_path": str(queue_summary_path),
        "scored_path": str(scored_path),
        "config_used_path": str(config_used_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def predict_repayment_probability(file_path: str, model_path: str, output_dir: str | None = None, config_path: str | None = None) -> dict[str, Any]:
    return optimize_collection_policy(file_path=file_path, model_path=model_path, output_dir=output_dir or None, config_path=config_path)


def build_collection_strategy_report(file_path: str, output_dir: str | None = None, config_path: str | None = None) -> dict[str, Any]:
    result = _train_and_score(file_path=file_path, output_dir=output_dir, config_path=config_path)
    return {
        "model_path": result["model_path"],
        "report_path": result["report_path"],
        "metrics_path": result["metrics_path"],
        "queue_summary_path": result["queue_summary_path"],
        "champion_summary_path": result["champion_summary_path"],
        "holdout_comparison_path": result["holdout_comparison_path"],
        "feature_importance_path": result["feature_importance_path"],
        "config_used_path": result["config_used_path"],
        "scored_test_path": result["scored_test_path"],
        "best_model": result["metrics"]["best_model"],
        "baseline_model": result["metrics"]["baseline_model"],
        "test_metrics": result["metrics"]["test_metrics"],
        "baseline_test_metrics": result["metrics"]["baseline_test_metrics"],
        "policy_summary": result["metrics"]["policy_summary"],
        "agent_vs_baseline": result["metrics"]["agent_vs_baseline"],
    }


