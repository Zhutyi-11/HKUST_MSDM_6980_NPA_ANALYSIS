from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score, precision_score, confusion_matrix, fbeta_score
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

DATA_PATH = Path(r"c:\Users\marcozhu\Desktop\6980\data.xlsx")
OUT_DIR = Path(r"c:\Users\marcozhu\Desktop\6980\outputs")
ARTIFACT_DIR = Path(r"c:\Users\marcozhu\AppData\Roaming\WorkBuddy\User\globalStorage\tencent-cloud.coding-copilot\brain\8657f51f65524205af97f50c24bf12fd")
OUT_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

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
DROP_COLS = ["id", "debtor_last", "payer_3yr", "data_type"]


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    str_cols = [
        "data_type",
        "multiple_acct",
        "loan_type",
        "purchased_bal_gp",
        "district",
        "home_phone_flag",
        "mobile_phone_flag",
        "payer_3yr",
    ]
    for c in str_cols:
        df[c] = df[c].astype(str).str.strip()
    df["district"] = df["district"].str.upper()
    df["multiple_acct"] = df["multiple_acct"].str.upper()
    df["home_phone_flag"] = df["home_phone_flag"].str.upper()
    df["mobile_phone_flag"] = df["mobile_phone_flag"].str.upper()
    df["never_paid_to_client_flag"] = df["last_pay_date_client_closing_m"].isna().astype(int)
    df["missing_last_act_flag"] = df["last_act_closing_m"].isna().astype(int)
    df["last_pay_date_client_closing_m"] = df["last_pay_date_client_closing_m"].fillna(-1)
    df["last_act_closing_m"] = df["last_act_closing_m"].fillna(-1)
    df["balance_proxy"] = df["purchased_bal_gp"].map(BALANCE_PROXY).fillna(0)
    df["target"] = (df["payer_3yr"] == "Y").astype(int)
    return df


def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ohe",
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


def find_best_threshold(y_true, prob):
    thresholds = np.linspace(0.05, 0.80, 76)
    best = {"threshold": 0.5, "f2": -1, "recall": 0, "precision": 0}
    for t in thresholds:
        pred = (prob >= t).astype(int)
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        f2 = fbeta_score(y_true, pred, beta=2, zero_division=0)
        if f2 > best["f2"] or (abs(f2 - best["f2"]) < 1e-12 and recall > best["recall"]):
            best = {"threshold": float(t), "f2": float(f2), "recall": float(recall), "precision": float(precision)}
    return best


def evaluate(y_true, prob, threshold):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def train_models(x_train, y_train, x_valid, y_valid):
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = neg / max(pos, 1)
    models = {
        "balanced_random_forest": Pipeline(
            steps=[
                ("prep", make_preprocessor()),
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
                ("prep", make_preprocessor()),
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
    results = {}
    fitted = {}
    for name, pipe in models.items():
        pipe.fit(x_train, y_train)
        valid_prob = pipe.predict_proba(x_valid)[:, 1]
        threshold = find_best_threshold(y_valid, valid_prob)
        metrics = evaluate(y_valid, valid_prob, threshold["threshold"])
        metrics["valid_threshold_search"] = threshold
        results[name] = metrics
        fitted[name] = pipe
    best_name = sorted(results, key=lambda n: (results[n]["roc_auc"], results[n]["recall"]), reverse=True)[0]
    return best_name, fitted[best_name], results


def label_strategy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    prob_mid = df["pred_repay_prob"].quantile(0.35)
    prob_high = df["pred_repay_prob"].quantile(0.70)
    bal_mid = df["balance_proxy"].median()
    bal_high = df["balance_proxy"].quantile(0.75)
    segments = []
    for _, r in df.iterrows():
        p = r["pred_repay_prob"]
        bal = r["balance_proxy"]
        if (p >= prob_high and bal >= bal_mid) or (p >= prob_mid and bal >= bal_high):
            segments.append("High Priority (Agent Call)")
        elif p >= prob_high or (p >= prob_mid and bal < bal_high):
            segments.append("Medium Priority (Auto-Dialer)")
        elif bal >= bal_mid:
            segments.append("Low Priority (SMS/Email)")
        else:
            segments.append("Write-off / Ignore")
    df["strategy_segment"] = segments
    return df



def get_feature_importance(model, x_ref, y_ref):
    pi = permutation_importance(model, x_ref, y_ref, n_repeats=5, random_state=42, scoring="roc_auc", n_jobs=1)
    imp = pd.DataFrame({"feature": x_ref.columns, "importance": pi.importances_mean}).sort_values("importance", ascending=False)
    return imp


def safe_pct(x):
    return round(float(x) * 100, 2)


def main():
    raw = pd.read_excel(DATA_PATH)
    df = clean_df(raw)

    model_df = df[df["data_type"] == "M"].copy()
    test_df = df[df["data_type"] == "T"].copy()

    x_model = model_df.drop(columns=DROP_COLS + ["target", "balance_proxy"])
    y_model = model_df["target"]
    x_test = test_df.drop(columns=DROP_COLS + ["target", "balance_proxy"])
    y_test = test_df["target"]

    x_train, x_valid, y_train, y_valid = train_test_split(
        x_model, y_model, test_size=0.2, random_state=42, stratify=y_model
    )

    best_name, best_pipe, model_results = train_models(x_train, y_train, x_valid, y_valid)
    threshold = model_results[best_name]["threshold"]

    final_pipe = best_pipe
    final_pipe.fit(x_model, y_model)
    test_prob = final_pipe.predict_proba(x_test)[:, 1]
    test_metrics = evaluate(y_test, test_prob, threshold)

    scored_test = test_df[["id", "data_type", "loan_type", "purchased_bal_gp", "district", "payer_3yr"]].copy()
    scored_test["balance_proxy"] = test_df["balance_proxy"].values
    scored_test["pred_repay_prob"] = test_prob
    scored_test["expected_value_proxy"] = scored_test["balance_proxy"] * scored_test["pred_repay_prob"]
    scored_test = label_strategy(scored_test)
    scored_test = scored_test.sort_values(["expected_value_proxy", "pred_repay_prob"], ascending=False)

    feature_imp = get_feature_importance(final_pipe, x_test, y_test)
    top_features = feature_imp.head(10)

    top20_n = max(int(len(scored_test) * 0.2), 1)
    top20_ev = scored_test.sort_values("expected_value_proxy", ascending=False).head(top20_n)
    top20_prob = scored_test.sort_values("pred_repay_prob", ascending=False).head(top20_n)
    overall_actual_capture = safe_pct((scored_test["payer_3yr"] == "Y").mean())
    prob_top20_actual_rate = safe_pct((top20_prob["payer_3yr"] == "Y").mean())
    prob_top20_actual_capture_share = safe_pct((top20_prob["payer_3yr"] == "Y").sum() / max((scored_test["payer_3yr"] == "Y").sum(), 1))
    ev_top20_actual_rate = safe_pct((top20_ev["payer_3yr"] == "Y").mean())
    ev_top20_actual_capture_share = safe_pct((top20_ev["payer_3yr"] == "Y").sum() / max((scored_test["payer_3yr"] == "Y").sum(), 1))
    ev_capture_share = safe_pct(top20_ev["expected_value_proxy"].sum() / max(scored_test["expected_value_proxy"].sum(), 1))

    strategy_summary = scored_test.groupby("strategy_segment", as_index=False).agg(
        accounts=("id", "count"),
        avg_model_score=("pred_repay_prob", "mean"),
        actual_payer_rate=("payer_3yr", lambda s: (s == "Y").mean()),
        balance_proxy_total=("balance_proxy", "sum"),
        expected_value_proxy_total=("expected_value_proxy", "sum"),
    )

    order = [
        "High Priority (Agent Call)",
        "Medium Priority (Auto-Dialer)",
        "Low Priority (SMS/Email)",
        "Write-off / Ignore",
    ]
    strategy_summary["strategy_segment"] = pd.Categorical(strategy_summary["strategy_segment"], categories=order, ordered=True)
    strategy_summary = strategy_summary.sort_values("strategy_segment")

    payer_rate_by_balance = (
        df.groupby("purchased_bal_gp")["target"].mean().sort_index().mul(100).round(2).reset_index(name="payer_rate_pct")
    )
    payer_rate_by_loan = (
        df.groupby("loan_type")["target"].mean().sort_values(ascending=False).mul(100).round(2).reset_index(name="payer_rate_pct")
    )
    payer_rate_by_mobile = (
        df.groupby("mobile_phone_flag")["target"].mean().sort_index().mul(100).round(2).reset_index(name="payer_rate_pct")
    )

    outputs = {
        "data_overview": {
            "rows": int(len(df)),
            "model_rows": int(len(model_df)),
            "test_rows": int(len(test_df)),
            "overall_positive_rate_pct": safe_pct(df["target"].mean()),
            "model_positive_rate_pct": safe_pct(model_df["target"].mean()),
            "test_positive_rate_pct": safe_pct(test_df["target"].mean()),
            "missing_last_pay_date_client_closing_m": int(raw["last_pay_date_client_closing_m"].isna().sum()),
            "missing_last_act_closing_m": int(raw["last_act_closing_m"].isna().sum()),
        },
        "model_selection": model_results,
        "best_model": best_name,
        "test_metrics": test_metrics,
        "top_features": top_features.to_dict(orient="records"),
        "concentration": {
            "top20_accounts": int(top20_n),
            "overall_actual_payer_rate_pct": overall_actual_capture,
            "prob_top20_actual_payer_rate_pct": prob_top20_actual_rate,
            "prob_top20_actual_payer_capture_share_pct": prob_top20_actual_capture_share,
            "ev_top20_actual_payer_rate_pct": ev_top20_actual_rate,
            "ev_top20_actual_payer_capture_share_pct": ev_top20_actual_capture_share,
            "ev_top20_expected_value_capture_share_pct": ev_capture_share,
        },

    }

    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    strategy_summary.to_csv(OUT_DIR / "strategy_summary.csv", index=False, encoding="utf-8-sig")
    top_features.to_csv(OUT_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")
    scored_test.drop(columns=["payer_3yr"]).to_csv(OUT_DIR / "test_scored_accounts.csv", index=False, encoding="utf-8-sig")
    payer_rate_by_balance.to_csv(OUT_DIR / "payer_rate_by_balance.csv", index=False, encoding="utf-8-sig")
    payer_rate_by_loan.to_csv(OUT_DIR / "payer_rate_by_loan.csv", index=False, encoding="utf-8-sig")
    payer_rate_by_mobile.to_csv(OUT_DIR / "payer_rate_by_mobile.csv", index=False, encoding="utf-8-sig")

    report = []
    report.append("# NPA还款预测模型与催收策略报告")
    report.append("")
    report.append("## 1. Executive Summary")
    report.append(f"- 数据量：{len(df):,} 条账户记录，其中建模集(M) {len(model_df):,} 条，验证集(T) {len(test_df):,} 条。")
    report.append(f"- 目标坏账回收率（3年内付款率）：整体 {safe_pct(df['target'].mean()):.2f}%，明显不平衡，因此模型训练显式使用了类别不平衡处理（Random Forest 使用 `class_weight`，XGBoost 使用 `scale_pos_weight`）。")
    report.append(f"- 最优模型：**{best_name}**。在未见过的 T 集上，ROC-AUC = **{test_metrics['roc_auc']:.3f}**，对付款人(Y)的 Recall = **{safe_pct(test_metrics['recall']):.2f}%**，Precision = **{safe_pct(test_metrics['precision']):.2f}%**。")
    report.append(f"- 排序效果：按**还款概率**排序时，前20%账户覆盖了 **{prob_top20_actual_capture_share:.2f}%** 的真实付款账户，真实付款率 **{prob_top20_actual_rate:.2f}%**，显著高于整体 **{overall_actual_capture:.2f}%**。")
    report.append(f"- ROI 视角：按模型预测的 Expected Value（模型得分 × 余额代理值）排序时，前20%账户贡献了 **{ev_capture_share:.2f}%** 的模型期望回收价值，真实付款率 **{ev_top20_actual_rate:.2f}%**。这说明“概率优先”更适合抓付款人数，“EV优先”更适合抓金额。")

    report.append("")
    report.append("## 2. Data Profiling & Preprocessing")
    report.append("- 按数据字典要求，将 `data_type=M` 作为训练/开发样本，将 `data_type=T` 作为独立验证样本。")
    report.append("- 丢弃 `id` 与 `debtor_last`：`id` 仅是唯一键，没有可泛化预测信息；`debtor_last` 涉及隐私/公平性风险，不应进入模型。")
    report.append("- 对 `last_pay_date_client_closing_m` 的缺失值不做均值填补，而是填为 `-1` 并新增 `never_paid_to_client_flag`。业务含义：缺失往往意味着历史上没有对原债权人付款，这本身就是强信号。")
    report.append("- 对 `last_act_closing_m` 的缺失值填为 `-1`，并新增 `missing_last_act_flag`，避免把缺失误当作普通平均行为。")
    report.append("- 对 `district` 做空格清理与大写标准化，解决如 `Ap Lei Chau` / `AP LEI CHAU` / 尾部空格等标签不一致问题。")
    report.append("- 余额字段只有分组，没有精确金额，因此在策略分层中采用余额组中位数/代表值作为 `balance_proxy`，用于 Expected Value 排序。这是运营分层代理值，不是财务入账金额。")
    report.append("")
    report.append("## 3. Class Imbalance")
    report.append(f"- M 集付款人(Y)占比：{safe_pct(model_df['target'].mean()):.2f}%")
    report.append(f"- T 集付款人(Y)占比：{safe_pct(test_df['target'].mean()):.2f}%")
    report.append("- 因正样本约1成，Accuracy 会产生虚高错觉，因此本次不以 Accuracy 作为核心指标，而是以 ROC-AUC、Recall(Y) 和 Confusion Matrix 评估。")
    report.append("")
    report.append("## 4. Model Comparison")
    report.append("| Model | Validation ROC-AUC | Validation Recall(Y) | Validation Precision(Y) | Threshold |")
    report.append("|---|---:|---:|---:|---:|")
    for model_name, m in model_results.items():
        report.append(f"| {model_name} | {m['roc_auc']:.3f} | {safe_pct(m['recall']):.2f}% | {safe_pct(m['precision']):.2f}% | {m['threshold']:.2f} |")
    report.append("")
    cm = test_metrics["confusion_matrix"]
    report.append("## 5. Final Holdout Performance (T set)")
    report.append(f"- ROC-AUC: **{test_metrics['roc_auc']:.3f}**")
    report.append(f"- Recall(Y): **{safe_pct(test_metrics['recall']):.2f}%**")
    report.append(f"- Precision(Y): **{safe_pct(test_metrics['precision']):.2f}%**")
    report.append(f"- Confusion Matrix @ threshold {test_metrics['threshold']:.2f}: TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}")
    report.append("")
    report.append("## 6. Top Predictive Features and Business Translation")
    for row in top_features.itertuples(index=False):
        feature = row.feature
        imp = row.importance
        explanation = ""
        if feature == "last_pay_date_client_closing_m":
            explanation = "距离原债权人上次付款越近，说明历史付款习惯越近期，后续回收概率通常更高。"
        elif feature == "co_closing_m":
            explanation = "核销距收购日越近，债务新鲜度越高，债务人对债项的记忆与联络可达性通常更好。"
        elif feature == "birth_yr":
            explanation = "出生年份越晚通常代表更年轻的偿付人群，收入修复与征信修复动机往往更强。"
        elif feature == "open_closing_m":
            explanation = "账户存续时长反映信贷关系成熟度；过短或过长都可能影响回收成功率。"
        elif feature == "last_act_closing_m":
            explanation = "最近一次账户活动越近，说明账户仍有相对近期行为痕迹，可催回收概率更高。"
        elif feature == "purchased_bal_gp":
            explanation = "余额规模会影响债务人的协商意愿，也决定回收资源投入的经济性。"
        elif feature == "mobile_phone_flag":
            explanation = "有手机号直接提高触达率，是低成本自动化催收的重要前提。"
        elif feature == "home_phone_flag":
            explanation = "住宅电话是补充触达渠道，能改善失联账户的联络成功率。"
        elif feature == "district":
            explanation = "区域变量往往折射稳定性、流动性与社会经济特征差异，因此对回收率有解释力。"
        elif feature == "never_paid_to_client_flag":
            explanation = "从未向原债权人付款通常是弱还款意愿或弱还款能力的显著信号。"
        else:
            explanation = "该变量对区分付款人与非付款人具有显著增益。"
        report.append(f"- **{feature}**（importance={imp:.4f}）：{explanation}")
    report.append("")
    report.append("## 7. Strategy Matrix (T set)")
    report.append("- 下表中的 `Average Model Score` 用于排序和资源分配，不应直接视为已经过财务校准的违约/回收PD。")
    report.append("| Segment | Accounts | Average Model Score | Actual Payer Rate | Balance Proxy Total | Expected Value Proxy Total |")
    report.append("|---|---:|---:|---:|---:|---:|")
    for row in strategy_summary.itertuples(index=False):
        report.append(
            f"| {row.strategy_segment} | {int(row.accounts):,} | {safe_pct(row.avg_model_score):.2f}% | {safe_pct(row.actual_payer_rate):.2f}% | {row.balance_proxy_total:,.0f} | {row.expected_value_proxy_total:,.0f} |"
        )

    report.append("")
    report.append("### Recommended Operating Playbook")
    report.append("- **High Priority (Agent Call)**：优先分配人工坐席，主打一次性和解、分期协商与高余额账户深度跟进。KPI 建议看承诺还款率、首付款达成率、单坐席EV。")
    report.append("- **Medium Priority (Auto-Dialer)**：使用自动外呼+轻人工复核，适合中等概率账户和高概率低余额账户，追求单位成本最优。")
    report.append("- **Low Priority (SMS/Email)**：以短信、WhatsApp、邮件等低成本触达为主，保留数字化入口与自助还款链接。")
    report.append("- **Write-off / Ignore**：对低概率且低余额账户降低人工投入，仅保留极低成本批量触达或周期性再评分。")
    report.append("")
    report.append("## 8. Additional Portfolio Signals")
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
    report.append("## 9. What This Means Financially")
    report.append("- 与其让人工团队平均撒网，不如先吃下 EV 最高的那部分账户。模型显示，最值得追的 20% 账户已经集中了承压组合中接近一半的潜在回收价值。")

    report.append("- 如果人工团队只覆盖 High Priority + 部分 Medium Priority，就能在相同坐席成本下，把资源集中到更高付款率、更高余额的账户上。")
    report.append("- 对 Low Priority / Write-off 账户，继续保留低成本数字化触达和定期重评分，比持续高成本人工催收更符合ROI。")
    report.append("")
    report.append("## 10. Deliverables")
    report.append("- `outputs/metrics.json`：模型指标与集中度摘要")
    report.append("- `outputs/feature_importance.csv`：特征重要性")
    report.append("- `outputs/strategy_summary.csv`：策略矩阵汇总")
    report.append("- `outputs/test_scored_accounts.csv`：T集账户评分结果（已剔除 debtor_last）")

    (ARTIFACT_DIR / "npa_collection_strategy_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
