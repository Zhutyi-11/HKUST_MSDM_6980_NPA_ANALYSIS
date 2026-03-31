---
name: npa-repayment-agent
description: This skill should be used when the task involves unsecured debt, credit card, or NPA repayment prediction, historical collection analysis, portfolio scoring, production policy optimization, or translating model outputs into ROI-driven collection strategies. Trigger especially when a user provides portfolio files such as .xlsx/.csv and asks for repayment modeling, probability calibration, collection segmentation, queue optimization, or collection strategy reports.
---

# NPA Repayment Agent

## Overview

Standardize unsecured debt/NPA repayment modeling into a production-oriented, explainable workflow. Use the bundled MCP service to preprocess data, train calibrated repayment models, compare champion-challenger candidates, optimize queue allocation under cost assumptions, and produce collection strategy outputs that a risk director or collection operations lead can act on.

## Workflow Decision Tree

- If the user provides a historical labeled portfolio and wants a full management deliverable, call `build_collection_strategy_report_tool` first.
- If the user wants to inspect data quality or validate assumptions before modeling, call `preprocess_npa_data_tool` first.
- If the user wants a reusable production model, call `train_repayment_model_tool`, save the returned `model_path`, then use `predict_repayment_probability_tool` for future pools.
- If the user already has a model but wants to change economics or capacity assumptions, call `optimize_collection_policy_tool` with a `config_path`.
- If the user only wants scoring for a new pool, require a trained `model_path` and call `predict_repayment_probability_tool`.

## Standard Operating Workflow

### 1. Confirm the modeling objective

Interpret the default target as `payer_3yr`, meaning whether the debtor paid within three years after deal closing. Preserve this target unless the user explicitly defines another label.

### 2. Enforce preprocessing policy

Apply the following rules consistently:

- Exclude `id` and `debtor_last` from modeling.
- Treat missing `last_pay_date_client_closing_m` as a business signal for “never paid original creditor”; encode with `-1` plus `never_paid_to_client_flag`.
- Treat missing `last_act_closing_m` as unknown recent activity; encode with `-1` plus `missing_last_act_flag`.
- Normalize `district` text before modeling.
- Split development and validation strictly by `data_type=M/T` when available.

### 3. Enforce production-grade model evaluation policy

Judge model quality by both predictive power and deployment usefulness:

- ROC-AUC
- Recall for the payer class (`Y`)
- Precision for the payer class
- Confusion matrix
- Brier score
- Log loss
- Expected net recovery under configured economics

Do not present Accuracy as the primary metric for this use case because the class distribution is typically highly imbalanced.

### 4. Calibrate probabilities before using them operationally

Do not treat raw model scores as production probabilities. Use the calibrated probability output when assigning collector queues, estimating expected recovery, or comparing channel economics.

### 5. Translate model output into collection action

Convert calibrated scores into an operating matrix rather than stopping at probabilities:

- `High Priority (Agent Call)` for high expected net recovery accounts that deserve scarce human collectors.
- `Medium Priority (Auto-Dialer)` for accounts that remain attractive under lower-touch economics.
- `Low Priority (SMS/Email)` for low-cost digital outreach.
- `Write-off / Ignore` for accounts whose expected net recovery is negative or not worth current capacity.

### 6. Write the management summary

Structure the final summary in business language:

- Which model is champion and why
- Whether calibration improved deployment quality
- How recoverable value is concentrated
- Which queue should receive collector capacity first
- What can be deprioritized without materially hurting recovery economics

## Tool Usage Notes

### `build_collection_strategy_report_tool`

Use for one-shot production delivery. Expect report markdown, calibrated metrics, champion-challenger summary, queue summary, scored accounts, and feature importance outputs.

### `preprocess_npa_data_tool`

Use when the user questions missing values, field quality, data split logic, or wants the production configuration snapshot before training.

### `train_repayment_model_tool`

Use when the user wants a reusable production model artifact. Save the returned `model_path` and mention it explicitly in the final handoff.

### `predict_repayment_probability_tool`

Use after a trained model already exists. Explain that the output contains calibrated repayment probability, expected net recovery proxy, and recommended queue.

### `optimize_collection_policy_tool`

Use when the model remains fixed but business assumptions change, such as channel cost, settlement rate, or collector capacity.

## Files and References

- Read `references/workflow.md` for the detailed operating playbook, config assumptions, and expected deliverables.
- Use `scripts/run_full_workflow.py` when a local CLI execution path is preferable to MCP invocation.

## Output Rules

- Never expose `debtor_last` or row-level private commentary in business summaries.
- Present grouped findings and operating recommendations.
- Keep the final answer concise, direct, and ROI-oriented.
- Explicitly distinguish among raw score, calibrated probability, and economic value.
- When `config_path` is provided, state which economic assumptions drove the queue recommendation.
