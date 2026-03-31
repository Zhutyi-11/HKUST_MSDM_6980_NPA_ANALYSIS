---
name: npa-repayment-agent
description: This skill should be used when the task involves unsecured debt, credit card, or NPA repayment prediction, historical collection analysis, portfolio scoring, or translating model outputs into ROI-driven collection strategies. Trigger especially when a user provides portfolio files such as .xlsx/.csv and asks for repayment modeling, probability scoring, collection segmentation, or collection strategy reports.
---

# NPA Repayment Agent

## Overview

Standardize unsecured debt/NPA repayment modeling into an explainable, ROI-oriented workflow. Use the bundled MCP service to preprocess data, train repayment models, score new portfolios, and produce collection strategy outputs that a risk director or collection operations lead can act on.

## Workflow Decision Tree

- If the user provides a historical labeled portfolio and wants a full analysis, call `build_collection_strategy_report_tool` first.
- If the user wants to inspect data quality or validate assumptions before modeling, call `preprocess_npa_data_tool` first.
- If the user wants a reusable production model, call `train_repayment_model_tool`, save the returned model path, then use `predict_repayment_probability_tool` for future pools.
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

### 3. Enforce model evaluation policy

Judge model quality by:

- ROC-AUC
- Recall for the payer class (`Y`)
- Precision for the payer class
- Confusion matrix

Do not present Accuracy as the primary metric for this use case because the class distribution is typically highly imbalanced.

### 4. Translate model output into collection action

Convert scores into an operating matrix rather than stopping at probabilities:

- `High Priority (Agent Call)` for high-probability and/or high-EV accounts that deserve human collectors.
- `Medium Priority (Auto-Dialer)` for decent score but lower-touch economics.
- `Low Priority (SMS/Email)` for low-cost digital outreach.
- `Write-off / Ignore` for low score and low balance accounts where incremental labor is not justified.

### 5. Write the management summary

Structure the final summary in business language:

- How concentrated the recoverable value is
- Which top features matter and why
- Which queue should receive collector capacity first
- What can be deprioritized without materially hurting recovery economics

## Tool Usage Notes

### `build_collection_strategy_report_tool`

Use for one-shot delivery. Expect report markdown, metrics JSON, scored test accounts, strategy summary, and feature importance outputs.

### `preprocess_npa_data_tool`

Use when the user questions missing values, field quality, data split logic, or needs a clean data extract before modeling.

### `train_repayment_model_tool`

Use when the user wants a reusable model artifact. Save the returned `model_path` and mention it explicitly in the final handoff.

### `predict_repayment_probability_tool`

Use after a trained model already exists. Always explain that the output contains a model score and an EV proxy, not guaranteed cash collections.

## Files and References

- Read `references/workflow.md` for the detailed operating playbook and expected deliverables.
- Use `scripts/run_full_workflow.py` when a local CLI execution path is preferable to MCP invocation.

## Output Rules

- Never expose `debtor_last` or row-level private commentary in business summaries.
- Present grouped findings and operating recommendations.
- Keep the final answer concise, direct, and ROI-oriented.
- Explicitly distinguish between probability-oriented ranking and EV-oriented ranking when both are shown.
