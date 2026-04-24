#!/usr/bin/env python3
"""
NPA Repayment Analyzer — Full Pipeline Entry Point
Usage:
  python run_full_workflow.py <data.xlsx> [--output-dir ./output] [--config config.json]
"""

from __future__ import annotations

import json, os, sys, time
from pathlib import Path

# ── Add skill directory to path for imports ──
SKILL_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SKILL_DIR.parent  # The npa-repayment-analyzer/ directory
for _p in [str(SKILL_DIR), str(SKILL_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.pipeline import build_collection_strategy_report


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_full_workflow.py <data.xlsx> [--output-dir DIR] [--config CONFIG.json]")
        sys.exit(1)

    file_path = Path(sys.argv[1]).resolve()
    output_dir = None
    config_path = None

    for i, arg in enumerate(sys.argv):
        if arg in ('--output-dir', '-o') and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
        elif arg in ('--config', '-c') and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    print("=" * 60)
    print("  NPA Repayment Analyzer — Full Pipeline")
    print("=" * 60)
    print(f"  Data file : {file_path.name} ({file_path.stat().st_size / 1024:.1f} KB)")
    print(f"  Output dir: {output_dir or 'auto'}")
    print(f"  Config    : {config_path or 'default'}")
    print()

    t0 = time.time()
    result = build_collection_strategy_report(
        file_path=str(file_path),
        output_dir=output_dir,
        config_path=config_path,
    )
    elapsed = time.time() - t0

    # Print summary
    m = result.get('metrics', {})
    tm = m.get('test_metrics', {})
    best = result.get('best_model', 'unknown')

    print()
    print("-" * 60)
    print("  PIPELINE COMPLETE")
    print("-" * 60)
    print(f"  Champion model     : {best}")
    print(f"  Test ROC-AUC        : {tm.get('roc_auc', 0):.4f}")
    print(f"  Test Brier Score    : {tm.get('brier', 0):.4f}")
    print(f"  Test Recall(Y)      : {tm.get('recall', 0)*100:.2f}%")
    print(f"  Net Recovery (T)   : {m.get('policy_summary', {}).get('expected_net_recovery_total', 0):,.0f}")
    print(f"  Expected ROI        : {m.get('policy_summary', {}).get('expected_roi', 0):.2f}x")
    print(f"  Elapsed             : {elapsed:.1f}s")
    print()
    print("  Output files:")
    for key in ['model_path', 'metrics_path', 'scored_test_path',
                'report_path', 'queue_summary_path', 'champion_summary_path']:
        val = result.get(key, '')
        if val:
            fname = os.path.basename(val)
            sz = os.path.getsize(val) / 1024 if os.path.exists(val) else 0
            print(f"    {fname:<40s} ({sz:>6.1f} KB)")

    print()
    print("Next step:")
    print(f"  python {SKILL_DIR / 'generate_dashboard.py'}")

    return result


if __name__ == '__main__':
    main()
