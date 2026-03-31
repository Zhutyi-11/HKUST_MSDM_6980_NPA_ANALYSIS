from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from npa_repayment_agent.pipeline import build_collection_strategy_report



def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full NPA repayment workflow and generate a collection strategy report.")
    parser.add_argument("file_path", help="Path to the source Excel file")
    parser.add_argument("--output-dir", default="", help="Optional output directory")
    parser.add_argument("--config-path", default="", help="Optional JSON file with production economics and capacity assumptions")
    args = parser.parse_args()

    result = build_collection_strategy_report(
        file_path=args.file_path,
        output_dir=args.output_dir or None,
        config_path=args.config_path or None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))



if __name__ == "__main__":
    main()
