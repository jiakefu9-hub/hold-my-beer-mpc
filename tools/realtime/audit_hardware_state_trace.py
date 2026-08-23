#!/usr/bin/env python3
"""Audit a persisted state-only trace; never changes verification flags."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from right_arm_runtime.hardware_state_replay import audit_state_trace_files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument(
        "--source-kind",
        required=True,
        choices=("unverified_real_capture", "synthetic_test_fixture"),
    )
    parser.add_argument("--bridge-summary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_state_trace_files(
        args.trace,
        source_kind=args.source_kind,
        bridge_summary_path=args.bridge_summary,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(encoded)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
