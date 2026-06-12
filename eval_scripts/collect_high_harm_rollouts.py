"""
Collect high-harm baseline rollouts into one JSON file.

This script reads Llama and Qwen baseline JSONL files and saves all rollouts with
score > threshold into a single JSON file with separate top-level sections:

{
  "threshold": 0.2,
  "llama": [...],
  "qwen": [...]
}

Usage:
  python collect_high_harm_rollouts.py \
    --llama-file llama_baseline.jsonl \
    --qwen-file qwen_baseline.jsonl \
    --threshold 0.2 \
    --output-file high_harm_rollouts_gt_0p2.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_high_harm_rollouts(path: str, threshold: float) -> List[Dict[str, Any]]:
    high_harm: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping malformed JSON line {line_idx} in {path}: {e}")
                continue

            score = record.get("score", None)
            if score is None:
                continue

            try:
                score_f = float(score)
            except (TypeError, ValueError):
                continue

            if score_f > threshold:
                # Keep original record, but add source line metadata for traceability.
                record = dict(record)
                record["_source_file"] = path
                record["_source_line"] = line_idx
                high_harm.append(record)

    high_harm.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
    return high_harm


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = [float(r["score"]) for r in records if r.get("score") is not None]

    if not scores:
        return {
            "count": 0,
            "max_score": None,
            "mean_score": None,
            "min_score": None,
        }

    return {
        "count": len(scores),
        "max_score": max(scores),
        "mean_score": sum(scores) / len(scores),
        "min_score": min(scores),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-file", type=str, required=True)
    parser.add_argument("--qwen-file", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--output-file", type=str, default="high_harm_rollouts_gt_0p2.json")
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Optional: keep only the top N high-harm rollouts per model.",
    )

    args = parser.parse_args()

    llama_records = load_high_harm_rollouts(args.llama_file, args.threshold)
    qwen_records = load_high_harm_rollouts(args.qwen_file, args.threshold)

    if args.top_n is not None:
        llama_records = llama_records[: args.top_n]
        qwen_records = qwen_records[: args.top_n]

    output = {
        "threshold": args.threshold,
        "comparison_rule": "score > threshold",
        "summary": {
            "llama": summarize(llama_records),
            "qwen": summarize(qwen_records),
        },
        "llama": llama_records,
        "qwen": qwen_records,
    }

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {output_path}")
    print(f"Llama high-harm rollouts: {len(llama_records)}")
    print(f"Qwen high-harm rollouts: {len(qwen_records)}")


if __name__ == "__main__":
    main()
