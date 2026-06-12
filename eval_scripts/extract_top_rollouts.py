"""
Extract the highest-scoring rollouts from one or more rollout files.

Works with:
1. CEM-style JSON files that contain a list of rollout dictionaries:
   [
     {"round": 1, "rollout": 1, "score": 0.1, "sim_out": {...}},
     ...
   ]

2. Baseline JSONL files with one rollout per line:
   {"rollout": 1, "score": 0.1, "sim_out": {...}}
   {"rollout": 2, "score": 0.2, "sim_out": {...}}

Output format:
{
  "top_n": 5,
  "sections": {
    "qwen": [...],
    "llama": [...]
  }
}

Example:
  python extract_top_rollouts.py \
    --files qwen=cem_qwen_50r_20e_80rol_halfway.json llama=llama_baseline.jsonl \
    --top-n 5 \
    --output-file top5_rollouts.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def parse_file_args(file_args: Sequence[str]) -> List[Tuple[str, str]]:
    """
    Accept either:
      --files qwen.json llama.jsonl
    or:
      --files qwen=qwen.json llama=llama.jsonl
    """
    parsed: List[Tuple[str, str]] = []

    for item in file_args:
        if "=" in item:
            label, path = item.split("=", 1)
            parsed.append((label.strip(), path.strip()))
        else:
            path = item.strip()
            label = Path(path).stem
            parsed.append((label, path))

    return parsed


def load_rollouts(path: str) -> List[Dict[str, Any]]:
    """
    Load either:
    - JSON array file
    - JSON object file
    - JSONL file
    """
    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {path}")

    text = path_obj.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # Try normal JSON first. This handles CEM JSON files that are lists.
    try:
        data = json.loads(text)

        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]

        if isinstance(data, dict):
            # Some outputs may already have sections.
            # Flatten any list-valued sections if present.
            if "sections" in data and isinstance(data["sections"], dict):
                out: List[Dict[str, Any]] = []
                for _, records in data["sections"].items():
                    if isinstance(records, list):
                        out.extend([x for x in records if isinstance(x, dict)])
                return out

            # Single rollout object.
            return [data]

    except json.JSONDecodeError:
        pass

    # Fall back to JSONL.
    records: List[Dict[str, Any]] = []
    with open(path_obj, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping malformed JSONL line {line_idx} in {path}: {e}")
                continue

            if isinstance(record, dict):
                records.append(record)

    return records


def get_score(record: Dict[str, Any]) -> float | None:
    score = record.get("score", None)

    if score is None:
        return None

    try:
        return float(score)
    except (TypeError, ValueError):
        return None


def extract_top_n(
    records: List[Dict[str, Any]],
    *,
    label: str,
    source_file: str,
    top_n: int,
) -> List[Dict[str, Any]]:
    scored: List[Tuple[float, Dict[str, Any]]] = []

    for idx, record in enumerate(records):
        score = get_score(record)
        if score is None:
            continue

        copied = dict(record)
        copied["_source_label"] = label
        copied["_source_file"] = source_file
        copied["_source_index"] = idx
        scored.append((score, copied))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [record for _, record in scored[:top_n]]


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = [get_score(r) for r in records]
    scores = [s for s in scores if s is not None]

    if not scores:
        return {
            "count": 0,
            "max_score": None,
            "min_score": None,
            "mean_score": None,
        }

    return {
        "count": len(scores),
        "max_score": max(scores),
        "min_score": min(scores),
        "mean_score": sum(scores) / len(scores),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="Input rollout files. Use label=path pairs, e.g. qwen=qwen.json llama=llama.jsonl.",
    )
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--output-file", type=str, default="top5_rollouts.json")

    args = parser.parse_args()

    sections: Dict[str, List[Dict[str, Any]]] = {}
    summary: Dict[str, Dict[str, Any]] = {}

    for label, path in parse_file_args(args.files):
        records = load_rollouts(path)
        top_records = extract_top_n(
            records,
            label=label,
            source_file=path,
            top_n=args.top_n,
        )

        sections[label] = top_records
        summary[label] = {
            "input_file": path,
            "total_loaded_records": len(records),
            "top_records": summarize(top_records),
        }

        print(f"{label}: loaded {len(records)} records; selected top {len(top_records)}")

    output = {
        "top_n": args.top_n,
        "selection_rule": "highest score values, sorted descending within each section",
        "summary": summary,
        "sections": sections,
    }

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
