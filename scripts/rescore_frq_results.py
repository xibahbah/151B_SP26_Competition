#!/usr/bin/env python3
"""Re-score saved FRQ generations with postprocessing and error labels."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frq_tools import score_frq_item
from judger import Judger


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/public.jsonl")
    parser.add_argument("--input", default="results/frq_baseline_full.jsonl")
    parser.add_argument("--output", default="results/frq_baseline_full_rescored.jsonl")
    parser.add_argument("--errors", default="results/frq_baseline_errors.jsonl")
    args = parser.parse_args()

    data_by_id = {row["id"]: row for row in read_jsonl(Path(args.data))}
    rows = read_jsonl(Path(args.input))
    judger = Judger(strict_extract=False)

    rescored = []
    for row in rows:
        item = data_by_id[row["id"]]
        if item.get("options"):
            continue
        response = row["response"]
        score_info = score_frq_item(judger, item, response)
        rescored.append({
            **row,
            "is_mcq": False,
            "gold": item["answer"],
            **score_info,
        })

    errors = [row for row in rescored if not row["correct"]]
    write_jsonl(Path(args.output), rescored)
    write_jsonl(Path(args.errors), errors)

    counts = collections.Counter(row["error_type"] for row in rescored)
    correct = sum(row["correct"] for row in rescored)
    raw_correct = sum(row["raw_correct"] for row in rescored)
    total = len(rescored)
    print(f"Raw FRQ accuracy: {raw_correct}/{total} ({raw_correct / total * 100:.2f}%)")
    print(f"Postprocessed FRQ accuracy: {correct}/{total} ({correct / total * 100:.2f}%)")
    print("Error types:")
    for key, value in sorted(counts.items()):
        print(f"  {key:24s} {value:4d}")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.errors}")


if __name__ == "__main__":
    main()
