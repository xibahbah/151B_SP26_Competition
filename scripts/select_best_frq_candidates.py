#!/usr/bin/env python3
"""Select one scored FRQ candidate per question from multi-generation results.

This is a public-data development helper. It may use gold-derived `correct`
fields already written by eval/repair scripts to estimate oracle upside. Do not
use this selector on private predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def score_key(row: dict[str, Any]) -> tuple[int, float, int]:
    correct = 1 if row.get("correct") is True else 0
    quality = float(row.get("repair_quality") or 0.0)
    # Prefer lower candidate index as the tie-breaker for reproducibility.
    candidate_idx = int(row.get("candidate_idx") or 0)
    return (correct, quality, -candidate_idx)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    grouped: dict[Any, list[dict[str, Any]]] = {}
    for row in read_jsonl(Path(args.input)):
        grouped.setdefault(row["id"], []).append(row)

    selected = []
    for rows in grouped.values():
        selected.append(max(rows, key=score_key))
    selected.sort(key=lambda row: row["id"])
    write_jsonl(Path(args.output), selected)

    total = len(selected)
    correct = sum(row.get("correct") is True for row in selected)
    raw = sum(row.get("raw_correct") is True for row in selected)
    post = sum(row.get("postprocess_correct") is True for row in selected)
    repair = sum(row.get("repair_correct") is True for row in selected)
    print(f"Selected {total} questions from {sum(len(rows) for rows in grouped.values())} candidates")
    print(f"Final selected accuracy: {correct}/{total} ({correct / total * 100:.2f}%)")
    print(f"Raw selected accuracy: {raw}/{total} ({raw / total * 100:.2f}%)")
    print(f"Postprocess selected accuracy: {post}/{total} ({post / total * 100:.2f}%)")
    print(f"Repair selected accuracy: {repair}/{total} ({repair / total * 100:.2f}%)")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
