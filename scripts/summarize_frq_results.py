#!/usr/bin/env python3
"""Summarize FRQ JSONL result files by answer accuracy and error bucket."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def pct(count: int, total: int) -> str:
    return f"{count / total * 100:.2f}%" if total else "0.00%"


def count_truthy(rows: list[dict[str, Any]], field: str) -> int | None:
    if not any(field in row for row in rows):
        return None
    return sum(bool(row.get(field)) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--ids-file", default=None, help="Optional JSON list of IDs to summarize.")
    args = parser.parse_args()

    rows = [row for row in read_jsonl(Path(args.input)) if not row.get("is_mcq")]
    if args.ids_file:
        ids = set(json.loads(Path(args.ids_file).read_text()))
        rows = [row for row in rows if row.get("id") in ids]

    total = len(rows)
    print(f"Input: {args.input}")
    if args.ids_file:
        print(f"IDs: {args.ids_file}")
    print(f"Rows: {total}")

    for label, field in (
        ("Final", "correct"),
        ("Raw", "raw_correct"),
        ("Postprocess", "postprocess_correct"),
        ("Baseline", "baseline_correct"),
        ("Repair selected", "repair_correct"),
        ("Oracle repair", "oracle_repair_correct"),
    ):
        count = count_truthy(rows, field)
        if count is not None:
            print(f"{label}: {count}/{total} ({pct(count, total)})")

    errors = collections.Counter(row.get("error_type", "unknown") for row in rows)
    print("Error types:")
    for error_type, count in errors.most_common():
        print(f"  {error_type:28s} {count:5d}")

    repair_sources = collections.Counter(
        row.get("repair_source")
        for row in rows
        if row.get("repair_source") and (not row.get("baseline_correct")) and row.get("repair_correct")
    )
    if repair_sources:
        print("Selected repair gains:")
        for source, count in repair_sources.most_common():
            print(f"  {source:28s} {count:5d}")


if __name__ == "__main__":
    main()
