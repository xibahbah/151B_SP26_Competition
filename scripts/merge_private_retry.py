#!/usr/bin/env python3
"""Merge targeted private FRQ retry generations into a submission CSV.

This is conservative: it keeps the original full response trace unless the
original final answer is obviously malformed and the retry has a cleaner boxed
answer with the expected answer count.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from private_retry_ids import BAD_FINALS, boxed_entries, split_top_level_commas


def clean_answer_count(answer: str, expected_count: int) -> bool:
    if not answer or answer.lower() in BAD_FINALS:
        return False
    parts = [p for p in split_top_level_commas(answer) if p.strip()]
    return len(parts) == expected_count and len(answer) <= 900


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-data", default="data/private.jsonl")
    parser.add_argument("--base-csv", required=True)
    parser.add_argument("--retry-jsonl", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    private_rows = {
        int(item["id"]): item
        for item in read_jsonl(Path(args.private_data))
    }
    retry_rows = read_jsonl(Path(args.retry_jsonl)) if Path(args.retry_jsonl).exists() else []
    retry_by_id: dict[int, dict] = {}
    for row in retry_rows:
        item_id = int(row["id"])
        answer = str(row.get("repair_response") or row.get("postprocessed_response") or row.get("response") or "")
        boxes = boxed_entries(answer)
        final = boxes[-1] if boxes else ""
        item = private_rows[item_id]
        expected = max(1, str(item.get("question", "")).count("[ANS]"))
        if clean_answer_count(final, expected):
            previous = retry_by_id.get(item_id)
            score = 1 if row.get("repair_response") else 0
            if previous is None or score > previous["_score"]:
                row["_score"] = score
                row["_final"] = final
                retry_by_id[item_id] = row

    with Path(args.base_csv).open(newline="") as f:
        rows = list(csv.DictReader(f))

    changed = 0
    for row in rows:
        item_id = int(row["id"])
        retry = retry_by_id.get(item_id)
        if not retry:
            continue
        item = private_rows[item_id]
        if item.get("options"):
            continue
        expected = max(1, str(item.get("question", "")).count("[ANS]"))
        current_boxes = boxed_entries(row.get("response", ""))
        current = current_boxes[-1] if current_boxes else ""
        current_bad = not clean_answer_count(current, expected)
        if current_bad:
            final = retry["_final"]
            retry_trace = str(retry.get("response") or "").strip()
            row["response"] = (
                row.get("response", "").rstrip()
                + "\n\nTargeted retry trace:\n"
                + retry_trace
                + f"\n\nFinal answer: \\boxed{{{final}}}"
            )
            changed += 1

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "response"])
        writer.writeheader()
        writer.writerows(rows)

    final_okay = 0
    boxed = 0
    for row in rows:
        boxes = boxed_entries(row.get("response", ""))
        if boxes:
            boxed += 1
            final_okay += boxes[-1].lower() == "okay"
    print(f"Base CSV: {args.base_csv}")
    print(f"Retry JSONL: {args.retry_jsonl}")
    print(f"Output CSV: {output_path}")
    print(f"Accepted retry replacements: {changed}")
    print(f"Rows: {len(rows)}")
    print(f"Rows with boxed answer: {boxed}")
    print(f"Final boxed Okay rows: {final_okay}")


if __name__ == "__main__":
    main()
