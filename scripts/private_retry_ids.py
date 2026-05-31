#!/usr/bin/env python3
"""Find private FRQ rows worth a targeted second inference pass."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


BAD_FINALS = {"", "okay", "ok", "done", "answer", "final answer"}


def boxed_entries(text: str) -> list[str]:
    entries: list[str] = []
    start = 0
    while True:
        idx = text.find("\\boxed{", start)
        if idx < 0:
            break
        i = idx + len("\\boxed{")
        depth = 1
        j = i
        while j < len(text) and depth:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            entries.append(text[i : j - 1].strip())
        start = max(j, idx + 1)
    return entries


def split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}" and depth:
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def looks_suspicious(answer: str, expected_count: int) -> bool:
    stripped = answer.strip()
    if stripped.lower() in BAD_FINALS:
        return True
    if len(stripped) > 900:
        return True
    parts = [p for p in split_top_level_commas(stripped) if p.strip()]
    if len(parts) != expected_count:
        return True
    if expected_count == 1 and re.fullmatch(r"[A-Za-z ]{8,}", stripped):
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-data", default="data/private.jsonl")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-ids", type=int, default=80)
    args = parser.parse_args()

    private_rows = {}
    for line in Path(args.private_data).open():
        if not line.strip():
            continue
        item = json.loads(line)
        private_rows[int(item["id"])] = item

    with Path(args.csv).open(newline="") as f:
        csv_rows = list(csv.DictReader(f))

    retry_ids: list[int] = []
    reasons: list[dict] = []
    for row in csv_rows:
        item_id = int(row["id"])
        item = private_rows.get(item_id)
        if not item or item.get("options"):
            continue
        expected_count = max(1, str(item.get("question", "")).count("[ANS]"))
        boxes = boxed_entries(row.get("response", ""))
        answer = boxes[-1] if boxes else ""
        if looks_suspicious(answer, expected_count):
            retry_ids.append(item_id)
            reasons.append({
                "id": item_id,
                "expected_count": expected_count,
                "final_box": answer[:200],
                "reason": "suspicious_final",
            })

    retry_ids = retry_ids[: args.max_ids]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(retry_ids))
    reason_path = Path(args.output).with_suffix(".reasons.jsonl")
    with reason_path.open("w") as f:
        for row in reasons[: args.max_ids]:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(retry_ids)} retry IDs to {args.output}")
    print(f"Wrote reasons to {reason_path}")
    if retry_ids:
        print("IDs:", ",".join(map(str, retry_ids)))


if __name__ == "__main__":
    main()
