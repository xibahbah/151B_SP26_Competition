#!/usr/bin/env python3
"""Merge MCQ and FRQ private inference outputs into one JSONL in data order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/private.jsonl")
    parser.add_argument("--mcq", required=True)
    parser.add_argument("--frq", required=True)
    parser.add_argument("--output", default="results/private_predictions_merged.jsonl")
    parser.add_argument(
        "--response-field",
        default="postprocessed_response",
        choices=["response", "postprocessed_response"],
        help="Field to use as the final response in the merged output.",
    )
    args = parser.parse_args()

    data = read_jsonl(Path(args.data))
    by_id = {}
    for row in read_jsonl(Path(args.mcq)) + read_jsonl(Path(args.frq)):
        by_id[row["id"]] = row

    missing = [item["id"] for item in data if item["id"] not in by_id]
    if missing:
        raise SystemExit(f"Missing predictions for {len(missing)} ids, first few: {missing[:10]}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for item in data:
            pred = by_id[item["id"]]
            response = pred.get(args.response_field) or pred.get("response", "")
            f.write(json.dumps({
                "id": item["id"],
                "is_mcq": bool(item.get("options")),
                "response": response,
            }) + "\n")

    print(f"Wrote {len(data)} merged predictions to {out_path}")


if __name__ == "__main__":
    main()
