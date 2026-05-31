#!/usr/bin/env python3
"""Run private inference and write the Kaggle submission CSV.

This wrapper intentionally runs MCQ and FRQ inference as separate subprocesses
so vLLM releases GPU memory between phases.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def run_step(name: str, cmd: list[str]) -> None:
    print(f"\n=== {name} ===", flush=True)
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def csv_response_text(record: dict[str, Any]) -> str:
    raw = str(record.get("response") or "").strip()
    final = str(
        record.get("repair_response")
        or record.get("postprocessed_response")
        or ""
    ).strip()
    if not final or final in raw:
        return raw
    return raw.rstrip() + "\n\nFinal answer: " + final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/private.jsonl")
    parser.add_argument("--output-csv", default="submission.csv")
    parser.add_argument("--results-dir", default="results/private_run")
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--frq-max-tokens", type=int, default=4096)
    parser.add_argument("--frq-batch-size", type=int, default=5)
    parser.add_argument("--mcq-max-tokens", type=int, default=32768)
    parser.add_argument("--mcq-batch-size", type=int, default=None)
    parser.add_argument("--skip-mcq", action="store_true")
    parser.add_argument("--skip-frq", action="store_true")
    args = parser.parse_args()

    data_path = Path(args.data)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    mcq_path = results_dir / "private_mcq.jsonl"
    mcq_errors = results_dir / "private_mcq_errors.jsonl"
    frq_path = results_dir / "private_frq_raw.jsonl"
    frq_errors = results_dir / "private_frq_raw_errors.jsonl"
    frq_repaired = results_dir / "private_frq_repaired.jsonl"
    frq_repaired_errors = results_dir / "private_frq_repaired_errors.jsonl"
    merged_path = results_dir / "private_predictions_merged.jsonl"

    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    if not args.skip_mcq:
        mcq_cmd = [
            sys.executable,
            "-u",
            "scripts/eval_mcq_vllm.py",
            "--data",
            str(data_path),
            "--output",
            str(mcq_path),
            "--errors",
            str(mcq_errors),
            "--model-id",
            args.model_id,
            "--max-tokens",
            str(args.mcq_max_tokens),
        ]
        if args.mcq_batch_size is not None:
            mcq_cmd.extend(["--batch-size", str(args.mcq_batch_size)])
        run_step("MCQ private inference", mcq_cmd)

    if not args.skip_frq:
        frq_cmd = [
            sys.executable,
            "-u",
            "scripts/eval_frq_vllm.py",
            "--data",
            str(data_path),
            "--output",
            str(frq_path),
            "--errors",
            str(frq_errors),
            "--model-id",
            args.model_id,
            "--max-tokens",
            str(args.frq_max_tokens),
            "--batch-size",
            str(args.frq_batch_size),
        ]
        run_step("FRQ private inference", frq_cmd)

        repair_cmd = [
            sys.executable,
            "-u",
            "scripts/repair_frq_outputs.py",
            "--data",
            str(data_path),
            "--input",
            str(frq_path),
            "--output",
            str(frq_repaired),
            "--errors",
            str(frq_repaired_errors),
        ]
        run_step("FRQ private repair", repair_cmd)

    by_id: dict[Any, dict[str, Any]] = {}
    if mcq_path.exists():
        for row in read_jsonl(mcq_path):
            by_id[row["id"]] = row
    if frq_repaired.exists():
        for row in read_jsonl(frq_repaired):
            by_id[row["id"]] = row

    data = read_jsonl(data_path)
    missing = [row["id"] for row in data if row["id"] not in by_id]
    if missing:
        raise SystemExit(f"Missing predictions for {len(missing)} IDs, first few: {missing[:10]}")

    merged = [
        {
            "id": item["id"],
            "is_mcq": bool(item.get("options")),
            "response": csv_response_text(by_id[item["id"]]),
        }
        for item in data
    ]
    write_jsonl(merged_path, merged)

    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "response"])
        writer.writeheader()
        for row in merged:
            writer.writerow({"id": row["id"], "response": row["response"]})

    secondary_csv = Path("results/submission.csv")
    if secondary_csv.resolve() != csv_path.resolve():
        secondary_csv.parent.mkdir(parents=True, exist_ok=True)
        with secondary_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "response"])
            writer.writeheader()
            for row in merged:
                writer.writerow({"id": row["id"], "response": row["response"]})

    print(f"\nWrote merged JSONL to {merged_path}", flush=True)
    print(f"Wrote Kaggle CSV to {csv_path}", flush=True)
    print(f"Wrote Kaggle CSV copy to {secondary_csv}", flush=True)
    print(f"Rows: {len(merged)}", flush=True)


if __name__ == "__main__":
    main()
