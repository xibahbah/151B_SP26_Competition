#!/usr/bin/env python3
"""Create a fixed public-FRQ train/holdout split and SFT train file."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


FRQ_SYSTEM_PROMPT = """You are solving free-response math problems for an automatic grader.

Each problem may contain one or more [ANS] blanks. Solve efficiently, then output the final answers in the exact format expected by the grader.

Keep your reasoning concise. Do not debate multiple strategies, do not repeat calculations, and do not explain basic definitions unless needed. After you have the answers, stop reasoning and write the boxed final answer immediately.

Rules:
1. End with exactly one boxed answer and write nothing after it.
   For one answer: \\boxed{answer}
   For multiple answers: \\boxed{answer1, answer2, answer3}
2. Put answers in the same order as the blanks or subquestions. If there is no [ANS] blank, infer the single requested final answer.
3. Use plain text math inside the box: ^ for powers, * for multiplication, same variable names as the problem, and functions like sqrt, sin, cos, tan, ln, log, atan.
4. Do not include units unless the problem explicitly requires units in the answer.
5. Do not use thousands separators, since commas separate multiple answers. Write 5850000, not 5,850,000.
6. Prefer exact expressions when natural. Keep clean fractions as reduced fractions, not decimals. Leave products unexpanded when asked.
7. For decimals, give about 12-15 significant digits unless the problem explicitly says to round. Obey nearest integer, cents, decimal-place, and significant-figure instructions exactly.
8. For embedded choice blanks, output only the requested letter or letters. If multiple letters are selected for one blank, concatenate them alphabetically with no spaces, e.g. CF.
9. For money, include $ only if the problem explicitly says the answer must begin with a dollar sign. For percent blanks, include % only when the problem explicitly asks for percent notation.
10. Before finalizing, check the answer count, order, signs, rounding, and formatting. Then output the boxed answer immediately.

Final response must end with exactly one line and no trailing explanation:
\\boxed{...}"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def clean_question(question: str) -> str:
    question = question.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in question.splitlines()).strip()


def answer_text(answer: Any) -> str:
    if isinstance(answer, list):
        return ", ".join(str(item).strip() for item in answer)
    return str(answer).strip()


def to_sft_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "messages": [
            {"role": "system", "content": FRQ_SYSTEM_PROMPT},
            {"role": "user", "content": clean_question(row["question"])},
            {"role": "assistant", "content": f"\\boxed{{{answer_text(row['answer'])}}}"},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/public.jsonl")
    parser.add_argument("--seed", type=int, default=151)
    parser.add_argument("--holdout-size", type=int, default=200)
    parser.add_argument("--train-output", default="data/public_frq_train.jsonl")
    parser.add_argument("--holdout-output", default="data/public_frq_holdout.jsonl")
    parser.add_argument("--holdout-ids-output", default="data/public_frq_holdout_ids.json")
    parser.add_argument("--sft-output", default="data/public_frq_train_sft.jsonl")
    args = parser.parse_args()

    rows = [row for row in read_jsonl(Path(args.input)) if not row.get("options")]
    rng = random.Random(args.seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)

    holdout_size = min(args.holdout_size, len(shuffled))
    holdout = shuffled[:holdout_size]
    train = shuffled[holdout_size:]
    holdout_ids = [row["id"] for row in holdout]
    train_sft = [to_sft_record(row) for row in train]

    write_jsonl(Path(args.train_output), train)
    write_jsonl(Path(args.holdout_output), holdout)
    write_jsonl(Path(args.sft_output), train_sft)
    Path(args.holdout_ids_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.holdout_ids_output).write_text(json.dumps(holdout_ids, indent=2) + "\n")

    print(f"Public FRQ rows: {len(rows)}")
    print(f"Train FRQ rows: {len(train)} -> {args.train_output}")
    print(f"Holdout FRQ rows: {len(holdout)} -> {args.holdout_output}")
    print(f"Holdout IDs: {args.holdout_ids_output}")
    print(f"Train SFT records: {len(train_sft)} -> {args.sft_output}")


if __name__ == "__main__":
    main()
