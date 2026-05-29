#!/usr/bin/env python3
"""Prepare FRQ SFT data from correct baseline generations.

This is meant for cautious LoRA experiments:
- hold out a fixed public FRQ sample for validation
- train on non-holdout examples where the baseline was already correct
- use the baseline's own correct final boxed answer as the assistant target

The goal is to preserve the base model's math behavior while nudging it toward
the local dataset/prompt format, instead of teaching final-answer guessing or
long chain-of-thought style outputs.
"""

from __future__ import annotations

import argparse
import json
import random
import re
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
    question = "\n".join(line.rstrip() for line in question.splitlines())
    return question.strip()


def answer_text(answer: Any) -> str:
    if isinstance(answer, list):
        return ", ".join(str(item).strip() for item in answer)
    return str(answer).strip()


def last_boxed_response(response: str) -> str:
    """Return only the last complete boxed answer, if present."""
    last = response.rfind("\\boxed{")
    if last < 0:
        return response.strip()
    depth = 0
    i = last
    while i < len(response):
        if response[i] == "{":
            depth += 1
        elif response[i] == "}":
            depth -= 1
            if depth == 0:
                return response[last : i + 1].strip()
        i += 1
    return response.strip()


def normalize_teacher_response(row: dict[str, Any], fallback_answer: Any, max_chars: int) -> str:
    if row.get("raw_correct", False):
        response = str(row.get("response") or "").strip()
    elif row.get("postprocess_correct", False):
        response = str(row.get("postprocessed_response") or "").strip()
    elif row.get("repair_correct", False):
        response = str(row.get("repair_response") or "").strip()
    else:
        response = ""

    if not response:
        response = f"\\boxed{{{answer_text(fallback_answer)}}}"

    response = last_boxed_response(response)
    response = re.sub(r"\n{3,}", "\n\n", response).strip()
    if len(response) > max_chars:
        response = f"\\boxed{{{answer_text(fallback_answer)}}}"
    if "\\boxed{" not in response:
        response = f"\\boxed{{{response}}}"
    return response


def is_trainable_correct(result: dict[str, Any], include_postprocess: bool, include_repair: bool) -> tuple[bool, str]:
    if bool(result.get("raw_correct")):
        return True, "raw_correct"
    if include_postprocess and bool(result.get("postprocess_correct")):
        return True, "postprocess_correct"
    if include_repair and bool(result.get("repair_correct")):
        return True, "repair_correct"
    return False, "wrong"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/public.jsonl")
    parser.add_argument("--baseline-results", required=True)
    parser.add_argument("--output", default="data/self_distill_frq_sft.jsonl")
    parser.add_argument("--holdout-output", default="data/self_distill_holdout_ids.json")
    parser.add_argument("--holdout-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=151)
    parser.add_argument("--max-teacher-chars", type=int, default=6000)
    parser.add_argument("--include-postprocess-correct", action="store_true")
    parser.add_argument(
        "--include-repair-correct",
        action="store_true",
        help="Also train on non-holdout rows where the offline repair extracted a correct answer.",
    )
    args = parser.parse_args()

    data = [row for row in read_jsonl(Path(args.data)) if not row.get("options")]
    by_id = {row["id"]: row for row in data}
    baseline_rows = {row["id"]: row for row in read_jsonl(Path(args.baseline_results))}

    rng = random.Random(args.seed)
    holdout = rng.sample(data, min(args.holdout_size, len(data)))
    holdout_ids = {row["id"] for row in holdout}
    train_ids = set(by_id) - holdout_ids

    records = []
    skipped_wrong = 0
    teacher_sources: dict[str, int] = {}
    for row_id in sorted(train_ids):
        item = by_id[row_id]
        result = baseline_rows.get(row_id)
        if result is None:
            continue
        correct, source = is_trainable_correct(
            result,
            include_postprocess=args.include_postprocess_correct,
            include_repair=args.include_repair_correct,
        )
        if not correct:
            skipped_wrong += 1
            continue
        teacher_sources[source] = teacher_sources.get(source, 0) + 1

        records.append({
            "id": row_id,
            "messages": [
                {"role": "system", "content": FRQ_SYSTEM_PROMPT},
                {"role": "user", "content": clean_question(item["question"])},
                {
                    "role": "assistant",
                    "content": normalize_teacher_response(result, item["answer"], args.max_teacher_chars),
                },
            ],
        })

    rng.shuffle(records)
    write_jsonl(Path(args.output), records)
    Path(args.holdout_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.holdout_output).write_text(json.dumps(sorted(holdout_ids), indent=2))

    print(f"Public FRQ rows: {len(data)}")
    print(f"Holdout rows: {len(holdout_ids)} -> {args.holdout_output}")
    print(f"Trainable correct baseline rows: {len(records)} -> {args.output}")
    for source, count in sorted(teacher_sources.items()):
        print(f"  {source}: {count}")
    print(f"Skipped wrong/non-correct train rows: {skipped_wrong}")


if __name__ == "__main__":
    main()
