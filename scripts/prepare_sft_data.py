#!/usr/bin/env python3
"""Prepare FRQ chat data for SFT/LoRA training."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


FRQ_SYSTEM_PROMPT = """You are solving free-response math problems for an automatic grader.

Each problem may contain one or more [ANS] blanks. Solve carefully, then output the final answers in the exact format expected by the grader.

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
10. Before finalizing, check the answer count, order, signs, rounding, and formatting.

Final response must end with exactly one line:
\\boxed{...}"""


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def answer_text(answer) -> str:
    if isinstance(answer, list):
        return ", ".join(str(x).strip() for x in answer)
    return str(answer).strip()


def clean_question(question: str) -> str:
    question = question.replace("\r\n", "\n").replace("\r", "\n")
    question = "\n".join(line.rstrip() for line in question.splitlines())
    return question.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="Training JSONL file. May be passed more than once.")
    parser.add_argument("--output", default="data/train_sft_frq.jsonl")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=151)
    parser.add_argument("--include-mcq", action="store_true")
    args = parser.parse_args()

    rows = []
    for path in args.input:
        rows.extend(read_jsonl(Path(path)))

    if not args.include_mcq:
        rows = [row for row in rows if not row.get("options")]

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.sample_size is not None:
        rows = rows[: args.sample_size]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            final_answer = answer_text(row["answer"])
            record = {
                "id": row.get("id"),
                "messages": [
                    {"role": "system", "content": FRQ_SYSTEM_PROMPT},
                    {"role": "user", "content": clean_question(row["question"])},
                    {"role": "assistant", "content": f"\\boxed{{{final_answer}}}"},
                ],
            }
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {len(rows)} SFT records to {out_path}")


if __name__ == "__main__":
    main()
