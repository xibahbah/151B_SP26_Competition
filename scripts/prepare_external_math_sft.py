#!/usr/bin/env python3
"""Prepare external math datasets as FRQ-style chat SFT JSONL.

This script streams from Hugging Face datasets and writes a small, filtered
training file without downloading the full dataset.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

try:
    from datasets import load_dataset
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: datasets. Install with `pip install -r requirements-lora.txt` "
        "inside the RunPod environment."
    ) from exc


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


PRESETS = {
    "math": {
        "dataset": "hendrycks/competition_math",
        "config": None,
        "split": "train",
        "problem_field": "problem",
        "solution_field": "solution",
        "answer_field": None,
    },
    "openr1": {
        "dataset": "open-r1/OpenR1-Math-220k",
        "config": "default",
        "split": "train",
        "problem_field": "problem",
        "solution_field": "solution",
        "answer_field": "answer",
    },
}


def clean_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def find_boxed_entries(text: str) -> list[str]:
    entries = []
    start = 0
    while True:
        idx = text.find("\\boxed{", start)
        if idx < 0:
            idx = text.find("\\fbox{", start)
            marker = "\\fbox{"
        else:
            marker = "\\boxed{"
        if idx < 0:
            break
        brace_start = idx + len(marker)
        depth = 1
        i = brace_start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            entries.append(text[brace_start : i - 1].strip())
        start = max(i, idx + 1)
    return entries


def clean_answer(answer: Any, solution: str) -> str:
    if isinstance(answer, list):
        answer_text = ", ".join(clean_text(item) for item in answer)
    else:
        answer_text = clean_text(answer)

    if not answer_text:
        boxes = find_boxed_entries(solution)
        answer_text = boxes[-1] if boxes else ""

    answer_text = re.sub(r"^\\boxed\{(.*)\}$", r"\1", answer_text.strip())
    answer_text = answer_text.strip(" $.")
    return answer_text


def looks_like_mcq(problem: str) -> bool:
    patterns = [
        r"\n\s*\(?A\)",
        r"\n\s*A\.",
        r"answer choices",
        r"multiple choice",
        r"choose the best",
    ]
    return any(re.search(pattern, problem, flags=re.IGNORECASE) for pattern in patterns)


def candidate_records(dataset: Iterable[dict[str, Any]], args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    for row in dataset:
        problem = clean_text(row.get(args.problem_field))
        solution = clean_text(row.get(args.solution_field))
        answer = clean_answer(row.get(args.answer_field) if args.answer_field else None, solution)

        if not problem or not answer:
            continue
        if len(problem) < args.min_problem_chars or len(problem) > args.max_problem_chars:
            continue
        if len(solution) > args.max_solution_chars:
            continue
        if not args.include_mcq_like and looks_like_mcq(problem):
            continue

        if args.assistant_mode == "solution" and solution:
            assistant = solution
            if "\\boxed{" not in assistant and "\\fbox{" not in assistant:
                assistant = f"{assistant.rstrip()}\n\n\\boxed{{{answer}}}"
        else:
            assistant = f"\\boxed{{{answer}}}"

        yield {
            "source": args.dataset,
            "source_config": args.config,
            "source_split": args.split,
            "messages": [
                {"role": "system", "content": FRQ_SYSTEM_PROMPT},
                {"role": "user", "content": problem},
                {"role": "assistant", "content": assistant},
            ],
            "answer": answer,
        }


def reservoir_sample(records: Iterable[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    sample = []
    for seen, record in enumerate(records, start=1):
        if len(sample) < sample_size:
            sample.append(record)
            continue
        j = rng.randrange(seen)
        if j < sample_size:
            sample[j] = record
    rng.shuffle(sample)
    return sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), default="math")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--problem-field", default=None)
    parser.add_argument("--solution-field", default=None)
    parser.add_argument("--answer-field", default=None)
    parser.add_argument("--output", default="data/external_math_sft.jsonl")
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=151)
    parser.add_argument("--assistant-mode", choices=["final", "solution"], default="final")
    parser.add_argument("--min-problem-chars", type=int, default=20)
    parser.add_argument("--max-problem-chars", type=int, default=2500)
    parser.add_argument("--max-solution-chars", type=int, default=12000)
    parser.add_argument("--include-mcq-like", action="store_true")
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    for key in ("dataset", "config", "split", "problem_field", "solution_field", "answer_field"):
        cli_key = key.replace("_", "-")
        value = getattr(args, key)
        if value is None:
            setattr(args, key, preset[key])

    load_kwargs = {"path": args.dataset, "split": args.split, "streaming": True}
    if args.config:
        load_kwargs["name"] = args.config
    dataset = load_dataset(**load_kwargs)

    records = reservoir_sample(candidate_records(dataset, args), args.sample_size, args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for i, record in enumerate(records):
            record["id"] = f"{args.dataset}:{args.split}:{i}"
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {len(records)} records to {out_path}")
    print(f"Dataset: {args.dataset} config={args.config} split={args.split}")
    print(f"Assistant mode: {args.assistant_mode}")


if __name__ == "__main__":
    main()
