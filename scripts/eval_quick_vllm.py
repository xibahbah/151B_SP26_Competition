#!/usr/bin/env python3
"""Run a quick mixed MCQ + FRQ vLLM accuracy check in one model load."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frq_tools import score_frq_item
from judger import Judger
from tqdm import tqdm
from vllm import LLM, SamplingParams


SYSTEM_PROMPT_MATH = """You are solving free-response math problems for an automatic grader.

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


SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def extract_letter(text: str) -> str:
    boxed = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if boxed:
        return boxed.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def build_prompt(item: dict) -> tuple[str, str]:
    if item.get("options"):
        labels = [chr(65 + i) for i in range(len(item["options"]))]
        opts_text = "\n".join(
            f"{label}. {option.strip()}" for label, option in zip(labels, item["options"])
        )
        return SYSTEM_PROMPT_MCQ, f"{item['question']}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, item["question"]


def sample_rows(rows: list[dict], sample_size: int | None, rng: random.Random) -> list[dict]:
    if sample_size is None:
        return rows
    return rng.sample(rows, min(sample_size, len(rows)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/public.jsonl")
    parser.add_argument("--output", default="results/quick_mixed_eval.jsonl")
    parser.add_argument("--errors", default="results/quick_mixed_errors.jsonl")
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--mcq-sample-size", type=int, default=100)
    parser.add_argument("--frq-sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=151)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.78)
    parser.add_argument("--max-model-len", type=int, default=8192)
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("VLLM_USE_V1", "0")

    rows = read_jsonl(Path(args.data))
    rng = random.Random(args.seed)
    mcq_rows = sample_rows([row for row in rows if row.get("options")], args.mcq_sample_size, rng)
    frq_rows = sample_rows([row for row in rows if not row.get("options")], args.frq_sample_size, rng)
    data = mcq_rows + frq_rows

    print(f"Evaluation set: {len(mcq_rows)} MCQ, {len(frq_rows)} FRQ")

    llm = LLM(
        model=args.model_id,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enable_prefix_caching=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        max_num_seqs=4,
        max_num_batched_tokens=args.max_model_len,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=1.0 if args.temperature == 0.0 else 0.95,
        top_k=-1 if args.temperature == 0.0 else 20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
    )

    prompts = []
    for item in data:
        system, user = build_prompt(item)
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    responses = []
    for start in tqdm(range(0, len(prompts), args.batch_size), desc="Generating"):
        batch = prompts[start : start + args.batch_size]
        outputs = llm.generate(batch, sampling_params=sampling_params)
        responses.extend(output.outputs[0].text.strip() for output in outputs)

    judger = Judger(strict_extract=False)
    results = []
    for item, response in tqdm(zip(data, responses), total=len(data), desc="Scoring"):
        is_mcq = bool(item.get("options"))
        if is_mcq:
            pred = extract_letter(response)
            gold = str(item["answer"]).strip().upper()
            correct = pred == gold
            score_info = {
                "postprocessed_answer": pred,
                "raw_correct": correct,
                "postprocess_correct": correct,
                "correct": correct,
                "error_type": "correct" if correct else "wrong_letter",
            }
        else:
            score_info = score_frq_item(judger, item, response)

        results.append(
            {
                "id": item["id"],
                "is_mcq": is_mcq,
                "gold": item["answer"],
                "response": response,
                **score_info,
            }
        )

    errors = [row for row in results if not row["correct"]]
    write_jsonl(Path(args.output), results)
    write_jsonl(Path(args.errors), errors)

    mcq_results = [row for row in results if row["is_mcq"]]
    frq_results = [row for row in results if not row["is_mcq"]]

    def pct(correct: int, total: int) -> float:
        return correct / total * 100 if total else 0.0

    mcq_correct = sum(row["correct"] for row in mcq_results)
    frq_raw = sum(row["raw_correct"] for row in frq_results)
    frq_correct = sum(row["correct"] for row in frq_results)
    total_correct = sum(row["correct"] for row in results)

    print("=" * 50)
    print("QUICK EVAL RESULTS")
    print("=" * 50)
    print(f"MCQ             : {mcq_correct:4d} / {len(mcq_results):4d} ({pct(mcq_correct, len(mcq_results)):.2f}%)")
    print(f"FRQ raw         : {frq_raw:4d} / {len(frq_results):4d} ({pct(frq_raw, len(frq_results)):.2f}%)")
    print(f"FRQ postprocess : {frq_correct:4d} / {len(frq_results):4d} ({pct(frq_correct, len(frq_results)):.2f}%)")
    print(f"Overall         : {total_correct:4d} / {len(results):4d} ({pct(total_correct, len(results)):.2f}%)")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.errors}")


if __name__ == "__main__":
    main()
