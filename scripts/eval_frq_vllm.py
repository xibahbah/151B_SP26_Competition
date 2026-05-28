#!/usr/bin/env python3
"""Run FRQ-only vLLM evaluation and write answer-accuracy JSONL results."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frq_tools import postprocess_response, score_frq_item
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


def build_user_content(question: str) -> str:
    """Append a structural, gold-free directive to the question.

    The hint is derived only from the question text (the number of [ANS]
    blanks), so it behaves identically on public and private data and does not
    overfit to gold answers. It targets the two biggest error buckets:
    wrong_answer_count and precision_rounding.
    """
    n_blanks = question.count("[ANS]")
    lines = [question.strip(), ""]
    if n_blanks >= 1:
        plural = "s" if n_blanks != 1 else ""
        lines.append(
            f"This problem has exactly {n_blanks} [ANS] blank{plural}. "
            f"Output exactly {n_blanks} value{plural} inside a single "
            f"\\boxed{{...}}, comma-separated, in the same order as the blanks."
        )
    else:
        lines.append(
            "Output the single requested final answer inside one \\boxed{...}."
        )
    lines.append(
        "Carry full precision through every step and do not round intermediate "
        "results. Unless the problem explicitly states how to round, give at "
        "least 10 significant digits for any decimal answer."
    )
    return "\n".join(lines)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/public.jsonl")
    parser.add_argument("--output", default="results/frq_baseline_full.jsonl")
    parser.add_argument("--errors", default="results/frq_baseline_errors.jsonl")
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=-1, help="Use -1 for a fresh random sample seed.")
    parser.add_argument("--ids-file", default=None, help="JSON file containing item IDs to evaluate.")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.78)
    parser.add_argument("--max-model-len", type=int, default=8192)
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("VLLM_USE_V1", "0")

    data = [row for row in read_jsonl(Path(args.data)) if not row.get("options")]
    if args.ids_file:
        ids = set(json.loads(Path(args.ids_file).read_text()))
        data = [row for row in data if row["id"] in ids]
    sample_seed = int(time.time_ns() % (2**32)) if args.seed < 0 else args.seed
    if args.sample_size is not None:
        rng = random.Random(sample_seed)
        data = rng.sample(data, min(args.sample_size, len(data)))
    print(f"Sample seed: {sample_seed}")

    llm_kwargs = {
        "model": args.model_id,
        "quantization": "bitsandbytes",
        "load_format": "bitsandbytes",
        "enable_prefix_caching": False,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "trust_remote_code": True,
        "max_num_seqs": 4,
        "max_num_batched_tokens": args.max_model_len,
    }
    if args.lora_path:
        llm_kwargs["enable_lora"] = True

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
    )

    lora_request = None
    if args.lora_path:
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest("frq_lora", 1, args.lora_path)

    prompts = []
    for item in data:
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT_MATH},
                {"role": "user", "content": build_user_content(item["question"])},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)

    responses = []
    for start in tqdm(range(0, len(prompts), args.batch_size), desc="Generating"):
        batch = prompts[start : start + args.batch_size]
        outputs = llm.generate(batch, sampling_params=sampling_params, lora_request=lora_request)
        responses.extend(output.outputs[0].text.strip() for output in outputs)

    has_gold = all("answer" in item for item in data)
    judger = Judger(strict_extract=False) if has_gold else None
    results = []
    for item, response in tqdm(zip(data, responses), total=len(data), desc="Scoring"):
        if has_gold:
            score_info = score_frq_item(judger, item, response)
            gold_fields = {"gold": item["answer"]}
        else:
            expected_count = max(1, item["question"].count("[ANS]"))
            post = postprocess_response(response, item["question"], expected_count)
            score_info = {
                "raw_correct": None,
                "postprocess_correct": None,
                "correct": None,
                "postprocessed_response": post["response"],
                "postprocessed_answer": post["answer_text"],
                "postprocess_notes": post["notes"],
                "error_type": "unscored",
            }
            gold_fields = {}
        results.append({
            "id": item["id"],
            "is_mcq": False,
            "response": response,
            "sample_seed": sample_seed,
            **gold_fields,
            **score_info,
        })

    errors = [row for row in results if row["correct"] is False]
    write_jsonl(Path(args.output), results)
    write_jsonl(Path(args.errors), errors)

    total = len(results)
    if has_gold:
        raw_correct = sum(row["raw_correct"] for row in results)
        final_correct = sum(row["correct"] for row in results)
        print(f"Raw FRQ accuracy: {raw_correct}/{total} ({raw_correct / total * 100:.2f}%)")
        print(f"Postprocessed FRQ accuracy: {final_correct}/{total} ({final_correct / total * 100:.2f}%)")
    else:
        print(f"FRQ private inference complete: {total} predictions")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.errors}")


if __name__ == "__main__":
    main()
