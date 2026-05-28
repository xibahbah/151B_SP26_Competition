#!/usr/bin/env python3
"""Run FRQ-only vLLM evaluation and write answer-accuracy JSONL results."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frq_tools import score_frq_item
from judger import Judger
from tqdm import tqdm
from vllm import LLM, SamplingParams


SYSTEM_PROMPT_MATH = """You are solving free-response math problems for an automatic grader.

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
    parser.add_argument("--seed", type=int, default=151)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.78)
    parser.add_argument("--max-model-len", type=int, default=8192)
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("VLLM_USE_V1", "0")

    data = [row for row in read_jsonl(Path(args.data)) if not row.get("options")]
    if args.sample_size is not None:
        rng = random.Random(args.seed)
        data = rng.sample(data, min(args.sample_size, len(data)))

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
                {"role": "user", "content": item["question"]},
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

    judger = Judger(strict_extract=False)
    results = []
    for item, response in tqdm(zip(data, responses), total=len(data), desc="Scoring"):
        score_info = score_frq_item(judger, item, response)
        results.append({
            "id": item["id"],
            "is_mcq": False,
            "gold": item["answer"],
            "response": response,
            **score_info,
        })

    errors = [row for row in results if not row["correct"]]
    write_jsonl(Path(args.output), results)
    write_jsonl(Path(args.errors), errors)

    raw_correct = sum(row["raw_correct"] for row in results)
    final_correct = sum(row["correct"] for row in results)
    total = len(results)
    print(f"Raw FRQ accuracy: {raw_correct}/{total} ({raw_correct / total * 100:.2f}%)")
    print(f"Postprocessed FRQ accuracy: {final_correct}/{total} ({final_correct / total * 100:.2f}%)")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.errors}")


if __name__ == "__main__":
    main()
