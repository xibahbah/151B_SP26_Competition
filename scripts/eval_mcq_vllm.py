#!/usr/bin/env python3
"""Run MCQ-only vLLM evaluation and write answer-accuracy JSONL results."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

from tqdm import tqdm
from vllm import LLM, SamplingParams


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


def build_user_prompt(question: str, options: list[str]) -> str:
    labels = [chr(65 + i) for i in range(len(options))]
    opts_text = "\n".join(f"{label}. {option.strip()}" for label, option in zip(labels, options))
    return f"{question}\n\nOptions:\n{opts_text}"


def extract_letter(text: str) -> str:
    patterns = [
        r"\\boxed\{\s*([A-J])\s*\}",
        r"Final Answer\s*:?\s*(?:\$\$)?\s*(?:\\boxed\{)?\s*([A-J])\b",
        r"correct (?:answer|option) is\s*:?\s*(?:\*\*)?([A-J])\b",
        r"(?:answer|option)\s+(?:is\s+)?(?:\*\*)?([A-J])\b",
        r"\*\*([A-J])\.\*\*",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if matches:
            return matches[-1].upper()
    matches = re.findall(r"\b([A-J])\b", text.upper())
    return matches[-1] if matches else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/public.jsonl")
    parser.add_argument("--output", default="results/mcq_baseline_full.jsonl")
    parser.add_argument("--errors", default="results/mcq_baseline_errors.jsonl")
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=-1, help="Use -1 for a fresh random sample seed.")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.78)
    parser.add_argument("--max-model-len", type=int, default=8192)
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("VLLM_USE_V1", "0")

    data = [row for row in read_jsonl(Path(args.data)) if row.get("options")]
    sample_seed = int(time.time_ns() % (2**32)) if args.seed < 0 else args.seed
    if args.sample_size is not None:
        rng = random.Random(sample_seed)
        data = rng.sample(data, min(args.sample_size, len(data)))
    print(f"Sample seed: {sample_seed}")

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
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
    )

    prompts = []
    for item in data:
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT_MCQ},
                {"role": "user", "content": build_user_prompt(item["question"], item["options"])},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)

    responses = []
    for start in tqdm(range(0, len(prompts), args.batch_size), desc="Generating"):
        batch = prompts[start : start + args.batch_size]
        outputs = llm.generate(batch, sampling_params=sampling_params)
        responses.extend(output.outputs[0].text.strip() for output in outputs)

    results = []
    for item, response in zip(data, responses):
        pred = extract_letter(response)
        gold = str(item["answer"]).strip().upper()
        correct = pred == gold
        results.append(
            {
                "id": item["id"],
                "is_mcq": True,
                "gold": gold,
                "response": response,
                "predicted_letter": pred,
                "correct": correct,
                "error_type": "correct" if correct else "wrong_letter",
                "sample_seed": sample_seed,
            }
        )

    errors = [row for row in results if not row["correct"]]
    write_jsonl(Path(args.output), results)
    write_jsonl(Path(args.errors), errors)

    correct = sum(row["correct"] for row in results)
    total = len(results)
    print(f"MCQ accuracy: {correct}/{total} ({correct / total * 100:.2f}%)")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.errors}")


if __name__ == "__main__":
    main()
