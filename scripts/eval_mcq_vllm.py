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
    match = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if match:
        return match.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/public.jsonl")
    parser.add_argument("--output", default="results/mcq_baseline_full.jsonl")
    parser.add_argument("--errors", default="results/mcq_baseline_errors.jsonl")
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=-1, help="Use -1 for a fresh random sample seed.")
    parser.add_argument("--batch-size", type=int, default=None, help="Defaults to all prompts in one vLLM call, matching the starter notebook.")
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.50)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("VLLM_USE_V1", "0")

    data = [row for row in read_jsonl(Path(args.data)) if row.get("options")]
    sample_seed = int(time.time_ns() % (2**32)) if args.seed < 0 else args.seed
    if args.sample_size is not None:
        rng = random.Random(sample_seed)
        data = rng.sample(data, min(args.sample_size, len(data)))
    print(f"Sample seed: {sample_seed}")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model_id,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enable_prefix_caching=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
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
    if args.batch_size is None:
        outputs = llm.generate(prompts, sampling_params=sampling_params)
        responses = [output.outputs[0].text.strip() for output in outputs]
    else:
        for start in tqdm(range(0, len(prompts), args.batch_size), desc="Generating"):
            batch = prompts[start : start + args.batch_size]
            outputs = llm.generate(batch, sampling_params=sampling_params)
            responses.extend(output.outputs[0].text.strip() for output in outputs)

    has_gold = all("answer" in item for item in data)
    results = []
    for item, response in zip(data, responses):
        pred = extract_letter(response)
        row = {
            "id": item["id"],
            "is_mcq": True,
            "response": response,
            "postprocessed_response": f"\\boxed{{{pred}}}" if pred else response,
            "postprocessed_answer": pred,
            "predicted_letter": pred,
            "sample_seed": sample_seed,
        }
        if has_gold:
            gold = str(item["answer"]).strip().upper()
            correct = pred == gold
            row.update({
                "gold": gold,
                "correct": correct,
                "error_type": "correct" if correct else "wrong_letter",
            })
        else:
            row.update({"correct": None, "error_type": "unscored"})
        results.append(row)

    errors = [row for row in results if row["correct"] is False]
    write_jsonl(Path(args.output), results)
    write_jsonl(Path(args.errors), errors)

    total = len(results)
    if has_gold:
        correct = sum(row["correct"] for row in results)
        print(f"MCQ accuracy: {correct}/{total} ({correct / total * 100:.2f}%)")
    else:
        print(f"MCQ private inference complete: {total} predictions")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.errors}")


if __name__ == "__main__":
    main()
