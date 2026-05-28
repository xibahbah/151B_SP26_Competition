# CSE 151B Competition — Starter Code

Open **`starter_code_cse151b_comp.ipynb`** to get started.

The notebook covers environment setup, inference with Qwen3-4B-Thinking (INT8), and scoring against the public dataset.

## Contents

| File | Description |
|---|---|
| `starter_code_cse151b_comp.ipynb` | Main entry point |
| `judger.py` | Response scoring logic |
| `utils.py` | Utilities used by `judger.py` |
| `data/public.jsonl` | Public dataset with ground-truth answers |
| `results/` | Output JSONL files written at runtime |

## FRQ-first workflow

Install the RunPod environment:

```bash
pip install -r requirements-lora.txt
```

Run a 20-question smoke test:

```bash
python scripts/eval_frq_vllm.py \
  --sample-size 20 \
  --output results/frq_smoke.jsonl \
  --errors results/frq_smoke_errors.jsonl
```

Run the full public FRQ baseline:

```bash
python scripts/eval_frq_vllm.py \
  --output results/frq_baseline_full.jsonl \
  --errors results/frq_baseline_errors.jsonl
```

Re-score saved generations without regenerating:

```bash
python scripts/rescore_frq_results.py \
  --input results/frq_baseline_full.jsonl \
  --output results/frq_baseline_full_rescored.jsonl \
  --errors results/frq_baseline_errors.jsonl
```

Analyze the remaining FRQ failures:

```bash
python scripts/analyze_frq_errors.py \
  --results results/frq_baseline_full_rescored.jsonl \
  --examples results/frq_error_examples.jsonl \
  --limit 3
```

Run the offline repair pass on saved FRQ outputs:

```bash
python scripts/repair_frq_outputs.py \
  --input results/frq_baseline_full.jsonl \
  --output results/frq_baseline_full_repaired.jsonl \
  --errors results/frq_baseline_full_repaired_errors.jsonl \
  --oracle-errors results/frq_baseline_full_oracle_errors.jsonl
```

Prepare local training data for SFT/LoRA:

```bash
python scripts/prepare_sft_data.py \
  --input path/to/your_training_set.jsonl \
  --output data/train_sft_frq.jsonl
```

Prepare external FRQ-style math data without downloading the full dataset:

```bash
python scripts/prepare_external_math_sft.py \
  --preset math \
  --sample-size 2000 \
  --assistant-mode final \
  --min-level 3 \
  --balanced-by-topic \
  --output data/external_math_sft_2k.jsonl
```

For a larger reasoning dataset, stream a small sample from OpenR1:

```bash
python scripts/prepare_external_math_sft.py \
  --preset openr1 \
  --sample-size 2000 \
  --assistant-mode final \
  --output data/openr1_math_sft_2k.jsonl
```

Train the LoRA adapter:

```bash
python scripts/train_lora.py \
  --train-file data/external_math_sft_2k.jsonl \
  --output-dir outputs/qwen-frq-lora
```

Evaluate the LoRA adapter:

```bash
python scripts/eval_frq_vllm.py \
  --lora-path outputs/qwen-frq-lora \
  --output results/frq_lora_full.jsonl \
  --errors results/frq_lora_errors.jsonl
```
