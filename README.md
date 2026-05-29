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

Create a fixed public-FRQ holdout before LoRA experiments:

```bash
python scripts/split_public_frq_holdout.py \
  --seed 151 \
  --holdout-size 200 \
  --train-output data/public_frq_train.jsonl \
  --holdout-output data/public_frq_holdout.jsonl \
  --holdout-ids-output data/public_frq_holdout_ids.json \
  --sft-output data/public_frq_train_sft.jsonl
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

Evaluate the LoRA adapter on the fixed holdout:

```bash
python scripts/eval_frq_vllm.py \
  --lora-path outputs/qwen-frq-lora \
  --ids-file data/public_frq_holdout_ids.json \
  --output results/frq_lora_holdout.jsonl \
  --errors results/frq_lora_holdout_errors.jsonl
```

## Controlled LoRA A/B

Use this only for FRQ. MCQ should stay on the baseline vLLM path.

The current public-FRQ baseline to beat is the repaired 4096-token run:

```text
results/frq_full_4096_v2_repaired_latest.jsonl
```

Run Adapter A, the safer self-distill format LoRA:

```bash
bash scripts/run_controlled_lora_ab.sh prepare
bash scripts/run_controlled_lora_ab.sh train-a
tail -f results/train_lora_selfdistill_safe.log
bash scripts/run_controlled_lora_ab.sh eval-a
tail -f results/frq_lora_selfdistill_holdout.log
bash scripts/run_controlled_lora_ab.sh repair-a
```

Only if Adapter A beats the baseline holdout, run Adapter B:

```bash
bash scripts/run_controlled_lora_ab.sh prepare-mixed
bash scripts/run_controlled_lora_ab.sh train-b
tail -f results/train_lora_mixed_safe.log
bash scripts/run_controlled_lora_ab.sh eval-b
tail -f results/frq_lora_mixed_holdout.log
bash scripts/run_controlled_lora_ab.sh repair-b
```

Check active jobs:

```bash
bash scripts/run_controlled_lora_ab.sh status
```

Reject an adapter if the repaired holdout is below the baseline holdout. If it
passes holdout, run full FRQ before trusting it.
