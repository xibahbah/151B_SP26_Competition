#!/usr/bin/env bash
set -euo pipefail

cmd="${1:-help}"

BASELINE="${BASELINE:-results/frq_full_4096_v2_repaired_latest.jsonl}"
SELF_SFT="${SELF_SFT:-data/self_distill_frq_sft.jsonl}"
HOLDOUT_IDS="${HOLDOUT_IDS:-data/self_distill_holdout_ids.json}"
EXTERNAL_SFT="${EXTERNAL_SFT:-data/external_math_final_1k.jsonl}"
MIXED_SFT="${MIXED_SFT:-data/mixed_selfdistill_external_sft.jsonl}"
A_DIR="${A_DIR:-outputs/qwen-frq-lora-selfdistill-safe}"
B_DIR="${B_DIR:-outputs/qwen-frq-lora-mixed-safe}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Missing required directory: $1" >&2
    exit 1
  fi
}

run_bg() {
  local pid_file="$1"
  local log_file="$2"
  shift 2
  mkdir -p "$(dirname "$pid_file")" "$(dirname "$log_file")"
  nohup "$@" > "$log_file" 2>&1 &
  local pid=$!
  echo "$pid" > "$pid_file"
  echo "Started PID $pid"
  echo "Log: $log_file"
}

tail_hint() {
  echo "Watch with: tail -f $1"
}

case "$cmd" in
  prepare)
    require_file "$BASELINE"
    python scripts/prepare_self_distill_sft.py \
      --data data/public.jsonl \
      --baseline-results "$BASELINE" \
      --output "$SELF_SFT" \
      --holdout-output "$HOLDOUT_IDS" \
      --holdout-size 200 \
      --seed 151 \
      --include-postprocess-correct \
      --max-teacher-chars 2500
    python scripts/summarize_frq_results.py --input "$BASELINE" --ids-file "$HOLDOUT_IDS"
    ;;

  train-a)
    require_file "$SELF_SFT"
    run_bg results/train_lora_selfdistill_safe.pid results/train_lora_selfdistill_safe.log \
      python -u scripts/train_lora.py \
        --train-file "$SELF_SFT" \
        --output-dir "$A_DIR" \
        --epochs 0.5 \
        --learning-rate 5e-5 \
        --max-length 4096 \
        --batch-size 1 \
        --grad-accum 16 \
        --lora-r 8 \
        --lora-alpha 16 \
        --lora-dropout 0.05
    tail_hint results/train_lora_selfdistill_safe.log
    ;;

  eval-a)
    require_file "$HOLDOUT_IDS"
    require_dir "$A_DIR"
    run_bg results/frq_lora_selfdistill_holdout.pid results/frq_lora_selfdistill_holdout.log \
      python -u scripts/eval_frq_vllm.py \
        --lora-path "$A_DIR" \
        --ids-file "$HOLDOUT_IDS" \
        --max-tokens 4096 \
        --output results/frq_lora_selfdistill_holdout.jsonl \
        --errors results/frq_lora_selfdistill_holdout_errors.jsonl
    tail_hint results/frq_lora_selfdistill_holdout.log
    ;;

  repair-a)
    require_file results/frq_lora_selfdistill_holdout.jsonl
    python scripts/repair_frq_outputs.py \
      --input results/frq_lora_selfdistill_holdout.jsonl \
      --output results/frq_lora_selfdistill_holdout_repaired.jsonl \
      --errors results/frq_lora_selfdistill_holdout_repaired_errors.jsonl
    python scripts/summarize_frq_results.py --input results/frq_lora_selfdistill_holdout_repaired.jsonl
    ;;

  prepare-mixed)
    require_file "$SELF_SFT"
    python scripts/prepare_external_math_sft.py \
      --preset math \
      --sample-size 1000 \
      --assistant-mode final \
      --min-level 2 \
      --max-level 4 \
      --balanced-by-topic \
      --output "$EXTERNAL_SFT"
    SELF_SFT="$SELF_SFT" EXTERNAL_SFT="$EXTERNAL_SFT" MIXED_SFT="$MIXED_SFT" python - <<'PY'
import json
import os
import random

paths = [os.environ["SELF_SFT"], os.environ["EXTERNAL_SFT"]]
rows = []
for path in paths:
    with open(path) as f:
        rows.extend(json.loads(line) for line in f if line.strip())
random.Random(151).shuffle(rows)
out = os.environ["MIXED_SFT"]
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")
print(f"Wrote {len(rows)} records to {out}")
PY
    ;;

  train-b)
    require_file "$MIXED_SFT"
    run_bg results/train_lora_mixed_safe.pid results/train_lora_mixed_safe.log \
      python -u scripts/train_lora.py \
        --train-file "$MIXED_SFT" \
        --output-dir "$B_DIR" \
        --epochs 0.5 \
        --learning-rate 3e-5 \
        --max-length 4096 \
        --batch-size 1 \
        --grad-accum 16 \
        --lora-r 8 \
        --lora-alpha 16 \
        --lora-dropout 0.05
    tail_hint results/train_lora_mixed_safe.log
    ;;

  eval-b)
    require_file "$HOLDOUT_IDS"
    require_dir "$B_DIR"
    run_bg results/frq_lora_mixed_holdout.pid results/frq_lora_mixed_holdout.log \
      python -u scripts/eval_frq_vllm.py \
        --lora-path "$B_DIR" \
        --ids-file "$HOLDOUT_IDS" \
        --max-tokens 4096 \
        --output results/frq_lora_mixed_holdout.jsonl \
        --errors results/frq_lora_mixed_holdout_errors.jsonl
    tail_hint results/frq_lora_mixed_holdout.log
    ;;

  repair-b)
    require_file results/frq_lora_mixed_holdout.jsonl
    python scripts/repair_frq_outputs.py \
      --input results/frq_lora_mixed_holdout.jsonl \
      --output results/frq_lora_mixed_holdout_repaired.jsonl \
      --errors results/frq_lora_mixed_holdout_repaired_errors.jsonl
    python scripts/summarize_frq_results.py --input results/frq_lora_mixed_holdout_repaired.jsonl
    ;;

  baseline-holdout)
    require_file "$BASELINE"
    require_file "$HOLDOUT_IDS"
    python scripts/summarize_frq_results.py --input "$BASELINE" --ids-file "$HOLDOUT_IDS"
    ;;

  status)
    ps aux | grep -E 'train_lora|eval_frq_vllm' | grep -v grep || true
    nvidia-smi || true
    ;;

  help|*)
    cat <<'EOF'
Usage:
  bash scripts/run_controlled_lora_ab.sh prepare
  bash scripts/run_controlled_lora_ab.sh train-a
  tail -f results/train_lora_selfdistill_safe.log
  bash scripts/run_controlled_lora_ab.sh eval-a
  tail -f results/frq_lora_selfdistill_holdout.log
  bash scripts/run_controlled_lora_ab.sh repair-a

If Adapter A beats the baseline holdout:
  bash scripts/run_controlled_lora_ab.sh prepare-mixed
  bash scripts/run_controlled_lora_ab.sh train-b
  bash scripts/run_controlled_lora_ab.sh eval-b
  bash scripts/run_controlled_lora_ab.sh repair-b

Helpful:
  bash scripts/run_controlled_lora_ab.sh baseline-holdout
  bash scripts/run_controlled_lora_ab.sh status

Override paths with env vars, for example:
  BASELINE=results/frq_full_4096_v2_repaired_latest.jsonl bash scripts/run_controlled_lora_ab.sh prepare
EOF
    ;;
esac
