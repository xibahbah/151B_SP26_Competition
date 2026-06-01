# CSE 151B Kaggle Competition Submission

Final code submission for the math reasoning Kaggle competition.

## Entry Point

The required single entry point is:

```python
from run_inference import run_inference

run_inference(
    data_path="data/private.jsonl",
    output_csv="submission.csv",
)
```

This performs the complete pipeline end to end:

1. Loads `Qwen/Qwen3-4B-Thinking-2507` with vLLM and bitsandbytes.
2. Runs MCQ inference with the original baseline MCQ prompt and settings.
3. Runs FRQ inference with the final concise FRQ prompt and settings.
4. Applies deterministic FRQ answer repair/post-processing.
5. Writes the Kaggle CSV with columns `id,response`.

No LoRA or fine-tuned checkpoint is used in the final submission.

## Setup

Use a CUDA GPU environment. The final runs were performed on RunPod with:

- GPU: NVIDIA A40, 48 GB VRAM
- CUDA driver shown by RunPod: CUDA 12.8
- Approximate full private-set inference time: 10-12 hours

Install dependencies:

```bash
pip install -r requirements.txt
```

The model weights are downloaded automatically from Hugging Face by vLLM on the
first run and cached by the environment. No manually uploaded model weights are
required.

## Reproduce Submission

Place the private dataset at:

```text
data/private.jsonl
```

Run:

```bash
python run_inference.py \
  --data data/private.jsonl \
  --output-csv submission.csv \
  --results-dir results/private_run
```

For a long remote run, use:

```bash
mkdir -p results/logs
nohup python -u run_inference.py \
  --data data/private.jsonl \
  --output-csv submission.csv \
  --results-dir results/private_run \
  > results/logs/run_inference.log 2>&1 &
```

The same pipeline is also called from `starter_code_cse151b_comp.ipynb`.

## Final Hyperparameters

MCQ:

- Model: `Qwen/Qwen3-4B-Thinking-2507`
- Quantization/load format: `bitsandbytes`
- Temperature: `0.6`
- Top-p: `0.95`
- Top-k: `20`
- Max tokens: `32768`
- Max model length: `16384`
- GPU memory utilization: `0.50`

FRQ:

- Model: `Qwen/Qwen3-4B-Thinking-2507`
- Quantization/load format: `bitsandbytes`
- Temperature: `0.0`
- Top-p: `1.0`
- Top-k: `-1`
- Max tokens: `4096`
- Max model length: `8192`
- Batch size: `5`
- GPU memory utilization: `0.78`

Post-processing is implemented in `scripts/repair_frq_outputs.py` and is called
inside `run_inference()`.
