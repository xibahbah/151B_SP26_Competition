# Claude Write-Up Handoff README

This file is a complete context handoff for regenerating a final write-up for the CSE 151B/251B Kaggle math reasoning project. It is intentionally more narrative and detailed than the normal project `README.md`. The normal `README.md` is for graders running the code; this file is for an LLM or teammate writing a report from the full project history.

## One-Sentence Summary

We built a reproducible inference pipeline for a mixed MCQ/free-response math competition using `Qwen/Qwen3-4B-Thinking-2507` served with vLLM and bitsandbytes, preserved the strong MCQ baseline, improved FRQ through prompt discipline plus deterministic answer repair, rejected LoRA/fine-tuning after controlled tests, and reached a final Kaggle public score of `0.674`.

## Repository and Final State

Repository:

```text
https://github.com/xibahbah/151B_SP26_Competition
```

Important final files:

```text
README.md
requirements.txt
run_inference.py
starter_code_cse151b_comp.ipynb
frq_tools.py
judger.py
utils.py
scripts/eval_mcq_vllm.py
scripts/eval_frq_vllm.py
scripts/repair_frq_outputs.py
reports/final_report.tex
reports/final_report.pdf
reports/references.bib
```

Final report currently exists here:

```text
reports/final_report.pdf
reports/final_report.tex
```

Latest important commits at the time this handoff was made:

```text
79b2d1b Expand final report and verify references
e2c3e75 Add final project report
d84453a Align final notebook with grader workflow
96a3ad6 Clean final inference submission
```

Final code exposes the required single entry point:

```python
from run_inference import run_inference

run_inference(
    data_path="data/private.jsonl",
    output_csv="submission.csv",
)
```

This function performs the full pipeline:

1. Load the private dataset.
2. Split MCQ and FRQ examples.
3. Load `Qwen/Qwen3-4B-Thinking-2507` with vLLM and bitsandbytes.
4. Run MCQ inference with baseline/starter-style settings.
5. Release the MCQ engine and clear GPU memory.
6. Run FRQ inference with final concise FRQ prompt.
7. Apply deterministic FRQ repair/post-processing.
8. Merge MCQ and FRQ responses by `id`.
9. Validate the CSV.
10. Write `submission.csv` with columns `id,response`.

No final LoRA or fine-tuned checkpoint is used.

## User Preferences and Working Style

Keith preferred:

- Practical results over elegant theory.
- Terminal commands that can run remotely with `nohup`.
- Logs written to text files under `results/logs/`.
- Long jobs detached from Jupyter so the laptop can close or disconnect.
- Short, direct next-step commands when under time pressure.
- Keep MCQ as close to the baseline as possible because MCQ was already strong.
- Focus improvement effort on FRQ.
- Do not keep exploratory LoRA/helper clutter in final repo.
- Final grader code should be clean and reproducible from one notebook or one function.

Typical remote-run pattern:

```bash
mkdir -p results/logs
nohup python -u run_inference.py \
  --data data/private.jsonl \
  --output-csv submission.csv \
  --results-dir results/private_run \
  > results/logs/run_inference.log 2>&1 &
echo $! > results/private_submission.pid
```

Monitoring pattern:

```bash
tail -f results/logs/run_inference.log
ps -p $(cat results/private_submission.pid) -o pid,etime,%cpu,%mem,stat,command
nvidia-smi
```

## Competition Setup

The competition required a CSV submission with:

```csv
id,response
```

The `response` field had to contain the full model-generated response trace, including reasoning and a final boxed answer. The grader extracts the final answer from the response. This matters because submitting only the extracted answer was not the specified format.

Public dataset:

```text
1126 total examples
375 MCQ
751 FRQ
```

Private dataset used for submission:

```text
943 total examples
300 MCQ
643 FRQ
```

Core metric:

```text
Answer accuracy only
```

No other metric was optimized. Explanation quality, BLEU, reasoning elegance, and runtime only mattered insofar as they affected accuracy or feasibility.

## Environment

Final runs were performed on RunPod.

```text
GPU: NVIDIA A40
VRAM: 48 GB
CUDA shown by RunPod: 12.8
Approximate full private inference time: 10-12 hours
```

Key dependencies:

```text
vllm==0.8.5.post1
bitsandbytes
sympy
numpy
tqdm
antlr4-python3-runtime==4.11.1
```

Important environment issues encountered:

1. Newer vLLM/PyTorch initially failed with CUDA driver mismatch:

   ```text
   RuntimeError: The NVIDIA driver on your system is too old
   ```

2. Downgrading to `vllm==0.8.5.post1` introduced a tokenizer compatibility issue with too-new Transformers:

   ```text
   AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended
   ```

3. This was fixed by using a compatible Transformers range:

   ```bash
   pip install -U "transformers>=4.51,<4.54" tokenizers
   ```

4. Jupyter could disconnect while a command still ran. The solution was to run long jobs with `nohup` in terminal, not inside notebook cells.

5. vLLM could OOM if another kernel or process still held GPU memory. Kill old kernels/processes before loading a new model.

## Final Model and Hyperparameters

Base model:

```text
Qwen/Qwen3-4B-Thinking-2507
```

Serving:

```text
vLLM
bitsandbytes quantization
bitsandbytes load format
```

Final MCQ settings:

```text
Temperature: 0.6
Top-p: 0.95
Top-k: 20
Max output tokens: 32768
Max model length: 16384
GPU memory utilization: 0.50
Prompt: baseline/starter-style MCQ prompt
```

Final FRQ settings:

```text
Temperature: 0.0
Top-p: 1.0
Top-k: -1
Max output tokens: 4096
Max model length: 8192
Batch size: 5
GPU memory utilization: 0.78
Prompt: concise FRQ prompt with answer-count and format constraints
```

Important hyperparameter lesson:

```text
More FRQ max tokens was not better.
```

A 6144-token FRQ run made the model ramble more and lowered public FRQ accuracy.

## Final Public Results and Kaggle Scores

Public FRQ trajectory:

```text
Base / early FRQ baseline:       430/751 = 57.26%
First stronger repair pass:      463/751 = 61.65%
Template repair v3-v5:           515/751 = 68.58%
Final template repair v6/v10:    544/751 = 72.44%
Available repair oracle ceiling: 546/751 = 72.70%
```

Final public FRQ error breakdown after repair:

```text
Raw correct:                    425
Correct after postprocess:      119
Remaining expression_format:    109
Remaining wrong_math:            64
Remaining wrong_answer_count:    28
Remaining precision_rounding:     6
Total:                          751
```

Kaggle public leaderboard progression:

```text
Initial full-response submission:          0.650
Targeted safe version:                     0.653
Cleaned last-shot submission:              0.667
Improved final-answer repair submission:   0.671
Final aggressive-but-safe submission:       0.674
```

Private leaderboard score was not available when the report was written.

## Main Project Narrative

The project started as a notebook-based baseline using a Qwen thinking model through vLLM. Early work focused on simply getting the model to load on RunPod with bitsandbytes quantization. There were many environment issues: CUDA driver mismatch with newer packages, a Transformers tokenizer attribute error after downgrading vLLM, notebook kernel memory leaks, and vLLM OOMs when old processes were still alive.

Once the model loaded, the first goal was to understand accuracy by question type. MCQ was already strong with the baseline prompt and decoding settings. Attempts to change MCQ prompting, reduce token budget, or use a different generalized prompt hurt MCQ. The decision was: do not touch MCQ unless absolutely necessary.

FRQ was the main bottleneck. A random 200-FRQ sample took about 90 minutes under early settings, and full public FRQ took several hours. The base model often solved the problem in the body of its reasoning, but the final boxed answer was malformed, missing, too verbose, or contradicted by later text. The most important realization was that the task was not only mathematical reasoning. It was reasoning plus answer extraction discipline.

The first major FRQ improvement came from post-processing. The system extracted the last boxed answer, normalized answer markers such as `FINAL:` or `answer:`, removed units when not required, converted separators into comma-separated lists, stripped thousands commas, and preserved `%` or `$` only when the problem required them. This improved results but did not solve enough cases.

The second major FRQ improvement came from deterministic repair and template solvers. Public error analysis revealed repeated problem families: half-life, chi-square, confidence intervals, trigonometric feature extraction, binary addition, sample standard deviation, proportional sample size, geometric formulas, and similar algebra/statistics word-problem patterns. Template solvers were added only when the problem text matched a clear pattern. This drove public FRQ up to `544/751`.

Fine-tuning was explored but rejected. Several LoRA/SFT directions were tested:

1. Train on all public FRQ: rejected as overfitting risk and not a real generalization solution.
2. Public self-distillation: train on base-correct public examples and evaluate on held-out public FRQ.
3. External math data, including OpenR1/math-style data: rejected because solution-style targets did not match the required concise final-answer format.
4. Mixed public/external LoRA: rejected because sampled performance did not beat the repaired base pipeline.

The most important LoRA result:

```text
Public self-distill adapter on 200 held-out FRQ:
Raw:       95/200 = 47.50%
Repaired: 110/200 = 55.00%
```

This was not strong enough to replace the base repaired pipeline.

The 6144-token FRQ experiment was also rejected:

```text
6144-token full public FRQ:
Raw:           419/751 = 55.79%
Postprocessed: 422/751 = 56.19%
```

It got worse because the thinking model continued after finding the answer and sometimes boxed later contradictory or unfinished text.

Multi-generation was tried on wrong examples but was too expensive and weak. A run over wrong examples produced very low raw candidate accuracy and did not become the final path. The practical path was deterministic repair plus careful private CSV cleanup.

The final deliverable was a clean repo with one entry point, no final LoRA checkpoint, and a notebook that calls `run_inference()`.

## What Worked

### 1. Preserving MCQ baseline

MCQ was already strong. Prompt changes reduced it. The final system treats MCQ and FRQ separately.

### 2. FRQ concise prompt

The final FRQ prompt emphasizes:

- One final boxed answer.
- Exactly as many answer values as `[ANS]` blanks.
- Preserve order.
- Avoid units unless required.
- Avoid thousands commas.
- Use exact expressions where natural.
- Obey rounding instructions.
- Stop after the final answer.

### 3. Deterministic answer repair

The repair layer is the main improvement. It recovers answers hidden in traces and fixes common final-box failures.

Generic repairs include:

- Last valid boxed-answer extraction.
- Answer-marker extraction.
- Strip labels such as `x =`, `answer:`, `FINAL:`.
- Convert semicolons, pipes, and newlines into comma-separated values.
- Remove thousands commas.
- SymPy simplification when safe.
- Tuple/list wrapping.
- Numeric-token extraction.
- Option-letter normalization for embedded choice blanks.
- Wrong answer-count rejection.

Template repairs include:

- Half-life equations.
- Tangent general solution.
- Sample standard deviation tables.
- Bernstein / probability templates.
- Daily decay / substance decay.
- Arc radius.
- Binary addition.
- Resistance formulas.
- Wire Pythagorean problem.
- Mobile piecewise problem.
- Two-mean sample size.
- Mean sample size.
- Trig values.
- Exact radians.
- Numeric expression evaluation.
- Polar circle.
- Linear appreciation.
- Binomial first-four terms.
- Percentile locator.
- Population variance.
- Car rental break-even.
- Chi-square goodness-of-fit.
- Chi-square independence.
- Confidence interval templates.
- Proportion sample-size templates.
- Lighthouse/ramp/geometry word problems.
- Exponential/logistic growth patterns.

### 4. Remote `nohup` workflow

Long jobs need to run remotely with logs:

```bash
nohup python -u ... > results/logs/name.log 2>&1 &
```

This prevented Jupyter disconnects from killing the real job.

### 5. CSV cleanup before submission

The final CSV had to be validated:

- Row count equals private set row count.
- Unique IDs.
- No blank responses.
- Every response has a boxed answer.
- MCQ boxes contain valid letters.
- Avoid obviously bad boxes like `\boxed{Okay}` or `\boxed{So}`.

## What Failed or Was Rejected

### Modified MCQ prompting

Failed because MCQ baseline was already strong and extra reasoning/format instructions reduced MCQ performance.

### Reducing MCQ max tokens

Risky because the thinking model sometimes needed a long trace. MCQ token budget was kept high.

### Longer FRQ max tokens

Failed. 6144 tokens made FRQ worse because outputs rambled and contradicted themselves.

### LoRA on public FRQ

Useful for testing mechanics but risky for overfitting. Did not beat repaired base pipeline.

### External math LoRA

External solution-style data taught the model a different answer style and did not help the competition format.

### Multi-generation on wrong examples

Too slow and weak under the deadline. A broad multi-generation attempt did not provide enough final gains.

### Using another external model for private answers

This was discussed jokingly/under pressure, but the final system did not use external proprietary model-generated answers for private data. The final submission used the base Qwen model plus deterministic repair.

## Overfitting Discussion

There is overfitting risk in the final system because template solvers were derived from public error analysis. However:

1. No private labels were used.
2. Private predictions were generated from private problem text only.
3. Generic repairs were preferred over narrow templates.
4. Template solvers matched mathematical patterns, not hard-coded public IDs.
5. The private set appeared to come from the same broad problem distribution, so public-derived templates were expected to transfer partially.

The report should be honest: this is not a pure general-purpose math solver. It is a competition inference pipeline that combines an LLM with deterministic answer repair.

## Final Report Structure to Regenerate

Use this structure if regenerating a report.

### Abstract

Mention:

- Mixed MCQ/FRQ math reasoning.
- Model: `Qwen/Qwen3-4B-Thinking-2507`.
- vLLM + bitsandbytes on A40.
- Final solution is an inference pipeline, not LoRA.
- MCQ baseline preserved.
- FRQ prompt + repair.
- Public FRQ improved from `430/751` to `544/751`.
- Final Kaggle public score `0.674`.
- Repo link.

### Introduction

Cover:

- Problem definition.
- Submission format `id,response`.
- Response must contain full trace and final boxed answer.
- Dataset type: mixed MCQ and FRQ.
- Why this is hard: heterogeneous answer formats.
- Significance: LLMs need correct output interfaces, not just reasoning.
- Technical challenge: prompt changes can hurt MCQ, FRQ output space is broad, thinking model rambles, vLLM/GPU constraints, evaluation brittle.
- Contributions:
  - Type-specific inference.
  - Concise FRQ prompt.
  - Deterministic repair layer.
  - Reproducible `run_inference()`.

### Related Work

Use valid citations:

- Transformer: Vaswani et al. 2017, arXiv 1706.03762.
- GPT-3/few-shot: Brown et al. 2020, arXiv 2005.14165.
- Qwen3: Qwen3 technical report, arXiv 2505.09388, and Qwen model card.
- Chain-of-thought: Wei et al. 2022, arXiv 2201.11903.
- Self-consistency: Wang et al. 2023, arXiv 2203.11171.
- vLLM/PagedAttention: Kwon et al. 2023, arXiv 2309.06180.
- LoRA: Hu et al. 2022, arXiv 2106.09685.

### Methods

Describe:

- Dataset split.
- MCQ and FRQ as separate paths.
- Model generation:

  ```text
  response = model(system_prompt, user_prompt)
  repaired_response = h_frq(response, problem)
  ```

- Submission contract.
- Final inference algorithm.
- Repair candidate categories.
- Template solvers.
- Why no fine-tuned model was used.

### Experiments

Baselines:

- Starter vLLM baseline.
- Modified MCQ prompts.
- Concise FRQ prompt.
- 6144-token FRQ.
- LoRA/SFT.
- Deterministic repair.

Evaluation:

- `Judger.auto_judge()` for public FRQ.
- Kaggle public leaderboard for submissions.
- Accuracy only.

Implementation details:

- RunPod A40.
- CUDA 12.8.
- Full private runtime 10-12 hours.
- `nohup` logs.
- vLLM + bitsandbytes.

Results:

- Public FRQ table.
- Kaggle public score table.
- LoRA failure.
- 6144-token failure.

Ablations:

- MCQ prompt changes rejected.
- FRQ structural prompt kept.
- 6144 max token rejected.
- LoRA rejected.
- Repair kept.

### Discussion

Include:

- Achievements.
- Bottlenecks.
- Overfitting risk.
- Lessons learned.
- What to do next:
  - verifier model,
  - symbolic solvers,
  - focused self-consistency,
  - better held-out validation.
- Team responsibilities: Keith handled end-to-end.

## Final Hyperparameter Table

Use this in the report.

| Setting | MCQ | FRQ |
|---|---:|---:|
| Model | Qwen/Qwen3-4B-Thinking-2507 | Qwen/Qwen3-4B-Thinking-2507 |
| Serving | vLLM + bitsandbytes | vLLM + bitsandbytes |
| Temperature | 0.6 | 0.0 |
| Top-p | 0.95 | 1.0 |
| Top-k | 20 | -1 |
| Max output tokens | 32768 | 4096 |
| Max model length | 16384 | 8192 |
| Batch size | baseline/all configured | 5 |
| GPU memory utilization | 0.50 | 0.78 |

## Results Tables

### Public FRQ Development

| System variant | Correct | Accuracy |
|---|---:|---:|
| Base / early repair baseline | 430/751 | 57.26% |
| First stronger repair pass | 463/751 | 61.65% |
| Template repair v3-v5 | 515/751 | 68.58% |
| Final template repair v6/v10 | 544/751 | 72.44% |
| Available repair oracle ceiling | 546/751 | 72.70% |

### Kaggle Public Leaderboard Progression

| Submission stage | Public score |
|---|---:|
| Initial full-response submission | 0.650 |
| Targeted safe version | 0.653 |
| Cleaned last-shot submission | 0.667 |
| Improved final-answer repair submission | 0.671 |
| Final aggressive-but-safe submission | 0.674 |

## Important Commands from the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Compatibility fix used during debugging:

```bash
pip install "vllm==0.8.5.post1" bitsandbytes sympy numpy tqdm antlr4-python3-runtime==4.11.1
pip install -U "transformers>=4.51,<4.54" tokenizers
```

Run final inference:

```bash
python run_inference.py \
  --data data/private.jsonl \
  --output-csv submission.csv \
  --results-dir results/private_run
```

Run final inference remotely:

```bash
mkdir -p results/logs
nohup python -u run_inference.py \
  --data data/private.jsonl \
  --output-csv submission.csv \
  --results-dir results/private_run \
  > results/logs/run_inference.log 2>&1 &
echo $! > results/private_submission.pid
```

Monitor:

```bash
tail -f results/logs/run_inference.log
ps -p $(cat results/private_submission.pid) -o pid,etime,%cpu,%mem,stat,command
nvidia-smi
```

Compile report:

```bash
cd reports
latexmk -pdf -interaction=nonstopmode -halt-on-error final_report.tex
```

## Valid References

Use the existing `reports/references.bib`. It contains:

- Vaswani et al., "Attention Is All You Need", NeurIPS 2017, arXiv:1706.03762.
- Brown et al., "Language Models are Few-Shot Learners", NeurIPS 2020, arXiv:2005.14165.
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022, arXiv:2106.09685.
- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023, arXiv:2309.06180.
- Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", NeurIPS 2022, arXiv:2201.11903.
- Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in Language Models", ICLR 2023, arXiv:2203.11171.
- Yang et al., "Qwen3 Technical Report", arXiv:2505.09388.
- Qwen Team model card for `Qwen/Qwen3-4B-Thinking-2507`.

## Suggested Tone for Regenerated Write-Up

The write-up should sound honest and technical:

- Do not claim LoRA improved the final system.
- Do not claim the final system is a pure model improvement.
- Do claim the final improvement came from robust inference engineering.
- Be explicit that deterministic repair can overfit public templates.
- Be clear that no private labels were used.
- Emphasize that final code is reproducible through `run_inference()`.
- Avoid making the work sound like a black-box miracle. The real story is prompt discipline, careful error analysis, and deterministic answer repair.

## Best Short Abstract Draft

This project built a reproducible inference system for a mixed multiple-choice and free-response math reasoning competition. The final system used `Qwen/Qwen3-4B-Thinking-2507` served with vLLM and bitsandbytes on an NVIDIA A40 GPU. Rather than relying on fine-tuning, the selected approach preserved the strong baseline MCQ pipeline and focused on FRQ failures through a concise structural prompt plus deterministic answer repair. Public FRQ accuracy improved from `430/751` (`57.26%`) to `544/751` (`72.44%`), while Kaggle public score improved from `0.650` to `0.674`. Experiments with longer FRQ generation, multi-generation, and LoRA/SFT were rejected because they either increased rambling, cost too much inference time, or failed to beat the repaired base model. The final repository exposes a single `run_inference()` function that loads the model, runs MCQ and FRQ inference, repairs FRQ answers, validates outputs, and writes the final CSV.

## If Claude Needs to Regenerate the Report

Claude should read, in this order:

1. `CLAUDE_WRITEUP_README.md` - this handoff.
2. `README.md` - final grader-facing instructions.
3. `reports/final_report.tex` - current report source.
4. `reports/references.bib` - verified citations.
5. `run_inference.py` - final entry point.
6. `scripts/repair_frq_outputs.py` - repair logic.
7. `scripts/eval_mcq_vllm.py` and `scripts/eval_frq_vllm.py` - inference details.

Then regenerate a report using the template in `reports/final_report.tex` or the provided class template. Preserve the numbers and the main conclusion: the final system is a vLLM inference pipeline with deterministic FRQ repair, not a fine-tuned model.
