# Reproducing ValueActionLens: Measuring the Value–Action Gap in LLMs

**Authors:** Jahanvi Jeswani, Ziming Lin, Xunmei Liu (University of Washington)

Reproducibility project for [Shen et al. (2025), "Mind the Value-Action Gap: Do LLMs Act in Alignment with Their Values?"](https://arxiv.org/pdf/2501.15463), EMNLP 2025.

We reproduce the central empirical claims using two models:
- **GPT-4o-mini** (OpenAI API, dynamic re-run) — scripts in `src/tasks/gpt_4o_mini_eval/`
- **Gemma-2-9B-it** (local Hugging Face inference on UW Hyak cluster) — scripts in `src/tasks/gemma_2_9b_eval/`

Original authors' repository: https://github.com/huashen218/value_action_gap

---

## Installation & Setup

### 1. Clone this repository

```bash
git clone https://github.com/Xunmei-liu/value_action_gap.git
cd value_action_gap
```

### 2. Create and activate a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For **local Gemma-2-9B-it inference** (requires CUDA-capable GPUs), additionally install:

```bash
pip install torch transformers accelerate
```

### 4. Set up API keys

For GPT-4o-mini:
```bash
export OPENAI_API_KEY=<your_openai_api_key>
```

For Gemma on the UW Hyak cluster, edit `src/tasks/gemma_2_9b_eval/setup_env.sh` with your own HF token, then:
```bash
source src/tasks/gemma_2_9b_eval/setup_env.sh
```

---

## Data Download

The VIA dataset is released in the original authors' repository. Clone it to get the data:

```bash
git clone https://github.com/huashen218/value_action_gap.git original_repo
cp -r original_repo/outputs/ outputs/
```

The key paired-action file used for Task 2 inference is:
```
src/outputs/full_data/value_action_gap_full_data_gpt_4o_generation.csv
```
(14,784 rows — 132 scenarios × 56 values × 2 polarities)

The static released model outputs (reused as the request grid for dynamic re-running) are:
```
outputs/evaluation/gpt-4o-mini_t1.csv
outputs/evaluation/gpt-4o-mini_t2.csv
```

---

## Repository Structure (our additions)

```
src/tasks/
├── gpt_4o_mini_eval/                 # GPT-4o-mini dynamic reproduction (ours)
│   ├── run_task1.py                  # Re-run Task 1 via current OpenAI API
│   ├── run_task2.py                  # Re-run Task 2 via current OpenAI API
│   ├── eval_alignment_rate.py        # Alignment Rate (F1)
│   ├── eval_alignment_distance.py    # Alignment Distance (ℓ₁)
│   ├── eval_alignment_ranking.py     # Alignment Ranking
│   └── eval_direction_check.py       # Diagnose Task 2 polarity convention
└── gemma_2_9b_eval/                  # Gemma-2-9B-it local reproduction (ours)
    ├── run_task1.py                  # Task 1 multi-GPU parallel inference (12 × 11 scenarios)
    ├── run_task2.py                  # Task 2 multi-GPU parallel inference + resume support
    ├── setup_env.sh                  # Cluster environment variables (set HF_TOKEN before use)
    ├── eval_alignment_rate.py        # Alignment Rate (F1)
    ├── eval_alignment_distance.py    # Alignment Distance (ℓ₁)
    ├── eval_alignment_ranking.py     # Alignment Ranking
    ├── eval_table4.py                # Table 4 cross-task inconsistency counts
    └── eval_alignment_full.py        # Combined alignment analysis (Rate + Distance + Ranking)
```

---

## Preprocessing

No standalone preprocessing step is required. The VIA dataset is used directly for inference. Format conversion from Gemma's wide-format outputs to the prompt-level format expected by evaluation scripts is handled internally within the evaluation scripts.

---

## Inference (Generation)

> **Note:** This project is inference-only. No model training is performed.

### GPT-4o-mini — Dynamic Re-run

All scripts are in `src/tasks/gpt_4o_mini_eval/`. Run from that directory.

```bash
cd src/tasks/gpt_4o_mini_eval

# Task 1 — re-run value statement rating with current API
python run_task1.py
# Output: dynamic_gpt4omini_t1.csv  (shape: 1056 × 4)
#         columns: country, topic, response, prompt_index

# Task 2 — re-run pairwise action selection with current API (prompt_index=5)
python run_task2.py
# Output: dynamic_gpt4omini_t2.csv  (shape: ~14330 × 7)
#         columns: country, topic, value, polarity, generation_prompt, model_choice, prompt_index
# Error log: dynamic_gpt4omini_t2_errors.csv  (small number of malformed prompts from source data)
```

**Important — Task 2 polarity convention for dynamic outputs:**
The dynamic file uses `positive → 0, negative → 1` (opposite to the released static file). This was verified by `eval_direction_check.py`. The evaluation scripts apply this convention automatically.

### Gemma-2-9B-it — Local Multi-GPU Inference

**Pretrained model:** `google/gemma-2-9b-it`
- Access: https://huggingface.co/google/gemma-2-9b-it (gated — requires HF account approval)
- ~18 GB disk, loaded in `bfloat16` with `device_map="auto"`, seed=42
- Hardware used: 4× NVIDIA L40S 48GB GPUs on UW Hyak cluster

**Task 1** — 4 GPU shards, deterministic decoding (temperature=0.0), full 12 countries × 11 topics, automatically merges shards on completion:
```bash
cd src/tasks/gemma_2_9b_eval
python run_task1.py
# Shard outputs: src/outputs/task1_gemma_statements_full_shard{0..3}.csv
# Merged output: src/outputs/task1_gemma_statements_full_parallel.csv
```

**Task 2** — 4× NVIDIA L40S GPUs, temperature=0.2, top_p=0.95. Supports resuming from a partial run automatically:
```bash
cd src/tasks/gemma_2_9b_eval
python run_task2.py
# Shard outputs: src/outputs/task2_results_gemma9b_full_shard{0..3}.csv
# Merged output: src/outputs/task2_results_gemma9b_full.csv
```

---

## Evaluation

### GPT-4o-mini Dynamic Evaluation

All scripts read from files generated by `run_task1.py` and `run_task2.py`. Run from `src/tasks/gpt_4o_mini_eval/`.

```bash
cd src/tasks/gpt_4o_mini_eval

# (Optional) Confirm Task 2 polarity direction convention
python eval_direction_check.py
# Output: dynamic_direction_check_summary.csv

# Alignment Rate (F1)
python eval_alignment_rate.py
# Outputs: dynamic_full_country_results.csv, dynamic_full_t1_pd.csv, dynamic_full_t2_pd.csv

# Alignment Distance (ℓ₁)
python eval_alignment_distance.py
# Outputs: dynamic_full_distance_country.csv (12×56), dynamic_full_distance_topic.csv (11×56)
#          dynamic_full_avg_distance_by_{country,topic,value_country,value_topic}.csv

# Alignment Ranking
python eval_alignment_ranking.py
# Outputs: dynamic_full_ranking_{country,topic}.csv
#          dynamic_full_ranking_{country,topic}_top{1,5}_counts.csv
```

### Gemma-2-9B-it Evaluation

Update `T1_PATH` / `T2_PATH` at the top of each script to point to your output files, then run from `src/tasks/gemma_2_9b_eval/`.

```bash
cd src/tasks/gemma_2_9b_eval

# Alignment Rate (F1)
python eval_alignment_rate.py
# Outputs: gemma_full_country_results.csv, gemma_full_t1_pd.csv, gemma_full_t2_pd.csv

# Alignment Distance (ℓ₁)
python eval_alignment_distance.py
# Outputs: distance CSVs by country, topic, and value

# Alignment Ranking
python eval_alignment_ranking.py
# Outputs: ranking CSVs (top-1 and top-5 counts by country and topic)

# Table 4 cross-task inconsistency counts
python eval_table4.py
# Outputs to: src/outputs/table4_verify_gemma2_full/
#   table4_summary.csv      — total misaligned counts and %
#   task1_task2_joined.csv  — merged Task 1 & Task 2 binary labels per (country, topic, value)

# Combined alignment analysis (Rate + Distance + Ranking)
python eval_alignment_full.py
# Outputs to: src/outputs/alignment_analysis_gemma9b_corrected/
```

Expected Table 4 results:
```
Paper (full Gemma):  (A,D)=497,  (D,A)=695, Total=1192 (16.13%)
Ours (full Gemma):   (A,D)=529,  (D,A)=671, Total=1200 (16.37%)
```

---

## Results Summary

| Model | Reproduced F1 | Paper's Reported F1 |
|---|---|---|
| GPT-4o-mini (dynamic, March 2026) | 0.5262 | 0.564 |
| Gemma-2-9B-it (full) | 0.461 | 0.461 |

Both models exhibit a nontrivial value–action gap with systematic variation across countries and values, supporting the paper's central claims. The small GPT-4o-mini difference is attributable to API model drift between 2025 and March 2026.

---

---

# Original Repository: Value Alignment in LLMs

<!-- # value_action_data -->

This repository includes codes for two papers presented in [EMNLP 2025](https://2025.emnlp.org/) regarding **value alignment in LLMs**:

- [EMNLP 2025 Main](https://arxiv.org/pdf/2501.15463?): [Mind the Value-Action Gap: Do LLMs Act in Alignment with Their Values?](https://arxiv.org/pdf/2501.15463?)
- [EMNLP 2025 WiNLP Workshop](https://arxiv.org/pdf/2409.09586): [ValueCompass: A Framework for Measuring Contextual Value Alignment Between Human and LLMs](https://arxiv.org/pdf/2409.09586)

These two papers aim to answer the core concerning questions related to value alignment:
- **RQ1**: *How can we systematically capture human values and evaluate the extent to which LLM aligns with them?* (Value Alignment between **Humans & LLMs**, [EMNLP 2025 WiNLP Workshop](https://arxiv.org/pdf/2409.09586))
- **RQ2**: *To what extent do LLM-generated value statements align with their value-informed actions?* (Value Alignment between LLM's **value claim & the corresponding actions**, [EMNLP 2025 Main](https://arxiv.org/pdf/2501.15463?))

## Overview

To address **RQ1** -- *Human-AI Value Alignment* -- we propose **ValueCompass**, a framework for systematically measuring value alignment between LLMs and humans across contextual scenarios. See below figure for an overview.

<img src="figures/figure-valuecompass.png" width="58%" alt="Description">

To address **RQ2** -- *LLM's Value-Action Alignment* -- we propose **ValueActionLens** Framework, associated with the VIA (Value-Informed Dataset) dataset, to assess the alignment between LLMs' stated values & value-informed actions. See below figure for an example of GPT4o's Value-Action Gap.

<img src="figures/figure-value-action-gap.png" width="58%" alt="Description">

## Evaluating Value Alignment

To evaluate the value alignment in LLMs, codes are released in this directory: [Value-Action Alignment Tasks](https://github.com/huashen218/value_action_gap/tree/main/src/tasks).

## VIA Dataset

The full VIA dataset can be accessed in this directory: [Value-Informed Dataset (VIA)](https://github.com/huashen218/value_action_gap/tree/main/outputs/data_release)

```bibtex
@article{shen2024valuecompass,
    title={Valuecompass: A framework of fundamental values for human-ai alignment},
    author={Shen, Hua and Knearem, Tiffany and Ghosh, Reshmi and Yang, Yu-Ju and Mitra, Tanushree and Huang, Yun},
    journal={arXiv preprint arXiv:2409.09586},
    year={2024}
}

@article{shen2025mind,
  title={Mind the Value-Action Gap: Do LLMs Act in Alignment with Their Values?},
  author={Shen, Hua and Clark, Nicholas and Mitra, Tanushree},
  journal={arXiv preprint arXiv:2501.15463},
  year={2025}
}
