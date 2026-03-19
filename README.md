# Value Alignment in LLMs
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




To address **RQ2** -- *LLM's Value-Action Alignment* -- we propose **ValueActionLens** Framework, associated with the VIA (Value-Informed Dataset) dataset, to assess the alignment between LLMs’ stated values & value-informed actions. See below figure for an example of GPT4o's Value-Action Gap. 

<img src="figures/figure-value-action-gap.png" width="58%" alt="Description">



## Evaluating Value Alignment

To evaluate the value alignment in LLMs, codes are released in this directory: [Value-Action Alignment Tasks](https://github.com/huashen218/value_action_gap/tree/main/src/tasks).




## VIA Dataset

The full VIA dataset can be accessed in this directory: [Value-Informed Dataset (VIA)](https://github.com/huashen218/value_action_gap/tree/main/outputs/data_release)



---

## Reproducibility: Reproducing ValueActionLens (CSE 517)

**Authors:** Jahanvi Jeswani, Ziming Lin, Xunmei Liu (University of Washington)

This section documents our reproduction of the central empirical claims of Shen et al. [2025] using two models:
- **GPT-4o-mini** (OpenAI API, dynamic re-run)
- **Gemma-2-9B-it** (local Hugging Face inference on UW Hyak cluster)

Scripts added for this reproduction live in `src/tasks/` and are marked below.

---

### Repository Structure (our additions)

```
src/tasks/
├── task1/
│   ├── eval_llm_statement_gemma9b_full.py   # Gemma Task 1 full inference (12 countries × 11 topics)
│   └── eval_parallel.py                     # Gemma Task 1 multi-GPU parallel inference + shard merge
├── task2/
│   ├── eval_gemma9b.py                      # Gemma Task 2 inference (single GPU)
│   ├── eval_gemma9b_parallel_all.py         # Gemma Task 2 multi-GPU parallel (4× L40S)
│   ├── eval_gemma9b_parallel_all_resume.py  # Task 2 resume from partial shard outputs
│   └── launch.sh                            # Cluster env vars (set HF_TOKEN before use)
├── analyze_table4.py                        # Table 4 inconsistency counts for Gemma
├── analyze_alignment_gemma9b.py             # Full alignment analysis (Rate, Distance, Ranking)
├── eval_gemma_full_rate_adapted.py          # Alignment Rate (F1) adapted for Gemma wide-format
├── eval_dynamic_full_ranking_gemma_adapted.py  # Alignment Ranking adapted for Gemma
└── eval_dynamic_full_distance_gemma_adapted.py # Alignment Distance (ℓ₁) adapted for Gemma
```

---

### Dependencies

Install Python dependencies (Python 3.10+ recommended):

```bash
pip install -r requirements.txt
```

Key packages: `openai==1.57.2`, `pandas==2.2.3`, `numpy`, `tqdm`, `python-dotenv`

For local Gemma inference, additionally install:
```bash
pip install torch transformers accelerate
```

---

### Data Download

The VIA dataset is released in the original repository under `outputs/data_release/`. Clone the original repo to get it:

```bash
git clone https://github.com/huashen218/value_action_gap.git original_repo
cp -r original_repo/outputs/ outputs/
```

The key paired-action file for Task 2 is:
```
src/outputs/full_data/value_action_gap_full_data_gpt_4o_generation.csv
```
(14,784 rows — 132 scenarios × 56 values × 2 polarities)

---

### Preprocessing

No standalone preprocessing step is required. The VIA dataset is used directly for inference. Format conversion from Gemma's wide-format outputs to prompt-level format is handled internally by the evaluation scripts.

---

### Inference (Generation)

> **Note:** This project is inference-only. No model training is performed.

#### GPT-4o-mini (API-based)

```bash
export OPENAI_API_KEY=<your_openai_api_key>

# Task 1 — value statement rating
cd src/tasks/task1
python eval_llm_statement.py
# Output: src/outputs/openai:gpt-4o-mini_t1.csv  (shape: 1056 × 4)

# Task 2 — value-informed action selection
cd src/tasks/task2
python eval.py
# Output: src/outputs/openai:gpt-4o-mini_t2.csv  (shape: ~14330 × 7)
```

#### Gemma-2-9B-it (Local, multi-GPU)

First set environment variables (edit `launch.sh` and set your own `HF_TOKEN`):
```bash
source src/tasks/task2/launch.sh
```

**Task 1** — full 12 countries × 11 topics, deterministic decoding (temperature=0.0):
```bash
cd src/tasks/task1
python eval_llm_statement_gemma9b_full.py
# Output: src/outputs/task1_gemma_statements_full.csv  (wide format: 132 rows × 10 cols)
```

Multi-GPU parallel version (4 shards, then merged):
```bash
python eval_parallel.py
# Merged output: src/outputs/task1_gemma_statements_full_parallel.csv
```

**Task 2** — full paired-action evaluation, 4× L40S GPUs (temperature=0.2, top_p=0.95):
```bash
cd src/tasks/task2
python eval_gemma9b_parallel_all.py
# Shard outputs: src/outputs/task2_results_gemma9b_full_shard{0..3}.csv
# Merged output: src/outputs/task2_results_gemma9b_full.csv
```

Resume from a partial run:
```bash
python eval_gemma9b_parallel_all_resume.py
```

**Pretrained model:** `google/gemma-2-9b-it`
- Access: https://huggingface.co/google/gemma-2-9b-it (gated — requires HF account approval)
- Disk: ~18 GB, loaded in `bfloat16`, `device_map="auto"`
- Hardware used: 4× NVIDIA L40S 48GB GPUs on UW Hyak cluster

---

### Evaluation

All scripts read from inference output CSVs. Update `T1_PATH` / `T2_PATH` constants at the top of each script to point to your output files.

#### Alignment Rate (F1)

```bash
cd src/tasks
python eval_gemma_full_rate_adapted.py
# Outputs: gemma_dynamic_full_country_results.csv
#          gemma_dynamic_full_t1_pd.csv
#          gemma_dynamic_full_t2_pd.csv
```

#### Alignment Distance (ℓ₁)

```bash
python eval_dynamic_full_distance_gemma_adapted.py
# Outputs: distance CSVs by country and topic
```

#### Alignment Ranking

```bash
python eval_dynamic_full_ranking_gemma_adapted.py
# Outputs: ranking CSVs (top-1 and top-5 counts by country and topic)
```

#### Table 4 Inconsistency Counts

Reproduces cross-task agree/disagree inconsistency counts from paper Table 4:

```bash
python analyze_table4.py
# Outputs to: src/outputs/table4_verify_gemma2_full/
#   table4_summary.csv      — total misaligned counts and percentages
#   task1_task2_joined.csv  — merged Task 1 & Task 2 binary labels per (country, topic, value)
```

Expected results:
```
Paper (full Gemma):  (A,D)=497,  (D,A)=695, Total=1192 (16.13%)
Ours (full Gemma):   (A,D)=529,  (D,A)=671, Total=1200 (16.37%)
```

#### Full Alignment Analysis

```bash
python analyze_alignment_gemma9b.py
# Outputs to: src/outputs/alignment_analysis_gemma9b_corrected/
```

---

### Results Summary

| Model | Reproduced F1 | Paper's Reported F1 |
|---|---|---|
| GPT-4o-mini (dynamic, March 2026) | 0.5262 | 0.564 |
| Gemma-2-9B-it (full) | 0.461 | 0.461 |

Both models exhibit a nontrivial value–action gap with systematic variation across countries and values, supporting the paper's central claims.

---

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
