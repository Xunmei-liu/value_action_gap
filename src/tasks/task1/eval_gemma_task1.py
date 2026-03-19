import os
import json
import random
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForCausalLM

from statement_prompting import StatementPrompting

load_dotenv()

# Optional: set HF cache to scratch on cluster
os.environ.setdefault("HF_HOME", "/gscratch/cse/xunmei/hf_home")
os.environ.setdefault("HF_HUB_CACHE", "/gscratch/cse/xunmei/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/gscratch/cse/xunmei/hf_cache")

MODEL_NAME = "google/gemma-2-9b-it"
NUM_GPUS = 4

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

tokenizer = None
model = None


def init_model(local_rank):
    global tokenizer, model

    torch.cuda.set_device(local_rank)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map={"": local_rank}
    )
    model.eval()
    print(f"[GPU {local_rank}] Loaded Gemma-2-9B-it successfully")


def generate_with_gemma(user_prompt, max_new_tokens=1024, temperature=0.0):
    global tokenizer, model

    messages = [{"role": "user", "content": user_prompt}]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs.get("attention_mask", None),
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.95

    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    prompt_len = inputs["input_ids"].shape[-1]
    generated_tokens = outputs[0][prompt_len:]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return text.strip()


def eval_value_statement(country, topic, temperature=0.0):
    """
    For one (country, topic), run 8 Task-1 prompt variants.
    Each output should contain ratings for all Schwartz values.
    """
    prompting_method = StatementPrompting()

    row = {
        "country": country,
        "topic": topic,
    }

    for prompt_index in range(8):
        prompt = prompting_method.generate_prompt(country, topic, prompt_index)

        prompt = (
            prompt
            + "\n\nReturn only one valid JSON object. "
              "Do not include markdown fences. "
              "Use double quotes for all keys and values."
        )

        generated_text = generate_with_gemma(
            prompt,
            max_new_tokens=1024,
            temperature=temperature
        )

        row[f"evaluation_{prompt_index}"] = generated_text

    return row


def full_annotation():
    countries = [
        "United States", "India", "Pakistan", "Nigeria", "Philippines",
        "United Kingdom", "Germany", "Uganda", "Canada", "Egypt",
        "France", "Australia"
    ]

    topics = [
        "Politics",
        "Social Networks",
        "Social Inequality",
        "Family & Changing Gender Roles",
        "Work Orientation",
        "Religion",
        "Environment",
        "National Identity",
        "Citizenship",
        "Leisure Time and Sports",
        "Health and Health Care"
    ]

    return countries, topics


def build_cases():
    countries, topics = full_annotation()
    cases = []
    for country in countries:
        for topic in topics:
            cases.append((country, topic))
    return cases


def split_cases(cases, num_splits):
    chunk_size = math.ceil(len(cases) / num_splits)
    return [cases[i * chunk_size:(i + 1) * chunk_size] for i in range(num_splits)]


def worker(local_rank, case_splits):
    init_model(local_rank)

    my_cases = case_splits[local_rank]
    print(f"[GPU {local_rank}] Processing {len(my_cases)} Task-1 cases")

    results = []
    shard_output_path = (
        f"/mmfs1/gscratch/cse/xunmei/value_action_gap/src/outputs/"
        f"task1_gemma_statements_full_shard{local_rank}.csv"
    )

    for country, topic in tqdm(my_cases, desc=f"GPU {local_rank}", position=local_rank):
        row = eval_value_statement(
            country=country,
            topic=topic,
            temperature=0.0
        )
        results.append(row)

        pd.DataFrame(results).to_csv(shard_output_path, index=False)

    print(f"[GPU {local_rank}] Saved shard results to {shard_output_path}")


def main():
    smoke_test = False

    if smoke_test:
        init_model(0)
        row = eval_value_statement(
            country="Philippines",
            topic="Politics",
            temperature=0.0
        )
        output_path = "/mmfs1/gscratch/cse/xunmei/value_action_gap/src/outputs/task1_gemma_statements_smoke.csv"
        pd.DataFrame([row]).to_csv(output_path, index=False)
        print(f"Saved smoke test to {output_path}")
        return

    cases = build_cases()
    print(f"Running full Task 1 set: {len(cases)} cases across {NUM_GPUS} GPUs")

    case_splits = split_cases(cases, NUM_GPUS)

    mp.spawn(
        worker,
        args=(case_splits,),
        nprocs=NUM_GPUS,
        join=True
    )

    shard_paths = [
        f"/mmfs1/gscratch/cse/xunmei/value_action_gap/src/outputs/task1_gemma_statements_full_shard{i}.csv"
        for i in range(NUM_GPUS)
    ]

    shard_dfs = []
    for path in shard_paths:
        if os.path.exists(path):
            shard_dfs.append(pd.read_csv(path))
        else:
            print(f"Warning: shard file not found: {path}")

    if len(shard_dfs) == 0:
        print("No shard outputs found. Nothing to merge.")
        return

    result_df = pd.concat(shard_dfs, ignore_index=True)
    output_path = "/mmfs1/gscratch/cse/xunmei/value_action_gap/src/outputs/task1_gemma_statements_full_parallel.csv"
    result_df.to_csv(output_path, index=False)
    print(f"Saved merged results to {output_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()