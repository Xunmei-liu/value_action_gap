import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import math

import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForCausalLM

from prompting import StatementPrompting
from utils import parse_json

load_dotenv()

# ====== 建议在集群上把 HF cache 放到 scratch ======
os.environ.setdefault("HF_HOME", "/gscratch/cse/xunmei/hf_home")
os.environ.setdefault("HF_HUB_CACHE", "/gscratch/cse/xunmei/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/gscratch/cse/xunmei/hf_cache")

MODEL_NAME = "google/gemma-2-9b-it"
NUM_GPUS = 4

# 可复现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 全局变量，worker 里初始化
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


def generate_with_gemma(user_prompt, max_new_tokens=256, temperature=0.2):
    """
    用本地 Gemma 做 chat inference
    """
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


def normalize_action(displayed_action, reverse_order):
    """
    把模型在“显示顺序”下选的 Option 1/2
    映射回原始 option1 / option2 的语义顺序
    """
    if displayed_action not in ["Option 1", "Option 2"]:
        return displayed_action

    if not reverse_order:
        return displayed_action

    # 如果 prompt 中把顺序反过来了：
    # displayed Option 1 实际对应原始 option2
    # displayed Option 2 实际对应原始 option1
    if displayed_action == "Option 1":
        return "Option 2"
    elif displayed_action == "Option 2":
        return "Option 1"


def eval_value_action(country, topic, value, option1, option2, temperature=0.2):
    prompting_method = StatementPrompting()

    outputs = {
        "country": country,
        "topic": topic,
        "value": value,
        "option1": option1,
        "option2": option2,
        "evaluation_0": None,
        "evaluation_1": None,
        "evaluation_2": None,
        "evaluation_3": None,
        "evaluation_4": None,
        "evaluation_5": None,
        "evaluation_6": None,
        "evaluation_7": None,
    }

    for prompt_index in range(8):
        prompt_text, reverse_order = prompting_method.generate_prompt(
            country=country,
            topic=topic,
            value=value,
            option1=option1,
            option2=option2,
            index=prompt_index
        )

        # 强化 JSON 输出要求，减少 parse 失败
        prompt_text = (
            prompt_text
            + "\n\nReturn only one valid JSON object. "
              "Do not include markdown fences or extra text. "
              "Use double quotes for all keys and values."
        )

        response_text = generate_with_gemma(
            prompt_text,
            max_new_tokens=256,
            temperature=temperature
        )

        r = parse_json(response_text)
        if r is None:
            raise ValueError(f"Failed to parse response: {response_text}")

        displayed_action = r.get("action")
        normalized_action = normalize_action(displayed_action, reverse_order)

        # 保留原始显示结果 + 归一化结果，后面分析更安全
        r["displayed_action"] = displayed_action
        r["action"] = normalized_action
        r["reverse_order"] = reverse_order

        outputs[f"evaluation_{prompt_index}"] = r
        print(f"{prompt_index} r={r}")

    return outputs


def build_grouped_cases(df):
    grouped = df.groupby(["country", "topic", "value"])
    cases = []

    for (country, topic, value), group in grouped:
        if len(group) != 2:
            print(f"== skip len(group) != 2 for {country} | {topic} | {value} ==")
            continue

        group = group.sort_values("polarity")

        assert group.iloc[0]["polarity"] == "negative"
        assert group.iloc[1]["polarity"] == "positive"

        try:
            option1 = parse_json(group.iloc[0]["generation_prompt"])["Human Action"]  # negative
            option2 = parse_json(group.iloc[1]["generation_prompt"])["Human Action"]  # positive
        except Exception as e:
            print(f"Skip due to parse error in generation_prompt: {e}")
            continue

        cases.append((country, topic, value, option1, option2))

    return cases


def split_cases(cases, num_splits):
    chunk_size = math.ceil(len(cases) / num_splits)
    return [cases[i * chunk_size:(i + 1) * chunk_size] for i in range(num_splits)]


def worker(local_rank, case_splits):
    init_model(local_rank)

    my_cases = case_splits[local_rank]
    print(f"[GPU {local_rank}] Processing {len(my_cases)} cases")

    results = []
    shard_output_path = (
        f"/mmfs1/gscratch/cse/xunmei/value_action_gap/src/outputs/"
        f"task2_results_gemma9b_full_shard{local_rank}.csv"
    )

    for country, topic, value, option1, option2 in tqdm(
        my_cases, desc=f"GPU {local_rank}", position=local_rank
    ):
        outputs = eval_value_action(
            country=country,
            topic=topic,
            value=value,
            option1=option1,
            option2=option2,
            temperature=0.2
        )
        results.append(outputs)

        cases = [
            (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
            (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)
        ]

        print("===========")
        print("OPTION 1\n")
        for i in range(8):
            if outputs[f"evaluation_{i}"]["action"] == "Option 1":
                print(f"Prompt {cases[i]}\n")

        print("===========")
        print("OPTION 2\n")
        for i in range(8):
            if outputs[f"evaluation_{i}"]["action"] == "Option 2":
                print(f"Prompt {cases[i]}\n")
        print("===========")

        # 每个 worker 持续覆盖写自己的 shard，避免中途丢失
        pd.DataFrame(results).to_csv(shard_output_path, index=False)

    print(f"[GPU {local_rank}] Saved shard results to {shard_output_path}")


def main():
    sub_sample = False

    if sub_sample:
        sample_size = 50
        df = pd.read_csv(
            "/mmfs1/gscratch/cse/xunmei/value_action_gap/src/outputs/full_data/value_action_gap_full_data_gpt_4o_generation.csv"
        )

        even_numbers = list(range(2, len(df), 2))
        sampled_evens = random.sample(even_numbers, sample_size)
        sampled_odds = list(map(lambda x: x + 1, sampled_evens))
        index_list = np.sort(sampled_evens + sampled_odds)

        filtered_df = df.loc[index_list]
        filtered_df.to_csv(
            "/mmfs1/gscratch/cse/xunmei/value_action_gap/src/outputs/filtered_sample_value_action_evaluation_gpt_4o_mini.csv",
            index=False
        )
        print("Saved filtered sample CSV")
        return

    else:
        df = pd.read_csv(
            "/mmfs1/gscratch/cse/xunmei/value_action_gap/src/outputs/full_data/value_action_gap_full_data_gpt_4o_generation.csv"
        )

        print(f"Loaded dataframe with {len(df)} rows")

        cases = build_grouped_cases(df)
        print(f"Total valid cases: {len(cases)}")

        case_splits = split_cases(cases, NUM_GPUS)

        mp.spawn(
            worker,
            args=(case_splits,),
            nprocs=NUM_GPUS,
            join=True
        )

        # 合并 4 个 shard
        shard_paths = [
            f"/mmfs1/gscratch/cse/xunmei/value_action_gap/src/outputs/task2_results_gemma9b_full_shard{i}.csv"
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
        output_path = "/mmfs1/gscratch/cse/xunmei/value_action_gap/src/outputs/task2_results_gemma9b_full.csv"
        result_df.to_csv(output_path, index=False)
        print(f"Saved merged results to {output_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()