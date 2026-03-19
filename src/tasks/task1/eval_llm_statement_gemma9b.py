import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from statement_prompting import StatementPrompting

load_dotenv()

# Optional: set HF cache to scratch on cluster
os.environ.setdefault("HF_HOME", "/gscratch/cse/xunmei/hf_home")
os.environ.setdefault("HF_HUB_CACHE", "/gscratch/cse/xunmei/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/gscratch/cse/xunmei/hf_cache")

MODEL_NAME = "google/gemma-2-9b-it"

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

print("Loaded Gemma-2-9B-it successfully")


def generate_with_gemma(user_prompt, max_new_tokens=1024, temperature=0.0):
    """
    Local inference with Gemma chat template.
    Uses tokenize=False first for compatibility across transformers versions.
    """
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


def eval_value_statement(country, topic, outputs, temperature=0.0):
    """
    Evaluate Gemma's stated values for a given (country, topic)
    across 8 prompt variants.
    """
    prompting_method = StatementPrompting()

    row = {
        "country": country,
        "topic": topic,
    }

    for prompt_index in tqdm(range(8), desc=f"{country} | {topic}"):
        prompt = prompting_method.generate_prompt(country, topic, prompt_index)

        # Stronger formatting instruction helps local models stay in JSON
        prompt = (
            prompt
            + "\n\nReturn only one valid JSON object. "
              "Do not include markdown fences. "
              "Use double quotes for all keys and values."
        )

        print(f"\n======== Prompt {prompt_index} ========\n{prompt[:1200]}\n")

        generated_text = generate_with_gemma(
            prompt,
            max_new_tokens=1024,
            temperature=temperature
        )

        row[f"evaluation_{prompt_index}"] = generated_text
        print(f"======== Output {prompt_index} ========\n{generated_text[:1000]}\n")

    for k, v in row.items():
        outputs[k].append(v)


def human_annotation():
    countries = ["United States", "Philippines"]

    topics = [
        "Politics",
        "Social Inequality",
        "Family & Changing Gender Roles",
        "Leisure Time and Sports",
    ]

    schwartz_values = {
        "Power": ["Authority"],
        "Achievement": ["Intelligent"],
        "Hedonism": ["Enjoying life"],
        "Stimulation": ["An exciting life"],
        "Self-direction": ["Choosing own goals"],
        "Universalism": ["Broad-minded"],
        "Benevolence": ["Responsible"],
        "Tradition": ["Humble"],
        "Conformity": ["Obedient"],
        "Security": ["Family security"]
    }
    return countries, topics, schwartz_values


def main():
    """
    Task 1 in this repo elicits value statements / value ratings.
    For each (country, topic), it runs 8 prompt variants and saves outputs.
    """

    countries, topics, schwartz_values = human_annotation()

    outputs = {
        "country": [],
        "topic": [],
        "evaluation_0": [],
        "evaluation_1": [],
        "evaluation_2": [],
        "evaluation_3": [],
        "evaluation_4": [],
        "evaluation_5": [],
        "evaluation_6": [],
        "evaluation_7": [],
    }

    # Smoke test first
    smoke_test = False

    if smoke_test:
        eval_value_statement(
            country="Philippines",
            topic="Politics",
            outputs=outputs,
            temperature=0.0
        )
    else:
        for country in countries:
            for topic in topics:
                eval_value_statement(
                    country=country,
                    topic=topic,
                    outputs=outputs,
                    temperature=0.0
                )

    output_path = "/mmfs1/gscratch/cse/xunmei/value_action_gap/src/outputs/task1_gemma_statements.csv"
    df = pd.DataFrame(outputs)
    df.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()