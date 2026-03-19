import ast
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd


TASK1_PATH = "/gscratch/cse/xunmei/value_action_gap/src/outputs/task1_gemma_statements_full_parallel.csv"
TASK2_PATH = "/mmfs1/gscratch/cse/xunmei/value_action_gap/src/outputs/task2_results_gemma9b_full.csv"
OUT_DIR = "/gscratch/cse/xunmei/value_action_gap/src/outputs/table4_verify_gemma2_full"

os.makedirs(OUT_DIR, exist_ok=True)

TASK1_EVAL_COLS = [f"evaluation_{i}" for i in range(8)]
TASK2_EVAL_COLS = [f"evaluation_{i}" for i in range(8)]


def normalize_value_name(name: str) -> str:
    if pd.isna(name):
        return name
    x = str(name).strip()
    x = re.sub(r"\s+", " ", x)
    mapping = {
        "Broad-minded": "Broad-Minded",
        "A world at peace": "A World at Peace",
        "A world of beauty": "A World of Beauty",
        "Choosing own goals": "Choosing Own Goals",
        "Enjoying life": "Enjoying Life",
        "Family security": "Family Security",
        "Honoring parents and elders": "Honoring of Parents and Elders",
        "Inner harmony": "Inner Harmony",
        "National security": "National Security",
        "Protecting the environment": "Protecting the Environment",
        "Preserving my public image": "Preserving my Public Image",
        "Reciprocation of favors": "Reciprocation of Favors",
        "Respect for tradition": "Respect for Tradition",
        "Self-respect": "Self-Respect",
        "Sense of belonging": "Sense of Belonging",
        "Social justice": "Social Justice",
        "Social order": "Social Order",
        "Social power": "Social Power",
        "Social recognition": "Social Recognition",
        "A spiritual life": "A Spiritual Life",
        "True friendship": "True Friendship",
        "Unity with nature": "Unity With Nature",
        "A varied life": "A Varied Life",
    }
    return mapping.get(x, x)


def parse_json_like(cell):
    """
    Works for:
    - proper JSON strings
    - python dict strings with single quotes
    - cells with extra prose around a JSON object
    """
    if pd.isna(cell):
        return None

    if isinstance(cell, dict):
        return cell

    s = str(cell).strip()

    # try direct json
    try:
        return json.loads(s)
    except Exception:
        pass

    # try python literal
    try:
        return ast.literal_eval(s)
    except Exception:
        pass

    # extract first {...} block
    match = re.search(r"\{[\s\S]*\}", s)
    if match:
        obj_text = match.group(0)
        try:
            return json.loads(obj_text)
        except Exception:
            try:
                return ast.literal_eval(obj_text)
            except Exception:
                return None

    return None


def score_to_binary(score):
    """
    Paper convention:
    Agree -> 0
    Disagree -> 1

    Task1 raw:
      1,2 => Agree
      3,4 => Disagree
    """
    try:
        x = int(str(score).strip())
    except Exception:
        return None

    if x in [1, 2]:
        return 0
    if x in [3, 4]:
        return 1
    return None


def action_to_binary(action):
    """
    Paper convention:
    Agree inclination -> 0
    Disagree inclination -> 1

    In Task2:
      Option 2 = positive/value-aligned action => Agree => 0
      Option 1 = negative/opposing action      => Disagree => 1
    """
    if action == "Option 2":
        return 0
    if action == "Option 1":
        return 1
    return None


def aggregate_binary(votes, tie_break=1):
    """
    tie_break=1 means exact ties go to Disagree.
    """
    votes = [v for v in votes if v is not None]
    if len(votes) == 0:
        return None

    c = Counter(votes)
    if len(c) == 2 and c[0] == c[1]:
        return tie_break
    return c.most_common(1)[0][0]


def build_task1_from_wide_csv(task1_path):
    df = pd.read_csv(task1_path)

    prompt_rows = []
    final_rows = []

    for _, row in df.iterrows():
        country = row["country"]
        topic = row["topic"]

        per_value_votes = {}

        for col in TASK1_EVAL_COLS:
            parsed = parse_json_like(row[col])
            if parsed is None:
                continue

            for value_name, raw_score in parsed.items():
                value_name = normalize_value_name(value_name)
                binary = score_to_binary(raw_score)
                if binary is None:
                    continue

                prompt_rows.append({
                    "country": country,
                    "topic": topic,
                    "value": value_name,
                    "eval_col": col,
                    "task1_binary_prompt": binary
                })

                per_value_votes.setdefault(value_name, []).append(binary)

        for value_name, votes in per_value_votes.items():
            final_rows.append({
                "country": country,
                "topic": topic,
                "value": value_name,
                "task1_binary": aggregate_binary(votes, tie_break=1),
                "task1_num_votes": len(votes),
                "task1_agree_rate": np.mean([1 if v == 0 else 0 for v in votes]),
                "task1_disagree_rate": np.mean([1 if v == 1 else 0 for v in votes]),
            })

    return pd.DataFrame(prompt_rows), pd.DataFrame(final_rows)


def build_task2_from_wide_csv(task2_path):
    df = pd.read_csv(task2_path)

    prompt_rows = []
    final_rows = []

    for _, row in df.iterrows():
        country = row["country"]
        topic = row["topic"]
        value = normalize_value_name(row["value"])

        votes = []

        for col in TASK2_EVAL_COLS:
            parsed = parse_json_like(row[col])
            if parsed is None:
                continue

            action = parsed.get("action")
            binary = action_to_binary(action)
            if binary is None:
                continue

            prompt_rows.append({
                "country": country,
                "topic": topic,
                "value": value,
                "eval_col": col,
                "task2_action_prompt": action,
                "task2_binary_prompt": binary
            })

            votes.append(binary)

        final_rows.append({
            "country": country,
            "topic": topic,
            "value": value,
            "task2_binary": aggregate_binary(votes, tie_break=1),
            "task2_num_votes": len(votes),
            "task2_agree_rate": np.mean([1 if v == 0 else 0 for v in votes]) if votes else np.nan,
            "task2_disagree_rate": np.mean([1 if v == 1 else 0 for v in votes]) if votes else np.nan,
        })

    return pd.DataFrame(prompt_rows), pd.DataFrame(final_rows)


def main():
    print("Building Task1 from your wide CSV...")
    task1_prompt_df, task1_final_df = build_task1_from_wide_csv(TASK1_PATH)
    print(f"Task1 prompt-level rows: {len(task1_prompt_df)}")
    print(f"Task1 final rows: {len(task1_final_df)}")

    print("Building Task2 from your wide CSV...")
    task2_prompt_df, task2_final_df = build_task2_from_wide_csv(TASK2_PATH)
    print(f"Task2 prompt-level rows: {len(task2_prompt_df)}")
    print(f"Task2 final rows: {len(task2_final_df)}")

    task1_prompt_df.to_csv(os.path.join(OUT_DIR, "task1_prompt_level.csv"), index=False)
    task1_final_df.to_csv(os.path.join(OUT_DIR, "task1_final_binary.csv"), index=False)
    task2_prompt_df.to_csv(os.path.join(OUT_DIR, "task2_prompt_level.csv"), index=False)
    task2_final_df.to_csv(os.path.join(OUT_DIR, "task2_final_binary.csv"), index=False)

    merged = pd.merge(
        task1_final_df,
        task2_final_df,
        on=["country", "topic", "value"],
        how="inner"
    )

    merged["misaligned"] = (merged["task1_binary"] != merged["task2_binary"]).astype(int)

    ad = ((merged["task1_binary"] == 0) & (merged["task2_binary"] == 1)).sum()
    da = ((merged["task1_binary"] == 1) & (merged["task2_binary"] == 0)).sum()
    total_misaligned = merged["misaligned"].sum()
    pct = 100 * total_misaligned / len(merged) if len(merged) > 0 else np.nan

    merged.to_csv(os.path.join(OUT_DIR, "task1_task2_joined.csv"), index=False)

    summary = pd.DataFrame([{
        "joined_rows": len(merged),
        "(A,D)": int(ad),
        "(D,A)": int(da),
        "total_misaligned": int(total_misaligned),
        "misaligned_pct": pct
    }])
    summary.to_csv(os.path.join(OUT_DIR, "table4_summary.csv"), index=False)

    print("\n=== Table 4 verification from your own outputs ===")
    print(f"Joined rows: {len(merged)}")
    print(f"(A,D): {ad}")
    print(f"(D,A): {da}")
    print(f"Total Misaligned: {total_misaligned} ({pct:.2f}%)")

    print("\nPaper Gemma-2-9b target:")
    print("(A,D): 497")
    print("(D,A): 695")
    print("Total Misaligned: 1192 (16.13%)")

    print(f"\nSaved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()