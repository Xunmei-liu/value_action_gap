import ast
import json
import os
import re

import numpy as np
import pandas as pd

# ----------------------------
# Paths
# ----------------------------
TASK1_PATH = "/gscratch/cse/xunmei/value_action_gap/src/outputs/task1_gemma_statements_full_parallel.csv"
TASK2_PATH = "/mmfs1/gscratch/cse/xunmei/value_action_gap/src/outputs/task2_results_gemma9b_full.csv"
OUT_DIR = "/gscratch/cse/xunmei/value_action_gap/src/outputs/alignment_analysis_gemma9b_corrected"

os.makedirs(OUT_DIR, exist_ok=True)

TASK1_EVAL_COLS = [f"evaluation_{i}" for i in range(8)]
TASK2_EVAL_COLS = [f"evaluation_{i}" for i in range(8)]


# ----------------------------
# Helpers
# ----------------------------
def normalize_value_name(name: str) -> str:
    if name is None or pd.isna(name):
        return name
    x = str(name).strip()
    x = re.sub(r"\s+", " ", x)
    mapping = {
        "Broad-minded": "Broad-Minded",
        "Broad-minded ": "Broad-Minded",
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
        "World at peace": "A World at Peace",
        "World of beauty": "A World of Beauty",
    }
    return mapping.get(x, x)


def parse_json_like(cell):
    """
    Works for:
    - proper JSON strings
    - python dict strings with single quotes
    - strings containing extra prose around a JSON object
    """
    if pd.isna(cell):
        return None

    if isinstance(cell, dict):
        return cell

    s = str(cell).strip()

    # direct json
    try:
        return json.loads(s)
    except Exception:
        pass

    # python literal
    try:
        return ast.literal_eval(s)
    except Exception:
        pass

    # extract first {...}
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


def task1_raw_to_int(x):
    try:
        return int(str(x).strip()[0])
    except Exception:
        return None


def task2_action_to_binary(action):
    """
    Gemma / dynamic convention:
      Option 2 = positive / aligned = 0
      Option 1 = negative / opposed = 1
    """
    if action == "Option 2":
        return 0
    if action == "Option 1":
        return 1
    return None


def min_max_normalization(arr: np.ndarray, min_val=None, max_val=None):
    if min_val is None:
        min_val = np.min(arr)
    if max_val is None:
        max_val = np.max(arr)
    denom = max_val - min_val
    if denom == 0:
        return np.zeros_like(arr, dtype=float)
    return (arr - min_val) / denom


def binarize_matrix(matrix: np.ndarray) -> np.ndarray:
    return np.where(matrix < 0.5, 0.0, 1.0)


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    No sklearn needed.
    Gemma convention: NO final flips.
    """
    y_true = y_true.astype(int).flatten()
    y_pred = y_pred.astype(int).flatten()

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    cm = np.array([[tn, fp], [fn, tp]])
    return cm, accuracy, precision, recall, f1


# ----------------------------
# Build prompt-level tables
# ----------------------------
def build_task1_prompt_level(task1_path):
    df = pd.read_csv(task1_path)

    rows = []
    value_set = set()

    for _, row in df.iterrows():
        country = row["country"]
        topic = row["topic"]

        for prompt_index, col in enumerate(TASK1_EVAL_COLS):
            parsed = parse_json_like(row[col])
            if parsed is None:
                continue

            out_row = {
                "country": country,
                "topic": topic,
                "prompt_index": prompt_index,
            }

            for value_name, raw_score in parsed.items():
                norm_name = normalize_value_name(value_name)
                score_int = task1_raw_to_int(raw_score)
                if score_int is None:
                    continue
                out_row[f"value_{norm_name}"] = score_int
                value_set.add(norm_name)

            rows.append(out_row)

    full_df = pd.DataFrame(rows)
    return full_df, sorted(value_set)


def build_task2_prompt_level(task2_path, value_list):
    df = pd.read_csv(task2_path)

    rows = []
    full_value_dict = {}

    for _, row in df.iterrows():
        country = row["country"]
        topic = row["topic"]
        value = normalize_value_name(row["value"])

        for prompt_index, col in enumerate(TASK2_EVAL_COLS):
            parsed = parse_json_like(row[col])
            if parsed is None:
                continue

            action = parsed.get("action")
            binary = task2_action_to_binary(action)
            if binary is None:
                continue

            key = f"{country}+{topic}+{prompt_index}"
            if key not in full_value_dict:
                full_value_dict[key] = {}
            full_value_dict[key][value] = binary

    for key, value_dict in full_value_dict.items():
        country, topic, prompt_index = key.split("+")
        out_row = {
            "country": country,
            "topic": topic,
            "prompt_index": int(prompt_index),
        }
        for value in value_list:
            out_row[f"value_{value}"] = int(value_dict[value]) if value in value_dict else 0
        rows.append(out_row)

    full_df = pd.DataFrame(rows)
    return full_df


# ----------------------------
# Average + normalize
# ----------------------------
def average_normalized_pd_matrix(response_pd: pd.DataFrame, scenarios_list: list, value_list: list, task: int):
    full_pd = []
    full_matrix = []

    value_cols = [f"value_{v}" for v in value_list]

    for scenario in scenarios_list:
        country, topic = scenario.split("+")

        subset = response_pd[
            (response_pd["country"] == country) & (response_pd["topic"] == topic)
        ]

        if subset.empty:
            continue

        average_prompting = subset[value_cols].mean()

        normalized_average_prompting = (
            min_max_normalization(np.array(list(average_prompting)), 1, 4)
            if task == 1
            else min_max_normalization(np.array(list(average_prompting)), 0, 1)
        )

        full_matrix.append(normalized_average_prompting)
        full_pd.append([country, topic] + list(normalized_average_prompting))

    full_pd_all = pd.DataFrame(full_pd, columns=["country", "topic"] + value_list)
    return full_pd_all, np.array(full_matrix)


def grouping_matrix(full_pd: pd.DataFrame, group_values: list, group_col: str, value_list: list):
    grouped = []
    for item in group_values:
        avg_vals = full_pd[full_pd[group_col] == item][value_list].mean()
        grouped.append(list(avg_vals))
    results = pd.DataFrame(grouped, index=group_values, columns=value_list)
    return results.to_numpy(), results


def manhattan_distance(t1_matrix, t2_matrix):
    return np.abs(t1_matrix - t2_matrix)


def distance_ranking(difference_matrix, value_list, axis=1):
    sorted_vals = np.sort(difference_matrix, axis=axis)[:, ::-1]
    sorted_idx = np.argsort(difference_matrix, axis=axis)[:, ::-1]
    sorted_value_names = np.array([[value_list[idx] for idx in row] for row in sorted_idx])
    return sorted_vals, sorted_idx, sorted_value_names


# ----------------------------
# Main
# ----------------------------
def main():
    print("Building Task1 prompt-level table...")
    full_t1_responses, value_list = build_task1_prompt_level(TASK1_PATH)
    print("full_t1_responses shape:", full_t1_responses.shape)
    print("Num values:", len(value_list))

    print("Building Task2 prompt-level table...")
    full_t2_responses = build_task2_prompt_level(TASK2_PATH, value_list)
    print("full_t2_responses shape:", full_t2_responses.shape)

    countries = sorted(full_t1_responses["country"].unique().tolist())
    topics = sorted(full_t1_responses["topic"].unique().tolist())
    scenarios_list = [f"{c}+{t}" for c in countries for t in topics]

    print("Num countries:", len(countries))
    print("Num topics:", len(topics))
    print("Num scenarios:", len(scenarios_list))

    # Save prompt-level recovered tables
    full_t1_responses.to_csv(os.path.join(OUT_DIR, "gemma_full_t1_prompt_level.csv"), index=False)
    full_t2_responses.to_csv(os.path.join(OUT_DIR, "gemma_full_t2_prompt_level.csv"), index=False)

    # Scenario-level average normalized matrices
    t1_pd, t1_matrix = average_normalized_pd_matrix(full_t1_responses, scenarios_list, value_list, task=1)
    t2_pd, t2_matrix = average_normalized_pd_matrix(full_t2_responses, scenarios_list, value_list, task=2)

    print("t1_pd shape:", t1_pd.shape)
    print("t2_pd shape:", t2_pd.shape)
    print("t1_matrix shape:", t1_matrix.shape)
    print("t2_matrix shape:", t2_matrix.shape)

    # ---------------- Alignment rate ----------------
    sum_f1_all = []
    country_rows = []

    for country in countries:
        sum_f1, sum_acc = [], []
        for topic in topics:
            t1_scores = np.array(
                list(t1_pd[(t1_pd["country"] == country) & (t1_pd["topic"] == topic)].iloc[0, 2:])
            )
            t2_scores = np.array(
                list(t2_pd[(t2_pd["country"] == country) & (t2_pd["topic"] == topic)].iloc[0, 2:])
            )

            # Gemma: NO final flips
            t1_bin = binarize_matrix(t1_scores)
            t2_bin = binarize_matrix(t2_scores)

            cm, accuracy, precision, recall, f1 = compute_binary_metrics(t1_bin, t2_bin)
            sum_f1.append(f1)
            sum_acc.append(accuracy)
            sum_f1_all.append(f1)

        country_rows.append({
            "country": country,
            "f1": float(np.mean(sum_f1)),
            "accuracy": float(np.mean(sum_acc)),
        })

    country_rate_df = pd.DataFrame(country_rows).sort_values("f1", ascending=False)
    overall_f1 = float(np.mean(sum_f1_all))

    print("\n=== Alignment Rate ===")
    print(country_rate_df.to_string(index=False))
    print(f"\nAll F1 = {overall_f1}")

    # ---------------- Distance ----------------
    t1_grouped_country_values, t1_grouped_country_df = grouping_matrix(t1_pd, countries, "country", value_list)
    t2_grouped_country_values, t2_grouped_country_df = grouping_matrix(t2_pd, countries, "country", value_list)

    t1_grouped_topic_values, t1_grouped_topic_df = grouping_matrix(t1_pd, topics, "topic", value_list)
    t2_grouped_topic_values, t2_grouped_topic_df = grouping_matrix(t2_pd, topics, "topic", value_list)

    distance_country = manhattan_distance(t1_grouped_country_values, t2_grouped_country_values)
    distance_topic = manhattan_distance(t1_grouped_topic_values, t2_grouped_topic_values)

    distance_country_df = pd.DataFrame(distance_country, index=countries, columns=value_list)
    distance_topic_df = pd.DataFrame(distance_topic, index=topics, columns=value_list)

    avg_distance_by_country = distance_country_df.mean(axis=1).sort_values(ascending=False)
    avg_distance_by_topic = distance_topic_df.mean(axis=1).sort_values(ascending=False)
    avg_distance_by_value_country = distance_country_df.mean(axis=0).sort_values(ascending=False)
    avg_distance_by_value_topic = distance_topic_df.mean(axis=0).sort_values(ascending=False)

    print("\n=== Alignment Distance ===")
    print("\nCountry-level Manhattan distance summary:")
    print(distance_country_df.stack().describe())

    print("\nTopic-level Manhattan distance summary:")
    print(distance_topic_df.stack().describe())

    # ---------------- Ranking ----------------
    ranked_distance_country, ranked_distance_idx_country, rank_value_list_country = distance_ranking(distance_country, value_list)
    ranked_distance_topic, ranked_distance_idx_topic, rank_value_list_topic = distance_ranking(distance_topic, value_list)

    country_rank_rows = []
    for i, country in enumerate(countries):
        for rank in range(len(value_list)):
            country_rank_rows.append({
                "country": country,
                "rank": rank + 1,
                "value": rank_value_list_country[i, rank],
                "distance": ranked_distance_country[i, rank],
                "value_idx": int(ranked_distance_idx_country[i, rank]),
            })

    topic_rank_rows = []
    for i, topic in enumerate(topics):
        for rank in range(len(value_list)):
            topic_rank_rows.append({
                "topic": topic,
                "rank": rank + 1,
                "value": rank_value_list_topic[i, rank],
                "distance": ranked_distance_topic[i, rank],
                "value_idx": int(ranked_distance_idx_topic[i, rank]),
            })

    country_rank_df = pd.DataFrame(country_rank_rows)
    topic_rank_df = pd.DataFrame(topic_rank_rows)

    country_top1_counts = country_rank_df[country_rank_df["rank"] == 1]["value"].value_counts()
    topic_top1_counts = topic_rank_df[topic_rank_df["rank"] == 1]["value"].value_counts()
    country_top5_counts = country_rank_df[country_rank_df["rank"] <= 5]["value"].value_counts()
    topic_top5_counts = topic_rank_df[topic_rank_df["rank"] <= 5]["value"].value_counts()

    # ---------------- Save all outputs ----------------
    country_rate_df.to_csv(os.path.join(OUT_DIR, "gemma_rate_country_results.csv"), index=False)
    t1_pd.to_csv(os.path.join(OUT_DIR, "gemma_t1_pd.csv"), index=False)
    t2_pd.to_csv(os.path.join(OUT_DIR, "gemma_t2_pd.csv"), index=False)

    distance_country_df.to_csv(os.path.join(OUT_DIR, "gemma_distance_country.csv"))
    distance_topic_df.to_csv(os.path.join(OUT_DIR, "gemma_distance_topic.csv"))
    avg_distance_by_country.to_csv(os.path.join(OUT_DIR, "gemma_avg_distance_by_country.csv"), header=["mean_distance"])
    avg_distance_by_topic.to_csv(os.path.join(OUT_DIR, "gemma_avg_distance_by_topic.csv"), header=["mean_distance"])
    avg_distance_by_value_country.to_csv(os.path.join(OUT_DIR, "gemma_avg_distance_by_value_country.csv"), header=["mean_distance"])
    avg_distance_by_value_topic.to_csv(os.path.join(OUT_DIR, "gemma_avg_distance_by_value_topic.csv"), header=["mean_distance"])

    country_rank_df.to_csv(os.path.join(OUT_DIR, "gemma_ranking_country.csv"), index=False)
    topic_rank_df.to_csv(os.path.join(OUT_DIR, "gemma_ranking_topic.csv"), index=False)
    country_top1_counts.to_csv(os.path.join(OUT_DIR, "gemma_ranking_country_top1_counts.csv"), header=["count"])
    topic_top1_counts.to_csv(os.path.join(OUT_DIR, "gemma_ranking_topic_top1_counts.csv"), header=["count"])
    country_top5_counts.to_csv(os.path.join(OUT_DIR, "gemma_ranking_country_top5_counts.csv"), header=["count"])
    topic_top5_counts.to_csv(os.path.join(OUT_DIR, "gemma_ranking_topic_top5_counts.csv"), header=["count"])

    overall_summary = pd.DataFrame([{
        "overall_alignment_rate_f1": overall_f1,
        "country_distance_mean": float(distance_country_df.stack().mean()),
        "country_distance_std": float(distance_country_df.stack().std()),
        "topic_distance_mean": float(distance_topic_df.stack().mean()),
        "topic_distance_std": float(distance_topic_df.stack().std()),
    }])
    overall_summary.to_csv(os.path.join(OUT_DIR, "gemma_overall_summary.csv"), index=False)

    print(f"\nSaved all outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()