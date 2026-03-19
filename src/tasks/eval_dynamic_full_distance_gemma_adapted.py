import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ====== INPUT PATHS ======
T1_PATH = "/gscratch/cse/xunmei/value_action_gap/src/outputs/task1_gemma_statements.csv"
T2_PATH = "/gscratch/cse/xunmei/value_action_gap/src/outputs/task2_results_gemma2.csv"

# ====== OUTPUT DIR ======
OUT_DIR = Path("/gscratch/cse/xunmei/value_action_gap/src/outputs/subset/gemma_dynamic_eval_distance_subset")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASK1_EVAL_COLS = [f"evaluation_{i}" for i in range(8)]
TASK2_EVAL_COLS = [f"evaluation_{i}" for i in range(8)]


def parse_json_like(x):
    """
    Robust parser for:
    - proper JSON strings
    - python dict strings with single quotes
    - strings containing embedded {...}
    """
    s = str(x).strip()

    try:
        return json.loads(s)
    except Exception:
        pass

    try:
        return ast.literal_eval(s)
    except Exception:
        pass

    left = s.find("{")
    right = s.rfind("}")
    if left != -1 and right != -1 and right >= left:
        s2 = s[left:right + 1]
        try:
            return json.loads(s2)
        except Exception:
            try:
                return ast.literal_eval(s2)
            except Exception:
                return None

    return None


def clean_value_response(x) -> str:
    return str(x).strip()


def normalize_value_name(name: str) -> str:
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


def recover_value_list_from_task1(t1_df: pd.DataFrame):
    """
    Recover value order from first parseable Task1 row.
    """
    for _, row in t1_df.iterrows():
        for col in TASK1_EVAL_COLS:
            if col not in t1_df.columns:
                continue
            obj = parse_json_like(row[col])
            if isinstance(obj, dict) and len(obj) > 0:
                return [normalize_value_name(v) for v in obj.keys()]
    raise RuntimeError("Could not recover value_list from Task1 file.")


def build_full_t1_responses_from_wide(t1_measures: pd.DataFrame, value_list: list) -> pd.DataFrame:
    """
    Convert your wide Task1:
      country, topic, evaluation_0...evaluation_7
    to prompt-level matrix:
      country, topic, prompt_index, value_...
    """
    rows = []
    bad_rows = []

    for _, row in t1_measures.iterrows():
        country = row["country"]
        topic = row["topic"]

        for col in TASK1_EVAL_COLS:
            if col not in t1_measures.columns:
                continue

            prompt_index = int(col.split("_")[1])

            try:
                obj = parse_json_like(row[col])
                if not isinstance(obj, dict):
                    raise ValueError("Task1 cell did not parse to dict")

                obj_norm = {normalize_value_name(k): v for k, v in obj.items()}

                value_response_list = []
                for value in value_list:
                    raw = obj_norm[value]
                    value_response_list.append(int(clean_value_response(raw)[0]))

                rows.append([country, topic, prompt_index] + value_response_list)

            except Exception:
                bad_rows.append((country, topic, prompt_index))
                continue

    if bad_rows:
        print("Bad T1 rows skipped:")
        for x in bad_rows[:20]:
            print("  ", x)
        if len(bad_rows) > 20:
            print(f"  ... and {len(bad_rows)-20} more")

    return pd.DataFrame(
        rows,
        columns=["country", "topic", "prompt_index"] + [f"value_{v}" for v in value_list]
    )


def build_full_t2_responses_from_wide(t2_measures: pd.DataFrame, value_list: list) -> pd.DataFrame:
    """
    Convert your wide Task2:
      country, topic, value, option1, option2, evaluation_0...evaluation_7
    to prompt-level matrix:
      country, topic, prompt_index, value_...

    Dynamic convention:
      Option 2 -> positive -> 0
      Option 1 -> negative -> 1
    """
    full_value_dict = {}

    for _, row in t2_measures.iterrows():
        country = row["country"]
        topic = row["topic"]
        value = normalize_value_name(row["value"])

        for col in TASK2_EVAL_COLS:
            if col not in t2_measures.columns:
                continue

            prompt_index = int(col.split("_")[1])
            parsed = parse_json_like(row[col])

            if not isinstance(parsed, dict):
                continue

            action = parsed.get("action")

            if action == "Option 2":
                polarity = 0
            elif action == "Option 1":
                polarity = 1
            else:
                continue

            key = f"{country}+{topic}+{prompt_index}"
            if key not in full_value_dict:
                full_value_dict[key] = {}
            full_value_dict[key][value] = polarity

    rows = []
    for key, value_dict in full_value_dict.items():
        country, topic, prompt_index = key.split("+")
        value_response_list = [
            int(value_dict[value]) if value in value_dict else 0
            for value in value_list
        ]
        rows.append([country, topic, int(prompt_index)] + value_response_list)

    return pd.DataFrame(
        rows,
        columns=["country", "topic", "prompt_index"] + [f"value_{v}" for v in value_list]
    )


def min_max_normalization(matrix: np.array, min=None, max=None):
    if min is None:
        min = np.min(matrix)
    if max is None:
        max = np.max(matrix)
    return (matrix - min) / (max - min)


def average_normalized_pd_matrix(response_pd: pd.DataFrame, scenarios_list: list, value_list: list, task: int):
    full_pd, full_matrix = [], []

    for scenario in scenarios_list:
        country, topic = scenario.split("+")

        subset = response_pd[
            (response_pd["country"] == country) & (response_pd["topic"] == topic)
        ]
        if subset.empty:
            continue

        average_prompting = subset.iloc[:, 3:].mean()

        normalized_average_prompting = (
            min_max_normalization(np.array(list(average_prompting)), 1, 4)
            if task == 1
            else min_max_normalization(np.array(list(average_prompting)), 0, 1)
        )

        full_matrix.append(normalized_average_prompting)
        full_pd.append([country, topic] + list(normalized_average_prompting))

    full_pd_all = pd.DataFrame(full_pd, columns=["country", "topic"] + [f"{value}" for value in value_list])
    return full_pd_all, np.array(full_matrix)


def grouping_matrix(full_pd: pd.DataFrame, group_values: list, scenario_name: str, value_list: list, starting_idx: int = 2):
    grouped = []
    for item in group_values:
        average_scenarios = full_pd[(full_pd[scenario_name] == item)].iloc[:, starting_idx:].mean()
        grouped.append(list(average_scenarios))
    results = pd.DataFrame(grouped, columns=[f"{value}" for value in value_list])
    return results.to_numpy(), results


def manhattan_distance(t1_matrix, t2_matrix):
    return np.abs(t1_matrix - t2_matrix)


def main():
    if not Path(T1_PATH).exists():
        raise FileNotFoundError(f"Missing {T1_PATH}")
    if not Path(T2_PATH).exists():
        raise FileNotFoundError(f"Missing {T2_PATH}")

    t1_measures = pd.read_csv(T1_PATH)
    t2_measures = pd.read_csv(T2_PATH)

    value_list = recover_value_list_from_task1(t1_measures)

    countries = sorted(t1_measures["country"].unique().tolist())
    topics = sorted(t1_measures["topic"].unique().tolist())
    scenarios_list = [f"{c}+{t}" for c in countries for t in topics]

    print("Num countries:", len(countries))
    print("Num topics:", len(topics))
    print("Num scenarios:", len(scenarios_list))
    print("Num values:", len(value_list))

    full_t1_responses = build_full_t1_responses_from_wide(t1_measures, value_list)
    full_t2_responses = build_full_t2_responses_from_wide(t2_measures, value_list)

    t1_pd, t1_matrix = average_normalized_pd_matrix(full_t1_responses, scenarios_list, value_list, 1)
    t2_pd, t2_matrix = average_normalized_pd_matrix(full_t2_responses, scenarios_list, value_list, 2)

    print("\nt1_pd shape:", t1_pd.shape)
    print("t2_pd shape:", t2_pd.shape)
    print("t1_matrix shape:", t1_matrix.shape)
    print("t2_matrix shape:", t2_matrix.shape)

    t1_grouped_country_values, t1_grouped_country_df = grouping_matrix(
        t1_pd, countries, "country", value_list, starting_idx=2
    )
    t2_grouped_country_values, t2_grouped_country_df = grouping_matrix(
        t2_pd, countries, "country", value_list, starting_idx=2
    )

    t1_grouped_topic_values, t1_grouped_topic_df = grouping_matrix(
        t1_pd, topics, "topic", value_list, starting_idx=2
    )
    t2_grouped_topic_values, t2_grouped_topic_df = grouping_matrix(
        t2_pd, topics, "topic", value_list, starting_idx=2
    )

    print("\nt1_grouped_country_values shape:", t1_grouped_country_values.shape)
    print("t2_grouped_country_values shape:", t2_grouped_country_values.shape)
    print("t1_grouped_topic_values shape:", t1_grouped_topic_values.shape)
    print("t2_grouped_topic_values shape:", t2_grouped_topic_values.shape)

    distance_country = manhattan_distance(t1_grouped_country_values, t2_grouped_country_values)
    distance_topic = manhattan_distance(t1_grouped_topic_values, t2_grouped_topic_values)

    print("\ndistance_country shape:", distance_country.shape)
    print("distance_topic shape:", distance_topic.shape)

    distance_country_df = pd.DataFrame(distance_country, index=countries, columns=value_list)
    distance_topic_df = pd.DataFrame(distance_topic, index=topics, columns=value_list)

    print("\nCountry-level Manhattan distance summary:")
    print(distance_country_df.stack().describe())

    print("\nTopic-level Manhattan distance summary:")
    print(distance_topic_df.stack().describe())

    avg_distance_by_country = distance_country_df.mean(axis=1).sort_values(ascending=False)
    avg_distance_by_topic = distance_topic_df.mean(axis=1).sort_values(ascending=False)
    avg_distance_by_value_country = distance_country_df.mean(axis=0).sort_values(ascending=False)
    avg_distance_by_value_topic = distance_topic_df.mean(axis=0).sort_values(ascending=False)

    print("\nTop 10 countries by mean distance:")
    print(avg_distance_by_country.head(10).to_string())

    print("\nTop 10 topics by mean distance:")
    print(avg_distance_by_topic.head(10).to_string())

    print("\nTop 10 values by mean country-level distance:")
    print(avg_distance_by_value_country.head(10).to_string())

    print("\nTop 10 values by mean topic-level distance:")
    print(avg_distance_by_value_topic.head(10).to_string())

    distance_country_df.to_csv(OUT_DIR / "gemma_dynamic_full_distance_country.csv")
    distance_topic_df.to_csv(OUT_DIR / "gemma_dynamic_full_distance_topic.csv")
    avg_distance_by_country.to_csv(OUT_DIR / "gemma_dynamic_full_avg_distance_by_country.csv", header=["mean_distance"])
    avg_distance_by_topic.to_csv(OUT_DIR / "gemma_dynamic_full_avg_distance_by_topic.csv", header=["mean_distance"])
    avg_distance_by_value_country.to_csv(OUT_DIR / "gemma_dynamic_full_avg_distance_by_value_country.csv", header=["mean_distance"])
    avg_distance_by_value_topic.to_csv(OUT_DIR / "gemma_dynamic_full_avg_distance_by_value_topic.csv", header=["mean_distance"])
    t1_pd.to_csv(OUT_DIR / "gemma_dynamic_full_t1_pd.csv", index=False)
    t2_pd.to_csv(OUT_DIR / "gemma_dynamic_full_t2_pd.csv", index=False)

    print("\nSaved all outputs to:")
    print(f" - {OUT_DIR}")
    print("Files:")
    print(" - gemma_dynamic_full_distance_country.csv")
    print(" - gemma_dynamic_full_distance_topic.csv")
    print(" - gemma_dynamic_full_avg_distance_by_country.csv")
    print(" - gemma_dynamic_full_avg_distance_by_topic.csv")
    print(" - gemma_dynamic_full_avg_distance_by_value_country.csv")
    print(" - gemma_dynamic_full_avg_distance_by_value_topic.csv")
    print(" - gemma_dynamic_full_t1_pd.csv")
    print(" - gemma_dynamic_full_t2_pd.csv")


if __name__ == "__main__":
    main()