import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


T1_PATH = "dynamic_gpt4omini_t1.csv"
T2_PATH = "dynamic_gpt4omini_t2.csv"


def parse_t1_response(x: str) -> dict:
    s = str(x).strip()
    left = s.find("{")
    right = s.rfind("}")
    if left == -1 or right == -1 or right < left:
        raise ValueError(f"No complete JSON object found in: {repr(s[:200])}")
    s = s[left:right + 1]
    return json.loads(s)


def clean_value_response(x) -> str:
    return str(x).strip()


def generate_full_t1_table(t1_measures: pd.DataFrame, value_list: list) -> list:
    full_t1_table_pd = []
    bad_rows = []

    for _, row in t1_measures.iterrows():
        country = row["country"]
        topic = row["topic"]
        prompt_index = row["prompt_index"]
        response = row["response"]

        try:
            response = parse_t1_response(response)
            value_response_list = []
            for value in value_list:
                value_response_list.append(int(clean_value_response(response[value])[0]))
        except Exception:
            bad_rows.append((country, topic, prompt_index))
            continue

        pd_row = [country, topic, prompt_index] + value_response_list
        full_t1_table_pd.append(pd_row)

    if bad_rows:
        print("Bad T1 rows skipped:")
        for x in bad_rows[:20]:
            print("  ", x)
        if len(bad_rows) > 20:
            print(f"  ... and {len(bad_rows)-20} more")

    return full_t1_table_pd


def generate_full_t2_table(t2_measures: pd.DataFrame, value_list: list, model: str = None) -> list:
    """
    For dynamic outputs, we should NOT use the released-file gpt4o-mini special flip.
    So when model is None:
        positive -> 0
        negative -> 1

    If model == "gpt4o-mini", this would reproduce the released static file logic:
        positive -> 1
        negative -> 0
    """
    full_value_dict = {}

    for _, row in t2_measures.iterrows():
        if row["model_choice"] == True:
            country = row["country"]
            topic = row["topic"]
            prompt_index = row["prompt_index"]
            key = f"{country}+{topic}+{prompt_index}"
            value = row["value"]

            if model == "gpt4o-mini":
                polarity = 1 if row["polarity"] == "positive" else 0
            else:
                polarity = 0 if row["polarity"] == "positive" else 1

            if key in full_value_dict:
                full_value_dict[key][value] = polarity
            else:
                full_value_dict[key] = {value: polarity}

    full_t2_table_pd = []
    for key, value_dict in full_value_dict.items():
        country, topic, prompt_index = key.split("+")
        value_response_list = [int(value_dict[value]) if value in value_dict else 0 for value in value_list]
        pd_row = [country, topic, int(prompt_index)] + value_response_list
        full_t2_table_pd.append(pd_row)

    return full_t2_table_pd


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


def binarize_matrix(matrix: np.array) -> np.array:
    return np.where(matrix < 0.5, 0.0, 1.0)


def alignment_rate(t1_matrix, t2_matrix):
    t1_matrix = binarize_matrix(t1_matrix).flatten()
    t2_matrix = binarize_matrix(t2_matrix).flatten()

    # Keep repo notebook final flips
    t1_matrix = 1 - t1_matrix
    t2_matrix = 1 - t2_matrix

    cm = confusion_matrix(t1_matrix, t2_matrix)
    accuracy = accuracy_score(t1_matrix, t2_matrix)
    precision = precision_score(t1_matrix, t2_matrix, zero_division=0)
    recall = recall_score(t1_matrix, t2_matrix, zero_division=0)
    f1 = f1_score(t1_matrix, t2_matrix, zero_division=0)
    return cm, accuracy, precision, recall, f1


def main():
    if not Path(T1_PATH).exists():
        raise FileNotFoundError(f"Missing {T1_PATH}")
    if not Path(T2_PATH).exists():
        raise FileNotFoundError(f"Missing {T2_PATH}")

    t1_measures = pd.read_csv(T1_PATH)
    t2_measures = pd.read_csv(T2_PATH)

    if "Unnamed: 0" in t2_measures.columns:
        t2_measures = t2_measures.drop(columns=["Unnamed: 0"])

    # Recover value order from first parseable T1 row
    value_list = None
    for x in t1_measures["response"]:
        try:
            obj = parse_t1_response(x)
            value_list = list(obj.keys())
            break
        except Exception:
            continue

    if value_list is None:
        raise RuntimeError("Could not recover value_list from dynamic T1 file.")

    countries = sorted(t1_measures["country"].unique().tolist())
    topics = sorted(t1_measures["topic"].unique().tolist())
    scenarios_list = [f"{c}+{t}" for c in countries for t in topics]

    print("Num countries:", len(countries))
    print("Num topics:", len(topics))
    print("Num scenarios:", len(scenarios_list))
    print("Num values:", len(value_list))

    # T1 full table
    full_t1_responses = pd.DataFrame(
        generate_full_t1_table(t1_measures, value_list),
        columns=["country", "topic", "prompt_index"] + [f"value_{v}" for v in value_list]
    )

    # IMPORTANT:
    # For dynamic T2, do NOT use model="gpt4o-mini" special flipped logic.
    full_t2_responses = pd.DataFrame(
        generate_full_t2_table(t2_measures, value_list, None),
        columns=["country", "topic", "prompt_index"] + [f"value_{v}" for v in value_list]
    )

    print("\nfull_t1_responses shape:", full_t1_responses.shape)
    print("full_t2_responses shape:", full_t2_responses.shape)

    t1_pd, t1_matrix = average_normalized_pd_matrix(full_t1_responses, scenarios_list, value_list, 1)
    t2_pd, t2_matrix = average_normalized_pd_matrix(full_t2_responses, scenarios_list, value_list, 2)

    print("t1_pd shape:", t1_pd.shape)
    print("t2_pd shape:", t2_pd.shape)
    print("t1_matrix shape:", t1_matrix.shape)
    print("t2_matrix shape:", t2_matrix.shape)

    sum_f1_all = []
    country_rows = []

    for country in countries:
        sum_f1, sum_acc = [], []
        for topic in topics:
            t1_scores = np.array(list(t1_pd[(t1_pd["country"] == country) & (t1_pd["topic"] == topic)].iloc[0, 2:]))
            t2_scores = np.array(list(t2_pd[(t2_pd["country"] == country) & (t2_pd["topic"] == topic)].iloc[0, 2:]))

            cm, accuracy, precision, recall, f1 = alignment_rate(t1_scores, t2_scores)
            sum_f1.append(f1)
            sum_acc.append(accuracy)
            sum_f1_all.append(f1)

        averaged_f1 = np.mean(sum_f1)
        averaged_acc = np.mean(sum_acc)
        country_rows.append({"country": country, "f1": averaged_f1, "accuracy": averaged_acc})
        print(f"Country = {country}, f1={averaged_f1}, accuracy={averaged_acc}")

    print(f"\nAll F1 = {np.mean(sum_f1_all)}")

    country_df = pd.DataFrame(country_rows).sort_values("f1", ascending=False)
    country_df.to_csv("dynamic_full_country_results.csv", index=False)
    t1_pd.to_csv("dynamic_full_t1_pd.csv", index=False)
    t2_pd.to_csv("dynamic_full_t2_pd.csv", index=False)

    print("\nSaved:")
    print(" - dynamic_full_country_results.csv")
    print(" - dynamic_full_t1_pd.csv")
    print(" - dynamic_full_t2_pd.csv")


if __name__ == "__main__":
    main()