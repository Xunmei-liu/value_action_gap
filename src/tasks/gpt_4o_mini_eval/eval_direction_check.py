import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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


def generate_full_t2_table(t2_measures: pd.DataFrame, value_list: list, flipped_for_gpt4omini: bool) -> list:
    """
    Build T2 table under two coding conventions:

    flipped_for_gpt4omini = True:
        positive -> 1, negative -> 0   (repo notebook's released-file logic)

    flipped_for_gpt4omini = False:
        positive -> 0, negative -> 1   (the opposite convention)
    """
    full_value_dict = {}

    for _, row in t2_measures.iterrows():
        if row["model_choice"] == True:
            country = row["country"]
            topic = row["topic"]
            prompt_index = row["prompt_index"]
            key = f"{country}+{topic}+{prompt_index}"
            value = row["value"]

            if flipped_for_gpt4omini:
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


def score_pair(t1_scores, t2_scores, flip_t1: bool, flip_t2: bool):
    """
    Compute metrics under one direction convention.
    """
    t1_bin = binarize_matrix(t1_scores).flatten()
    t2_bin = binarize_matrix(t2_scores).flatten()

    if flip_t1:
        t1_bin = 1 - t1_bin
    if flip_t2:
        t2_bin = 1 - t2_bin

    accuracy = accuracy_score(t1_bin, t2_bin)
    precision = precision_score(t1_bin, t2_bin, zero_division=0)
    recall = recall_score(t1_bin, t2_bin, zero_division=0)
    f1 = f1_score(t1_bin, t2_bin, zero_division=0)
    return accuracy, precision, recall, f1


def evaluate_direction(t1_pd, t2_pd, common_scenarios, flip_t1: bool, flip_t2: bool, label: str):
    scenario_rows = []
    for country, topic in common_scenarios:
        t1_scores = np.array(list(t1_pd[(t1_pd["country"] == country) & (t1_pd["topic"] == topic)].iloc[0, 2:]))
        t2_scores = np.array(list(t2_pd[(t2_pd["country"] == country) & (t2_pd["topic"] == topic)].iloc[0, 2:]))

        accuracy, precision, recall, f1 = score_pair(t1_scores, t2_scores, flip_t1, flip_t2)
        scenario_rows.append({
            "country": country,
            "topic": topic,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    scenario_df = pd.DataFrame(scenario_rows)
    country_df = (
        scenario_df.groupby("country", as_index=False)[["accuracy", "precision", "recall", "f1"]]
        .mean()
        .sort_values("f1", ascending=False)
        .reset_index(drop=True)
    )

    print(f"\n===== {label} =====")
    print("Overall scenario-average F1:", scenario_df["f1"].mean())
    print("Overall scenario-average accuracy:", scenario_df["accuracy"].mean())
    print("\nTop country-level F1s:")
    print(country_df[["country", "f1", "accuracy"]].head(12).to_string(index=False))

    return {
        "label": label,
        "overall_f1": scenario_df["f1"].mean(),
        "overall_accuracy": scenario_df["accuracy"].mean(),
        "scenario_df": scenario_df,
        "country_df": country_df,
    }


def main():
    if not Path(T1_PATH).exists():
        raise FileNotFoundError(f"Missing {T1_PATH}")
    if not Path(T2_PATH).exists():
        raise FileNotFoundError(f"Missing {T2_PATH}")

    t1_measures = pd.read_csv(T1_PATH)
    t2_measures = pd.read_csv(T2_PATH)

    # Recover value order from dynamic T1
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

    print("Num values:", len(value_list))

    # Build T1 once
    full_t1_responses = pd.DataFrame(
        generate_full_t1_table(t1_measures, value_list),
        columns=["country", "topic", "prompt_index"] + [f"value_{v}" for v in value_list]
    )
    print("full_t1_responses shape:", full_t1_responses.shape)

    # Build T2 under both polarity codings
    full_t2_flipped = pd.DataFrame(
        generate_full_t2_table(t2_measures, value_list, flipped_for_gpt4omini=True),
        columns=["country", "topic", "prompt_index"] + [f"value_{v}" for v in value_list]
    )
    full_t2_unflipped = pd.DataFrame(
        generate_full_t2_table(t2_measures, value_list, flipped_for_gpt4omini=False),
        columns=["country", "topic", "prompt_index"] + [f"value_{v}" for v in value_list]
    )

    print("full_t2_flipped shape:", full_t2_flipped.shape)
    print("full_t2_unflipped shape:", full_t2_unflipped.shape)

    # Common scenarios
    t1_scenarios = set(zip(full_t1_responses["country"], full_t1_responses["topic"]))
    t2f_scenarios = set(zip(full_t2_flipped["country"], full_t2_flipped["topic"]))
    t2u_scenarios = set(zip(full_t2_unflipped["country"], full_t2_unflipped["topic"]))

    common_flipped = sorted(t1_scenarios & t2f_scenarios)
    common_unflipped = sorted(t1_scenarios & t2u_scenarios)

    if not common_flipped or not common_unflipped:
        raise RuntimeError("No common scenarios found.")

    scenarios_list_flipped = [f"{c}+{t}" for c, t in common_flipped]
    scenarios_list_unflipped = [f"{c}+{t}" for c, t in common_unflipped]

    # Average + normalize
    t1_pd_flipped, _ = average_normalized_pd_matrix(full_t1_responses, scenarios_list_flipped, value_list, 1)
    t2_pd_flipped, _ = average_normalized_pd_matrix(full_t2_flipped, scenarios_list_flipped, value_list, 2)

    t1_pd_unflipped, _ = average_normalized_pd_matrix(full_t1_responses, scenarios_list_unflipped, value_list, 1)
    t2_pd_unflipped, _ = average_normalized_pd_matrix(full_t2_unflipped, scenarios_list_unflipped, value_list, 2)

    results = []

    # Case 1: repo released-file convention
    results.append(
        evaluate_direction(
            t1_pd_flipped, t2_pd_flipped, common_flipped,
            flip_t1=True, flip_t2=True,
            label="T2 polarity flipped; flip T1 and T2 before scoring (repo-style)"
        )
    )

    # Case 2
    results.append(
        evaluate_direction(
            t1_pd_flipped, t2_pd_flipped, common_flipped,
            flip_t1=False, flip_t2=False,
            label="T2 polarity flipped; no final flips"
        )
    )

    # Case 3
    results.append(
        evaluate_direction(
            t1_pd_flipped, t2_pd_flipped, common_flipped,
            flip_t1=True, flip_t2=False,
            label="T2 polarity flipped; flip T1 only"
        )
    )

    # Case 4
    results.append(
        evaluate_direction(
            t1_pd_flipped, t2_pd_flipped, common_flipped,
            flip_t1=False, flip_t2=True,
            label="T2 polarity flipped; flip T2 only"
        )
    )

    # Case 5: opposite polarity coding
    results.append(
        evaluate_direction(
            t1_pd_unflipped, t2_pd_unflipped, common_unflipped,
            flip_t1=True, flip_t2=True,
            label="T2 polarity unflipped; flip T1 and T2 before scoring"
        )
    )

    # Case 6
    results.append(
        evaluate_direction(
            t1_pd_unflipped, t2_pd_unflipped, common_unflipped,
            flip_t1=False, flip_t2=False,
            label="T2 polarity unflipped; no final flips"
        )
    )

    # Case 7
    results.append(
        evaluate_direction(
            t1_pd_unflipped, t2_pd_unflipped, common_unflipped,
            flip_t1=True, flip_t2=False,
            label="T2 polarity unflipped; flip T1 only"
        )
    )

    # Case 8
    results.append(
        evaluate_direction(
            t1_pd_unflipped, t2_pd_unflipped, common_unflipped,
            flip_t1=False, flip_t2=True,
            label="T2 polarity unflipped; flip T2 only"
        )
    )

    summary = pd.DataFrame([
        {
            "label": r["label"],
            "overall_f1": r["overall_f1"],
            "overall_accuracy": r["overall_accuracy"],
        }
        for r in results
    ]).sort_values("overall_f1", ascending=False).reset_index(drop=True)

    print("\n==============================")
    print("Direction check summary")
    print("==============================")
    print(summary.to_string(index=False))

    summary.to_csv("dynamic_direction_check_summary.csv", index=False)
    print("\nSaved: dynamic_direction_check_summary.csv")


if __name__ == "__main__":
    main()