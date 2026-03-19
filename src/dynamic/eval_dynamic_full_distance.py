import json
from pathlib import Path

import numpy as np
import pandas as pd


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
    For dynamic outputs, do NOT use released-file special flip.
    model=None  => positive -> 0, negative -> 1
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

        average_prompting = response_pd[
            (response_pd["country"] == country) & (response_pd["topic"] == topic)
        ].iloc[:, 3:].mean()

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

    if "Unnamed: 0" in t2_measures.columns:
        t2_measures = t2_measures.drop(columns=["Unnamed: 0"])

    # recover value order from first parseable T1 row
    value_list = None
    for x in t1_measures["response"]:
        try:
            obj = parse_t1_response(x)
            value_list = list(obj.keys())
            break
        except Exception:
            continue

    if value_list is None:
        raise RuntimeError("Could not recover value_list from T1.")

    countries = sorted(t1_measures["country"].unique().tolist())
    topics = sorted(t1_measures["topic"].unique().tolist())
    scenarios_list = [f"{c}+{t}" for c in countries for t in topics]

    print("Num countries:", len(countries))
    print("Num topics:", len(topics))
    print("Num scenarios:", len(scenarios_list))
    print("Num values:", len(value_list))

    full_t1_responses = pd.DataFrame(
        generate_full_t1_table(t1_measures, value_list),
        columns=["country", "topic", "prompt_index"] + [f"value_{v}" for v in value_list]
    )

    # IMPORTANT: dynamic T2 uses model=None here
    full_t2_responses = pd.DataFrame(
        generate_full_t2_table(t2_measures, value_list, None),
        columns=["country", "topic", "prompt_index"] + [f"value_{v}" for v in value_list]
    )

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

    distance_country_df.to_csv("dynamic_full_distance_country.csv")
    distance_topic_df.to_csv("dynamic_full_distance_topic.csv")
    avg_distance_by_country.to_csv("dynamic_full_avg_distance_by_country.csv", header=["mean_distance"])
    avg_distance_by_topic.to_csv("dynamic_full_avg_distance_by_topic.csv", header=["mean_distance"])
    avg_distance_by_value_country.to_csv("dynamic_full_avg_distance_by_value_country.csv", header=["mean_distance"])
    avg_distance_by_value_topic.to_csv("dynamic_full_avg_distance_by_value_topic.csv", header=["mean_distance"])

    print("\nSaved:")
    print(" - dynamic_full_distance_country.csv")
    print(" - dynamic_full_distance_topic.csv")
    print(" - dynamic_full_avg_distance_by_country.csv")
    print(" - dynamic_full_avg_distance_by_topic.csv")
    print(" - dynamic_full_avg_distance_by_value_country.csv")
    print(" - dynamic_full_avg_distance_by_value_topic.csv")


if __name__ == "__main__":
    main()