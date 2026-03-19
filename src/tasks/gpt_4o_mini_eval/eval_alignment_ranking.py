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
    model=None => positive -> 0, negative -> 1
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


def distance_ranking(difference_matrix, value_list, axis=1):
    sorted_list = np.sort(difference_matrix, axis=axis)[:, ::-1]
    sorted_list_idx = np.argsort(difference_matrix, axis=axis)[:, ::-1]
    sorted_values = np.array([[value_list[idx] for idx in row] for row in sorted_list_idx])
    return sorted_list, sorted_list_idx, sorted_values


def main():
    if not Path(T1_PATH).exists():
        raise FileNotFoundError(f"Missing {T1_PATH}")
    if not Path(T2_PATH).exists():
        raise FileNotFoundError(f"Missing {T2_PATH}")

    t1_measures = pd.read_csv(T1_PATH)
    t2_measures = pd.read_csv(T2_PATH)

    if "Unnamed: 0" in t2_measures.columns:
        t2_measures = t2_measures.drop(columns=["Unnamed: 0"])

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

    t1_grouped_country_values, _ = grouping_matrix(t1_pd, countries, "country", value_list, starting_idx=2)
    t2_grouped_country_values, _ = grouping_matrix(t2_pd, countries, "country", value_list, starting_idx=2)

    t1_grouped_topic_values, _ = grouping_matrix(t1_pd, topics, "topic", value_list, starting_idx=2)
    t2_grouped_topic_values, _ = grouping_matrix(t2_pd, topics, "topic", value_list, starting_idx=2)

    distance_country = manhattan_distance(t1_grouped_country_values, t2_grouped_country_values)
    distance_topic = manhattan_distance(t1_grouped_topic_values, t2_grouped_topic_values)

    ranked_distance_country, ranked_distance_idx_country, rank_value_list_country = distance_ranking(distance_country, value_list)
    ranked_distance_topic, ranked_distance_idx_topic, rank_value_list_topic = distance_ranking(distance_topic, value_list)

    print("ranked_distance_country shape:", ranked_distance_country.shape)
    print("rank_value_list_country shape:", rank_value_list_country.shape)
    print("ranked_distance_topic shape:", ranked_distance_topic.shape)
    print("rank_value_list_topic shape:", rank_value_list_topic.shape)

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

    country_rank_df.to_csv("dynamic_full_ranking_country.csv", index=False)
    topic_rank_df.to_csv("dynamic_full_ranking_topic.csv", index=False)

    print("\nTop 10 ranked values for first 5 countries:\n")
    for country in countries[:5]:
        sub = country_rank_df[country_rank_df["country"] == country].head(10)
        print(f"Country: {country}")
        print(sub[["rank", "value", "distance"]].to_string(index=False))
        print()

    print("\nTop 10 ranked values for first 5 topics:\n")
    for topic in topics[:5]:
        sub = topic_rank_df[topic_rank_df["topic"] == topic].head(10)
        print(f"Topic: {topic}")
        print(sub[["rank", "value", "distance"]].to_string(index=False))
        print()

    country_top1_counts = country_rank_df[country_rank_df["rank"] == 1]["value"].value_counts()
    topic_top1_counts = topic_rank_df[topic_rank_df["rank"] == 1]["value"].value_counts()

    country_top5_counts = country_rank_df[country_rank_df["rank"] <= 5]["value"].value_counts()
    topic_top5_counts = topic_rank_df[topic_rank_df["rank"] <= 5]["value"].value_counts()

    print("\nMost frequent top-1 country-ranking values:")
    print(country_top1_counts.head(10).to_string())

    print("\nMost frequent top-1 topic-ranking values:")
    print(topic_top1_counts.head(10).to_string())

    print("\nMost frequent top-5 country-ranking values:")
    print(country_top5_counts.head(15).to_string())

    print("\nMost frequent top-5 topic-ranking values:")
    print(topic_top5_counts.head(15).to_string())

    country_top1_counts.to_csv("dynamic_full_ranking_country_top1_counts.csv", header=["count"])
    topic_top1_counts.to_csv("dynamic_full_ranking_topic_top1_counts.csv", header=["count"])
    country_top5_counts.to_csv("dynamic_full_ranking_country_top5_counts.csv", header=["count"])
    topic_top5_counts.to_csv("dynamic_full_ranking_topic_top5_counts.csv", header=["count"])

    print("\nSaved:")
    print(" - dynamic_full_ranking_country.csv")
    print(" - dynamic_full_ranking_topic.csv")
    print(" - dynamic_full_ranking_country_top1_counts.csv")
    print(" - dynamic_full_ranking_topic_top1_counts.csv")
    print(" - dynamic_full_ranking_country_top5_counts.csv")
    print(" - dynamic_full_ranking_topic_top5_counts.csv")


if __name__ == "__main__":
    main()