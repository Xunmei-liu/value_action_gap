Dynamic Reproduction README
===========================

Purpose
-------
This folder contains the dynamic reproduction of value_action_gap using the current OpenAI API version of gpt-4o-mini.

“Dynamic reproduction” means:
- the current gpt-4o-mini was called again through the API,
- new Task 1 and Task 2 outputs were generated,
- and those new outputs were evaluated using our local evaluation scripts.

What was reused
---------------
We reused the repository data/setup:
- the released Task 1 file to recover the intended request grid,
- the released Task 2 file to recover the intended grouped comparisons and action candidates.

We did NOT regenerate the VIA dataset from scratch.

Main scripts
------------
Generation scripts:
1. run_dynamic_t1.py
   - Re-runs Task 1 on the current gpt-4o-mini.
   - Produces dynamic_gpt4omini_t1.csv

2. run_dynamic_t2.py
   - Re-runs Task 2 on the current gpt-4o-mini using pairwise action comparison.
   - Produces dynamic_gpt4omini_t2.csv
   - Also produces an error log if some grouped requests fail.

Evaluation scripts:
3. eval_dynamic_full_rate.py
   - Evaluates alignment rate on the dynamic outputs.

4. eval_dynamic_full_distance.py
   - Evaluates alignment distance on the dynamic outputs.

5. eval_dynamic_full_ranking.py
   - Evaluates alignment ranking on the dynamic outputs.

6. eval_dynamic_direction_check.py
   - Used to diagnose the correct polarity/direction convention for the dynamic Task 2 output.

Important methodological note
-----------------------------
Dynamic Task 2 does NOT use the same polarity coding as the released static gpt-4o-mini Task 2 file.

For the dynamic Task 2 file, the correct convention is:
- positive -> 0
- negative -> 1

Then, during alignment-rate scoring, we still apply the final repository-style flips:
- flip T1
- flip T2

This was verified by eval_dynamic_direction_check.py.

Do NOT reuse the released static gpt-4o-mini special flip directly for the dynamic Task 2 file.

Main dynamic result
-------------------
Final dynamic overall alignment-rate F1:
- 0.9153809656382943

This is slightly higher than the released static result:
- static released F1 = 0.9049320081421365

Main raw output files
---------------------
- dynamic_gpt4omini_t1.csv
  Full dynamic Task 1 output from current gpt-4o-mini.
  Shape: (1056, 4)

- dynamic_gpt4omini_t2.csv
  Full dynamic Task 2 output from current gpt-4o-mini.
  Shape: (14330, 7)

- dynamic_gpt4omini_t2_errors.csv
  Error log for Task 2 grouped requests that failed, mostly due to malformed generation_prompt strings in the repository data.

Main evaluation output files
----------------------------
Alignment rate:
- dynamic_full_country_results.csv
  Country-level F1 / accuracy summary.

- dynamic_full_t1_pd.csv
  Scenario-level normalized dynamic Task 1 matrix.

- dynamic_full_t2_pd.csv
  Scenario-level normalized dynamic Task 2 matrix.

Alignment distance:
- dynamic_full_distance_country.csv
  Country-level Manhattan distance matrix (12 x 56).

- dynamic_full_distance_topic.csv
  Topic-level Manhattan distance matrix (11 x 56).

- dynamic_full_avg_distance_by_country.csv
  Mean distance by country.

- dynamic_full_avg_distance_by_topic.csv
  Mean distance by topic.

- dynamic_full_avg_distance_by_value_country.csv
  Mean country-level distance by value.

- dynamic_full_avg_distance_by_value_topic.csv
  Mean topic-level distance by value.

Alignment ranking:
- dynamic_full_ranking_country.csv
  Full country-level ranking table.

- dynamic_full_ranking_topic.csv
  Full topic-level ranking table.

- dynamic_full_ranking_country_top1_counts.csv
  Frequency of top-1 ranked values across countries.

- dynamic_full_ranking_topic_top1_counts.csv
  Frequency of top-1 ranked values across topics.

- dynamic_full_ranking_country_top5_counts.csv
  Frequency of top-5 ranked values across countries.

- dynamic_full_ranking_topic_top5_counts.csv
  Frequency of top-5 ranked values across topics.

Direction check outputs
-----------------------
- dynamic_direction_check_summary.csv
  Summary table comparing multiple polarity/flip conventions.

- dynamic_direction_check.log
  Console log from the direction diagnostic.

Logs
----
- dynamic_t1_run.log
  Task 1 generation run log.

- dynamic_t2_run.log
  Task 2 generation run log.

- dynamic_full_rate.log
  Dynamic rate evaluation log.

- dynamic_full_distance.log
  Dynamic distance evaluation log.

- dynamic_full_ranking.log
  Dynamic ranking evaluation log.

How to use this folder
----------------------
If you only want the final dynamic rate result:
- read dynamic_full_country_results.csv

If you want the raw dynamic model outputs:
- read dynamic_gpt4omini_t1.csv
- read dynamic_gpt4omini_t2.csv

If you want the dynamic distance matrices:
- read dynamic_full_distance_country.csv
- read dynamic_full_distance_topic.csv

If you want the dynamic ranking outputs:
- read dynamic_full_ranking_country.csv
- read dynamic_full_ranking_topic.csv

If you want to understand why the polarity handling differs from the static released file:
- read dynamic_direction_check_summary.csv
- inspect eval_dynamic_direction_check.py

Recommended interpretation
--------------------------
The dynamic reproduction indicates that the current API version of gpt-4o-mini produces an overall alignment-rate result that is very close to the released static result, and slightly higher in our run.