import pandas as pd
import numpy as np
import re 
import random
from pandas import DataFrame
from scipy.stats import chisquare

jeopardy = pd.read_csv("JEOPARDY_CSV.csv", skipinitialspace=True)

def normalize(s: str) -> str:
    if pd.isna(s):
        return ""
    return re.sub(r'[^\w\s]', '', s).lower()

def normalize_dollar(s: str) -> str:
    try:
       return int(re.sub(r'[^\w\s]', '', s)) 
    except:
        return 0

def clean(row: DataFrame) -> float:
    split_answer = row["clean_answers"].split(" ")
    split_question = row["clean_questions"].split(" ")

    match_count = 0

    if "the" in split_answer:
        split_answer.remove("the")
    if len(split_answer) == 0:
        return 0

    for answer in split_answer:
        if answer in split_question:
            match_count += 1
    result = match_count / len(split_answer)

    return result

jeopardy["clean_questions"] = jeopardy["Question"].apply(normalize)
jeopardy["clean_answers"] = jeopardy["Answer"].apply(normalize)
jeopardy["clean_values"] = jeopardy["Value"].apply(normalize_dollar)
jeopardy["Air Date"] = pd.to_datetime(jeopardy["Air Date"])
jeopardy["answer_in_question"] = jeopardy.apply(clean, axis=1)
print(jeopardy.columns)
print(jeopardy[["Value", "clean_values"]].head())
print(jeopardy.columns)

question_overlap = []
terms_used = set()
jeopardy = jeopardy.sort_values("Air Date" ,ascending=True)

for index, row in jeopardy.iterrows():
    split_question = row["clean_questions"].split(" ")
    split_question = [w for w in split_question if len(w) >= 6]
    match_count = 0
    for word in split_question:
        if word in terms_used:
            match_count += 1 
        terms_used.add(word)

    if len(split_question) > 0:
        match_count = match_count / len(split_question)
    else:
        match_count = 0
    question_overlap.append(match_count)
jeopardy["question_overlap"] = question_overlap

print(jeopardy["question_overlap"].mean())

def classify_rows(row: DataFrame) -> int:
    value = 0

    if row["clean_values"] > 800:
        value = 1
    else:
        value = 0

    return value

jeopardy["high_value"] = jeopardy.apply(classify_rows, axis=1)

def get_high_value(word: str) -> tuple[int, int]:
    low_count = 0
    high_count = 0

    for _, row in jeopardy.iterrows():
        split_question = row["clean_questions"].split(" ")

        if word in split_question:
            if row["high_value"] == 1:
                high_count += 1
            else:
                low_count += 1

    return high_count, low_count

comparison_terms = random.sample(sorted(terms_used), 10)
observed_expected = []

for terms in comparison_terms:
    low_value_count, high_value_count = get_high_value(terms)
    observed_expected.append((low_value_count, high_value_count))

low_value_count = jeopardy[jeopardy["high_value"] == 1].shape[0]
high_value_count = jeopardy[jeopardy["high_value"] == 0].shape[0]

chi_squared = []

for observed in observed_expected:
    observed_high = observed[0]
    observed_low = observed[1]

    total = observed_high + observed_low

    expected_high = total * (61422/216930)
    expected_low = total * (155508/216930) 

    chi_squared.append(chisquare([observed_high, observed_low], [expected_high, expected_low]))

    print(chi_squared)