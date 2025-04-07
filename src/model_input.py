"""
This script defines the function `load_concept_data()` which loads:
1. Human-annotated similarity scores from the MTurk771 dataset
2. Human-like binary concept representations from a spiking neural network (SNN)

Notes:
- This is a required file for running `model.py`

File Requirements:
- `./data/processedData/AM_binarycode.pkl` — Binary vectors from SNN
- `./data/rawData/EN-MTurk-771.txt` — Human similarity scores
"""


import pickle
import pandas as pd

def load_concept_data():
    """
    Loads and processes concept data for a concept learning model.

    Returns:
        tuple:
            - concept_pairs (list of tuple): A list of tuples where each tuple 
              contains a pair of concepts (concept1, concept2).
            - Human_scores (pd.DataFrame): A DataFrame containing:
                - "Concept pairs": A string representation of the concept pairs.
                - "Human Scores": The human similarity scores for the concept pairs.
                - "Rank": The rank of the human similarity scores in descending order.
    """
    with open("./data/processedData/AM_binarycode.pkl", "rb") as f:
        concept_vectors = pickle.load(f)

    df = pd.read_csv('./data/rawData/EN-MTurk-771.txt', sep='\t', header=None, names=["Concept1", "Concept2", "HumanScore"])
    results = []

    for _, row in df.iterrows():
        concept1 = row["Concept1"]
        concept2 = row["Concept2"]
        human_score = row["HumanScore"]
        if concept1 not in concept_vectors or concept2 not in concept_vectors:
            continue
        concept1_vector = concept_vectors[concept1]
        concept2_vector = concept_vectors[concept2]
        results.append([concept1, concept2, human_score, concept1_vector, concept2_vector])

    Human_scores = pd.DataFrame({
        "Concept pairs": [f"{result[0]}-{result[1]}" for result in results],
        "Human Scores": [result[2] for result in results],
    })
    Human_scores["Rank"] = Human_scores["Human Scores"].rank(ascending=False, method='min').astype(int)
    Human_scores = Human_scores.sort_values(by="Rank")

    # test_data = pd.DataFrame({
    #     "Concept pairs": [f"{result[0]}-{result[1]}" for result in results],
    #     "Concept Vectors": [f"{result[3]}-{result[4]}" for result in results],
    # })

    # test_concepts = {c for r in results for c in (r[0], r[1])}
    # train_concepts = set(concept_vectors.keys()) - test_concepts

    # train_data = pd.DataFrame({
    #     "Concept": list(train_concepts),
    #     "Vector": [concept_vectors[c] for c in train_concepts]
    # })

    concept_pairs = [(r[0], r[1]) for r in results]

    return concept_pairs, Human_scores