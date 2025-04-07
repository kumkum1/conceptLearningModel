import pickle
import pandas as pd

def load_concept_data():
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

    concept_pairs = [(r[0], r[1]) for r in results]

    return concept_pairs, Human_scores
