import pickle
import pandas as pd

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

print(Human_scores)


test_data = pd.DataFrame({
    "Concept pairs": [f"{result[0]}-{result[1]}" for result in results],
    "Concept Vectors": [f"{result[3]}-{result[4]}" for result in results],
})


test_concepts = {c for r in results for c in (r[0], r[1])}
train_concepts = set(concept_vectors.keys()) - test_concepts

train_data = pd.DataFrame({
    "Concept": list(train_concepts),
    "Vector": [concept_vectors[c] for c in train_concepts]
})