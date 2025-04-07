from pyClarion import Atom, Atoms, Family, Chunk, Agent, ChunkStore, Input
import pandas as pd
import numpy as np

LC823_Merged = pd.read_excel("./data/processedData/LC823_Merged.xlsx")  
vectors = pd.read_pickle("./data/processedData/AM_binarycode.pkl")
concepts = list(LC823_Merged["Concept"])[:3]
concept_vectors = {concept: vectors[concept] for concept in concepts}
concept_vectors = pd.Series(concept_vectors).apply(lambda x: list(x))

# === Keyspace Definitions ===
class Concept(Atoms): 
    for concept in concepts:
        concept: Atom

class Feature(Atoms):
    auditory: Atom
    gustatory: Atom
    haptic: Atom
    olfactory: Atom
    visual: Atom

class Bits(Atoms):
    _0 : Atom
    _1 : Atom

class ConceptFeature(Family):
    concept: Concept
    feature: Feature
    bits: Bits

# === Model Construction ===
class ConceptAgent(Agent):
    d: ConceptFeature
    input: Input
    tlInput: Input
    store: ChunkStore

    def __init__(self, name: str) -> None:
        d = ConceptFeature()
        p = Family()
        e = Family()
        super().__init__(name, p=p, e=e, d=d)
        self.d = d
        for i in range(2500):
            d.feature[f"bit_{i}"] = Atom()
        with self:
            self.input = Input("input", (d, d))
            self.store = ChunkStore("store", d, d, d)
            self.tlInput = Input("tlInput", (self.store.chunks))
        
        self.store.td.input = self.tlInput.main
        self.store.bu.input = self.store.td.main

# === Knowledge Initialization ===
def init_stimuli(d: ConceptFeature, concepts: list[str], df: pd.DataFrame) -> list[Chunk]:
    feature, concept, bit = d.feature, d.concept, d.bits
    chunk_defs = []

    for concept in concepts:
        row = df[df["Concept"] == concept].iloc[0]
        bits = concept_vectors[concept]

        chunk = (
            concept ^ 
            + row["Auditory"] * (feature.auditory ** feature.auditory)
            + row["Gustatory"] * (feature.gustatory ** feature.gustatory)
            + row["Haptic"] * (feature.haptic ** feature.haptic)
            + row["Olfactory"] * (feature.olfactory ** feature.olfactory)
            + row["Visual"] * (feature.visual ** feature.visual)
      )
        for i in range(2500):
            if bits[i] == '1':
                chunk += (feature[f"bit_{i}"] ** bit._1)
            else:   
                chunk -= (feature[f"bit_{i}"] ** bit._0)

        chunk_defs.append(chunk)
    return chunk_defs


# === Event Processing ===
agent = ConceptAgent("agent")
stimuli = init_stimuli(agent.d, concepts, LC823_Merged)
agent.store.compile(*stimuli)

while agent.system.queue: 
    event = agent.system.advance()
    # print(event.describe())

probe = ~stimuli[1] 
agent.tlInput.send({probe: 1.0})  

while agent.system.queue: 
    event = agent.system.advance()
    # print(event.describe()) 

# print(agent.store.td.main[0])
# print(agent.store.bu.weights[0])

similarity_vector = agent.store.main[0] 

ranked = sorted(
    [(k, similarity_vector[k]) for k in similarity_vector],
    key=lambda x: x[1],
    reverse=True
)

print(f"\nSimilarity ranking for: {probe}")
for rank, (concept, score) in enumerate(ranked, 1):
    print(f"{rank}. {concept}: {score:.3f}")