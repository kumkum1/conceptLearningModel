"""
This script runs a simulation of human-like concept similarity judgment using the 
Clarion cognitive architecture. It integrates multisensory features and 
spiking-neural-network-derived binary vectors to represent concepts symbolically 
and simulate cognitive spreading activation.

The model compares Clarion-simulated similarity scores to human similarity judgments 
from the MTurk771 dataset, ranking the results and producing a performance comparison.

"""

from pyClarion import Atom, Atoms, Family, Chunk, Agent, ChunkStore, Input
import pandas as pd
import numpy as np
from model_input import load_concept_data

# Load Data
concept_pairs, Human_scores = load_concept_data()
LC823_Merged = pd.read_excel("./data/processedData/LC823_Merged.xlsx")  
vectors = pd.read_pickle("./data/processedData/AM_binarycode.pkl")
concepts = list(set([c for pair in concept_pairs for c in pair]))
concept_vectors = pd.Series({concept: vectors[concept] for concept in concepts}).apply(lambda x: list(x))

# === Keyspace Definitions ===
class Concept(Atoms): 
    """Concept space: one atom per concept"""
    for concept in concepts:
        concept: Atom

class Feature(Atoms):
    """Sensory modality atoms"""
    auditory: Atom
    gustatory: Atom
    haptic: Atom
    olfactory: Atom
    visual: Atom

class Bits(Atoms):
    """Binary representation atoms for SNN encodings"""
    _0 : Atom
    _1 : Atom

class ConceptFeature(Family):
    """Feature family combining concepts, sensory features, and binary bits"""
    concept: Concept
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

        # Dynamically declare 2500 bits as features
        for i in range(2500):
            d.feature[f"bit_{i}"] = Atom()

        with self:
            self.input = Input("input", (d, d))
            self.store = ChunkStore("store", d, d, d)
            self.tlInput = Input("tlInput", (self.store.chunks)) # Model Input
        
        # Connect top-down and bottom-up processes
        self.store.td.input = self.tlInput.main
        self.store.bu.input = self.store.td.main

# === Knowledge Initialization ===
def init_stimuli(d: ConceptFeature, concepts: list[str], df: pd.DataFrame) -> list[Chunk]:
    """
    Create Clarion chunks from sensory features and binary SNN vectors
    rgs:
        d: ConceptFeature defining concept, sensory, and binary atoms
        concepts: List of concepts to encode
        df: DataFrame of sensory features (LC823)
    Returns:
        chunk_list: List of chunks to compile
        chunk_dict: Dict mapping each concept to its chunk
    """
    feature, concept, bit = d.feature, d.concept, d.bits
    chunk_list = []
    chunk_dict = {}

    for concept in concepts:
        row = df[df["Concept"] == concept].iloc[0]
        bits = concept_vectors[concept]

        # Construct chunk from sensory weights
        chunk = ( concept ^ 
            + row["Auditory"] * (feature.auditory ** feature.auditory)
            + row["Gustatory"] * (feature.gustatory ** feature.gustatory)
            + row["Haptic"] * (feature.haptic ** feature.haptic)
            + row["Olfactory"] * (feature.olfactory ** feature.olfactory)
            + row["Visual"] * (feature.visual ** feature.visual)
        )

        # Add 2500-bit vector from SNN output
        for i in range(2500):
            if bits[i] == '1':
                chunk += (feature[f"bit_{i}"] ** bit._1)
            else:   
                chunk -= (feature[f"bit_{i}"] ** bit._0)

        
        chunk_list.append(chunk)
        chunk_dict[concept] = chunk
    return chunk_list, chunk_dict

# === Event Processing ===
agent = ConceptAgent("agent")
stimuli_list, stimuli_dict = init_stimuli(agent.d, concepts, LC823_Merged)
agent.store.compile(*stimuli_list)

# Run initial setup to populate system
while agent.system.queue: 
    agent.system.advance()

pair_scores = []

for c1, c2 in concept_pairs:
    # c1 as a top-down input
    probe = ~stimuli_dict[c1]  
    agent.tlInput.send({probe: 1.0})

    # Advance the Clarion system
    while agent.system.queue:
        agent.system.advance()

    # Measure similarity activation for c2
    activation = agent.store.main[0]
    score = activation[~stimuli_dict[c2]] if ~stimuli_dict[c2] in activation else 0.0
    pair_scores.append(((c1, c2), score))


# === Dispay results ===
pair_scores.sort(key=lambda x: x[1], reverse=True)

Model_scores = pd.DataFrame({
    "Concept pairs": [f"{c1}-{c2}" for (c1, c2), _ in pair_scores],
    "Model Scores": [score for (_, _), score in pair_scores]
})

Model_scores["Model Rank"] = Model_scores["Model Scores"].rank(ascending=False, method="min").astype(int)
Model_scores = Model_scores.sort_values(by="Model Rank")
comparison_df = pd.merge(Human_scores, Model_scores, on="Concept pairs")
print("\n=== Comparison Table ===")
print(comparison_df[["Concept pairs", "Human Scores", "Rank", "Model Scores", "Model Rank"]])