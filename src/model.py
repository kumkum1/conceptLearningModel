from pyClarion import Atom, Atoms, Family, Chunk, Agent, ChunkStore, Input, Choice
import pandas as pd
import pprint


# strat with 3 conspts
# 1. have a shuchkstore with 3 concptes each chuck is one cecnpet with the multisensory features (LC823_Merged)
# 2. send one concept top down and then connect that directly to a bottom up process this almost acts asactly like  cosince calualtion
# 3. by doing this you get the activations which is like the similarity score compairting the one conspt with each of the concepts in the chuckstore
# 4. now add the human-like repsentation in this as well 
#  - add it as a feature for each and each bit is an atom

#NEED TO MOTIVATE THE MODEL AND WHY IT IS IMPORTANT AND WHAT IT ADD TO THE EXISTING PAPER STUDY
#modulatiry 
#how it is able to learn to study the environemt as it exists within it 


LC823_Merged = pd.read_excel("./data/processedData/LC823_Merged.xlsx")  
concepts = list(LC823_Merged["Concept"])[:3]

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

class ConceptFeature(Family):
    concept: Concept
    feature: Feature

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
        with self:
            self.input = Input("input", (d, d))
            self.store = ChunkStore("store", d, d, d)
            self.tlInput = Input("tlInput", (self.store.chunks))
        
        self.store.td.input = self.tlInput.main
        self.store.bu.input = self.store.td.main

# === Knowledge Initialization ===
def init_stimuli(d: ConceptFeature, concepts: list[str], df: pd.DataFrame) -> list[Chunk]:
    feature, concept = d.feature, d.concept
    chunk_defs = []

    for concept in concepts:
        row = df[df["Concept"] == concept].iloc[0]
        chunk = ( concept ^ 
            + row["Auditory"] * (feature.auditory ** feature.auditory)
            + row["Gustatory"] * (feature.gustatory ** feature.gustatory)
            + row["Haptic"] * (feature.haptic ** feature.haptic)
            + row["Olfactory"] * (feature.olfactory ** feature.olfactory)
            + row["Visual"] * (feature.visual ** feature.visual)
        )
        chunk_defs.append(chunk)
    return chunk_defs

# === Event Processing ===
agent = ConceptAgent("agent")
stimuli = init_stimuli(agent.d, concepts, LC823_Merged)
agent.store.compile(*stimuli)

while agent.system.queue: 
    event = agent.system.advance()
    print(event.describe())
    #get the activation
    #figure out the events

# agent.input.send(stimuli[0])
# agent.tlInput.send(stimuli[0]) #send in activations; each chuck already has the coresponding concpet name in it so for each concpet have the activiation maped to it 
# print("Input sent:", stimuli[0])
# print("Input sent:", concepts[0])

# while agent.system.queue:
#     event = agent.system.advance()
#     print(event.describe())

# for i in range(3):
#     key = 'd:store:_'+str(i)
#     print(f"Similarity score {concepts[i]}: {agent.store.bu.main[0][key]}")

   