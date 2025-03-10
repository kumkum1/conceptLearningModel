from pyClarion import * 
from pyClarion import io
import pandas as pd
from dataProcessing.textData import get_text_embedding
from dataProcessing.sensoryData import get_sensory_representation

class Sensory(Atoms):
    auditory: Atom
    gustatory: Atom
    haptic: Atom 
    olfactory: Atom
    visual: Atom

# chunk_defs = [
#     + io.input ** Sensory.auditory
#     + io.input ** Sensory.gustatory
#     + io.input ** Sensory.haptic
#     + io.input ** Sensory.olfactory
#     + io.input ** Sensory.visual,
# ]

class Text(Atoms):
    word: Atom

class IO(Atoms):
    input: Atom
    output: Atom
    goal: Atom
