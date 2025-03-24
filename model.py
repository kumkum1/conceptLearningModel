from pyClarion import Chunk, Feature, Key, world, Agent
import numpy as np
import pickle

with open("AM_binarycode.pkl", "rb") as f:
    concept_vectors = pickle.load(f)
    
