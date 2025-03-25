from pyClarion import Atom, Atoms, Family, Chunk, Agent, ChunkStore
import pandas as pd
from src.model_input import vectors

class Feature(Atoms):
    pass

# Dynamically generate 128 bit atoms
for i in range(2500):
    setattr(Feature, f"bit_{i}", Atom)

class IO(Atoms):
    concept1: Atom
    concept2: Atom
    similarity: Atom

class ConceptData(Family):
    io: IO
    features: Feature

def string_to_chunk(binary_str, io_label, features):
    chunk = Chunk()
    for i, bit in enumerate(binary_str):
        if bit == '1':
            chunk += io_label ** getattr(features, f'bit_{i}')
    return chunk
