import numpy as np
from typing import Dict

def normalize_vector(vector: np.ndarray) -> np.ndarray:
    if np.all(vector == 0):
        return vector  
        
    v_min, v_max = vector.min(), vector.max()
    if v_min == v_max:
        return np.ones_like(vector)  
    
    return (vector - v_min) / (v_max - v_min)

def poisson_spike_train(rate: float, duration: int, dt: float = 1.0) -> np.ndarray:
    prob_spike = rate * dt
    return (np.random.random(duration) < prob_spike).astype(np.int8)

def encode_vector_poisson(
    vector: np.ndarray, 
    duration: int = 100, 
    dt: float = 1.0,
    normalize: bool = True
) -> np.ndarray:
    if normalize:
        vector = normalize_vector(vector)
    spike_trains = np.zeros((len(vector), duration), dtype=np.int8)
    
    for i, val in enumerate(vector):
        spike_trains[i] = poisson_spike_train(val, duration, dt)
    
    return spike_trains

def encode_concept(
    word: str,
    sensory_func, 
    text_func,
    duration: int = 100, 
    dt: float = 1.0
) -> Dict[str, np.ndarray]:
    sensory_vector = sensory_func(word)
    text_vector = text_func(word)
    sensory_spikes = encode_vector_poisson(sensory_vector, duration, dt)
    text_spikes = encode_vector_poisson(text_vector, duration, dt)
    
    return {
        'sensory': sensory_spikes,
        'text': text_spikes
    }
