# Human-like Concept Learning Computational Model

## Project Overview
Developing a computational model for human-like concept learning using the Clarion cognitive-science framework. The model integrates multisensory and text-derived representations of concepts, dynamically combining them to emulate human cognition closely.

## Datasets Used

### 1. LC823 Multisensory Dataset
- **Source**: Lynott & Connell (2009, 2013)
- **Data Format**: 5-dimensional vectors representing sensory intensity across:
  - Auditory
  - Gustatory
  - Haptic
  - Olfactory
  - Visual
- **Total Concepts**: 823 (423 adjectives + 400 nouns)

**Official Sources:**
- [Adjective Dataset](https://link.springer.com/article/10.3758/BRM.41.2.558)
- [Noun Dataset](https://link.springer.com/article/10.3758/s13428-012-0267-0)

### Text-derived Representation Datasets
#### GloVe Dataset
- **Source:** [GloVe embeddings](https://nlp.stanford.edu/projects/glove/)
- **Format**: 300-dimensional word embeddings (glove-wiki-gigaword-300)

### Word2Vec Dataset
- **Source:** [Word2Vec](https://code.google.com/archive/p/word2vec/)
- **Format:** 300-dimensional word embeddings

## Implementation Structure

### Training Phase
- 

### Testing and Evaluation
- Evaluate the model against human similarity judgments using datasets such as SimLex999, MEN, and MTurk771.
- Perform comparative analysis with:
  1. Multisensory-only model
  2. Text-derived-only model
  3. Original SNN model
  4. Proposed Clarion-based model (this project)