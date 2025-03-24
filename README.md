# Human-like Concept Learning Computational Model

## Project Overview
Developing a computational model for human-like concept learning using the Clarion cognitive-science framework. The model integrates multisensory and text-derived representations of concepts, dynamically combining them to emulate human cognition closely.

## Goal 
The goal is to investigate cognitive processes underlying human concept learning rather than solely optimizing model fit.

## Datasets Used

### Multisensory Representation Datasets
#### 1. LC823 Multisensory Dataset
- **Source**: Lynott & Connell (2009, 2013)
- **Data Format**: 5-dimensional vectors representing sensory intensity across:
  - Auditory
  - Gustatory
  - Haptic
  - Olfactory
  - Visual
- **Total Concepts**: 823 (423 adjectives + 400 nouns)
- **Official Sources:**
  - [Adjective Dataset](https://link.springer.com/article/10.3758/BRM.41.2.558)
  - [Noun Dataset](https://link.springer.com/article/10.3758/s13428-012-0267-0)

#### 2. BBSR Dataset
- **Source**: Binder et al. (2016)
- **Data Format**: 65-dimensional vectors covering perceptual, motor, spatial, temporal, emotional, social, and cognitive modalities
- **Total Concepts**: 535
- **Official Sources:**
  - 

### Text-derived Representation Datasets
#### GloVe Dataset
- **Source:** [GloVe embeddings](https://nlp.stanford.edu/projects/glove/)
- **Format**: 300-dimensional word embeddings (glove-wiki-gigaword-300)

### Word2Vec Dataset
- **Source:** [Word2Vec](https://code.google.com/archive/p/word2vec/)
- **Format:** 300-dimensional word embeddings


## Implementation Structure

### Data processing
- **SensoryData**: Multisensory vectors processed to represent concepts as sensory intensities.  
- **TextData**: Text-derived concept representations converted to dense vectors. 

### Generating Concept Representations (SNN Model)
Used [BrainCog](https://github.com/BrainCog-X/Brain-Cog) to build and run the **spiking neural network model**:
- Based on the methodology from Zeng et al. (2023) [DOI](https://doi.org/10.1016/j.patter.2023.100789)
- **Output**: Human-like concept vectors (binary/spike representations) saved in `.npy` or `.csv` for downstream use in Clarion.
@article{Zeng2023,
  doi = {10.1016/j.patter.2023.100789},
  url = {https://doi.org/10.1016/j.patter.2023.100789},
  year = {2023},
  month = jul,
  publisher = {Cell Press},
  pages = {100789},
  author = {Yi Zeng and Dongcheng Zhao and Feifei Zhao and Guobin Shen and Yiting Dong and Enmeng Lu and Qian Zhang and Yinqian Sun and Qian Liang and Yuxuan Zhao and Zhuoya Zhao and Hongjian Fang and Yuwei Wang and Yang Li and Xin Liu and Chengcheng Du and Qingqun Kong and Zizhe Ruan and Weida Bi},
  title = {{BrainCog}: A spiking neural network based,  brain-inspired cognitive intelligence engine for brain-inspired {AI} and brain simulation},
  journal = {Patterns}
}

### Clarion Model
- Concept vectors from BrainCog are loaded into Clarion as **perceptual chunks**.
- Each vector = one concept chunk, processed in Clarion’s **New Association Subsystem (NACS)**.
- These chunks are used in **categorization tasks**, simulating human concept learning and classification.

### Testing and Evaluation
- Evaluate the model against human similarity judgments using datasets such as SimLex999, MEN, and MTurk771.
- Perform comparative analysis with:
  1. Multisensory-only model
  2. Text-derived-only model
  3. Original SNN model
  4. Proposed Clarion-based model (this project)


## Requirements
This project uses two frameworks with **conflicting Python version requirements**.  
To ensure smooth operation, use **separate environments** for each tool.

### BrainCog Environment (Concept Vector Generation)
- **Python ≤ 3.9** 
- Required Packages:
  - `BrainCog`

### pyClarion Environment (Clarion Modeling)
- **Python ≥ 3.12** 
- Required Packages:
  - `pyClarion`
