# Human-like Concept Learning Computational Model
A cognitive architecture that models human-like concept learning using the Clarion framework, enhanced with human-like binary concept vectors derived from a spiking neural network (SNN). This approach integrates multisensory features with brain-inspired representations, simulating internal processes of similarity judgment and concept recall

## Goal 
The goal is to model and understand human concept learning mechanisms, using cognitively grounded processes rather than purely optimizing statistical fit

## Datasets 

### Multisensory Representation 
1. **LC823 Dataset** (Lynott & Connell, 2009, 2013)  
   - 5D perceptual vectors across:  
     `auditory`, `gustatory`, `haptic`, `olfactory`, `visual`  
   - [Adjective Dataset](https://link.springer.com/article/10.3758/BRM.41.2.558)  
   - [Noun Dataset](https://link.springer.com/article/10.3758/s13428-012-0267-0)

### Human-like Binary Representations
Derived using a **spiking neural network** (SNN) trained in **BrainCog** (https://github.com/BrainCog-X/Brain-Cog), based on the methodology in: 
**Wang, Y.**, Zeng, Y., et al. (2023).   *A Brain-inspired Computational Model for Human-like Concept Learning* DOI: [10.1016/j.patter.2023.100789](https://doi.org/10.1016/j.patter.2023.100789)

- Output: Binary vectors (`2500` bits) for each concept

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

## Evaluation Tasks

Evaluated on the **MTurk771 dataset**, where each concept pair has a human similarity score.  
The model:
- Inputs one concept from each pair
- Computes similarity to others using Clarion’s bottom-up spreading
- Ranks all stored concepts
- Outputs a ranked list to be compared with human-annotated rankings

### Comparative Evaluation
We compare:
1. **Clarion without SNN binary features**
2. **Clarion with SNN-derived binary features**
3. **Original human MTurk771 scores**


## Environment Setup

| Tool        | Purpose                        | Python Version | Installation              |
|-------------|--------------------------------|----------------|---------------------------|
| **BrainCog**| Generate SNN concept vectors   | ≤ 3.9          | `pip install braincog`    |
| **pyClarion**| Build and run Clarion model    | ≥ 3.12         | `pip install pyclarion`   |

Use separate environments for each tool

## Running the Simulation

### 1. Prerequisites
Download the following files:
- `LC823_Merged.xlsx` and `AM_binarycode.pkl` → `data/processedData/`
- `EN-MTurk-771.txt` → `data/rawData/`

#### Required Code Files:
- `model_input.py` – contains the `load_concept_data()` function
- `model.py` – the main simulation script  

### 2. Run the Model
```bash
cd src/
python model.py
```