"""
This script merges two key multisensory datasets from Lynott & Connell (2009, 2013)—
one containing adjectives and the other containing nouns. Each word is represented
by average perceptual strength ratings across five sensory modalities:
Auditory, Gustatory, Haptic, Olfactory, and Visual.

References:
- Lynott, D., & Connell, L. (2009). Modality exclusivity norms for 423 adjectives.
- Lynott, D., & Connell, L. (2013). Modality exclusivity norms for 400 nouns.
"""

import pandas as pd
import numpy as np

# Load both Excel datasets
noun_data = pd.read_excel("./data/rawData/lynott_connell_2013.xls")  # 400 nouns
adjective_data = pd.read_excel("./data/rawData/lynott_connell_2009.xls")  # 423 adjectives

# Standardize the column names for nouns dataset
noun_data = noun_data.rename(columns={
    "Noun": "Concept",
    "Auditory_mean": "Auditory",
    "Gustatory_mean": "Gustatory",
    "Haptic_mean": "Haptic",
    "Olfactory_mean": "Olfactory",
    "Visual_mean": "Visual"
})[["Concept", "Auditory", "Gustatory", "Haptic", "Olfactory", "Visual"]]  

# Standardize the column names for adjectives dataset
adjective_data = adjective_data.rename(columns={
    "Property": "Concept",
    "AuditoryStrengthMean": "Auditory",
    "GustatoryStrengthMean": "Gustatory",
    "HapticStrengthMean": "Haptic",
    "OlfactoryStrengthMean": "Olfactory",
    "VisualStrengthMean": "Visual"
})[["Concept", "Auditory", "Gustatory", "Haptic", "Olfactory", "Visual"]]  

# Merge the two datasets
merged_data = pd.concat([noun_data, adjective_data], ignore_index=True)

merged_data = merged_data.drop_duplicates(subset="Concept", keep="first")

# Save the merged dataset to CSV
merged_data.to_excel("./data/processedData/LC823_Merged.xlsx", index=True)


def get_sensory_representation(word):
    """
    Retrieves the sensory representation of a given word from the merged data.

    Args:
        word (str): The word for which the sensory representation is to be retrieved.

    Returns:
        numpy.ndarray: A NumPy array containing the sensory representation of the word, or
        an array of zeros if the word is not found.
    """
    row = merged_data[merged_data["Concept"].str.lower() == word.lower()]
    if not row.empty:
        return np.array(row.iloc[0, 1:], dtype=np.float32)  
    return np.zeros(5)  
