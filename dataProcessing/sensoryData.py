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

# Save the merged dataset to CSV
merged_data.to_csv("./data/processedData/LC823_Merged.csv", index=False)


def get_sensory_representation(word):
    row = merged_data[merged_data["Concept"].str.lower() == word.lower()]
    if not row.empty:
        return np.array(row.iloc[0, 1:], dtype=np.float32)  
    return np.zeros(5)  
