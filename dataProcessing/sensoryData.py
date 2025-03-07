import pandas as pd

# Load both Excel datasets
noun_data = pd.read_excel("./conceptLearningModel/data/rawData/lynott_connell_2013.xls")  # 400 nouns
adjective_data = pd.read_excel("./conceptLearningModel/data/rawData/lynott_connell_2009.xls")  # 423 adjectives

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
merged_data.to_csv("./conceptLearningModel/data/processedData/LC823_Merged.csv", index=False)

print("LC823 dataset successfully merged and saved as 'LC823_Merged.csv'.")
