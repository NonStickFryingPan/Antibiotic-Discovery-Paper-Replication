import pandas as pd

# 1. Load the CSV (skip the trailing empty column with usecols)
df = pd.read_csv("Table_S1B.csv", usecols=["Mean_Inhibition", "SMILES", "Name", "Activity"])

# 2. Binarize the Label: "Active" -> 1, everything else -> 0
# Rename columns to match Chemprop expectations
df_cleaned = df[df["SMILES"].notna()].copy()
df_cleaned["label"] = (df_cleaned["Activity"].str.strip() == "Active").astype(int)
df_cleaned = df_cleaned.rename(columns={"SMILES": "smiles"})[["smiles", "label"]]

# 3. Show stats (Important for your report: prove you have a balanced or imbalanced set)
print(f"Total molecules: {len(df_cleaned)}")
print(df_cleaned["label"].value_counts().to_string())

# 4. Save to a single CSV for Chemprop training
df_cleaned.to_csv("cleaned_training_data.csv", index=False)
print("\nSaved cleaned_training_data.csv")
