import pandas as pd
import subprocess

print("1. Cleaning Drug Repurposing Hub...")
# Load the Broad Institute data, skipping the 9 metadata lines
df_hub = pd.read_csv("repurposing_hub.txt", sep="\t", skiprows=9, low_memory=False)

# Keep only what we need and drop missing SMILES
df_hub = df_hub[['smiles', 'pert_iname', 'broad_id']].dropna()

# Some molecules have multiple SMILES separated by '|'. We take the first one.
df_hub['smiles'] = df_hub['smiles'].apply(lambda x: str(x).split('|')[0])

# Save it to a clean CSV for Chemprop
df_hub.to_csv("repurposing_clean.csv", index=False)
print(f"Saved {len(df_hub)} molecules for inference.")

print("2. Running Neural Network Inference on the GPU...")
# Run chemprop using the CLI
subprocess.run([
    "chemprop_predict",
    "--test_path", "repurposing_clean.csv",
    "--checkpoint_dir", "halicin_model",
    "--preds_path", "repurposing_predictions.csv",
    "--gpu", "0"
])

print("\n3. Searching for Halicin (SU3327)...")
# Load the predictions
preds = pd.read_csv("repurposing_predictions.csv")

# Find SU3327 (Halicin)
halicin = preds[preds['pert_iname'].str.contains('SU3327', case=False, na=False)]

print("\n--- RESULTS ---")
print(halicin[['pert_iname', 'smiles', 'label']])
print("----------------")
if float(halicin['label'].iloc[0]) > 0.5:
    print("FUCKING BRILLIANT! The model successfully predicted Halicin as an antibiotic!")
else:
    print("Score is low - check the training data balance.")
