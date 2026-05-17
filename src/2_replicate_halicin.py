#!/usr/bin/env python
"""
Phase 2: Replicate the Cell paper — discover Halicin from the Drug Repurposing Hub.
"""
import os, sys, subprocess
from pathlib import Path

import pandas as pd

from _init import init_project
from _paths import REPURPOSING_HUB, CHECKPOINT_DIR

init_project()

CLEAN_CSV = "repurposing_clean.csv"
PREDS_CSV = "repurposing_predictions.csv"

df_hub = pd.read_csv(str(REPURPOSING_HUB), sep="\t", skiprows=9, low_memory=False)
df_hub = df_hub[["smiles", "pert_iname", "broad_id"]].dropna()
df_hub["smiles"] = df_hub["smiles"].apply(lambda x: str(x).split("|")[0])
df_hub.to_csv(CLEAN_CSV, index=False)

subprocess.run([
    "chemprop_predict", "--test_path", CLEAN_CSV,
    "--checkpoint_dir", str(CHECKPOINT_DIR),
    "--preds_path", PREDS_CSV, "--gpu", "0",
])

preds = pd.read_csv(PREDS_CSV)
halicin = preds[preds["pert_iname"].str.contains("SU3327", case=False, na=False)]
print("\n--- HALICIN (SU3327) SCORE ---")
print(halicin[["pert_iname", "smiles", "label" if "label" in halicin.columns else preds.columns[-1]]])
