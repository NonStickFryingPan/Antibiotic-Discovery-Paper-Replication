#!/usr/bin/env python
"""
Phase 1: Data ingestion + SMILES validation. Run once.
"""
import pandas as pd
from rdkit import Chem

from _init import init_project
from _paths import DATA_DIR, TRAIN_CSV

init_project()

RAW_XLSX = str(DATA_DIR / "Table_S1B.xlsx")
OUT_CSV = str(TRAIN_CSV)

df_raw = pd.read_excel(RAW_XLSX, sheet_name="S1B", skiprows=1)
df = df_raw[["SMILES", "Activity"]].dropna().copy()
df["valid"] = df["SMILES"].apply(lambda s: Chem.MolFromSmiles(str(s)) is not None)
df_clean = df[df["valid"]].drop(columns=["valid"])
df_clean["label"] = df_clean["Activity"].apply(
    lambda x: 1 if str(x).strip().lower() in ("active", "1", "true") else 0
)
df_clean = df_clean[["SMILES", "label"]].rename(columns={"SMILES": "smiles"})
df_clean.to_csv(OUT_CSV, index=False)
print(f"[OK] {len(df_clean)} molecules → {OUT_CSV}")
