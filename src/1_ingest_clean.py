#!/usr/bin/env python
"""
Phase 1: Data Ingestion & Cleaning.
Reads the Stokes et al. training data, validates SMILES with RDKit,
and outputs cleaned_training_data.csv.

Run once:
    python src/1_ingest_clean.py
"""

import os
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem

PROJECT_DIR = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_DIR)

# ── Ingest ────────────────────────────────────────────────────────────────────
df_raw = pd.read_excel("data/Table_S1B.xlsx", sheet_name="S1B", skiprows=1)
print(f"[OK] Loaded {len(df_raw)} raw molecules")

# ── Keep only SMILES + Activity ──────────────────────────────────────────────
df = df_raw[["SMILES", "Activity"]].dropna().copy()

# ── Validate SMILES with RDKit ───────────────────────────────────────────────
df["valid"] = df["SMILES"].apply(lambda s: Chem.MolFromSmiles(str(s)) is not None)
df_clean = df[df["valid"]].drop(columns=["valid"])
print(f"[OK] {len(df_clean)} valid SMILES ({len(df) - len(df_clean)} dropped)")

# ── Binarize Activity ────────────────────────────────────────────────────────
df_clean["label"] = df_clean["Activity"].apply(
    lambda x: 1 if str(x).strip().lower() in ("active", "1", "true") else 0
)
df_clean = df_clean[["SMILES", "label"]].rename(columns={"SMILES": "smiles"})

# ── Save ─────────────────────────────────────────────────────────────────────
df_clean.to_csv("data/cleaned_training_data.csv", index=False)
print(f"[OK] Saved data/cleaned_training_data.csv ({len(df_clean)} molecules)")
