#!/usr/bin/env python
"""
Dream Pipeline — max GPU throughput, no Spark/metrics overhead.
D-MPNN on RTX 4060, batch_size=2048, one call to make_predictions.

Run:
    python dream/dream_pipeline.py 50000
    python dream/dream_pipeline.py 100000
"""

import os, time, warnings, sys
from pathlib import Path

import pandas as pd
import chemprop

warnings.filterwarnings("ignore")
os.chdir(Path(__file__).resolve().parent.parent)


def parse_limit(arg):
    if arg is None: return None
    arg = arg.strip().lower()
    if arg in ("all","full","-1"): return None
    try: return int(arg) if int(arg)>=1 else None
    except ValueError: return None


def main(molecule_limit):
    t_total = time.time()
    print(f"[*] Dream — GPU D-MPNN, batch=2048")

    # Load model
    args = chemprop.args.PredictArgs().parse_args([
        "--test_path","/dev/null","--preds_path","/dev/null",
        "--checkpoint_dir","model/halicin_model","--gpu","0","--batch_size","2048",
    ])
    model_objects = chemprop.train.load_model(args=args)

    # Load molecules
    print("[*] Loading molecules …")
    pdf = pd.read_csv("chembl_36_chemreps.txt", sep="\t",
        usecols=["chembl_id","canonical_smiles"], nrows=molecule_limit
    ).dropna(subset=["canonical_smiles"])
    print(f"[OK] {len(pdf):,} molecules")

    # Predict — one call, GPU, batch=2048
    print(f"[*] GPU inference …")
    t_infer = time.time()
    smiles_list = [[s] for s in pdf["canonical_smiles"].tolist()]
    preds = chemprop.train.make_predictions(
        args=args, model_objects=model_objects, smiles=smiles_list)
    t_infer = time.time() - t_infer
    rate = len(pdf) / t_infer

    scores = [float(p[0]) if isinstance(p,list) and len(p)>0 and isinstance(p[0],(float,int)) else 0.0 for p in preds]
    result = pd.DataFrame({"chembl_id": pdf["chembl_id"].values, "smiles": pdf["canonical_smiles"].values, "Antibiotic_Score": scores})

    os.makedirs("dream", exist_ok=True)
    result.to_parquet("dream/predictions.parquet", index=False)

    hits = (result["Antibiotic_Score"] > 0.5).sum()
    print(f"[OK] {len(pdf):,} mol in {t_infer:.0f}s ({rate:.0f} mol/s) — GPU")
    print(f"[OK] {hits:,} hits > 0.5")
    print(f"\n┌── TOP 5 ──┐")
    for _, r in result.nlargest(5,"Antibiotic_Score").iterrows():
        print(f"  {r['chembl_id']:15s} {r['Antibiotic_Score']:.4f}")
    print(f"└───────────┘")


if __name__ == "__main__":
    main(parse_limit(sys.argv[1] if len(sys.argv)>1 else None))
