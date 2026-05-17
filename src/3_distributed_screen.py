#!/usr/bin/env python
"""
Phase 3: Spark CPU Inference — Scalable ChEMBL Screening.
Per-partition RDD writes: each partition saves results independently.
Driver never accumulates all results — scales to 2.85M.

Run:
    python 2_spark_cpu_final.py 300000
"""

import os
import sys
import time
import glob
import shutil
from pathlib import Path

import chemprop
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import FloatType

PROJECT_DIR = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_DIR)
if "CONDA_PREFIX" in os.environ:
    os.environ["JAVA_HOME"] = os.environ["CONDA_PREFIX"]

# ── Config ────────────────────────────────────────────────────────────────────
P_CORE_THREADS       = 4
DRIVER_MEMORY        = "4g"
CHEMPROP_BATCH       = 100
TARGET_PER_PARTITION = 5_000   # slightly larger — fewer partitions to write

TEMP_PARQUET = "results/_part_parquets"
FINAL_PARQUET = "results/chEMBL_predictions.parquet"


def parse_limit(arg: str):
    if arg is None:
        return None
    arg = arg.strip().lower()
    if arg in ("all", "full", "-1"):
        return None
    try:
        n = int(arg)
        return n if n >= 1 else None
    except ValueError:
        return None


def predict_and_write_partition(partition_iterator):
    """
    RDD mapPartitions function.
    Loads model ONCE, processes all rows in this partition,
    writes results directly to a partition-specific parquet file.
    Returns nothing — results are on disk.
    """
    import os as _os
    pid = _os.getpid()
    partition_idx = None

    # Collect all rows first (RDD partitions are small enough)
    rows = []
    for row in partition_iterator:
        rows.append(row)
        if partition_idx is None:
            partition_idx = row["_pid"]

    if not rows:
        return iter([])

    smiles_list = [[row["smiles"]] for row in rows]

    # Load chemprop model once
    args = chemprop.args.PredictArgs().parse_args([
        "--test_path", "/dev/null",
        "--preds_path", "/dev/null",
        "--checkpoint_dir", "model/halicin_model",
        "--no_cuda",
        "--batch_size", str(CHEMPROP_BATCH),
    ])
    model_objects = chemprop.train.load_model(args=args)

    # Predict
    try:
        preds = chemprop.train.make_predictions(
            args=args, model_objects=model_objects, smiles=smiles_list,
        )
    except Exception:
        preds = [0.0] * len(rows)

    # Extract scores
    scores = []
    if isinstance(preds, list):
        for p in preds:
            if isinstance(p, list) and len(p) > 0 and isinstance(p[0], (float, int)):
                scores.append(float(p[0]))
            else:
                scores.append(0.0)
    else:
        scores = [0.0] * len(rows)

    # Build result table and write to parquet
    result_rows = []
    for row, score in zip(rows, scores):
        result_rows.append({
            "chembl_id": row["chembl_id"],
            "smiles": row["smiles"],
            "Antibiotic_Score": score,
        })

    result_pdf = pd.DataFrame(result_rows)
    out_path = _os.path.join(TEMP_PARQUET, f"part_{pid}_{partition_idx}.parquet")
    _os.makedirs(_os.path.dirname(out_path), exist_ok=True)
    result_pdf.to_parquet(out_path, index=False)

    return iter([])


def main(molecule_limit):
    t_start = time.time()

    spark = (
        SparkSession.builder
        .appName("ChEMBL_CPU_Scalable")
        .master(f"local[{P_CORE_THREADS}]")
        .config("spark.driver.memory", DRIVER_MEMORY)
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.driver.maxResultSize", "1g")
        .config("spark.sql.adaptive.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print(f"[OK] Spark CPU — {P_CORE_THREADS} threads | {DRIVER_MEMORY} heap | RDD writes")

    # ── Ingest ────────────────────────────────────────────────────────────────
    df_chembl = (
        spark.read.csv("chembl_36_chemreps.txt", sep="\t", header=True)
        .select(col("canonical_smiles").alias("smiles"), "chembl_id")
        .dropna()
    )
    if molecule_limit is not None:
        df_chembl = df_chembl.limit(molecule_limit)

    num_partitions = max(16, (molecule_limit or 500_000) // TARGET_PER_PARTITION)
    df_target = df_chembl.repartition(num_partitions)
    row_count = df_target.count()
    per_part = row_count // num_partitions
    print(f"[OK] {row_count:,} mol | {num_partitions} partitions | ~{per_part:,}/partition")

    # ── Clear temp ────────────────────────────────────────────────────────────
    if os.path.exists(TEMP_PARQUET):
        shutil.rmtree(TEMP_PARQUET)
    if os.path.exists(FINAL_PARQUET):
        shutil.rmtree(FINAL_PARQUET)
    os.makedirs(TEMP_PARQUET, exist_ok=True)

    # ── Add partition ID for file naming ──────────────────────────────────────
    from pyspark.sql.functions import spark_partition_id
    df_target = df_target.withColumn("_pid", spark_partition_id())

    # ── RDD-level: each partition writes its own parquet ──────────────────────
    print(f"[*] Per-partition inference + disk writes …")
    t_infer = time.time()

    rdd = df_target.rdd.mapPartitions(predict_and_write_partition)
    rdd.collect()  # force execution

    t_infer = time.time() - t_infer
    rate = row_count / t_infer if t_infer > 0 else 0
    print(f"[OK] Inference: {t_infer:.0f}s ({rate:.0f} mol/s)")

    # ── Merge partition files ─────────────────────────────────────────────────
    print("[*] Merging partition files …")
    import dask.dataframe as dd
    ddf = dd.read_parquet(TEMP_PARQUET + "/*.parquet")
    ddf.to_parquet(FINAL_PARQUET, write_index=False)
    shutil.rmtree(TEMP_PARQUET)
    print(f"[OK] Merged → {FINAL_PARQUET}")

    # ── Results ───────────────────────────────────────────────────────────────
    result_pdf = pd.read_parquet(FINAL_PARQUET)
    hit_count = (result_pdf["Antibiotic_Score"] > 0.5).sum()
    print(f"[OK] {hit_count:,} hits > 0.5 ({100 * hit_count / len(result_pdf):.1f}%)")

    print("\n┌── TOP 10 ──┐")
    top = result_pdf.nlargest(10, "Antibiotic_Score")[
        ["chembl_id", "smiles", "Antibiotic_Score"]
    ]
    for _, row in top.iterrows():
        print(f"  {row['chembl_id']:15s} {row['Antibiotic_Score']:.4f}  {row['smiles'][:60]}…")
    print("└───────────┘")

    total = time.time() - t_start
    print(f"[OK] Total: {total:.0f}s | {len(result_pdf):,} molecules")
    spark.stop()


if __name__ == "__main__":
    limit_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(parse_limit(limit_arg))
