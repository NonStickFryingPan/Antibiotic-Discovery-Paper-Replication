#!/usr/bin/env python
"""
Phase 3 (chunked): processes ChEMBL in 100K-molecule batches, merges results.
Each batch uses the proven Spark CPU pipeline. Resumes on crash.

Run:
    python src/4_chunked_screen.py
"""
import os, sys, time, glob, shutil
from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import FloatType

from _init import init_project
from _config import P_CORE_THREADS, DRIVER_MEMORY, ARROW_BATCH, CHEMPROP_BATCH
from _predict import make_predict_cpu_udf

init_project()

CHUNKS_DIR = "chunks"
TEMP_PARQUETS = "results/_chunk_parquets"
FINAL_PARQUET = "results/chEMBL_predictions_full.parquet"


def process_chunk(spark, chunk_path, output_dir):
    df = spark.read.csv(chunk_path, sep="\t", header=True) \
        .select(col("canonical_smiles").alias("smiles"), "chembl_id").dropna()
    predict_cpu = make_predict_cpu_udf()
    result = df.withColumn("Antibiotic_Score", predict_cpu(col("smiles")))
    out_path = os.path.join(output_dir, os.path.basename(chunk_path) + ".parquet")
    result.write.mode("overwrite").parquet(out_path)
    return df.count()


def main():
    t_start = time.time()
    chunk_files = sorted(glob.glob(os.path.join(CHUNKS_DIR, "data_chunk_*.csv")))
    if not chunk_files:
        print("[FATAL] No chunks found in chunks/"); sys.exit(1)

    completed = set()
    if os.path.exists(TEMP_PARQUETS):
        for f in glob.glob(os.path.join(TEMP_PARQUETS, "*.parquet")):
            key = os.path.basename(f).replace(".parquet", "")
            try:
                if pd.read_parquet(f).shape[0] > 0:
                    completed.add(key)
                else:
                    os.remove(f)
            except Exception:
                os.remove(f)

    remaining = [c for c in chunk_files if os.path.basename(c) not in completed]
    print(f"[OK] {len(chunk_files)} chunks, {len(completed)} done, {len(remaining)} remaining")
    os.makedirs(TEMP_PARQUETS, exist_ok=True)
    if os.path.exists(FINAL_PARQUET):
        os.remove(FINAL_PARQUET) if os.path.isfile(FINAL_PARQUET) else shutil.rmtree(FINAL_PARQUET)

    spark = SparkSession.builder.appName("ChEMBL_Chunked") \
        .master(f"local[{P_CORE_THREADS}]") \
        .config("spark.driver.memory", DRIVER_MEMORY) \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", str(ARROW_BATCH)) \
        .config("spark.driver.maxResultSize", "1g") \
        .config("spark.sql.adaptive.enabled", "false") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    print(f"[OK] Spark — {P_CORE_THREADS} threads, {DRIVER_MEMORY}")

    total = 0
    for i, ch in enumerate(remaining):
        t0 = time.time()
        n = process_chunk(spark, ch, TEMP_PARQUETS)
        total += n
        spark.catalog.clearCache()
        print(f"  Chunk {i+1}/{len(remaining)}: {n:,} mol, {time.time()-t0:.0f}s (total {total:,})")

    spark.stop()

    parts = sorted(glob.glob(os.path.join(TEMP_PARQUETS, "*.parquet")))
    combined = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    combined.to_parquet(FINAL_PARQUET, index=False)
    shutil.rmtree(TEMP_PARQUETS)

    hits = (combined["Antibiotic_Score"] > 0.5).sum()
    print(f"[OK] {hits:,} hits > 0.5")
    print("\n┌── TOP 10 ──┐")
    for _, r in combined.nlargest(10, "Antibiotic_Score").iterrows():
        print(f"  {r['chembl_id']:15s} {r['Antibiotic_Score']:.4f}")
    print("└───────────┘")
    print(f"[OK] Total: {time.time()-t_start:.0f}s | {FINAL_PARQUET}")


if __name__ == "__main__":
    main()
