#!/usr/bin/env python
"""
Phase 3: Chunked ChEMBL Screening — processes 100K-molecule chunks sequentially.
Each chunk uses the proven 4-thread Spark CPU pipeline.
Combines all results into final parquet.

Run:
    python 2_spark_chunked.py
"""

import os
import sys
import time
import glob
import shutil
from pathlib import Path
from typing import Iterator

import chemprop
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import FloatType

PROJECT_DIR = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_DIR)
if "CONDA_PREFIX" in os.environ:
    os.environ["JAVA_HOME"] = os.environ["CONDA_PREFIX"]

# ── Proven config (100K at 1,442 mol/s) ──────────────────────────────────────
P_CORE_THREADS       = 4
DRIVER_MEMORY        = "4g"
ARROW_BATCH          = 500
CHEMPROP_BATCH       = 100

CHUNKS_DIR    = "chunks"
TEMP_PARQUETS = "results/_chunk_parquets"
FINAL_PARQUET = "results/chEMBL_predictions_full.parquet"


def process_chunk(spark, chunk_path: str, output_dir: str):
    """Process one 100K chunk with the proven pipeline."""
    df = (
        spark.read.csv(chunk_path, sep="\t", header=True)
        .select(col("canonical_smiles").alias("smiles"), "chembl_id")
        .dropna()
    )
    row_count = df.count()

    @pandas_udf(FloatType())
    def predict_cpu(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
        args = chemprop.args.PredictArgs().parse_args([
            "--test_path", "/dev/null",
            "--preds_path", "/dev/null",
            "--checkpoint_dir", "model/halicin_model",
            "--no_cuda",
            "--batch_size", str(CHEMPROP_BATCH),
        ])
        try:
            model_objects = chemprop.train.load_model(args=args)
        except Exception as exc:
            sys.stderr.write(f"[FATAL] {exc}\n")
            for s in iterator:
                yield pd.Series([float("nan")] * len(s))
            return

        for smiles_series in iterator:
            try:
                smiles_list = [[s] for s in smiles_series.tolist()]
                preds = chemprop.train.make_predictions(
                    args=args, model_objects=model_objects, smiles=smiles_list,
                )
                results = []
                for p in preds:
                    if isinstance(p, list) and len(p) > 0 and isinstance(p[0], (float, int)):
                        results.append(float(p[0]))
                    else:
                        results.append(0.0)
            except Exception as exc:
                sys.stderr.write(f"[WARN] {exc}\n")
                results = [0.0] * len(smiles_series)
            yield pd.Series(results)

    result = df.withColumn("Antibiotic_Score", predict_cpu(col("smiles")))
    out_path = os.path.join(output_dir, f"{os.path.basename(chunk_path)}.parquet")
    result.write.mode("overwrite").parquet(out_path)
    return row_count


def main():
    t_start = time.time()

    # Find all chunk files
    chunk_files = sorted(glob.glob(os.path.join(CHUNKS_DIR, "data_chunk_*.csv")))
    if not chunk_files:
        print("[FATAL] No chunk files found in chunks/")
        sys.exit(1)

    print(f"[OK] Found {len(chunk_files)} chunks")

    # Check for already-completed chunks (resume support)
    completed = set()
    if os.path.exists(TEMP_PARQUETS):
        for f in glob.glob(os.path.join(TEMP_PARQUETS, "*.parquet")):
            chunk_key = os.path.basename(f).replace(".parquet", "")
            # Validate: parquet must have rows (not a partial/corrupt write)
            try:
                n = pd.read_parquet(f).shape[0]
                if n > 0:
                    completed.add(chunk_key)
                else:
                    os.remove(f)  # empty — reprocess
                    print(f"[WARN] Removing empty chunk: {chunk_key}")
            except Exception:
                os.remove(f)  # corrupt — reprocess
                print(f"[WARN] Removing corrupt chunk: {chunk_key}")
    
    if completed:
        print(f"[OK] Resuming — {len(completed)} chunks already completed")

    remaining = [c for c in chunk_files if os.path.basename(c) not in completed]
    if not remaining:
        print("[OK] All chunks already processed. Merging…")
    else:
        print(f"[*] {len(remaining)} chunks remaining")

    os.makedirs(TEMP_PARQUETS, exist_ok=True)

    # Always clear the final merged output (will be rebuilt)
    if os.path.exists(FINAL_PARQUET):
        if os.path.isdir(FINAL_PARQUET):
            shutil.rmtree(FINAL_PARQUET)
        else:
            os.remove(FINAL_PARQUET)

    # Create Spark session (reuse across chunks to avoid JVM startup overhead)
    spark = (
        SparkSession.builder
        .appName("ChEMBL_Chunked")
        .master(f"local[{P_CORE_THREADS}]")
        .config("spark.driver.memory", DRIVER_MEMORY)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", str(ARROW_BATCH))
        .config("spark.driver.maxResultSize", "1g")
        .config("spark.sql.adaptive.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print(f"[OK] Spark — {P_CORE_THREADS} threads, {DRIVER_MEMORY} heap\n")

    # Count already-completed rows
    total_processed = 0
    for c in completed:
        tmp_path = os.path.join(TEMP_PARQUETS, c + ".parquet")
        try:
            total_processed += pd.read_parquet(tmp_path).shape[0]
        except Exception:
            pass

    for i, chunk_path in enumerate(remaining):
        t_chunk = time.time()
        row_count = process_chunk(spark, chunk_path, TEMP_PARQUETS)
        total_processed += row_count
        elapsed = time.time() - t_chunk
        rate = row_count / elapsed if elapsed > 0 else 0

        # Clear cache between chunks
        spark.catalog.clearCache()

        overall_rate = total_processed / (time.time() - t_start) if (time.time() - t_start) > 0 else 0
        remaining = total_processed  # just for display
        print(f"  Chunk {i+1:2d}/{len(chunk_files)}: {row_count:,} mol | "
              f"{elapsed:.0f}s ({rate:.0f} mol/s) | "
              f"total {total_processed:,} | overall {overall_rate:.0f} mol/s")

    spark.stop()

    # ── Merge all chunk parquets ──────────────────────────────────────────────
    print(f"\n[*] Merging {len(chunk_files)} chunk parquets …")
    t_merge = time.time()

    all_parts = sorted(glob.glob(os.path.join(TEMP_PARQUETS, "*.parquet")))
    combined = pd.concat([pd.read_parquet(p) for p in all_parts], ignore_index=True)
    os.makedirs(os.path.dirname(FINAL_PARQUET), exist_ok=True)
    combined.to_parquet(FINAL_PARQUET, index=False)

    print(f"[OK] Merged in {time.time() - t_merge:.0f}s → {FINAL_PARQUET}")

    # ── Results ───────────────────────────────────────────────────────────────
    hit_count = (combined["Antibiotic_Score"] > 0.5).sum()
    print(f"[OK] {hit_count:,} hits > 0.5 ({100 * hit_count / len(combined):.1f}%)")

    print("\n┌── TOP 10 ──┐")
    top = combined.nlargest(10, "Antibiotic_Score")[
        ["chembl_id", "smiles", "Antibiotic_Score"]
    ]
    for _, row in top.iterrows():
        print(f"  {row['chembl_id']:15s} {row['Antibiotic_Score']:.4f}  {row['smiles'][:60]}…")
    print("└───────────┘")

    total = time.time() - t_start
    print(f"\n[OK] Total: {total:.0f}s | {len(combined):,} molecules")
    shutil.rmtree(TEMP_PARQUETS)


if __name__ == "__main__":
    main()
