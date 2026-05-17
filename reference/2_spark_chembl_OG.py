#!/usr/bin/env python
"""
Phase 3: Spark Distributed Inference — Screen ChEMBL for novel antibiotics.
**ORIGINAL WORKING VERSION** — successfully benchmarked 100K molecules.

Run from the pdc_lab conda environment:
    conda activate pdc_lab
    python 2_spark_chembl_OG.py              # default: 100K molecules
    python 2_spark_chembl_OG.py 50000        # custom chunk
"""

import os
import sys
import time
from pathlib import Path
from typing import Iterator

import chemprop
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import FloatType

# ── Ensure we run from the project directory (model path is relative) ─────────
PROJECT_DIR = Path(__file__).resolve().parent
os.chdir(PROJECT_DIR)

# ── Spark needs Java 17 from the conda env ────────────────────────────────────
if "CONDA_PREFIX" in os.environ:
    os.environ["JAVA_HOME"] = os.environ["CONDA_PREFIX"]

# ── Config ────────────────────────────────────────────────────────────────────
P_CORE_THREADS = 4            # WSL — limited to 4 threads
PARTITIONS = 16               # 4× threads — smaller per-partition batches
DRIVER_MEMORY = "4g"          # safe for 7.6 Gi system RAM
EXECUTOR_MEMORY = "512m"      # per-executor JVM heap (Python workers use system RAM)
ARROW_BATCH = 500             # max rows per Arrow batch (avoids giant transfers)
CHEMPROP_BATCH = 100          # molecules processed at once within each partition


def parse_chunk_size(arg: str):
    """Parse user-supplied chunk size."""
    if arg is None:
        return 100_000
    arg = arg.strip().lower()
    try:
        n = int(arg)
        return n if n >= 1 else 100_000
    except ValueError:
        return 100_000


def main(chunk_size):
    t_start = time.time()
    print(f"[INFO] Chunk: {chunk_size:,}, "
          f"partitions: {PARTITIONS}, batch: {CHEMPROP_BATCH}")

    # 1. Boot Spark — P-core-optimised local cluster ───────────────────────────
    spark = (
        SparkSession.builder
        .appName("ChEMBL_Distributed_Antibiotic_Screening")
        .master(f"local[{P_CORE_THREADS}]")
        .config("spark.driver.memory", DRIVER_MEMORY)
        .config("spark.executor.memory", EXECUTOR_MEMORY)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", str(ARROW_BATCH))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print(f"[OK] Spark ready — {P_CORE_THREADS} threads, "
          f"{DRIVER_MEMORY} driver / {EXECUTOR_MEMORY} executor.")

    # 2. Ingest ChEMBL (tab-separated, 2.85M molecules) ────────────────────────
    df_chembl = (
        spark.read.csv("chembl_36_chemreps.txt", sep="\t", header=True)
        .select(col("canonical_smiles").alias("smiles"), "chembl_id")
        .dropna()
    )
    if chunk_size is not None:
        df_chembl = df_chembl.limit(chunk_size)

    df_target = df_chembl.repartition(PARTITIONS)
    df_target.cache()
    row_count = df_target.count()
    per_partition = row_count // PARTITIONS
    print(f"[OK] Ingested {row_count:,} molecules "
          f"(~{per_partition:,}/partition).")

    # 3. Iterator Pandas UDF — one model load per executor, OOM-safe ───────────
    @pandas_udf(FloatType())
    def predict_antibiotic_udf(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
        args = chemprop.args.PredictArgs().parse_args([
            "--test_path", "/dev/null", "--preds_path", "/dev/null",
            "--checkpoint_dir", "halicin_model", "--no_cuda",
            "--batch_size", str(CHEMPROP_BATCH),
        ])
        try:
            model_objects = chemprop.train.load_model(args=args)
        except Exception as exc:
            sys.stderr.write(f"[FATAL] Model load failed: {exc}\n")
            for smiles_series in iterator:
                yield pd.Series([float("nan")] * len(smiles_series))
            return

        for smiles_series in iterator:
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
            yield pd.Series(results)

    # 4. Distributed inference ─────────────────────────────────────────────────
    print("[*] Running distributed inference …")
    t_infer = time.time()
    results_df = df_target.withColumn(
        "Antibiotic_Score", predict_antibiotic_udf(col("smiles"))
    )
    results_df.cache()
    _ = results_df.count()
    t_infer = time.time() - t_infer
    print(f"[OK] Inference: {t_infer:.1f}s ({row_count / t_infer:.0f} mol/s)")

    # 5. Top hits ──────────────────────────────────────────────────────────────
    print("\n┌── TOP 10 NOVEL ANTIBIOTIC CANDIDATES FROM ChEMBL ──┐")
    results_df.orderBy(col("Antibiotic_Score").desc()).limit(10).show(truncate=False)
    print("└──────────────────────────────────────────────────────┘")

    # 6. Summary ───────────────────────────────────────────────────────────────
    total = time.time() - t_start
    hit_count = results_df.filter(col("Antibiotic_Score") > 0.5).count()
    print(f"\nMolecules with score > 0.5: {hit_count:,}")
    print(f"Total wall-clock time: {total:.1f}s")
    spark.stop()


if __name__ == "__main__":
    chunk_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(parse_chunk_size(chunk_arg))
