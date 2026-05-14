#!/usr/bin/env python
"""
Phase 3: Spark Distributed Inference — Screen ChEMBL for novel antibiotics.

Run from the pdc_lab conda environment:
    conda activate pdc_lab
    python 2_spark_chembl.py              # default: 10,000 molecules
    python 2_spark_chembl.py 100000       # 100K molecules
    python 2_spark_chembl.py all          # full 2.85M dataset
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
    """Parse user-supplied chunk size: a number or 'all' for the full dataset."""
    if arg is None:
        return 100_000
    arg = arg.strip().lower()
    if arg in ("all", "full", "-1"):
        return None
    try:
        n = int(arg)
        if n < 1:
            print(f"[WARN] Invalid size '{arg}', using default 10,000.")
            return 10_000
        return n
    except ValueError:
        print(f"[WARN] Could not parse '{arg}', using default 10,000.")
        return 10_000


def main(chunk_size):
    t_start = time.time()
    print(f"[INFO] Chunk size: {chunk_size or 'FULL DATASET (2.85M)'}, "
          f"partitions: {PARTITIONS}, chemprop batch: {CHEMPROP_BATCH}")

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
    print(f"[OK] Spark cluster initialised — {P_CORE_THREADS} P-core threads, "
          f"{DRIVER_MEMORY} driver / {EXECUTOR_MEMORY} executor heap.")

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
    print(f"[OK] Data ingested: {row_count:,} molecules "
          f"(~{per_partition:,}/partition across {PARTITIONS} partitions).")

    # 3. Iterator Pandas UDF — one model load per executor, OOM-safe ───────────
    @pandas_udf(FloatType())
    def predict_antibiotic_udf(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
        """Load chemprop model once, then stream SMILES through it."""

        # --no_cuda keeps inference on CPU P-cores (no GPU contention)
        # --batch_size limits molecules per forward pass to avoid OOM
        args = chemprop.args.PredictArgs().parse_args([
            "--test_path", "/dev/null",
            "--preds_path", "/dev/null",
            "--checkpoint_dir", "halicin_model",
            "--no_cuda",
            "--batch_size", str(CHEMPROP_BATCH),
        ])

        try:
            model_objects = chemprop.train.load_model(args=args)
        except Exception as exc:
            sys.stderr.write(f"[FATAL] Worker failed to load model: {exc}\n")
            for smiles_series in iterator:
                yield pd.Series([float("nan")] * len(smiles_series))
            return

        for smiles_series in iterator:
            smiles_list = [[s] for s in smiles_series.tolist()]

            preds = chemprop.train.make_predictions(
                args=args,
                model_objects=model_objects,
                smiles=smiles_list,
            )

            results = []
            for p in preds:
                # Valid prediction: list of floats.  Invalid molecule: string.
                if isinstance(p, list) and len(p) > 0 and isinstance(p[0], (float, int)):
                    results.append(float(p[0]))
                else:
                    results.append(0.0)

            yield pd.Series(results)

    # 4. Distributed inference ─────────────────────────────────────────────────
    print("[*] Running distributed inference on P-core threads …")
    t_infer = time.time()

    results_df = df_target.withColumn(
        "Antibiotic_Score", predict_antibiotic_udf(col("smiles"))
    )
    results_df.cache()
    _ = results_df.count()  # materialise the cached DataFrame

    t_infer = time.time() - t_infer
    print(f"[OK] Inference complete in {t_infer:.1f}s "
          f"({row_count / t_infer:.0f} molecules/s)")

    # 5. Top hits ──────────────────────────────────────────────────────────────
    print("\n┌── TOP 10 NOVEL ANTIBIOTIC CANDIDATES FROM ChEMBL ──┐")
    top_hits = (
        results_df
        .orderBy(col("Antibiotic_Score").desc())
        .limit(10)
    )
    top_hits.show(truncate=False)
    print("└──────────────────────────────────────────────────────┘")

    # 6. Summary ───────────────────────────────────────────────────────────────
    total = time.time() - t_start
    hit_count = results_df.filter(col("Antibiotic_Score") > 0.5).count()
    print(f"\nMolecules with score > 0.5: {hit_count:,}")
    print(f"Total wall-clock time: {total:.1f}s")

    spark.stop()


if __name__ == "__main__":
    chunk_arg = sys.argv[1] if len(sys.argv) > 1 else None
    chunk_size = parse_chunk_size(chunk_arg)
    main(chunk_size)
