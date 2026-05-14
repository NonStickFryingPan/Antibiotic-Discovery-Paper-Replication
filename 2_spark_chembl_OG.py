#!/usr/bin/env python
"""
Phase 3: Spark Distributed Inference — Screen ChEMBL for novel antibiotics.

Run from the pdc_lab conda environment:
    conda activate pdc_lab
    python 2_spark_chembl.py
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

# ── Ensure we run from the project directory ───────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parent
os.chdir(PROJECT_DIR)

# ── Spark needs Java 17 from the conda env ────────────────────────────────

if "CONDA_PREFIX" in os.environ:
    os.environ["JAVA_HOME"] = os.environ["CONDA_PREFIX"]

# ── Config ────────────────────────────────────────────────────────────────

CHUNK_SIZE = 100_000
P_CORE_THREADS = 12
DRIVER_MEMORY = "12g"


def main():
    t_start = time.time()

    # 1. Boot Spark
    spark = (
        SparkSession.builder
        .appName("ChEMBL_Distributed_Antibiotic_Screening")
        .master(f"local[{P_CORE_THREADS}]")
        .config("spark.driver.memory", DRIVER_MEMORY)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    print(
        f"[OK] Spark cluster initialised — "
        f"{P_CORE_THREADS} P-core threads, "
        f"{DRIVER_MEMORY} driver heap."
    )

    # 2. Ingest ChEMBL
    df_chembl = (
        spark.read.csv(
            "chembl_36_chemreps.txt",
            sep="\t",
            header=True
        )
        .select(col("canonical_smiles").alias("smiles"), "chembl_id")
        .dropna()
    )

    if CHUNK_SIZE is not None:
        df_chembl = df_chembl.limit(CHUNK_SIZE)

    df_target = df_chembl.repartition(P_CORE_THREADS)
    df_target.cache()

    row_count = df_target.count()

    print(
        f"[OK] Data ingested: {row_count:,} molecules, "
        f"{P_CORE_THREADS} partitions."
    )

    # 3. Iterator Pandas UDF
    @pandas_udf(FloatType())
    def predict_antibiotic_udf(
        iterator: Iterator[pd.Series]
    ) -> Iterator[pd.Series]:
        """
        Load chemprop model once, then stream batches of SMILES through it.
        """

        args = chemprop.args.PredictArgs().parse_args([
            "--test_path", "/dev/null",
            "--preds_path", "/dev/null",
            "--checkpoint_dir", "halicin_model",
            "--no_cuda",
        ])

        try:
            model_objects = chemprop.train.load_model(args=args)

        except Exception as exc:
            sys.stderr.write(
                f"[FATAL] Worker failed to load model: {exc}\n"
            )

            for smiles_series in iterator:
                yield pd.Series(
                    [float("nan")] * len(smiles_series)
                )

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
                # Valid prediction: list of floats
                # Invalid molecule: string
                if (
                    isinstance(p, list)
                    and len(p) > 0
                    and isinstance(p[0], (float, int))
                ):
                    results.append(float(p[0]))
                else:
                    results.append(0.0)

            yield pd.Series(results)

    # 4. Distributed inference
    print("[*] Running distributed inference on P-core threads …")

    t_infer = time.time()

    results_df = df_target.withColumn(
        "Antibiotic_Score",
        predict_antibiotic_udf(col("smiles"))
    )

    results_df.cache()

    _ = results_df.count()

    t_infer = time.time() - t_infer

    print(
        f"[OK] Inference complete in {t_infer:.1f}s "
        f"({row_count / t_infer:.0f} molecules/s)"
    )

    # 5. Top hits
    print("\n┌── TOP 10 NOVEL ANTIBIOTIC CANDIDATES FROM ChEMBL ──┐")

    top_hits = (
        results_df
        .orderBy(col("Antibiotic_Score").desc())
        .limit(10)
    )

    top_hits.show(truncate=False)

    print("└──────────────────────────────────────────────────────┘")

    # 6. Summary
    total = time.time() - t_start

    hit_count = (
        results_df
        .filter(col("Antibiotic_Score") > 0.5)
        .count()
    )

    print(f"\nMolecules with score > 0.5: {hit_count:,}")
    print(f"Total wall-clock time: {total:.1f}s")

    spark.stop()


if __name__ == "__main__":
    main()
