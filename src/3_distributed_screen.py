#!/usr/bin/env python
"""
Phase 3: Spark CPU Inference — distributed ChEMBL screening.
Spark CSV → repartition → Arrow-streamed UDF → Parquet.
"""
import os, sys, time
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from _init import init_project
from _config import P_CORE_THREADS, DRIVER_MEMORY, ARROW_BATCH, TARGET_PER_PARTITION
from _paths import CHEMBL_CSV
from _predict import make_predict_cpu_udf

init_project()

RESULTS_PATH = "results/chEMBL_predictions.parquet"


def parse_limit(arg):
    if arg is None: return None
    arg = arg.strip().lower()
    if arg in ("all","full","-1"): return None
    try: n = int(arg); return n if n >= 1 else None
    except ValueError: return None


def main(molecule_limit):
    t_start = time.time()

    spark = SparkSession.builder.appName("ChEMBL_CPU") \
        .master(f"local[{P_CORE_THREADS}]") \
        .config("spark.driver.memory", DRIVER_MEMORY) \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", str(ARROW_BATCH)) \
        .config("spark.driver.maxResultSize", "2g") \
        .config("spark.sql.adaptive.enabled", "false") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    print(f"[OK] Spark — {P_CORE_THREADS} threads, {DRIVER_MEMORY}")

    df = spark.read.csv(str(CHEMBL_CSV), sep="\t", header=True) \
        .select(col("canonical_smiles").alias("smiles"), "chembl_id").dropna()
    if molecule_limit is not None:
        df = df.limit(molecule_limit)

    nparts = max(16, (molecule_limit or 500_000) // TARGET_PER_PARTITION)
    df = df.repartition(nparts)
    row_count = df.count()
    print(f"[OK] {row_count:,} mol, {nparts} partitions")

    predict_cpu = make_predict_cpu_udf()
    t_infer = time.time()
    df.withColumn("Antibiotic_Score", predict_cpu(col("smiles"))) \
      .write.mode("overwrite").parquet(RESULTS_PATH)
    t_infer = time.time() - t_infer
    print(f"[OK] {row_count:,} mol in {t_infer:.0f}s ({row_count/t_infer:.0f} mol/s)")

    r = spark.read.parquet(RESULTS_PATH)
    hits = r.filter(col("Antibiotic_Score") > 0.5).count()
    print(f"[OK] {hits:,} hits > 0.5")
    print("\n┌── TOP 10 ──┐")
    r.orderBy(col("Antibiotic_Score").desc()).limit(10).show(truncate=False)
    print("└───────────┘")
    print(f"[OK] Total: {time.time()-t_start:.0f}s")
    spark.stop()


if __name__ == "__main__":
    main(parse_limit(sys.argv[1] if len(sys.argv) > 1 else None))
