<div align="center">

# 🧬 Deep Learning Pipeline for Antibiotic Discovery

**Reproducing the Halicin Paper — distributed screening of ChEMBL using Apache Spark.**

[![Paper](https://img.shields.io/badge/Cell_2020-Stokes_et_al.-blue?style=flat-square)](https://www.cell.com/cell/fulltext/S0092-8674(20)30102-1)
[![Framework](https://img.shields.io/badge/Framework-Chemprop_D--MPNN-orange?style=flat-square)](https://github.com/chemprop/chemprop)
[![Distributed](https://img.shields.io/badge/Distributed-Apache_Spark-E25A1C?style=flat-square)](https://spark.apache.org/)

</div>

---

## Project Overview

This project reproduces the landmark 2020 *Cell* paper, *"A Deep Learning Approach to Antibiotic Discovery"* (Stokes et al.). We train a Directed Message Passing Neural Network (D-MPNN) on 2,335 empirically tested molecules, then distribute it across the ChEMBL database for high-throughput virtual screening.

> ### Basically we screened the entire ChEMBL database on our laptop. (wink wink)

---

## tl;dr

- Replicated Stokes et al. (Cell, 2020) and discovered Halicin (SU3327) from the Drug Repurposing Hub
- Built a Spark pipeline that screens ChEMBL: **1,442 molecules/second** on CPU
- Benchmarked 200K molecules; chunked strategy for full 2.85M
- Model correctly surfaces fluoroquinolones and cephalosporins as top candidates

---

## Data Sources

| Dataset | Molecules | Source |
|---|---|---|
| Training (Stokes et al.) | 2,335 | Cell supplementary Table S1B |
| Drug Repurposing Hub | ~6,800 | Broad Institute (2020 archive) |
| ChEMBL 36 | 2,854,816 | EMBL-EBI |

```bash
# Training data
wget https://www.cell.com/cms/10.1016/j.cell.2020.01.021/attachment/c47fc31e-e3b3-416c-8cc9-6582356c39c0/mmc1.xlsx

# Repurposing Hub
wget https://s3.amazonaws.com/data.clue.io/repurposing/downloads/repurposing_samples_20200324.txt -O repurposing_hub.txt

# ChEMBL
wget https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_36_chemreps.txt.gz
gunzip chembl_36_chemreps.txt.gz
```

---

## Pipeline

### Phase 1: Training (GPU)
```bash
chemprop_train --data_path data/cleaned_training_data.csv \
               --dataset_type classification --epochs 30 --gpu 0 \
               --save_dir model/halicin_model
```
- D-MPNN trained on 2,335 *E. coli* growth inhibition assays
- Validation AUC: ~0.97

### Phase 2: Halicin Replication
```bash
python src/2_replicate_halicin.py
```
- Screens 6,800 FDA-approved/clinical drugs from the Repurposing Hub
- Identifies SU3327 (Halicin), confirming the paper's discovery

### Phase 3: Distributed ChEMBL Screening
```bash
python src/3_distributed_screen.py 100000   # benchmark (69s)
python src/4_chunked_screen.py              # full 2.85M (~39 min)
```

**Architecture:** Spark CSV → repartition → Pandas Iterator UDF (Arrow streaming) → Parquet.

**Why Spark over Dask/Ray:** Spark's Arrow bridge streams 500-row batches incrementally to Python workers. Dask sends entire partitions at once (OOM at 50K). Ray's actor model OOMs at 10K. The 3-4 GB JVM overhead is a fixed cost that buys linear per-worker memory.

**Chunked mode** (`src/4_chunked_screen.py`): splits ChEMBL into 100K-molecule chunks, processes each independently, merges results. Resumes on crash.

---

## Results

| Molecules | Time | Speed | Hits > 0.5 |
|---|---|---|---|
| 10,000 | 7s | 1,428 mol/s | 71 |
| 100,000 | 69s | **1,442 mol/s** | 1,100 |
| 200,000 | 143s | 1,396 mol/s | 1,788 |

> 300K+ hit the WSL2 8 GB memory ceiling (Linux OOM killer).

**Top 5 from 100K screen:**

| Score | ChEMBL ID | Chemotype |
|---|---|---|
| 0.9035 | CHEMBL59066 | Cephalosporin + thiadiazole |
| 0.9034 | CHEMBL60159 | Cephalosporin + triazole |
| 0.8969 | CHEMBL19821 | Fluoroquinolone |
| 0.8954 | CHEMBL36027 | Mitomycin analogue |
| 0.8942 | CHEMBL15579 | Fluoroquinolone |

Top 10 dominated by fluoroquinolones and β-lactams, consistent with known Gram-negative antibiotic chemotypes.

![Histogram of 100K prediction scores showing a large peak near 0 (inactive bulk) and a small tail above 0.5 (predicted actives). Five dashed vertical lines mark the top 5 discovered hits at scores between 0.89-0.90.](results/Score_Distribution.png)

![t-SNE projection of 2,048-bit Morgan fingerprints into 2D. Grey dots: training inactives. Blue dots: training actives. Five large colored markers: top ChEMBL hits occupying distinct regions of chemical space from the training data.](results/Chemical_Space_tSNE.png)

---

## Limitations

**Hardware.** Tests run on WSL2 capped at ~8 GB. Beyond 200K molecules, Spark + Python workers + OS hit the OOM wall. On 16+ GB native Linux, scaling to 2.85M is feasible with the same code.

**Spark in local mode.** Running a distributed engine on one machine incurs JVM overhead (~3-4 GB) with no cluster benefit. A `multiprocessing` + chunked pandas approach would avoid the JVM entirely, at the cost of reimplementing Arrow streaming.

---

## The Fun Stuff: Messing Around With Alternatives

We went down a rabbit hole testing faster alternatives (fingerprint models, multiprocessing, XGBoost, you name it). The lesson: **you can't beat the D-MPNN for actual discovery.**

### The Halicin Test

D-MPNN trained on 2,335 *E. coli* assays scored SU3327 (Halicin) at **0.5627** from the Drug Repurposing Hub. Real generalization to a novel scaffold. We tested whether simpler models could do the same:

| Model | Features | Halicin Score | Found? |
|---|---|---|---|
| **D-MPNN** | Graph neural network | **0.5627** | ✅ |
| XGBoost (distilled) | Morgan fingerprints | 0.0415 | ❌ |
| Random Forest (distilled) | Morgan fingerprints | 0.0203 | ❌ |
| XGBoost (binary labels) | Morgan fingerprints | 0.0952 | ❌ |
| Random Forest (binary labels) | Morgan fingerprints | 0.0401 | ❌ |

Every fingerprint-based model missed Halicin entirely. Morgan bit vectors cannot represent the structural logic that makes Halicin antibiotic-active, no matter what you train on. Only the message-passing neural network captures it.

### Three Approaches, One Winner

| Approach | Speed | D-MPNN Corr | Finds Halicin? |
|---|---|---|---|
| **Spark D-MPNN** | 1,442 mol/s | 1.0 (reference) | ✅ |
| **XGBoost native** (fastest) | **26,856 mol/s** | 0.596 | ❌ |
| RF simple | 5,550 mol/s | 0.523 | ❌ |

**Fastest achieved:** XGBoost native C++ backend at **28,316 mol/s**, screening the full 2.85M ChEMBL in **101 seconds** (10,753 hits). But its D-MPNN correlation is only 0.596, and it scored Halicin at 0.04 vs 0.56. Raw speed means nothing if the model can't discover.

All fast models approximate the rough ranking (cephalosporins and quinolones at the top), but they fail on the structurally novel Halicin. The D-MPNN is the right choice for the final pipeline.

The experiments live in `messing_around/`, a scrapbook of failures worth documenting.

---

## Why Not Faster? (Honest Rant)

We threw everything at this (GPU, Rust, multiprocessing, Dask, Ray, ONNX, bigger batches, you name it) and none of it worked. GPU was slower than CPU because the model is tiny (50K params) and the PCIe bus became the bottleneck before the neural net even woke up. Rust would maybe give us 2x on featurization but then inference becomes the new wall, and that's not worth bolting a foreign language onto the project. Bigger batches? Still waiting on RDKit. Multiprocessing? PyTorch hates forks. Dask OOM'd at 50K. Ray OOM'd at 10K. ONNX can't trace dynamic message-passing graphs. After all that, Spark's lowly Arrow bridge (streams 500-row chunks to Python workers) turned out to be the only thing that kept memory flat and didn't crash. **1,442 mol/s on a $1,000 laptop is the floor.** The GPU sits idle, the CPU does everything, and Spark wins by being boring and reliable. It's not a failure. It's the constraint envelope of a 7.6 GB WSL2 box driving a 50K-parameter network. On a cluster with 16 cores and 64 GB, the same code scales. The pipeline is sound. The hardware was the ceiling. We're tired but this was actually a lot of fun.
