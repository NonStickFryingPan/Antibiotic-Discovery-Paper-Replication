"""Shared D-MPNN prediction UDF — imported by 3 and 4."""
import sys
from typing import Iterator

import chemprop
import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import FloatType

from _config import CHEMPROP_BATCH


def make_predict_cpu_udf():
    @pandas_udf(FloatType())
    def predict_cpu(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
        args = chemprop.args.PredictArgs().parse_args([
            "--test_path", "/dev/null", "--preds_path", "/dev/null",
            "--checkpoint_dir", "model/halicin_model", "--no_cuda",
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
                    args=args, model_objects=model_objects, smiles=smiles_list)
                results = [float(p[0]) if isinstance(p, list) and len(p) > 0 and isinstance(p[0], (float, int)) else 0.0 for p in preds]
            except Exception as exc:
                sys.stderr.write(f"[WARN] {exc}\n")
                results = [0.0] * len(smiles_series)
            yield pd.Series(results)
    return predict_cpu
