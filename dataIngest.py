from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
import pandas as pd
from rdkit import Chem

spark = SparkSession.builder \
    .appName("AntibioticDiscovery") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

# Load the training data (sheet S1B has SMILES; skip the title row)
df_train_raw = spark.createDataFrame(pd.read_excel("Table_S1B.xlsx", sheet_name="S1B", skiprows=1))

# PDC Task: Distributed Validation
@pandas_udf("boolean")
def check_smiles_validity(smiles_series: pd.Series) -> pd.Series:
    # RDKit check: can we actually parse this molecule?
    return smiles_series.apply(lambda x: Chem.MolFromSmiles(x) is not None)

df_cleaned = df_train_raw.filter(check_smiles_validity(df_train_raw.SMILES))
df_cleaned.cache() # Store in memory for speed
df_cleaned.select("SMILES", "Activity").toPandas().to_csv("train_clean.csv", index=False)
