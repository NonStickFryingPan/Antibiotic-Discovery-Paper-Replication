"""Project paths as Path objects. Single source of truth for data locations."""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "model"
RESULTS_DIR = PROJECT_DIR / "results"
CHUNKS_DIR = PROJECT_DIR / "chunks"
CHECKPOINT_DIR = MODEL_DIR / "halicin_model"
TRAIN_CSV = DATA_DIR / "cleaned_training_data.csv"
REPURPOSING_HUB = DATA_DIR / "repurposing_hub.txt"
CHEMBL_CSV = PROJECT_DIR / "chembl_36_chemreps.txt"
