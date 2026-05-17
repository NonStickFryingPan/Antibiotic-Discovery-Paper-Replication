"""Project initialization — call init_project() at the top of every script."""
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent


def init_project():
    os.chdir(PROJECT_DIR)
    if "CONDA_PREFIX" in os.environ:
        os.environ["JAVA_HOME"] = os.environ["CONDA_PREFIX"]
