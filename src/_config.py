"""Hardware config — all tuning knobs in one place.
Override via environment variables."""
import os

P_CORE_THREADS = int(os.environ.get("P_CORE_THREADS", "4"))
DRIVER_MEMORY = os.environ.get("DRIVER_MEMORY", "4g")
ARROW_BATCH = int(os.environ.get("ARROW_BATCH", "500"))
CHEMPROP_BATCH = int(os.environ.get("CHEMPROP_BATCH", "100"))
TARGET_PER_PARTITION = int(os.environ.get("TARGET_PER_PARTITION", "5000"))
