from datetime import datetime
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = BASE_DIR / "munder_difflin.db"
DB_URL = f"sqlite:///{DB_PATH}"

# Defaults
DEFAULT_SEED = 137
DEFAULT_INVENTORY_COVERAGE = 0.4
START_DATE = datetime(2025, 1, 1)


def data_path(filename: str) -> Path:
    """Helper to build absolute paths to files in the data directory."""
    return DATA_DIR / filename
