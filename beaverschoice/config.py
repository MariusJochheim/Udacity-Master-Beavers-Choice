import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = BASE_DIR / "munder_difflin.db"
DB_URL = f"sqlite:///{DB_PATH}"

# Defaults
DEFAULT_SEED = 137
DEFAULT_INVENTORY_COVERAGE = 0.4
START_DATE = datetime(2025, 1, 1)

# OpenAI settings (use .env or environment variables)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")


def data_path(filename: str) -> Path:
    """Helper to build absolute paths to files in the data directory."""
    return DATA_DIR / filename
