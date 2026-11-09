from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"


def list_datasets() -> list[Path]:
    """All .parquet files under data/raw (recursively)."""
    return sorted(RAW_DATA_DIR.rglob("*.parquet"))


def load(name: str | None = None) -> pd.DataFrame:
    """Load one file by name (with or without .parquet), or every file in data/raw combined."""
    if name is None:
        files = list_datasets()
        if not files:
            raise FileNotFoundError(f"No .parquet files in {RAW_DATA_DIR}")
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    path = RAW_DATA_DIR / (name if name.endswith(".parquet") else f"{name}.parquet")
    return pd.read_parquet(path)
