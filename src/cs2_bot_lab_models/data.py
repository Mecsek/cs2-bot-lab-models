from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
TABLES_DIR = RAW_DATA_DIR / "parquet"
MODELS_DIR = PROJECT_ROOT / "models"


def list_datasets() -> list[Path]:
    """All .parquet files under data/raw (recursively)."""
    return sorted(RAW_DATA_DIR.rglob("*.parquet"))


def list_tables() -> list[str]:
    """Table names under data/raw/parquet, e.g. 'rounds', 'round_state_samples'."""
    return sorted(p.name for p in TABLES_DIR.iterdir() if p.is_dir())


def _dataset(table: str) -> ds.Dataset:
    return ds.dataset(TABLES_DIR / table, format="parquet", partitioning="hive")


def load(
    table: str,
    batch_size: int = 100_000,
    columns: list[str] | None = None,
    filter: ds.Expression | None = None,
) -> Iterator[pd.DataFrame]:
    """Load one table incrementally in batches."""
    scanner = _dataset(table).scanner(
        columns=columns,
        filter=filter,
        batch_size=batch_size,
        use_threads=True,
    )
    for batch in scanner.to_batches():
        yield batch.to_pandas(types_mapper=pd.ArrowDtype)


def load_all(
    table: str,
    columns: list[str] | None = None,
    filter: ds.Expression | None = None,
) -> pd.DataFrame:
    """Load a whole table into memory with NumPy dtypes, ready for scikit-learn."""
    return _dataset(table).to_table(columns=columns, filter=filter).to_pandas()
