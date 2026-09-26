from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from .constants import ITEM_COLUMNS, QUERY_COLUMNS, TRAIN_COLUMNS


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_frames(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(data_dir)
    train = pd.read_parquet(root / "train.parquet")
    queries = pd.read_parquet(root / "benchmark_queries.parquet")
    items = pd.read_parquet(root / "benchmark_items.parquet")
    require_columns(train, TRAIN_COLUMNS, "train.parquet")
    require_columns(queries, ["query_id", *QUERY_COLUMNS], "benchmark_queries.parquet")
    require_columns(items, ITEM_COLUMNS, "benchmark_items.parquet")
    return train, queries, items


def require_columns(frame: pd.DataFrame, required: list[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label}: missing columns: {missing}")


def data_fingerprint(paths: list[str | Path], config: dict[str, Any] | None = None) -> str:
    """Fingerprint full parquet contents, metadata and model-affecting configuration."""
    payload: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        stat = path.stat()
        parquet = pq.ParquetFile(path)
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                hasher.update(chunk)
        payload.append(
            {
                "name": path.name,
                "size": stat.st_size,
                "sha256": hasher.hexdigest(),
                "rows": parquet.metadata.num_rows,
                "schema": str(parquet.schema_arrow),
            }
        )
    body = json.dumps(
        {"files": payload, "config": config or {}},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def source_fingerprint(package_dir: str | Path) -> str:
    hasher = hashlib.sha256()
    for path in sorted(Path(package_dir).glob("*.py")):
        hasher.update(path.name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


def write_json(payload: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        stream.write("\n")
