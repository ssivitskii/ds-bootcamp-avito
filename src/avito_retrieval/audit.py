from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from .constants import ITEM_ID_PATTERN, QUERY_ID_PATTERN
from .io import data_fingerprint, load_frames


def audit_data(data_dir: str | Path) -> dict[str, Any]:
    root = Path(data_dir)
    train, queries, items = load_frames(root)

    def frame_stats(frame: pd.DataFrame) -> dict[str, Any]:
        return {
            "rows": len(frame),
            "columns": list(frame.columns),
            "nulls": {column: int(count) for column, count in frame.isna().sum().items() if count},
        }

    train_ids = set(train["item_id"].astype(str))
    item_ids = items["item_id"].astype(str)
    query_ids = queries["query_id"].astype(str)
    return {
        "fingerprint": data_fingerprint(
            [root / "train.parquet", root / "benchmark_queries.parquet", root / "benchmark_items.parquet"]
        ),
        "frames": {
            "train": frame_stats(train),
            "benchmark_queries": frame_stats(queries),
            "benchmark_items": frame_stats(items),
        },
        "ids": {
            "duplicate_query_id": int(query_ids.duplicated().sum()),
            "duplicate_benchmark_item_id": int(item_ids.duplicated().sum()),
            "invalid_query_id": int(sum(not re.fullmatch(QUERY_ID_PATTERN, value) for value in query_ids)),
            "invalid_train_item_id": int(
                sum(not re.fullmatch(ITEM_ID_PATTERN, value) for value in train["item_id"].astype(str))
            ),
            "invalid_benchmark_item_id": int(
                sum(not re.fullmatch(ITEM_ID_PATTERN, value) for value in item_ids)
            ),
            "unique_train_items": len(train_ids),
            "unique_benchmark_items": item_ids.nunique(),
            "item_overlap": len(train_ids & set(item_ids)),
        },
    }
