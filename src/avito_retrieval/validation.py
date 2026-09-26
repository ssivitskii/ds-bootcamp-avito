from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from .constants import ITEM_ID_PATTERN, QUERY_ID_PATTERN


def validate_answer(
    answer_path: str | Path,
    benchmark_queries: pd.DataFrame,
    benchmark_items: pd.DataFrame,
) -> dict[str, Any]:
    answer = pd.read_csv(answer_path, dtype=str, keep_default_na=False)
    errors: list[str] = []
    if list(answer.columns) != ["query_id", "answer"]:
        errors.append(f"columns must be exactly ['query_id', 'answer'], got {list(answer.columns)!r}")
        return {"valid": False, "errors": errors, "rows": len(answer)}

    expected_ids = benchmark_queries["query_id"].astype(str).tolist()
    actual_ids = answer["query_id"].astype(str).tolist()
    if answer["query_id"].duplicated().any():
        errors.append("query_id contains duplicates")
    missing = sorted(set(expected_ids) - set(actual_ids))
    extra = sorted(set(actual_ids) - set(expected_ids))
    if missing:
        errors.append(f"missing query_id values: {len(missing)}")
    if extra:
        errors.append(f"unexpected query_id values: {len(extra)}")
    if len(answer) != len(expected_ids):
        errors.append(f"row count must be {len(expected_ids)}, got {len(answer)}")

    invalid_query_ids = [value for value in actual_ids if not re.fullmatch(QUERY_ID_PATTERN, value)]
    if invalid_query_ids:
        errors.append(f"invalid query_id format: {len(invalid_query_ids)}")

    corpus_ids = set(benchmark_items["item_id"].astype(str))
    invalid_format = 0
    unknown = 0
    duplicate_rows = 0
    over_limit = 0
    invalid_separator = 0
    empty_rows = 0
    for raw in answer["answer"].astype(str):
        if raw == "":
            item_ids: list[str] = []
            empty_rows += 1
        else:
            item_ids = raw.split(" ")
            if raw.strip() != raw or "" in item_ids or "\t" in raw or "\n" in raw:
                invalid_separator += 1
        if len(item_ids) > 50:
            over_limit += 1
        if len(item_ids) != len(set(item_ids)):
            duplicate_rows += 1
        invalid_format += sum(not re.fullmatch(ITEM_ID_PATTERN, item_id) for item_id in item_ids)
        unknown += sum(item_id not in corpus_ids for item_id in item_ids)
    if invalid_separator:
        errors.append(f"answers with separator other than one space: {invalid_separator}")
    if over_limit:
        errors.append(f"answers with more than 50 items: {over_limit}")
    if duplicate_rows:
        errors.append(f"answers containing duplicate item_id: {duplicate_rows}")
    if invalid_format:
        errors.append(f"item_id values with invalid lowercase hex format: {invalid_format}")
    if unknown:
        errors.append(f"item_id values absent from benchmark corpus: {unknown}")
    return {
        "valid": not errors,
        "errors": errors,
        "rows": len(answer),
        "empty_answers": empty_rows,
        "max_items": max((len(value.split(" ")) if value else 0 for value in answer["answer"]), default=0),
    }
