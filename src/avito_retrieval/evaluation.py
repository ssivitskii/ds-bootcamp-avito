from __future__ import annotations

import itertools
import time
from typing import Any

import numpy as np
import pandas as pd

from .constants import ITEM_COLUMNS, QUERY_COLUMNS
from .model import HybridRetriever
from .text import normalize_text, stable_fold, stable_rank


def make_query_units(train: pd.DataFrame, fold: int, modulo: int) -> pd.DataFrame:
    """Aggregate every full search tuple; relevant items are unique within a unit."""
    heldout = train[train["search_query"].map(lambda value: stable_fold(value, modulo)) == fold]
    units = (
        heldout.groupby(QUERY_COLUMNS, sort=True, dropna=False)["item_id"]
        .agg(lambda values: sorted(set(map(str, values))))
        .rename("relevant_items")
        .reset_index()
    )
    return units


def limit_units(units: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    if limit is None or limit <= 0 or len(units) <= limit:
        return units.reset_index(drop=True)
    ranked = units.copy()
    ranked["_rank"] = [
        stable_rank((*(getattr(row, column) for column in QUERY_COLUMNS),), seed)
        for row in ranked.itertuples(index=False)
    ]
    return ranked.sort_values("_rank", kind="mergesort").head(limit).drop(columns="_rank").reset_index(drop=True)


def training_without_folds(train: pd.DataFrame, folds: set[int], modulo: int) -> pd.DataFrame:
    mask = ~train["search_query"].map(lambda value: stable_fold(value, modulo)).isin(folds)
    return train.loc[mask].reset_index(drop=True)


def evaluation_corpus(
    benchmark_items: pd.DataFrame, heldout_rows: pd.DataFrame
) -> pd.DataFrame:
    """Ensure each offline relevant item is retrievable while retaining the real corpus."""
    existing = set(benchmark_items["item_id"].astype(str))
    additions = heldout_rows.loc[~heldout_rows["item_id"].astype(str).isin(existing), ITEM_COLUMNS]
    return (
        pd.concat([benchmark_items[ITEM_COLUMNS], additions], ignore_index=True)
        .drop_duplicates("item_id", keep="first")
        .reset_index(drop=True)
    )


def recall_at_k(predictions: list[list[str]], relevant: list[list[str]]) -> float:
    values = []
    for predicted, truth in zip(predictions, relevant, strict=True):
        truth_set = set(truth)
        values.append(len(set(predicted) & truth_set) / len(truth_set) if truth_set else 0.0)
    return float(np.mean(values)) if values else 0.0


def _grid(base: dict[str, float], tuning_grid: dict[str, list[float]]) -> list[dict[str, float]]:
    names = sorted(tuning_grid)
    combinations = []
    for values in itertools.product(*(tuning_grid[name] for name in names)):
        weights = dict(base)
        weights.update(dict(zip(names, values, strict=True)))
        combinations.append(weights)
    return combinations


def _slices(
    units: pd.DataFrame,
    predictions: list[list[str]],
    train_fit: pd.DataFrame,
) -> dict[str, dict[str, float | int]]:
    truth = units["relevant_items"].tolist()
    per_query = np.array(
        [
            len(set(predicted) & set(relevant)) / len(set(relevant))
            for predicted, relevant in zip(predictions, truth, strict=True)
        ],
        dtype=float,
    )
    location_counts = train_fit["search_location_id"].value_counts()
    nonzero_counts = location_counts[location_counts > 0]
    median_count = float(nonzero_counts.median()) if len(nonzero_counts) else 0.0
    definitions = {
        "filter_empty": units["search_infm_params_text"].map(normalize_text).eq(""),
        "filter_present": units["search_infm_params_text"].map(normalize_text).ne(""),
        "category_global": units["search_category"].eq(0),
        "category_scoped": units["search_category"].ne(0),
        "query_one_token": units["search_query"].map(lambda value: len(normalize_text(value).split()) <= 1),
        "query_multi_token": units["search_query"].map(lambda value: len(normalize_text(value).split()) > 1),
        "location_rare": units["search_location_id"].map(location_counts).fillna(0).lt(median_count),
        "location_common": units["search_location_id"].map(location_counts).fillna(0).ge(median_count),
    }
    history_item_ids = set(train_fit["item_id"].astype(str))
    all_seen = units["relevant_items"].map(
        lambda values: bool(values) and all(str(value) in history_item_ids for value in values)
    )
    definitions["relevant_all_history_seen"] = all_seen
    definitions["relevant_has_unseen"] = ~all_seen
    result: dict[str, dict[str, float | int]] = {}
    for name, mask in definitions.items():
        selected = mask.to_numpy(dtype=bool)
        result[name] = {
            "queries": int(selected.sum()),
            "recall_at_50": float(per_query[selected].mean()) if selected.any() else 0.0,
        }
    return result


def _fit_fold(
    train: pd.DataFrame,
    corpus: pd.DataFrame,
    config: dict[str, Any],
    fold: int,
    limit: int,
    excluded_folds: set[int],
    retriever: HybridRetriever | None = None,
) -> tuple[HybridRetriever, pd.DataFrame, list[dict[int, dict[str, float]]], pd.DataFrame, float]:
    modulo = int(config["fold_modulo"])
    units = limit_units(make_query_units(train, fold, modulo), limit, int(config["seed"]) + fold)
    heldout_texts = set(units["search_query"].map(normalize_text))
    # Exclude every row of every held-out text group, including tuples not selected by the query cap.
    train_fit = training_without_folds(train, excluded_folds, modulo)
    if set(train_fit["search_query"].map(normalize_text)) & heldout_texts:
        raise AssertionError("Cold split leakage: held-out text remains in training history")
    started = time.perf_counter()
    if retriever is None:
        retriever = HybridRetriever(config).fit(corpus, train_fit)
    else:
        retriever.refit_behavior(train_fit)
    features = retriever.retrieve_features(units[QUERY_COLUMNS])
    elapsed = time.perf_counter() - started
    return retriever, units, features, train_fit, elapsed


def evaluate_pipeline(
    train: pd.DataFrame,
    benchmark_items: pd.DataFrame,
    config: dict[str, Any],
    max_queries: int | None = None,
) -> dict[str, Any]:
    tune_limit = int(max_queries or config["max_tune_queries"])
    confirm_limit = int(max_queries or config["max_confirm_queries"])
    # The fixed offline catalog is independent of the sampled query units. It is
    # intentionally larger than the final benchmark catalog and contains every target.
    corpus = evaluation_corpus(benchmark_items, train)
    tune_fold = int(config["tune_fold"])
    confirm_fold = int(config["confirm_fold"])

    tune_model, tune_units, tune_features, _, tune_seconds = _fit_fold(
        train,
        corpus,
        config,
        tune_fold,
        tune_limit,
        excluded_folds={tune_fold, confirm_fold},
    )
    truth_tune = tune_units["relevant_items"].tolist()
    trials = []
    for weights in _grid(config["weights"], config["tuning_grid"]):
        predictions = tune_model.predict_from_features(tune_features, weights=weights)
        trials.append(
            {
                "weights": weights,
                "recall_at_50": recall_at_k(predictions, truth_tune),
            }
        )
    trials.sort(
        key=lambda row: (
            -float(row["recall_at_50"]),
            tuple((key, row["weights"][key]) for key in sorted(row["weights"])),
        )
    )
    selected_weights = dict(trials[0]["weights"])

    confirm_model, confirm_units, confirm_features, confirm_train, confirm_seconds = _fit_fold(
        train,
        corpus,
        config,
        confirm_fold,
        confirm_limit,
        excluded_folds={confirm_fold},
        retriever=tune_model,
    )
    truth_confirm = confirm_units["relevant_items"].tolist()
    ablation_weights = {
        "lexical_word": {name: (1.0 if name == "word" else 0.0) for name in selected_weights},
        "lexical_word_char": {
            name: (selected_weights[name] if name in {"word", "char"} else 0.0)
            for name in selected_weights
        },
        "lexical_with_filter": {
            name: (selected_weights[name] if name in {"word", "char", "filter"} else 0.0)
            for name in selected_weights
        },
        "hybrid_without_history": {**selected_weights, "history": 0.0},
        "hybrid_without_metadata": {
            **selected_weights,
            "category": 0.0,
            "location": 0.0,
        },
        "hybrid_full": selected_weights,
    }
    ablations: dict[str, float] = {}
    full_predictions: list[list[str]] = []
    for name, weights in ablation_weights.items():
        predictions = confirm_model.predict_from_features(confirm_features, weights=weights)
        ablations[name] = recall_at_k(predictions, truth_confirm)
        if name == "hybrid_full":
            full_predictions = predictions

    return {
        "protocol": {
            "split": "sha256(normalized search_query) modulo fold_modulo",
            "tune_fold": int(config["tune_fold"]),
            "confirm_fold": int(config["confirm_fold"]),
            "history_excludes_entire_heldout_text_group": True,
            "tune_history_also_excludes_confirm_fold": True,
            "evaluation_catalog": "benchmark items union all deduplicated train items",
            "metric": "macro Recall@50 over aggregated full search tuples",
        },
        "tune": {
            "queries": len(tune_units),
            "fit_and_retrieval_seconds": tune_seconds,
            "best_recall_at_50": float(trials[0]["recall_at_50"]),
            "trials": trials,
        },
        "confirm": {
            "queries": len(confirm_units),
            "fit_and_retrieval_seconds": confirm_seconds,
            "ablations": ablations,
            "error_slices": _slices(confirm_units, full_predictions, confirm_train),
        },
        "selected_weights": selected_weights,
    }
