from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .constants import QUERY_COLUMNS
from .evaluation import (
    evaluation_corpus,
    limit_units,
    make_query_units,
    training_without_folds,
)
from .io import data_fingerprint, source_fingerprint
from .model import FeatureRow, HybridRetriever
from .text import normalize_text

LOGGER = logging.getLogger(__name__)

FEATURE_NAMES = (
    "word",
    "char",
    "filter",
    "history",
    "category",
    "location",
    "popularity",
    "exact_word",
    "exact_char",
    "exact_filter",
    "title",
    "microcat",
)
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
RANK_MISSING = np.iinfo(np.uint16).max


@dataclass
class PackedRow:
    positions: np.ndarray
    values: np.ndarray
    ranks: dict[str, np.ndarray]
    always_source: np.ndarray


@dataclass
class PackedFeatureBundle:
    fold: int
    item_ids: np.ndarray
    rows: list[PackedRow]
    relevant: list[list[str]]
    slices: dict[str, np.ndarray]


def _atomic_joblib_dump(value: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        joblib.dump(value, temporary, compress=0)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def pack_feature_rows(
    model: HybridRetriever,
    feature_rows: list[FeatureRow],
    units: pd.DataFrame,
    train_fit: pd.DataFrame,
    fold: int,
    geo_options: tuple[int, ...],
) -> PackedFeatureBundle:
    assert model.item_ids is not None
    assert model.item_locations is not None
    rank_names = [
        "__global_word_rank",
        "__global_char_rank",
        "__global_filter_rank",
        *[
            name
            for geo in geo_options
            for name in (f"__geo{geo}_word_rank", f"__geo{geo}_char_rank")
        ],
    ]
    packed_rows: list[PackedRow] = []
    for feature_row in feature_rows:
        positions = np.array(sorted(feature_row), dtype=np.int32)
        values = np.zeros((len(positions), len(FEATURE_NAMES)), dtype=np.float64)
        ranks = {
            name: np.full(len(positions), RANK_MISSING, dtype=np.uint16)
            for name in rank_names
        }
        always = np.zeros(len(positions), dtype=bool)
        for local, pos in enumerate(positions):
            channels = feature_row[int(pos)]
            for name, column in FEATURE_INDEX.items():
                values[local, column] = float(channels.get(name, 0.0))
            for name in rank_names:
                if name in channels:
                    ranks[name][local] = np.uint16(int(channels[name]))
            always[local] = (
                "__history_source" in channels
                or "__fallback_source" in channels
                or "__dense_source" in channels
            )
        packed_rows.append(PackedRow(positions, values, ranks, always))

    relevant = units["relevant_items"].tolist()
    history_ids = set(train_fit["item_id"].astype(str))
    item_location = {
        str(item_id): int(model.item_locations[pos])
        for pos, item_id in enumerate(model.item_ids)
    }
    all_seen = np.array(
        [all(item_id in history_ids for item_id in values) for values in relevant], dtype=bool
    )
    all_same_location = np.array(
        [
            all(item_location.get(item_id) == int(row.search_location_id) for item_id in values)
            for row, values in zip(units.itertuples(index=False), relevant, strict=True)
        ],
        dtype=bool,
    )
    slices = {
        "filter_empty": units["search_infm_params_text"].map(normalize_text).eq("").to_numpy(),
        "filter_present": units["search_infm_params_text"].map(normalize_text).ne("").to_numpy(),
        "relevant_all_history_seen": all_seen,
        "relevant_has_unseen": ~all_seen,
        "relevant_all_exact_location": all_same_location,
        "relevant_has_cross_location": ~all_same_location,
    }
    return PackedFeatureBundle(
        fold=fold,
        item_ids=model.item_ids.copy(),
        rows=packed_rows,
        relevant=relevant,
        slices=slices,
    )


def _channel_masks(
    row: PackedRow, pool: int, geo_neighbors: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    word = (row.ranks["__global_word_rank"] <= pool) | (
        row.ranks[f"__geo{geo_neighbors}_word_rank"] <= pool
    )
    char = (row.ranks["__global_char_rank"] <= pool) | (
        row.ranks[f"__geo{geo_neighbors}_char_rank"] <= pool
    )
    filter_mask = row.ranks["__global_filter_rank"] <= pool
    eligible = word | char | filter_mask | row.always_source
    return word, char, filter_mask, eligible


def score_packed_row(
    row: PackedRow,
    settings: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    weights = settings["weights"]
    word_mask, char_mask, filter_mask, eligible = _channel_masks(
        row, int(settings["pool"]), int(settings["geo_neighbors"])
    )
    values = row.values
    if bool(settings["exact_rescore"]):
        word = values[:, FEATURE_INDEX["exact_word"]]
        char = values[:, FEATURE_INDEX["exact_char"]]
        filter_values = values[:, FEATURE_INDEX["exact_filter"]]
    else:
        word = values[:, FEATURE_INDEX["word"]] * word_mask
        char = values[:, FEATURE_INDEX["char"]] * char_mask
        filter_values = values[:, FEATURE_INDEX["filter"]] * filter_mask
    scores = (
        float(weights["word"]) * word
        + float(weights["char"]) * char
        + float(weights["filter"]) * filter_values
        + float(weights["history"]) * values[:, FEATURE_INDEX["history"]]
        + float(weights["category"]) * values[:, FEATURE_INDEX["category"]]
        + float(weights["location"]) * values[:, FEATURE_INDEX["location"]]
        + float(weights["popularity"]) * values[:, FEATURE_INDEX["popularity"]]
        + float(weights["title"]) * values[:, FEATURE_INDEX["title"]]
        + float(weights["microcat"]) * values[:, FEATURE_INDEX["microcat"]]
    )
    return scores, eligible


def predict_packed(
    bundle: PackedFeatureBundle,
    settings: dict[str, Any],
    top_k: int = 50,
) -> tuple[list[list[str]], np.ndarray, np.ndarray]:
    predictions: list[list[str]] = []
    recalls = np.zeros(len(bundle.rows), dtype=np.float64)
    oracle = np.zeros(len(bundle.rows), dtype=np.float64)
    for row_number, row in enumerate(bundle.rows):
        scores, eligible = score_packed_row(row, settings)
        local = np.flatnonzero(eligible)
        # item_ids are globally sorted, so corpus position is the exact ID tie-break.
        order = np.lexsort((row.positions[local], -scores[local]))
        selected_positions = row.positions[local[order[:top_k]]]
        selected = [str(bundle.item_ids[pos]) for pos in selected_positions]
        if len(selected) < top_k:
            seen = set(selected)
            for item_id in bundle.item_ids:
                candidate = str(item_id)
                if candidate not in seen:
                    selected.append(candidate)
                    seen.add(candidate)
                if len(selected) == top_k:
                    break
        predictions.append(selected)
        truth = set(bundle.relevant[row_number])
        if not truth:
            continue
        recalls[row_number] = len(set(selected) & truth) / len(truth)
        eligible_ids = {str(bundle.item_ids[pos]) for pos in row.positions[eligible]}
        oracle[row_number] = len(eligible_ids & truth) / len(truth)
    return predictions, recalls, oracle


def _settings_key(settings: dict[str, Any]) -> str:
    return json.dumps(settings, sort_keys=True, separators=(",", ":"))


def baseline_settings(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "exact_rescore": False,
        "pool": 120,
        "geo_neighbors": 3,
        "weights": copy.deepcopy(config["baseline_weights"]),
    }


def coordinate_tune(
    bundle: PackedFeatureBundle, config: dict[str, Any]
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, float],
    dict[str, Any],
]:
    current = baseline_settings(config)
    cache: dict[str, np.ndarray] = {}
    trials: list[dict[str, Any]] = []

    def recalls(settings: dict[str, Any]) -> np.ndarray:
        key = _settings_key(settings)
        if key not in cache:
            _, values, _ = predict_packed(bundle, settings)
            cache[key] = values
        return cache[key]

    minimum_queries = int(config["selection_slice_min_queries"])
    minimum_delta = float(config["selection_slice_min_delta"])
    baseline_values = recalls(current)
    guarded_slices = {
        name: np.asarray(mask, dtype=bool)
        for name, mask in bundle.slices.items()
        if int(np.asarray(mask).sum()) >= minimum_queries
    }
    baseline_slices = {
        name: float(baseline_values[mask].mean())
        for name, mask in guarded_slices.items()
    }

    def evaluation(settings: dict[str, Any]) -> tuple[float, float, dict[str, float]]:
        values = recalls(settings)
        deltas = {
            name: float(values[mask].mean()) - baseline_slices[name]
            for name, mask in guarded_slices.items()
        }
        return float(values.mean()), min(deltas.values(), default=0.0), deltas

    def feasible(settings: dict[str, Any]) -> bool:
        return evaluation(settings)[1] >= minimum_delta - 1e-12

    domains: list[tuple[str, list[Any]]] = [
        ("exact_rescore", [False, True]),
        ("pool", list(config["pool_options"])),
        ("geo_neighbors", list(config["geo_neighbor_options"])),
        *[(f"weight:{name}", list(values)) for name, values in config["weight_options"].items()],
    ]
    for pass_number in range(int(config["coordinate_passes"])):
        for name, options in domains:
            best = copy.deepcopy(current)
            best_recall = evaluation(best)[0] if feasible(best) else -1.0
            for option in options:
                candidate = copy.deepcopy(current)
                if name.startswith("weight:"):
                    candidate["weights"][name.split(":", 1)[1]] = float(option)
                else:
                    candidate[name] = option
                candidate_recall, guard_floor, _ = evaluation(candidate)
                candidate_feasible = feasible(candidate)
                trials.append(
                    {
                        "pass": pass_number + 1,
                        "parameter": name,
                        "value": option,
                        "recall_at_50": candidate_recall,
                        "eligible": candidate_feasible,
                        "minimum_guarded_slice_delta": guard_floor,
                    }
                )
                if candidate_feasible and candidate_recall > best_recall + 1e-12:
                    best = candidate
                    best_recall = candidate_recall
            current = best

    ablations: dict[str, float] = {}
    baseline = baseline_settings(config)
    for name in ("exact_rescore", "pool", "geo_neighbors"):
        candidate = copy.deepcopy(current)
        candidate[name] = baseline[name]
        ablations[name] = evaluation(candidate)[0]
    for name in config["weight_options"]:
        candidate = copy.deepcopy(current)
        candidate["weights"][name] = baseline["weights"][name]
        ablations[name] = evaluation(candidate)[0]
    ablations["selected"] = evaluation(current)[0]
    ablations["baseline"] = evaluation(baseline)[0]
    selected_overall, selected_floor, selected_deltas = evaluation(current)
    guard_report = {
        "minimum_queries": minimum_queries,
        "minimum_delta": minimum_delta,
        "guarded_slice_baselines": baseline_slices,
        "selected_slice_deltas": selected_deltas,
        "selected_minimum_delta": selected_floor,
        "selected_recall_at_50": selected_overall,
        "selected_eligible": feasible(current),
    }
    return current, trials, ablations, guard_report


def _bootstrap_ci(delta: np.ndarray, seed: int, samples: int) -> list[float]:
    if not len(delta):
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 200):
        count = min(200, samples - start)
        indices = rng.integers(0, len(delta), size=(count, len(delta)))
        means[start : start + count] = delta[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return [float(low), float(high)]


def compare_settings(
    bundle: PackedFeatureBundle,
    baseline: dict[str, Any],
    enhanced: dict[str, Any],
    *,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    _, baseline_recall, baseline_oracle = predict_packed(bundle, baseline)
    _, enhanced_recall, enhanced_oracle = predict_packed(bundle, enhanced)
    delta = enhanced_recall - baseline_recall
    slices: dict[str, Any] = {}
    for name, mask in bundle.slices.items():
        selected = np.asarray(mask, dtype=bool)
        slices[name] = {
            "queries": int(selected.sum()),
            "baseline_recall_at_50": float(baseline_recall[selected].mean())
            if selected.any()
            else 0.0,
            "enhanced_recall_at_50": float(enhanced_recall[selected].mean())
            if selected.any()
            else 0.0,
            "delta": float(delta[selected].mean()) if selected.any() else 0.0,
        }
    return {
        "queries": len(bundle.rows),
        "baseline_recall_at_50": float(baseline_recall.mean()),
        "enhanced_recall_at_50": float(enhanced_recall.mean()),
        "delta": float(delta.mean()),
        "paired_bootstrap_95_ci": _bootstrap_ci(delta, seed, bootstrap_samples),
        "wins": int((delta > 0).sum()),
        "losses": int((delta < 0).sum()),
        "ties": int((delta == 0).sum()),
        "baseline_candidate_oracle_recall": float(baseline_oracle.mean()),
        "enhanced_candidate_oracle_recall": float(enhanced_oracle.mean()),
        "slices": slices,
    }


def select_fold_units(
    train: pd.DataFrame,
    fold: int,
    modulo: int,
    limit: int,
    seed: int,
    sample_version: str,
) -> pd.DataFrame:
    all_units = make_query_units(train, fold, modulo)
    if sample_version == "standard-v1":
        return limit_units(all_units, limit, seed)
    if sample_version != "fresh-text-v2":
        raise ValueError(f"Unknown sample_version: {sample_version}")
    original = limit_units(all_units, limit, seed)
    original_texts = set(original["search_query"].map(normalize_text))
    remaining = all_units[
        ~all_units["search_query"].map(normalize_text).isin(original_texts)
    ]
    fresh = limit_units(remaining, limit, seed)
    if set(fresh["search_query"].map(normalize_text)) & original_texts:
        raise AssertionError("Fresh sample overlaps original text clusters")
    if len(fresh) < limit:
        raise ValueError(
            f"Fold {fold} has only {len(fresh)} units after fresh text exclusion"
        )
    return fresh


def _fold_bundle(
    model: HybridRetriever,
    train: pd.DataFrame,
    config: dict[str, Any],
    fold: int,
    excluded_folds: set[int],
    limit: int,
    cache_path: Path,
    rebuild: bool,
    sample_version: str = "standard-v1",
) -> PackedFeatureBundle:
    if cache_path.exists() and not rebuild:
        LOGGER.info("Loading fold %d feature bundle from %s", fold, cache_path)
        return joblib.load(cache_path)
    LOGGER.info(
        "Building fold %d bundle: excluded_folds=%s query_limit=%d sample=%s",
        fold,
        sorted(excluded_folds),
        limit,
        sample_version,
    )
    modulo = int(config["fold_modulo"])
    sample_seed = int(config["seed"]) + fold
    units = select_fold_units(
        train, fold, modulo, limit, sample_seed, sample_version
    )
    train_fit = training_without_folds(train, excluded_folds, modulo)
    heldout = set(units["search_query"].map(normalize_text))
    if heldout & set(train_fit["search_query"].map(normalize_text)):
        raise AssertionError("Cold split leakage in improvement fold")
    model.refit_behavior(train_fit)
    feature_rows = model.retrieve_improvement_features(
        units[QUERY_COLUMNS],
        max_pool=int(config["max_pool"]),
        geo_neighbor_options=tuple(map(int, config["geo_neighbor_options"])),
    )
    bundle = pack_feature_rows(
        model,
        feature_rows,
        units,
        train_fit,
        fold,
        tuple(map(int, config["geo_neighbor_options"])),
    )
    _atomic_joblib_dump(bundle, cache_path)
    LOGGER.info("Saved fold %d feature bundle to %s", fold, cache_path)
    return bundle


def run_improvement(
    train: pd.DataFrame,
    benchmark_items: pd.DataFrame,
    base_config: dict[str, Any],
    config: dict[str, Any],
    *,
    data_dir: str | Path,
    cache_dir: str | Path,
    max_queries: int | None = None,
    rebuild_cache: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    model_config = copy.deepcopy(base_config)
    model_config["max_geo_neighbors"] = max(map(int, config["geo_neighbor_options"]))
    model_config["microcat_neighbors"] = int(config["microcat_neighbors"])
    root = Path(data_dir)
    fingerprint = data_fingerprint(
        [root / "train.parquet", root / "benchmark_items.parquet"],
        {
            "improvement_cache_format": 1,
            "base_config": model_config,
            "improvement_config": config,
            "source_sha256": source_fingerprint(Path(__file__).parent),
        },
    )
    cache_root = Path(cache_dir) / "improvement" / fingerprint
    model_path = cache_root / "model.joblib"
    tune_fold = int(config["tune_fold"])
    confirm_fold = int(config["confirm_fold"])
    audit_fold = int(config["audit_fold"])
    tune_train = training_without_folds(
        train, {tune_fold, confirm_fold, audit_fold}, int(config["fold_modulo"])
    )
    if model_path.exists() and not rebuild_cache:
        LOGGER.info("Loading improvement model from %s", model_path)
        model = HybridRetriever.load(model_path)
    else:
        LOGGER.info("Building fixed improvement catalog")
        corpus = evaluation_corpus(benchmark_items, train)
        model = HybridRetriever(model_config).fit(corpus, tune_train)
        model.enable_improvement(tune_train)
        model.save(model_path)
        LOGGER.info("Saved improvement model to %s", model_path)

    limit = int(max_queries or config["max_queries_per_fold"])
    fold_specs = {
        tune_fold: {tune_fold, confirm_fold, audit_fold},
        confirm_fold: {confirm_fold, audit_fold},
        audit_fold: {audit_fold},
    }
    sample_versions = {
        tune_fold: "standard-v1",
        confirm_fold: "fresh-text-v2",
        audit_fold: "standard-v1",
    }
    bundles = {
        fold: _fold_bundle(
            model,
            train,
            config,
            fold,
            excluded,
            limit,
            cache_root
            / (
                f"fold-{fold}-fresh-text-v2-queries-{limit}.joblib"
                if sample_versions[fold] == "fresh-text-v2"
                else f"fold-{fold}-queries-{limit}.joblib"
            ),
            rebuild_cache,
            sample_versions[fold],
        )
        for fold, excluded in fold_specs.items()
    }
    selected, trials, tune_ablation, selection_guard = coordinate_tune(
        bundles[tune_fold], config
    )
    baseline = baseline_settings(config)
    comparisons = {
        name: compare_settings(
            bundles[fold],
            baseline,
            selected,
            seed=int(config["seed"]) + fold,
            bootstrap_samples=int(config["bootstrap_samples"]),
        )
        for name, fold in (
            ("tune", tune_fold),
            ("confirm", confirm_fold),
            ("audit", audit_fold),
        )
    }
    return {
        "protocol": {
            "folds": {"tune": tune_fold, "confirm": confirm_fold, "audit": audit_fold},
            "exclusions": {
                "tune": sorted(fold_specs[tune_fold]),
                "confirm": sorted(fold_specs[confirm_fold]),
                "audit": sorted(fold_specs[audit_fold]),
            },
            "selection_uses_only_tune": True,
            "catalog": "benchmark items union all deduplicated train items",
            "queries_per_fold": limit,
            "sampling": {
                "tune": sample_versions[tune_fold],
                "confirm": sample_versions[confirm_fold],
                "audit": sample_versions[audit_fold],
            },
        },
        "baseline_settings": baseline,
        "selected_settings": selected,
        "model_overrides": {
            "max_geo_neighbors": model_config["max_geo_neighbors"],
            "microcat_neighbors": model_config["microcat_neighbors"],
        },
        "tune_trials": trials,
        "tune_ablations": tune_ablation,
        "selection_guard": selection_guard,
        "folds": comparisons,
        "cache_fingerprint": fingerprint,
        "elapsed_seconds": time.perf_counter() - started,
    }


def predict_improved(
    model: HybridRetriever,
    queries: pd.DataFrame,
    settings: dict[str, Any],
    top_k: int = 50,
) -> list[list[str]]:
    geo = int(settings["geo_neighbors"])
    rows = model.retrieve_improvement_features(
        queries,
        max_pool=int(settings["pool"]),
        geo_neighbor_options=(geo,),
    )
    empty_units = queries.copy()
    empty_units["relevant_items"] = [[] for _ in range(len(queries))]
    # No labels are used for final prediction; the packed scorer only needs candidate arrays.
    bundle = pack_feature_rows(
        model,
        rows,
        empty_units,
        pd.DataFrame({"item_id": []}),
        fold=-1,
        geo_options=(geo,),
    )
    predictions, _, _ = predict_packed(bundle, settings, top_k=top_k)
    return predictions
