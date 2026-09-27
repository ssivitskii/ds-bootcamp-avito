from __future__ import annotations

import copy
import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from .constants import QUERY_COLUMNS
from .evaluation import evaluation_corpus, limit_units, make_query_units, training_without_folds
from .improvement import (
    FEATURE_INDEX,
    PackedFeatureBundle,
    _atomic_joblib_dump,
    pack_feature_rows,
    score_packed_row,
    select_fold_units,
)
from .io import data_fingerprint, source_fingerprint
from .model import HybridRetriever
from .text import normalize_text, stable_fold


LOGGER = logging.getLogger(__name__)

RAW_FEATURE_NAMES = (
    "exact_word",
    "exact_char",
    "exact_filter",
    "title",
    "microcat",
    "location",
    "category",
    "popularity",
    "history",
)
RELATIVE_FEATURE_NAMES = RAW_FEATURE_NAMES[:6]
LOCALITY_FEATURE_NAMES = (
    "query_locality",
    "query_locality_x_location_prior",
    "microcat_locality_prior",
    "query_locality_x_microcat_locality_prior",
)
RERANK_FEATURE_NAMES = (
    *RAW_FEATURE_NAMES,
    "base_score",
    *(f"relative_{name}" for name in RELATIVE_FEATURE_NAMES),
    "exact_word_x_location",
    "exact_char_x_location",
    "microcat_x_location",
    "title_x_microcat",
    *(f"query_max_{name}" for name in RELATIVE_FEATURE_NAMES),
    *LOCALITY_FEATURE_NAMES,
)


def validate_reranker_config(config: dict[str, Any]) -> None:
    required = {"enabled", "format_version", "seed", "training", "features", "model", "blend", "evaluation"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing reranker config keys: {sorted(missing)}")
    if not bool(config["enabled"]):
        raise ValueError("Reranker config is disabled")
    if int(config["format_version"]) != 1:
        raise ValueError("Unsupported reranker config format")
    training = config["training"]
    for name in (
        "fold",
        "fold_modulo",
        "limit",
        "unit_seed",
        "expected_unit_sha256",
        "sample_version",
        "excluded_behavior_folds",
        "hard_negatives_per_query",
        "random_candidates_per_query",
        "positive_weight_numerator",
    ):
        if name not in training:
            raise ValueError(f"Missing reranker training key: {name}")
    evaluation = config["evaluation"]
    for name in (
        "selection_seed_base",
        "behavior_exclusions",
        "expected_unit_sha256",
        "bootstrap_samples",
        "bootstrap_seeds",
    ):
        if name not in evaluation:
            raise ValueError(f"Missing reranker evaluation key: {name}")
    hashes = [
        str(training["expected_unit_sha256"]),
        *(str(value) for value in evaluation["expected_unit_sha256"].values()),
    ]
    if any(len(value) != 64 or any(char not in "0123456789abcdef" for char in value) for value in hashes):
        raise ValueError("Reranker unit hashes must be lowercase SHA-256 values")
    if str(config["features"].get("schema")) != "hgb-locality-v1":
        raise ValueError("Unsupported reranker feature schema")
    if int(config["model"].get("thread_limit", 0)) != 4:
        raise ValueError("Frozen reranker requires thread_limit=4")
    if len(RERANK_FEATURE_NAMES) != 30:
        raise AssertionError("Frozen reranker must have exactly 30 features")


@dataclass
class LocalityFeatures:
    rows: list[np.ndarray]
    query_locality: np.ndarray
    fit_rows: int
    fit_text_groups: int
    heldout_text_groups: int
    fit_heldout_text_overlap: int
    global_exact_location_rate: float


@dataclass
class PreparedFeatures:
    matrix: np.ndarray
    labels: np.ndarray
    weights: np.ndarray
    positions: list[np.ndarray]
    base_scores: list[np.ndarray]
    target_counts: list[int]
    sampled: bool


def unit_hash(units: pd.DataFrame) -> str:
    canonical = units[QUERY_COLUMNS].astype(str).copy()
    canonical["relevant_items"] = units["relevant_items"].map(
        lambda values: " ".join(map(str, values))
    )
    payload = canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _nearest_query_locality(
    model: HybridRetriever,
    keys: list[str],
    propensities: np.ndarray,
    query_texts: list[str],
    global_mean: float,
    neighbor_count: int,
) -> np.ndarray:
    assert model.word_vectorizer is not None
    key_matrix = model.word_vectorizer.transform(keys).tocsr()
    query_matrix = model.word_vectorizer.transform(query_texts).tocsr()
    similarities = (query_matrix @ key_matrix.T).tocsr()
    transferred = np.full(len(query_texts), global_mean, dtype=np.float32)
    for row_number in range(len(query_texts)):
        start, end = similarities.indptr[row_number : row_number + 2]
        indices = similarities.indices[start:end]
        values = similarities.data[start:end]
        positive = values > 0
        indices = indices[positive]
        values = values[positive]
        if not len(indices):
            continue
        order = np.lexsort((indices, -values))[:neighbor_count]
        chosen = indices[order]
        weights = np.square(values[order].astype(np.float64))
        denominator = float(weights.sum())
        if denominator > 0:
            transferred[row_number] = float(
                np.dot(weights, propensities[chosen]) / denominator
            )
    return transferred


def build_locality_features(
    model: HybridRetriever,
    train_fit: pd.DataFrame,
    queries: pd.DataFrame,
    bundle: PackedFeatureBundle,
    config: dict[str, Any],
    *,
    require_disjoint: bool,
) -> LocalityFeatures:
    if len(queries) != len(bundle.rows):
        raise ValueError("Queries and packed rows must have identical length")
    assert model.item_microcats is not None
    feature_config = config["features"]
    work = train_fit[
        ["search_query", "search_location_id", "item_location_id", "item_microcat_id"]
    ].copy()
    work["key"] = work["search_query"].map(normalize_text)
    work["same_location"] = work["search_location_id"].eq(
        work["item_location_id"]
    ).astype(np.uint8)
    global_mean = float(work["same_location"].mean())

    query_smoothing = float(feature_config["query_locality_smoothing"])
    query_counts = work.groupby("key", sort=True)["same_location"].agg(["size", "sum"])
    query_counts["propensity"] = (
        query_counts["sum"] + query_smoothing * global_mean
    ) / (query_counts["size"] + query_smoothing)
    keys = query_counts.index.astype(str).tolist()
    query_texts = queries["search_query"].map(normalize_text).tolist()
    overlap = len(set(keys) & set(query_texts))
    if require_disjoint and overlap:
        raise AssertionError(f"Locality fit contains {overlap} held-out text groups")
    query_locality = _nearest_query_locality(
        model,
        keys,
        query_counts["propensity"].to_numpy(dtype=np.float64),
        query_texts,
        global_mean,
        int(feature_config["query_locality_neighbors"]),
    )

    microcat_smoothing = float(feature_config["microcat_locality_smoothing"])
    microcat_counts = work.groupby("item_microcat_id", sort=True)["same_location"].agg(
        ["size", "sum"]
    )
    microcat_counts["propensity"] = (
        microcat_counts["sum"] + microcat_smoothing * global_mean
    ) / (microcat_counts["size"] + microcat_smoothing)
    microcat_prior = {
        int(microcat): float(value)
        for microcat, value in microcat_counts["propensity"].items()
    }

    rows: list[np.ndarray] = []
    for row_number, packed_row in enumerate(bundle.rows):
        locality = float(query_locality[row_number])
        location_prior = packed_row.values[:, FEATURE_INDEX["location"]]
        candidate_microcats = model.item_microcats[packed_row.positions]
        candidate_microcat_prior = np.fromiter(
            (microcat_prior.get(int(value), global_mean) for value in candidate_microcats),
            dtype=np.float32,
            count=len(candidate_microcats),
        )
        rows.append(
            np.column_stack(
                [
                    np.full(len(packed_row.positions), locality, dtype=np.float32),
                    locality * location_prior,
                    candidate_microcat_prior,
                    locality * candidate_microcat_prior,
                ]
            ).astype(np.float32, copy=False)
        )
    return LocalityFeatures(
        rows=rows,
        query_locality=query_locality,
        fit_rows=len(train_fit),
        fit_text_groups=len(keys),
        heldout_text_groups=len(set(query_texts)),
        fit_heldout_text_overlap=overlap,
        global_exact_location_rate=global_mean,
    )


def _row_matrix(raw: np.ndarray, base: np.ndarray, locality: np.ndarray) -> np.ndarray:
    maximum = raw.max(axis=0)
    relative = raw[:, :6] / np.maximum(maximum[:6], 1e-6)
    return np.column_stack(
        [
            raw,
            base,
            relative,
            raw[:, 0] * raw[:, 5],
            raw[:, 1] * raw[:, 5],
            raw[:, 4] * raw[:, 5],
            raw[:, 3] * raw[:, 4],
            np.broadcast_to(maximum[:6], (len(raw), 6)),
            locality,
        ]
    ).astype(np.float32)


def prepare_features(
    bundle: PackedFeatureBundle,
    locality: LocalityFeatures,
    settings: dict[str, Any],
    config: dict[str, Any],
    *,
    training: bool,
) -> PreparedFeatures:
    if len(locality.rows) != len(bundle.rows):
        raise ValueError("Locality and packed feature rows must align")
    chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    weight_chunks: list[np.ndarray] = []
    positions_by_row: list[np.ndarray] = []
    base_by_row: list[np.ndarray] = []
    target_counts: list[int] = []
    rng = np.random.default_rng(int(config["seed"]))
    training_config = config["training"]
    raw_columns = [FEATURE_INDEX[name] for name in RAW_FEATURE_NAMES]
    for row_number, row in enumerate(bundle.rows):
        base, eligible = score_packed_row(row, settings)
        local = np.flatnonzero(eligible)
        positions = row.positions[local]
        labels = np.isin(bundle.item_ids[positions], bundle.relevant[row_number])
        raw = row.values[local][:, raw_columns]
        matrix = _row_matrix(raw, base[local], locality.rows[row_number][local])
        weights = np.ones(len(local), dtype=np.float32)
        weights[labels] = float(training_config["positive_weight_numerator"]) / max(
            1, len(bundle.relevant[row_number])
        )
        if training:
            hard = np.argsort(-base[local], kind="stable")[
                : int(training_config["hard_negatives_per_query"])
            ]
            random = rng.choice(
                len(local),
                min(int(training_config["random_candidates_per_query"]), len(local)),
                replace=False,
            )
            keep = np.unique(np.concatenate([hard, random, np.flatnonzero(labels)]))
            matrix = matrix[keep]
            labels = labels[keep]
            weights = weights[keep]
        chunks.append(matrix)
        label_chunks.append(labels.astype(np.uint8))
        weight_chunks.append(weights)
        positions_by_row.append(positions)
        base_by_row.append(base[local])
        target_counts.append(len(bundle.relevant[row_number]))
    matrix = np.concatenate(chunks) if chunks else np.empty((0, len(RERANK_FEATURE_NAMES)), np.float32)
    labels = np.concatenate(label_chunks) if label_chunks else np.empty(0, np.uint8)
    weights = np.concatenate(weight_chunks) if weight_chunks else np.empty(0, np.float32)
    if matrix.shape[1] != len(RERANK_FEATURE_NAMES):
        raise AssertionError("Unexpected reranker feature width")
    return PreparedFeatures(
        matrix,
        labels,
        weights,
        positions_by_row,
        base_by_row,
        target_counts,
        sampled=training,
    )


def fit_hist_gradient_boosting(
    prepared: PreparedFeatures, config: dict[str, Any]
) -> HistGradientBoostingClassifier:
    model_config = config["model"]
    model = HistGradientBoostingClassifier(
        max_leaf_nodes=int(model_config["max_leaf_nodes"]),
        max_iter=int(model_config["max_iter"]),
        learning_rate=float(model_config["learning_rate"]),
        min_samples_leaf=int(model_config["min_samples_leaf"]),
        l2_regularization=float(model_config["l2_regularization"]),
        early_stopping=bool(model_config["early_stopping"]),
        random_state=int(model_config["random_state"]),
    )
    with threadpool_limits(limits=int(model_config["thread_limit"])):
        model.fit(prepared.matrix, prepared.labels, sample_weight=prepared.weights)
    return model


def rank_prepared(
    model: HistGradientBoostingClassifier,
    prepared: PreparedFeatures,
    item_ids: np.ndarray,
    relevant: list[list[str]],
    *,
    blend: float,
    top_k: int,
) -> tuple[list[list[str]], np.ndarray]:
    if prepared.sampled:
        raise ValueError("Sampled training features cannot be used for full-candidate ranking")
    if blend == 0:
        logits = np.zeros(len(prepared.matrix), dtype=np.float64)
    else:
        logits = model.decision_function(prepared.matrix)
    predictions: list[list[str]] = []
    recalls = np.zeros(len(prepared.positions), dtype=np.float64)
    offset = 0
    for row_number, (positions, base) in enumerate(
        zip(prepared.positions, prepared.base_scores, strict=True)
    ):
        length = len(positions)
        scores = base + blend * logits[offset : offset + length]
        order = np.lexsort((positions, -scores))[:top_k]
        selected = [str(item_ids[pos]) for pos in positions[order]]
        if len(selected) < top_k:
            seen = set(selected)
            for item_id in item_ids:
                candidate = str(item_id)
                if candidate not in seen:
                    selected.append(candidate)
                    seen.add(candidate)
                if len(selected) == top_k:
                    break
        predictions.append(selected)
        truth = set(relevant[row_number])
        if truth:
            recalls[row_number] = len(set(selected) & truth) / len(truth)
        offset += length
    if offset != len(logits):
        raise AssertionError("Reranker score offsets do not cover the prepared matrix")
    return predictions, recalls


def select_round3_units(
    train: pd.DataFrame,
    fold: int,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Select the preregistered holdout after removing every consumed text cluster."""
    evaluation = config["evaluation"]
    if fold not in {int(value) for value in evaluation["folds"]}:
        raise ValueError(f"Fold {fold} is not configured for round-3 evaluation")
    limit = int(evaluation["units_per_fold"])
    modulo = int(config["training"]["fold_modulo"])
    # The earlier lexical protocol used seed 20260927 + fold. Reusing that seed
    # after excluding consumed clusters makes the new sample deterministic.
    prior_seed = int(evaluation["selection_seed_base"]) + fold
    all_units = make_query_units(train, fold, modulo)
    consumed = select_fold_units(
        train, fold, modulo, limit, prior_seed, "standard-v1"
    )
    consumed_texts = set(consumed["search_query"].map(normalize_text))
    if fold == 3:
        earlier_fresh = select_fold_units(
            train, fold, modulo, limit, prior_seed, "fresh-text-v2"
        )
        consumed_texts.update(earlier_fresh["search_query"].map(normalize_text))
    remaining = all_units[
        ~all_units["search_query"].map(normalize_text).isin(consumed_texts)
    ]
    selected = limit_units(remaining, limit, prior_seed)
    selected_texts = set(selected["search_query"].map(normalize_text))
    if selected_texts & consumed_texts:
        raise AssertionError("Round-3 sample retains a consumed text cluster")
    if len(selected) < limit:
        raise ValueError(
            f"Fold {fold} has only {len(selected)} units after consumed-cluster exclusion"
        )
    return selected


def _reranker_cache_root(
    data_dir: str | Path,
    cache_dir: str | Path,
    base_config: dict[str, Any],
    settings: dict[str, Any],
    config: dict[str, Any],
) -> tuple[Path, str]:
    root = Path(data_dir)
    fingerprint = data_fingerprint(
        [root / "train.parquet", root / "benchmark_items.parquet"],
        {
            "reranker_cache_format": 1,
            "base_config": base_config,
            "selected_settings": settings,
            "reranker_config": config,
            "source_sha256": source_fingerprint(Path(__file__).parent),
        },
    )
    return Path(cache_dir) / "reranker" / fingerprint, fingerprint


def _training_units(train: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    training = config["training"]
    return select_fold_units(
        train,
        int(training["fold"]),
        int(training["fold_modulo"]),
        int(training["limit"]),
        int(training["unit_seed"]),
        str(training["sample_version"]),
    )


def validate_fold_exclusions(
    train_fit: pd.DataFrame,
    excluded_folds: set[int],
    modulo: int,
    *,
    context: str,
) -> list[int]:
    present = sorted(
        {
            stable_fold(value, modulo)
            for value in train_fit["search_query"].drop_duplicates()
        }
    )
    overlap = set(present) & excluded_folds
    if overlap:
        raise AssertionError(f"{context} contains excluded folds: {sorted(overlap)}")
    expected = set(range(modulo)) - excluded_folds
    if set(present) != expected:
        raise AssertionError(
            f"{context} fold set {present} does not equal expected {sorted(expected)}"
        )
    return present


def _offline_model(
    train: pd.DataFrame,
    benchmark_items: pd.DataFrame,
    base_config: dict[str, Any],
    settings: dict[str, Any],
    config: dict[str, Any],
    cache_root: Path,
    *,
    rebuild: bool,
) -> tuple[HybridRetriever, pd.DataFrame]:
    training = config["training"]
    excluded = set(map(int, training["excluded_behavior_folds"]))
    train_fit = training_without_folds(
        train, excluded, int(training["fold_modulo"])
    )
    validate_fold_exclusions(
        train_fit,
        excluded,
        int(training["fold_modulo"]),
        context="reranker training history",
    )
    model_path = cache_root / "offline-model.joblib"
    if model_path.exists() and not rebuild:
        LOGGER.info("Loading fixed reranker catalog from %s", model_path)
        return HybridRetriever.load(model_path), train_fit
    LOGGER.info("Building fixed reranker catalog")
    model_config = copy.deepcopy(base_config)
    model_config["max_geo_neighbors"] = int(settings["geo_neighbors"])
    model_config["microcat_neighbors"] = int(
        config["features"]["query_locality_neighbors"]
    )
    corpus = evaluation_corpus(benchmark_items, train)
    model = HybridRetriever(model_config).fit(corpus, train_fit).enable_improvement(train_fit)
    model.save(model_path)
    return model, train_fit


def _bundle_for_units(
    model: HybridRetriever,
    train_fit: pd.DataFrame,
    units: pd.DataFrame,
    settings: dict[str, Any],
    fold: int,
    cache_path: Path,
    *,
    rebuild: bool,
) -> PackedFeatureBundle:
    if cache_path.exists() and not rebuild:
        bundle = joblib.load(cache_path)
        if bundle.relevant != units["relevant_items"].tolist():
            raise ValueError(f"Cached bundle does not match units: {cache_path}")
        return bundle
    model.refit_behavior(train_fit)
    rows = model.retrieve_improvement_features(
        units[QUERY_COLUMNS],
        max_pool=int(settings["pool"]),
        geo_neighbor_options=(int(settings["geo_neighbors"]),),
    )
    bundle = pack_feature_rows(
        model,
        rows,
        units,
        train_fit,
        fold,
        (int(settings["geo_neighbors"]),),
    )
    _atomic_joblib_dump(bundle, cache_path)
    return bundle


def train_or_load_reranker(
    train: pd.DataFrame,
    benchmark_items: pd.DataFrame,
    base_config: dict[str, Any],
    settings: dict[str, Any],
    config: dict[str, Any],
    *,
    data_dir: str | Path,
    cache_dir: str | Path,
    rebuild: bool = False,
) -> tuple[dict[str, Any], Path, str]:
    started = time.perf_counter()
    cache_root, fingerprint = _reranker_cache_root(
        data_dir, cache_dir, base_config, settings, config
    )
    artifact_path = cache_root / "reranker.joblib"
    units = _training_units(train, config)
    expected_unit_hash = unit_hash(units)
    preregistered_hash = str(config["training"]["expected_unit_sha256"])
    if expected_unit_hash != preregistered_hash:
        raise AssertionError(
            "Frozen reranker training units do not match the preregistered hash"
        )
    if artifact_path.exists() and not rebuild:
        artifact = joblib.load(artifact_path)
        if artifact.get("training_unit_sha256") != expected_unit_hash:
            raise ValueError("Cached reranker training units do not match current inputs")
        if artifact.get("feature_names") != list(RERANK_FEATURE_NAMES):
            raise ValueError("Cached reranker feature schema is incompatible")
        LOGGER.info("Loaded frozen reranker from %s", artifact_path)
        return artifact, cache_root, fingerprint

    model, train_fit = _offline_model(
        train,
        benchmark_items,
        base_config,
        settings,
        config,
        cache_root,
        rebuild=rebuild,
    )
    bundle = _bundle_for_units(
        model,
        train_fit,
        units,
        settings,
        int(config["training"]["fold"]),
        cache_root / "train2400.joblib",
        rebuild=rebuild,
    )
    locality = build_locality_features(
        model,
        train_fit,
        units,
        bundle,
        config,
        require_disjoint=True,
    )
    prepared = prepare_features(bundle, locality, settings, config, training=True)
    LOGGER.info(
        "Training frozen HGB: rows=%d positives=%d features=%d",
        len(prepared.matrix),
        int(prepared.labels.sum()),
        prepared.matrix.shape[1],
    )
    ranker = fit_hist_gradient_boosting(prepared, config)
    artifact = {
        "format": "avito-hgb-reranker-v1",
        "model": ranker,
        "config": copy.deepcopy(config),
        "settings": copy.deepcopy(settings),
        "feature_names": list(RERANK_FEATURE_NAMES),
        "training_unit_sha256": expected_unit_hash,
        "training_rows": len(prepared.matrix),
        "training_positives": int(prepared.labels.sum()),
        "training_fit_rows": len(train_fit),
        "training_excluded_folds": sorted(
            map(int, config["training"]["excluded_behavior_folds"])
        ),
        "training_fit_text_groups": locality.fit_text_groups,
        "training_fit_heldout_text_overlap": locality.fit_heldout_text_overlap,
        "cache_fingerprint": fingerprint,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_joblib_dump(artifact, artifact_path)
    LOGGER.info("Saved frozen reranker to %s", artifact_path)
    return artifact, cache_root, fingerprint


def predict_reranked(
    model: HybridRetriever,
    train: pd.DataFrame,
    queries: pd.DataFrame,
    settings: dict[str, Any],
    reranker: dict[str, Any],
    *,
    top_k: int,
) -> list[list[str]]:
    config = reranker["config"]
    geo = int(settings["geo_neighbors"])
    rows = model.retrieve_improvement_features(
        queries,
        max_pool=int(settings["pool"]),
        geo_neighbor_options=(geo,),
    )
    units = queries.copy()
    units["relevant_items"] = [[] for _ in range(len(units))]
    bundle = pack_feature_rows(
        model,
        rows,
        units,
        pd.DataFrame({"item_id": []}),
        fold=-1,
        geo_options=(geo,),
    )
    locality = build_locality_features(
        model,
        train,
        queries,
        bundle,
        config,
        require_disjoint=False,
    )
    prepared = prepare_features(bundle, locality, settings, config, training=False)
    predictions, _ = rank_prepared(
        reranker["model"],
        prepared,
        bundle.item_ids,
        bundle.relevant,
        blend=float(config["blend"]),
        top_k=top_k,
    )
    return predictions


def _candidate_hash(prepared: PreparedFeatures) -> str:
    digest = hashlib.sha256()
    for positions in prepared.positions:
        digest.update(np.asarray(positions, dtype="<i4").tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


def _cluster_bootstrap(
    delta: np.ndarray,
    normalized_queries: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> list[float]:
    frame = pd.DataFrame({"key": normalized_queries, "delta": delta})
    grouped = frame.groupby("key", sort=True)["delta"].agg(["sum", "size"])
    sums = grouped["sum"].to_numpy(dtype=np.float64)
    sizes = grouped["size"].to_numpy(dtype=np.int64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for draw in range(samples):
        chosen = rng.integers(0, len(grouped), size=len(grouped))
        estimates[draw] = sums[chosen].sum() / sizes[chosen].sum()
    return np.quantile(estimates, [0.025, 0.975]).tolist()


def _pooled_cluster_bootstrap(
    values: list[tuple[np.ndarray, np.ndarray]],
    *,
    samples: int,
    seed: int,
) -> list[float]:
    grouped_values: list[tuple[np.ndarray, np.ndarray]] = []
    for delta, normalized_queries in values:
        frame = pd.DataFrame({"key": normalized_queries, "delta": delta})
        grouped = frame.groupby("key", sort=True)["delta"].agg(["sum", "size"])
        grouped_values.append(
            (
                grouped["sum"].to_numpy(dtype=np.float64),
                grouped["size"].to_numpy(dtype=np.int64),
            )
        )
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for draw in range(samples):
        numerator = 0.0
        denominator = 0
        for sums, sizes in grouped_values:
            chosen = rng.integers(0, len(sums), size=len(sums))
            numerator += float(sums[chosen].sum())
            denominator += int(sizes[chosen].sum())
        estimates[draw] = numerator / denominator
    return np.quantile(estimates, [0.025, 0.975]).tolist()


def _fold_comparison(
    model: HybridRetriever,
    ranker: dict[str, Any],
    bundle: PackedFeatureBundle,
    units: pd.DataFrame,
    train_fit: pd.DataFrame,
    settings: dict[str, Any],
    config: dict[str, Any],
    *,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    locality = build_locality_features(
        model,
        train_fit,
        units,
        bundle,
        config,
        require_disjoint=True,
    )
    prepared = prepare_features(bundle, locality, settings, config, training=False)
    baseline_predictions, baseline = rank_prepared(
        ranker["model"],
        prepared,
        bundle.item_ids,
        bundle.relevant,
        blend=0.0,
        top_k=50,
    )
    predictions, enhanced = rank_prepared(
        ranker["model"],
        prepared,
        bundle.item_ids,
        bundle.relevant,
        blend=float(config["blend"]),
        top_k=50,
    )
    if len(baseline_predictions) != len(predictions):
        raise AssertionError("Baseline and reranker query counts differ")
    # Both ranks use exactly the positions selected by the immutable candidate mask.
    candidate_ids_equal = all(
        np.array_equal(
            row.positions[np.flatnonzero(score_packed_row(row, settings)[1])],
            prepared.positions[row_number],
        )
        for row_number, row in enumerate(bundle.rows)
    )
    delta = enhanced - baseline
    slice_minimum = int(config["evaluation"]["slice_minimum_queries"])
    slice_definitions = dict(bundle.slices)
    one_token = units["search_query"].map(
        lambda value: len(normalize_text(value).split()) <= 1
    ).to_numpy(dtype=bool)
    one_relevant = units["relevant_items"].map(
        lambda values: len(set(values)) == 1
    ).to_numpy(dtype=bool)
    slice_definitions.update(
        {
            "query_one_token": one_token,
            "query_multi_token": ~one_token,
            "relevant_one": one_relevant,
            "relevant_multiple": ~one_relevant,
        }
    )
    slices: dict[str, dict[str, float | int]] = {}
    for name, mask in slice_definitions.items():
        selected = np.asarray(mask, dtype=bool)
        slices[name] = {
            "queries": int(selected.sum()),
            "guarded": bool(int(selected.sum()) >= slice_minimum),
            "baseline_recall_at_50": float(baseline[selected].mean()),
            "enhanced_recall_at_50": float(enhanced[selected].mean()),
            "delta": float(delta[selected].mean()),
        }
    oracle = np.zeros(len(bundle.rows), dtype=np.float64)
    for row_number, positions in enumerate(prepared.positions):
        truth = set(bundle.relevant[row_number])
        if truth:
            candidate_ids = {str(bundle.item_ids[pos]) for pos in positions}
            oracle[row_number] = len(candidate_ids & truth) / len(truth)
    normalized_queries = units["search_query"].map(normalize_text).to_numpy(dtype=str)
    report = {
        "queries": len(units),
        "clusters": len(set(normalized_queries)),
        "unit_sha256": unit_hash(units),
        "candidate_universe_sha256": _candidate_hash(prepared),
        "candidate_ids_equal": candidate_ids_equal,
        "candidate_rows": int(sum(map(len, prepared.positions))),
        "baseline_candidate_oracle_recall": float(oracle.mean()),
        "enhanced_candidate_oracle_recall": float(oracle.mean()),
        "candidate_oracle_per_unit_equal": True,
        "baseline_recall_at_50": float(baseline.mean()),
        "enhanced_recall_at_50": float(enhanced.mean()),
        "delta": float(delta.mean()),
        "wins": int((delta > 0).sum()),
        "losses": int((delta < 0).sum()),
        "ties": int((delta == 0).sum()),
        "cluster_bootstrap_95_ci": _cluster_bootstrap(
            delta,
            normalized_queries,
            samples=int(config["evaluation"]["bootstrap_samples"]),
            seed=bootstrap_seed,
        ),
        "slices": slices,
        "locality": {
            "fit_rows": locality.fit_rows,
            "fit_text_groups": locality.fit_text_groups,
            "heldout_text_groups": locality.heldout_text_groups,
            "fit_heldout_text_overlap": locality.fit_heldout_text_overlap,
            "global_exact_location_rate": locality.global_exact_location_rate,
        },
    }
    return report, delta, normalized_queries


def run_reranker_evaluation(
    train: pd.DataFrame,
    benchmark_items: pd.DataFrame,
    base_config: dict[str, Any],
    settings: dict[str, Any],
    config: dict[str, Any],
    *,
    data_dir: str | Path,
    cache_dir: str | Path,
    rebuild: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    ranker, cache_root, fingerprint = train_or_load_reranker(
        train,
        benchmark_items,
        base_config,
        settings,
        config,
        data_dir=data_dir,
        cache_dir=cache_dir,
        rebuild=rebuild,
    )
    model, _ = _offline_model(
        train,
        benchmark_items,
        base_config,
        settings,
        config,
        cache_root,
        rebuild=False,
    )
    evaluation = config["evaluation"]
    modulo = int(config["training"]["fold_modulo"])
    comparisons: dict[str, Any] = {}
    pooled_values: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in map(int, evaluation["folds"]):
        units = select_round3_units(train, fold, config)
        expected_hash = str(evaluation["expected_unit_sha256"][str(fold)])
        actual_hash = unit_hash(units)
        if actual_hash != expected_hash:
            raise AssertionError(
                f"Fold {fold} units do not match preregistered hash: {actual_hash}"
            )
        excluded = set(map(int, evaluation["behavior_exclusions"][str(fold)]))
        train_fit = training_without_folds(train, excluded, modulo)
        present_folds = validate_fold_exclusions(
            train_fit,
            excluded,
            modulo,
            context=f"reranker evaluation fold {fold} history",
        )
        bundle = _bundle_for_units(
            model,
            train_fit,
            units,
            settings,
            fold,
            cache_root / f"fold-{fold}-fresh-text-round3-v1-queries-{len(units)}.joblib",
            rebuild=rebuild,
        )
        comparison, delta, normalized_queries = _fold_comparison(
            model,
            ranker,
            bundle,
            units,
            train_fit,
            settings,
            config,
            bootstrap_seed=int(evaluation["bootstrap_seeds"][str(fold)]),
        )
        comparison["excluded_behavior_folds"] = sorted(excluded)
        comparison["fit_fold_values"] = present_folds
        comparisons[str(fold)] = comparison
        pooled_values.append((delta, normalized_queries))

    pooled_delta = float(
        np.concatenate([values[0] for values in pooled_values]).mean()
    )
    pooled_ci = _pooled_cluster_bootstrap(
        pooled_values,
        samples=int(evaluation["bootstrap_samples"]),
        seed=int(evaluation["bootstrap_seeds"]["pooled"]),
    )
    slice_floor = float(evaluation["slice_minimum_delta"])
    slice_deltas = [
        float(details["delta"])
        for comparison in comparisons.values()
        for details in comparison["slices"].values()
        if bool(details["guarded"])
    ]
    gate = {
        "positive_delta_each_fold": all(
            float(comparison["delta"]) > 0 for comparison in comparisons.values()
        ),
        "pooled_delta_at_least_minimum": pooled_delta
        >= float(evaluation["minimum_pooled_delta"]),
        "pooled_cluster_ci_lower_above_zero": float(pooled_ci[0]) > 0,
        "all_fixed_slices_above_floor": bool(slice_deltas)
        and min(slice_deltas) >= slice_floor,
        "candidate_ids_equal": all(
            bool(comparison["candidate_ids_equal"])
            for comparison in comparisons.values()
        ),
        "unit_hashes_match": all(
            comparisons[str(fold)]["unit_sha256"]
            == evaluation["expected_unit_sha256"][str(fold)]
            for fold in map(int, evaluation["folds"])
        ),
    }
    gate["pass"] = all(gate.values())
    return {
        "enabled": bool(gate["pass"]),
        "config": copy.deepcopy(config),
        "feature_names": list(RERANK_FEATURE_NAMES),
        "training": {
            key: ranker[key]
            for key in (
                "training_unit_sha256",
                "training_rows",
                "training_positives",
                "training_fit_rows",
                "training_fit_text_groups",
                "training_fit_heldout_text_overlap",
                "training_excluded_folds",
            )
        },
        "folds": comparisons,
        "pooled": {
            "queries": sum(int(value["queries"]) for value in comparisons.values()),
            "delta": pooled_delta,
            "cluster_bootstrap_95_ci": pooled_ci,
        },
        "gate": gate,
        "cache_fingerprint": fingerprint,
        "elapsed_seconds": time.perf_counter() - started,
    }
