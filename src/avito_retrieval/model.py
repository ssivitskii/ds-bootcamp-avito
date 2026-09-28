from __future__ import annotations

import math
import os
import logging
import time
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .text import (
    item_char_document,
    item_filter_document,
    item_word_document,
    normalize_text,
    query_document,
    query_filter_document,
    stem_tokenize,
)

LOGGER = logging.getLogger(__name__)


class MetadataPrior:
    """Train-only compatibility priors; no category or location is a hard filter."""

    def __init__(self) -> None:
        self.category: dict[tuple[int, int], float] = {}
        self.location: dict[tuple[int, int], float] = {}

    @staticmethod
    def _pair_prior(frame: pd.DataFrame, left: str, right: str) -> dict[tuple[int, int], float]:
        counts = frame.groupby([left, right], sort=True).size().rename("count").reset_index()
        if counts.empty:
            return {}
        maxima = counts.groupby(left)["count"].transform("max").clip(lower=1)
        scores = np.log1p(counts["count"].to_numpy()) / np.log1p(maxima.to_numpy())
        return {
            (int(lvalue), int(rvalue)): float(score)
            for lvalue, rvalue, score in zip(counts[left], counts[right], scores, strict=True)
        }

    def fit(self, train: pd.DataFrame) -> "MetadataPrior":
        self.category = self._pair_prior(train, "search_category", "item_category_id")
        self.location = self._pair_prior(train, "search_location_id", "item_location_id")
        return self


FeatureRow = dict[int, dict[str, float]]


class HybridRetriever:
    """Fielded sparse retrieval with train-only behavior and metadata signals."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.word_vectorizer: TfidfVectorizer | None = None
        self.char_vectorizer: TfidfVectorizer | None = None
        self.word_items: sparse.csr_matrix | None = None
        self.char_items: sparse.csr_matrix | None = None
        self.filter_items: sparse.csr_matrix | None = None
        self.title_items: sparse.csr_matrix | None = None
        self.word_items_t: sparse.csc_matrix | None = None
        self.char_items_t: sparse.csc_matrix | None = None
        self.filter_items_t: sparse.csc_matrix | None = None
        self.title_items_t: sparse.csc_matrix | None = None
        self.history_matrix: sparse.csr_matrix | None = None
        self.history_keys: list[str] = []
        self.history_key_to_pos: dict[str, int] = {}
        self.history_items: list[list[tuple[int, float]]] = []
        self.item_ids: np.ndarray | None = None
        self.item_id_to_pos: dict[str, int] = {}
        self.item_categories: np.ndarray | None = None
        self.item_locations: np.ndarray | None = None
        self.item_microcats: np.ndarray | None = None
        self.item_titles: np.ndarray | None = None
        self.positions_by_location: dict[int, np.ndarray] = {}
        self.compatible_locations: dict[int, list[int]] = {}
        self.metadata = MetadataPrior()
        self.popularity: np.ndarray | None = None
        self.fallback_positions: list[int] = []
        self.improvement_enabled = False
        self.microcat_keys: list[str] = []
        self.microcat_matrix: sparse.csr_matrix | None = None
        self.microcat_distributions: list[dict[int, float]] = []

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # Transposes are cheap views rebuilt once per process; omitting them avoids
        # duplicating hundreds of MB in the persistent joblib cache.
        for name in ("word_items_t", "char_items_t", "filter_items_t", "title_items_t"):
            state[name] = None
        return state

    def _ensure_improvement_transposes(self) -> None:
        assert self.word_items is not None
        assert self.char_items is not None
        assert self.filter_items is not None
        assert self.title_items is not None
        if self.word_items_t is None:
            self.word_items_t = self.word_items.T.tocsc(copy=False)
        if self.char_items_t is None:
            self.char_items_t = self.char_items.T.tocsc(copy=False)
        if self.filter_items_t is None:
            self.filter_items_t = self.filter_items.T.tocsc(copy=False)
        if self.title_items_t is None:
            self.title_items_t = self.title_items.T.tocsc(copy=False)

    def fit(self, items: pd.DataFrame, history: pd.DataFrame) -> "HybridRetriever":
        started = time.perf_counter()
        corpus = items.drop_duplicates("item_id", keep="first").copy()
        corpus["item_id"] = corpus["item_id"].astype(str)
        corpus = corpus.sort_values("item_id", kind="mergesort").reset_index(drop=True)
        self.item_ids = corpus["item_id"].to_numpy(dtype=str)
        self.item_id_to_pos = {item_id: pos for pos, item_id in enumerate(self.item_ids)}
        self.item_categories = corpus["item_category_id"].to_numpy(dtype=np.int64)
        self.item_locations = corpus["item_location_id"].to_numpy(dtype=np.int64)
        self.item_microcats = corpus["item_microcat_id"].to_numpy(dtype=np.int64)
        self.item_titles = corpus["item_title_raw"].fillna("").astype(str).to_numpy()
        self.positions_by_location = {
            int(location): np.flatnonzero(self.item_locations == location)
            for location in np.unique(self.item_locations)
        }

        params_chars = int(self.config["item_params_chars"])
        description_chars = int(self.config["description_chars"])
        word_docs = [
            item_word_document(row, description_chars, params_chars)
            for row in corpus.itertuples(index=False)
        ]
        char_docs = [item_char_document(row) for row in corpus.itertuples(index=False)]
        filter_docs = [
            item_filter_document(row, params_chars) for row in corpus.itertuples(index=False)
        ]
        LOGGER.info("Prepared field documents: items=%d seconds=%.1f", len(corpus), time.perf_counter() - started)

        self.word_vectorizer = TfidfVectorizer(
            tokenizer=stem_tokenize,
            token_pattern=None,
            lowercase=False,
            ngram_range=(1, 2),
            min_df=int(self.config["word_min_df"]),
            max_features=int(self.config["word_max_features"]),
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=int(self.config["char_min_df"]),
            max_features=int(self.config["char_max_features"]),
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.word_items = self.word_vectorizer.fit_transform(word_docs).tocsr()
        LOGGER.info(
            "Built word index: shape=%s nnz=%d seconds=%.1f",
            self.word_items.shape,
            self.word_items.nnz,
            time.perf_counter() - started,
        )
        self.char_items = self.char_vectorizer.fit_transform(char_docs).tocsr()
        LOGGER.info(
            "Built char index: shape=%s nnz=%d seconds=%.1f",
            self.char_items.shape,
            self.char_items.nnz,
            time.perf_counter() - started,
        )
        self.filter_items = self.word_vectorizer.transform(filter_docs).tocsr()
        LOGGER.info(
            "Built filter index: shape=%s nnz=%d seconds=%.1f",
            self.filter_items.shape,
            self.filter_items.nnz,
            time.perf_counter() - started,
        )

        self.refit_behavior(history)
        LOGGER.info(
            "Built history transfer: groups=%d seconds=%.1f",
            len(self.history_keys),
            time.perf_counter() - started,
        )
        return self

    def refit_behavior(self, history: pd.DataFrame) -> "HybridRetriever":
        """Refit train-only signals while reusing the expensive fixed lexical catalog."""
        self.metadata = MetadataPrior().fit(history)
        compatible: dict[int, list[tuple[float, int]]] = defaultdict(list)
        for (search_location, item_location), score in self.metadata.location.items():
            compatible[search_location].append((score, item_location))
        max_geo_neighbors = int(self.config.get("max_geo_neighbors", 3))
        self.compatible_locations = {
            search_location: [
                location
                for _, location in sorted(values, key=lambda pair: (-pair[0], pair[1]))[
                    :max_geo_neighbors
                ]
            ]
            for search_location, values in compatible.items()
        }
        self._fit_popularity(history)
        self._fit_history(history)
        if self.improvement_enabled:
            self._fit_microcat(history)
        return self

    def enable_improvement(self, history: pd.DataFrame) -> "HybridRetriever":
        """Build optional reranking-only indexes without changing legacy retrieval."""
        assert self.word_vectorizer is not None
        assert self.word_items is not None
        assert self.char_items is not None
        assert self.filter_items is not None
        assert self.item_titles is not None
        started = time.perf_counter()
        self.title_items = self.word_vectorizer.transform(
            [normalize_text(value) for value in self.item_titles]
        ).tocsr()
        self._ensure_improvement_transposes()
        self.improvement_enabled = True
        self._fit_microcat(history)
        LOGGER.info(
            "Built improvement indexes: title_nnz=%d microcat_groups=%d seconds=%.1f",
            self.title_items.nnz,
            len(self.microcat_keys),
            time.perf_counter() - started,
        )
        return self

    def _fit_microcat(self, history: pd.DataFrame) -> None:
        """Learn P(microcategory | normalized query) from every passed train row."""
        assert self.word_vectorizer is not None
        work = history[["search_query", "item_microcat_id"]].copy()
        work["key"] = work["search_query"].map(normalize_text)
        counts = (
            work.groupby(["key", "item_microcat_id"], sort=True)
            .size()
            .rename("count")
            .reset_index()
        )
        grouped: dict[str, dict[int, float]] = {}
        for key, group in counts.groupby("key", sort=True):
            total = float(group["count"].sum())
            grouped[str(key)] = {
                int(microcat): float(count) / total
                for microcat, count in group[["item_microcat_id", "count"]].itertuples(
                    index=False
                )
            }
        self.microcat_keys = sorted(grouped)
        self.microcat_distributions = [grouped[key] for key in self.microcat_keys]
        self.microcat_matrix = self.word_vectorizer.transform(self.microcat_keys).tocsr()

    def _fit_popularity(self, history: pd.DataFrame) -> None:
        assert self.item_ids is not None
        counts = history["item_id"].astype(str).value_counts()
        raw = np.array([float(counts.get(item_id, 0)) for item_id in self.item_ids])
        logged = np.log1p(raw)
        maximum = float(logged.max(initial=0.0))
        self.popularity = logged / maximum if maximum > 0 else logged
        fallback_count = max(int(self.config["fallback_items"]), int(self.config["top_k"]))
        self.fallback_positions = sorted(
            range(len(self.item_ids)),
            key=lambda pos: (-float(self.popularity[pos]), str(self.item_ids[pos])),
        )[:fallback_count]

    def _fit_history(self, history: pd.DataFrame) -> None:
        assert self.word_vectorizer is not None
        work = history[["search_query", "item_id"]].copy()
        work["key"] = work["search_query"].map(normalize_text)
        work["item_id"] = work["item_id"].astype(str)
        work = work[work["item_id"].isin(self.item_id_to_pos)]
        counts = work.groupby(["key", "item_id"], sort=True).size().rename("count").reset_index()
        limit = int(self.config["history_items_per_query"])
        grouped: dict[str, list[tuple[int, float]]] = {}
        for key, group in counts.groupby("key", sort=True):
            ordered = group.sort_values(["count", "item_id"], ascending=[False, True]).head(limit)
            max_count = max(int(ordered["count"].max()), 1)
            grouped[str(key)] = [
                (
                    self.item_id_to_pos[str(item_id)],
                    float(math.log1p(int(count)) / math.log1p(max_count)),
                )
                for item_id, count in ordered[["item_id", "count"]].itertuples(index=False)
            ]
        self.history_keys = sorted(grouped)
        self.history_key_to_pos = {key: pos for pos, key in enumerate(self.history_keys)}
        self.history_items = [grouped[key] for key in self.history_keys]
        self.history_matrix = self.word_vectorizer.transform(self.history_keys).tocsr()

    @staticmethod
    def _add(feature_rows: list[FeatureRow], row: int, pos: int, name: str, value: float) -> None:
        if value <= 0:
            return
        channels = feature_rows[row].setdefault(pos, {})
        channels[name] = max(channels.get(name, 0.0), float(value))

    @staticmethod
    def _add_rank(
        feature_rows: list[FeatureRow], row: int, pos: int, name: str, rank: int
    ) -> None:
        channels = feature_rows[row].setdefault(pos, {})
        channels[name] = min(channels.get(name, float("inf")), float(rank))

    @staticmethod
    def _top_indices(scores: np.ndarray, limit: int) -> np.ndarray:
        """Top positive positions with ascending corpus position as the exact tie-break."""
        positive = np.flatnonzero(scores > 0)
        if positive.size <= limit:
            chosen = positive
        else:
            threshold = np.partition(scores[positive], -limit)[-limit]
            above = positive[scores[positive] > threshold]
            tied = positive[scores[positive] == threshold]
            chosen = np.concatenate([above, tied[: limit - len(above)]])
        return np.array(sorted(chosen, key=lambda pos: (-float(scores[pos]), int(pos))), dtype=np.int64)

    def _nearest_channel(
        self,
        query_matrix: sparse.csr_matrix,
        item_matrix: sparse.csr_matrix,
        name: str,
        feature_rows: list[FeatureRow],
        *,
        limit: int | None = None,
        rank_name: str | None = None,
        item_matrix_t: sparse.csc_matrix | None = None,
    ) -> None:
        active_limit = min(
            int(limit or self.config["candidate_pool_per_channel"]), item_matrix.shape[0]
        )
        if active_limit <= 0:
            return
        transpose = item_matrix_t if item_matrix_t is not None else item_matrix.T
        batch_size = int(self.config["batch_size"])
        for start in range(0, query_matrix.shape[0], batch_size):
            score_block = (query_matrix[start : start + batch_size] @ transpose).toarray()
            for local_row, scores in enumerate(score_block):
                target_row = start + local_row
                for rank, pos in enumerate(self._top_indices(scores, active_limit), start=1):
                    self._add(feature_rows, target_row, int(pos), name, float(scores[pos]))
                    if rank_name is not None:
                        self._add_rank(feature_rows, target_row, int(pos), rank_name, rank)

    def _location_channel(
        self,
        queries: pd.DataFrame,
        query_matrix: sparse.csr_matrix,
        item_matrix: sparse.csr_matrix,
        name: str,
        feature_rows: list[FeatureRow],
        *,
        limit: int | None = None,
        neighbor_count: int | None = None,
        rank_name: str | None = None,
    ) -> None:
        """Add a lexical pool from plausible locations before metadata reranking."""
        assert self.item_ids is not None
        active_limit = int(limit or self.config["candidate_pool_per_channel"])
        for row_number, query in enumerate(queries.itertuples(index=False)):
            search_location = int(query.search_location_id)
            locations = self.compatible_locations.get(search_location, [search_location])
            if neighbor_count is not None:
                locations = locations[:neighbor_count]
            position_parts = [
                self.positions_by_location[location]
                for location in locations
                if location in self.positions_by_location
            ]
            if not position_parts:
                continue
            positions = np.unique(np.concatenate(position_parts))
            scores = (query_matrix[row_number] @ item_matrix[positions].T).toarray().ravel()
            for rank, local in enumerate(self._top_indices(scores, active_limit), start=1):
                item_pos = int(positions[local])
                self._add(
                    feature_rows,
                    row_number,
                    item_pos,
                    name,
                    float(scores[local]),
                )
                if rank_name is not None:
                    self._add_rank(feature_rows, row_number, item_pos, rank_name, rank)

    def _history_channel(
        self,
        queries: pd.DataFrame,
        query_word: sparse.csr_matrix,
        feature_rows: list[FeatureRow],
        *,
        marker_name: str | None = None,
    ) -> None:
        if not self.history_keys or self.history_matrix is None:
            return
        neighbors = min(int(self.config["history_neighbors"]), len(self.history_keys))
        score_matrix = query_word @ self.history_matrix.T
        for row_number, row in enumerate(queries.itertuples(index=False)):
            exact = self.history_key_to_pos.get(normalize_text(row.search_query))
            if exact is not None:
                for item_pos, click_score in self.history_items[exact]:
                    self._add(feature_rows, row_number, item_pos, "history", click_score)
                    if marker_name is not None:
                        self._add_rank(feature_rows, row_number, item_pos, marker_name, 1)
            scores = score_matrix.getrow(row_number).toarray().ravel()
            for history_pos in self._top_indices(scores, neighbors):
                similarity = float(scores[history_pos])
                for item_pos, click_score in self.history_items[int(history_pos)]:
                    self._add(
                        feature_rows,
                        row_number,
                        item_pos,
                        "history",
                        similarity * click_score,
                    )
                    if marker_name is not None:
                        self._add_rank(feature_rows, row_number, item_pos, marker_name, 1)

    def retrieve_features(self, queries: pd.DataFrame) -> list[FeatureRow]:
        if self.word_vectorizer is None or self.char_vectorizer is None:
            raise RuntimeError("Retriever must be fitted before prediction")
        assert self.word_items is not None
        assert self.char_items is not None
        assert self.filter_items is not None
        assert self.item_categories is not None
        assert self.item_locations is not None
        assert self.popularity is not None

        rows: list[FeatureRow] = [dict() for _ in range(len(queries))]
        query_word_docs = [query_document(row) for row in queries.itertuples(index=False)]
        query_filter_docs = [query_filter_document(row) for row in queries.itertuples(index=False)]
        query_word = self.word_vectorizer.transform(query_word_docs).tocsr()
        query_char = self.char_vectorizer.transform(query_word_docs).tocsr()
        query_filter = self.word_vectorizer.transform(query_filter_docs).tocsr()

        self._nearest_channel(query_word, self.word_items, "word", rows)
        self._nearest_channel(query_char, self.char_items, "char", rows)
        self._nearest_channel(query_filter, self.filter_items, "filter", rows)
        self._location_channel(queries, query_word, self.word_items, "word", rows)
        self._location_channel(queries, query_char, self.char_items, "char", rows)
        self._history_channel(queries, query_word, rows)

        for row_number, query in enumerate(queries.itertuples(index=False)):
            for pos in self.fallback_positions:
                self._add(rows, row_number, pos, "popularity", float(self.popularity[pos]))
            search_category = int(query.search_category)
            search_location = int(query.search_location_id)
            for pos, channels in rows[row_number].items():
                category_score = self.metadata.category.get(
                    (search_category, int(self.item_categories[pos])), 0.0
                )
                location_score = self.metadata.location.get(
                    (search_location, int(self.item_locations[pos])), 0.0
                )
                if category_score > 0:
                    channels["category"] = category_score
                if location_score > 0:
                    channels["location"] = location_score
        return rows

    def _microcat_scores(
        self, query_word: sparse.csr_matrix
    ) -> list[dict[int, float]]:
        if self.microcat_matrix is None or not self.microcat_keys:
            return [{} for _ in range(query_word.shape[0])]
        neighbor_count = min(
            int(self.config.get("microcat_neighbors", 16)), len(self.microcat_keys)
        )
        similarities = query_word @ self.microcat_matrix.T
        output: list[dict[int, float]] = []
        for row_number in range(query_word.shape[0]):
            scores = similarities.getrow(row_number).toarray().ravel()
            combined: defaultdict[int, float] = defaultdict(float)
            total_weight = 0.0
            for key_pos in self._top_indices(scores, neighbor_count):
                weight = float(scores[key_pos]) ** 2
                total_weight += weight
                for microcat, probability in self.microcat_distributions[int(key_pos)].items():
                    combined[microcat] += weight * probability
            output.append(
                {
                    microcat: score / total_weight
                    for microcat, score in combined.items()
                    if total_weight > 0
                }
            )
        return output

    def retrieve_improvement_features(
        self,
        queries: pd.DataFrame,
        *,
        max_pool: int,
        geo_neighbor_options: tuple[int, ...],
        extra_candidates: list[np.ndarray] | None = None,
    ) -> list[FeatureRow]:
        """Build one maximal candidate union with ranks and exact reranking features.

        ``extra_candidates`` holds per-query corpus positions from an external
        channel (the dense bi-encoder); they are always eligible, like history.
        """
        if not self.improvement_enabled:
            raise RuntimeError("Call enable_improvement before improved retrieval")
        self._ensure_improvement_transposes()
        assert self.word_vectorizer is not None
        assert self.char_vectorizer is not None
        assert self.word_items is not None and self.word_items_t is not None
        assert self.char_items is not None and self.char_items_t is not None
        assert self.filter_items is not None and self.filter_items_t is not None
        assert self.title_items_t is not None
        assert self.item_categories is not None
        assert self.item_locations is not None
        assert self.item_microcats is not None
        assert self.popularity is not None

        started = time.perf_counter()
        rows: list[FeatureRow] = [dict() for _ in range(len(queries))]
        query_word_docs = [query_document(row) for row in queries.itertuples(index=False)]
        query_filter_docs = [query_filter_document(row) for row in queries.itertuples(index=False)]
        query_word = self.word_vectorizer.transform(query_word_docs).tocsr()
        query_char = self.char_vectorizer.transform(query_word_docs).tocsr()
        query_filter = self.word_vectorizer.transform(query_filter_docs).tocsr()

        self._nearest_channel(
            query_word,
            self.word_items,
            "word",
            rows,
            limit=max_pool,
            rank_name="__global_word_rank",
            item_matrix_t=self.word_items_t,
        )
        self._nearest_channel(
            query_char,
            self.char_items,
            "char",
            rows,
            limit=max_pool,
            rank_name="__global_char_rank",
            item_matrix_t=self.char_items_t,
        )
        self._nearest_channel(
            query_filter,
            self.filter_items,
            "filter",
            rows,
            limit=max_pool,
            rank_name="__global_filter_rank",
            item_matrix_t=self.filter_items_t,
        )
        LOGGER.info(
            "Improvement global pools ready: queries=%d max_pool=%d seconds=%.1f",
            len(queries),
            max_pool,
            time.perf_counter() - started,
        )
        for geo_neighbors in sorted(set(geo_neighbor_options)):
            self._location_channel(
                queries,
                query_word,
                self.word_items,
                "word",
                rows,
                limit=max_pool,
                neighbor_count=geo_neighbors,
                rank_name=f"__geo{geo_neighbors}_word_rank",
            )
            self._location_channel(
                queries,
                query_char,
                self.char_items,
                "char",
                rows,
                limit=max_pool,
                neighbor_count=geo_neighbors,
                rank_name=f"__geo{geo_neighbors}_char_rank",
            )
            LOGGER.info(
                "Improvement geo pool ready: neighbors=%d seconds=%.1f",
                geo_neighbors,
                time.perf_counter() - started,
            )
        self._history_channel(
            queries, query_word, rows, marker_name="__history_source"
        )
        if extra_candidates is not None:
            if len(extra_candidates) != len(queries):
                raise ValueError("Extra candidates must align with queries")
            for row_number, positions in enumerate(extra_candidates):
                for pos in positions:
                    self._add_rank(rows, row_number, int(pos), "__dense_source", 1)

        for row_number, query in enumerate(queries.itertuples(index=False)):
            for pos in self.fallback_positions:
                popularity = float(self.popularity[pos])
                self._add(rows, row_number, pos, "popularity", popularity)
                if popularity > 0:
                    self._add_rank(rows, row_number, pos, "__fallback_source", 1)
            search_category = int(query.search_category)
            search_location = int(query.search_location_id)
            for pos, channels in rows[row_number].items():
                category_score = self.metadata.category.get(
                    (search_category, int(self.item_categories[pos])), 0.0
                )
                location_score = self.metadata.location.get(
                    (search_location, int(self.item_locations[pos])), 0.0
                )
                if category_score > 0:
                    channels["category"] = category_score
                if location_score > 0:
                    channels["location"] = location_score

        microcat_scores = self._microcat_scores(query_word)
        LOGGER.info(
            "Improvement transfer ready: candidates=%d seconds=%.1f",
            sum(len(row) for row in rows),
            time.perf_counter() - started,
        )
        for row_number, channels_by_pos in enumerate(rows):
            if not channels_by_pos:
                continue
            positions = np.array(sorted(channels_by_pos), dtype=np.int64)
            exact_matrices = {
                "exact_word": query_word[row_number] @ self.word_items_t[:, positions],
                "exact_char": query_char[row_number] @ self.char_items_t[:, positions],
                "exact_filter": query_filter[row_number] @ self.filter_items_t[:, positions],
                "title": query_word[row_number] @ self.title_items_t[:, positions],
            }
            for name, matrix in exact_matrices.items():
                values = np.asarray(matrix.toarray()).ravel()
                for local, value in enumerate(values):
                    self._add(rows, row_number, int(positions[local]), name, float(value))
            transferred = microcat_scores[row_number]
            for pos in positions:
                value = transferred.get(int(self.item_microcats[pos]), 0.0)
                self._add(rows, row_number, int(pos), "microcat", value)
        LOGGER.info(
            "Built maximal improvement features: queries=%d candidates=%d seconds=%.1f",
            len(rows),
            sum(len(row) for row in rows),
            time.perf_counter() - started,
        )
        return rows

    def predict_from_features(
        self,
        feature_rows: list[FeatureRow],
        weights: dict[str, float] | None = None,
        top_k: int | None = None,
    ) -> list[list[str]]:
        assert self.item_ids is not None
        active_weights = weights or self.config["weights"]
        limit = int(top_k or self.config["top_k"])
        answers: list[list[str]] = []
        for row in feature_rows:
            scored = []
            for pos, channels in row.items():
                score = sum(float(active_weights.get(name, 0.0)) * value for name, value in channels.items())
                scored.append((score, str(self.item_ids[pos])))
            scored.sort(key=lambda pair: (-pair[0], pair[1]))
            selected = [item_id for _, item_id in scored[:limit]]
            if len(selected) < limit:
                seen = set(selected)
                for item_id in self.item_ids:
                    candidate = str(item_id)
                    if candidate not in seen:
                        selected.append(candidate)
                        seen.add(candidate)
                    if len(selected) == limit:
                        break
            answers.append(selected)
        return answers

    def predict(
        self, queries: pd.DataFrame, weights: dict[str, float] | None = None
    ) -> list[list[str]]:
        return self.predict_from_features(self.retrieve_features(queries), weights=weights)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            joblib.dump(self, temporary, compress=0)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> "HybridRetriever":
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Unexpected cache type: {type(model)!r}")
        return model
