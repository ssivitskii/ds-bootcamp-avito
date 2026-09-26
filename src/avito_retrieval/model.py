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
        self.history_matrix: sparse.csr_matrix | None = None
        self.history_keys: list[str] = []
        self.history_key_to_pos: dict[str, int] = {}
        self.history_items: list[list[tuple[int, float]]] = []
        self.item_ids: np.ndarray | None = None
        self.item_id_to_pos: dict[str, int] = {}
        self.item_categories: np.ndarray | None = None
        self.item_locations: np.ndarray | None = None
        self.positions_by_location: dict[int, np.ndarray] = {}
        self.compatible_locations: dict[int, list[int]] = {}
        self.metadata = MetadataPrior()
        self.popularity: np.ndarray | None = None
        self.fallback_positions: list[int] = []

    def fit(self, items: pd.DataFrame, history: pd.DataFrame) -> "HybridRetriever":
        started = time.perf_counter()
        corpus = items.drop_duplicates("item_id", keep="first").copy()
        corpus["item_id"] = corpus["item_id"].astype(str)
        corpus = corpus.sort_values("item_id", kind="mergesort").reset_index(drop=True)
        self.item_ids = corpus["item_id"].to_numpy(dtype=str)
        self.item_id_to_pos = {item_id: pos for pos, item_id in enumerate(self.item_ids)}
        self.item_categories = corpus["item_category_id"].to_numpy(dtype=np.int64)
        self.item_locations = corpus["item_location_id"].to_numpy(dtype=np.int64)
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
        self.compatible_locations = {
            search_location: [
                location
                for _, location in sorted(values, key=lambda pair: (-pair[0], pair[1]))[:3]
            ]
            for search_location, values in compatible.items()
        }
        self._fit_popularity(history)
        self._fit_history(history)
        return self

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
    ) -> None:
        limit = min(int(self.config["candidate_pool_per_channel"]), item_matrix.shape[0])
        if limit <= 0:
            return
        batch_size = int(self.config["batch_size"])
        for start in range(0, query_matrix.shape[0], batch_size):
            score_block = (query_matrix[start : start + batch_size] @ item_matrix.T).toarray()
            for local_row, scores in enumerate(score_block):
                target_row = start + local_row
                for pos in self._top_indices(scores, limit):
                    self._add(feature_rows, target_row, int(pos), name, float(scores[pos]))

    def _location_channel(
        self,
        queries: pd.DataFrame,
        query_matrix: sparse.csr_matrix,
        item_matrix: sparse.csr_matrix,
        name: str,
        feature_rows: list[FeatureRow],
    ) -> None:
        """Add a lexical pool from plausible locations before metadata reranking."""
        assert self.item_ids is not None
        limit = int(self.config["candidate_pool_per_channel"])
        for row_number, query in enumerate(queries.itertuples(index=False)):
            search_location = int(query.search_location_id)
            locations = self.compatible_locations.get(search_location, [search_location])
            position_parts = [
                self.positions_by_location[location]
                for location in locations
                if location in self.positions_by_location
            ]
            if not position_parts:
                continue
            positions = np.unique(np.concatenate(position_parts))
            scores = (query_matrix[row_number] @ item_matrix[positions].T).toarray().ravel()
            for local in self._top_indices(scores, limit):
                self._add(
                    feature_rows,
                    row_number,
                    int(positions[local]),
                    name,
                    float(scores[local]),
                )

    def _history_channel(
        self, queries: pd.DataFrame, query_word: sparse.csr_matrix, feature_rows: list[FeatureRow]
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
