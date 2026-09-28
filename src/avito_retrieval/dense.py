"""Dense bi-encoder channel: fine-tuned sentence encoder, candidates and rerank features.

The encoder is fine-tuned on (query, clicked item) pairs from a fixed set of train
folds that excludes the reranker training fold and all evaluation folds.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .text import stable_fold

LOGGER = logging.getLogger(__name__)

DENSE_FEATURE_NAMES = (
    "dense_cos",
    "dense_cos_minus_global_top1",
    "dense_log_global_rank",
    "dense_cos_minus_local_top1",
    "dense_log_local_rank",
)
DENSE_TEXT_COLUMNS = ["item_title_raw", "item_infm_params_text", "item_description_raw"]


def _clean(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def query_texts(frame: pd.DataFrame) -> list[str]:
    query = _clean(frame["search_query"]).str.lower()
    params = _clean(frame["search_infm_params_text"])
    return [q if not p else f"{q} | {p}" for q, p in zip(query, params)]


def item_texts(frame: pd.DataFrame) -> list[str]:
    title = _clean(frame["item_title_raw"])
    params = _clean(frame["item_infm_params_text"]).str[:200]
    description = _clean(frame["item_description_raw"]).str[:300]
    return (title + " | " + params + " | " + description).tolist()


def _device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def train_encoder(train: pd.DataFrame, config: dict[str, Any], output: Path) -> Path:
    """Fine-tune the bi-encoder with in-batch negatives; returns the model directory."""
    import torch
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    model_dir = output / "model"
    if (model_dir / "config.json").exists():
        return model_dir
    folds = set(map(int, config["train_folds"]))
    mask = train["search_query"].map(lambda value: stable_fold(value, int(config["fold_modulo"])))
    rows = train.loc[mask.isin(folds)]
    frame = pd.DataFrame({"anchor": query_texts(rows), "positive": item_texts(rows)})
    frame = frame.drop_duplicates().sample(frac=1.0, random_state=int(config["seed"]))
    LOGGER.info("Fine-tuning dense encoder on %d pairs from folds %s", len(frame), sorted(folds))
    torch.manual_seed(int(config["seed"]))
    model = SentenceTransformer(str(config["base_model"]), device=_device())
    model.max_seq_length = int(config["max_seq_length"])
    args = SentenceTransformerTrainingArguments(
        output_dir=str(output / "checkpoints"),
        num_train_epochs=float(config["epochs"]),
        per_device_train_batch_size=int(config["batch_size"]),
        learning_rate=float(config["learning_rate"]),
        warmup_steps=0.05,
        lr_scheduler_type="linear",
        weight_decay=0.01,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=100,
        save_strategy="no",
        report_to=[],
        seed=int(config["seed"]),
        dataloader_num_workers=0,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_pandas(frame.reset_index(drop=True)),
        loss=MultipleNegativesRankingLoss(model, scale=20.0),
    )
    started = time.perf_counter()
    trainer.train()
    model.save(str(model_dir))
    (output / "training.json").write_text(
        json.dumps({"pairs": len(frame), "seconds": time.perf_counter() - started, "config": config}, indent=2)
    )
    return model_dir


@dataclass
class DenseSearch:
    """Per-query search results needed for candidates and features."""

    queries: np.ndarray
    search_locations: np.ndarray
    global_top: np.ndarray
    global_top_scores: np.ndarray
    local_top: list[np.ndarray]


class DenseIndex:
    """Normalized item embeddings aligned with ``HybridRetriever.item_ids``."""

    def __init__(
        self,
        encoder_dir: str | Path,
        item_ids: np.ndarray,
        embeddings: np.ndarray,
        item_locations: np.ndarray,
        config: dict[str, Any],
    ) -> None:
        if len(item_ids) != len(embeddings) or len(item_ids) != len(item_locations):
            raise ValueError("Dense index arrays must align")
        self.encoder_dir = str(encoder_dir)
        self.item_ids = np.asarray(item_ids)
        self.embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.item_locations = np.asarray(item_locations, dtype=np.int64)
        self.config = config
        order = np.argsort(self.item_locations, kind="stable")
        bounds = np.flatnonzero(np.diff(self.item_locations[order])) + 1
        self.location_items = {
            int(self.item_locations[group[0]]): group for group in np.split(order, bounds) if len(group)
        }
        self._encoder = None

    @classmethod
    def build(
        cls,
        encoder_dir: str | Path,
        corpus: pd.DataFrame,
        item_ids: np.ndarray,
        item_locations: np.ndarray,
        config: dict[str, Any],
        cache_path: Path,
    ) -> "DenseIndex":
        """Encode ``corpus`` rows in ``item_ids`` order (cached as float16)."""
        if cache_path.exists():
            payload = np.load(cache_path, allow_pickle=True)
            if np.array_equal(payload["item_ids"], np.asarray(item_ids).astype(str)):
                return cls(encoder_dir, item_ids, payload["embeddings"], item_locations, config)
            raise ValueError(f"Dense embedding cache item order differs: {cache_path}")
        frame = corpus.drop_duplicates("item_id", keep="first").set_index("item_id")
        frame = frame.loc[np.asarray(item_ids).astype(str), DENSE_TEXT_COLUMNS]
        texts = item_texts(frame)
        order = np.argsort([len(text) for text in texts], kind="stable")
        encoder = cls._load_encoder(encoder_dir, config)
        started = time.perf_counter()
        encoded = encoder.encode(
            [texts[index] for index in order],
            batch_size=512,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        embeddings = np.empty_like(encoded)
        embeddings[order] = encoded
        embeddings = embeddings.astype(np.float16)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp.npz")
        np.savez(temporary, item_ids=np.asarray(item_ids).astype(str), embeddings=embeddings)
        temporary.replace(cache_path)
        LOGGER.info("Encoded %d items in %.1fs", len(texts), time.perf_counter() - started)
        return cls(encoder_dir, item_ids, embeddings, item_locations, config)

    @staticmethod
    def _load_encoder(encoder_dir: str | Path, config: dict[str, Any]):
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(str(encoder_dir), device=_device())
        encoder.max_seq_length = int(config["max_seq_length"])
        return encoder

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_encoder"] = None
        return state

    def encode_queries(self, queries: pd.DataFrame) -> np.ndarray:
        if self._encoder is None:
            self._encoder = self._load_encoder(self.encoder_dir, self.config)
        return np.asarray(
            self._encoder.encode(
                query_texts(queries), batch_size=256, normalize_embeddings=True, show_progress_bar=False
            ),
            dtype=np.float32,
        )

    def _location_groups(self, search_locations: np.ndarray):
        order = np.argsort(search_locations, kind="stable")
        ordered = search_locations[order]
        for location in np.unique(search_locations):
            rows = order[np.searchsorted(ordered, location, "left") : np.searchsorted(ordered, location, "right")]
            yield rows, self.location_items.get(int(location))

    def search(self, queries: pd.DataFrame) -> DenseSearch:
        vectors = self.encode_queries(queries)
        locations = queries["search_location_id"].to_numpy(np.int64)
        depth = min(int(self.config["global_rank_depth"]), len(self.embeddings))
        global_top = np.empty((len(vectors), depth), dtype=np.int64)
        global_scores = np.empty((len(vectors), depth), dtype=np.float32)
        for start in range(0, len(vectors), 256):
            scores = vectors[start : start + 256] @ self.embeddings.T
            top = np.argpartition(-scores, depth - 1, axis=1)[:, :depth]
            top_scores = np.take_along_axis(scores, top, axis=1)
            order = np.lexsort((top, -top_scores), axis=1)
            global_top[start : start + 256] = np.take_along_axis(top, order, axis=1)
            global_scores[start : start + 256] = np.take_along_axis(top_scores, order, axis=1)
        local_limit = int(self.config["local_candidates"])
        local_top: list[np.ndarray] = [np.empty(0, dtype=np.int64)] * len(vectors)
        for rows, items in self._location_groups(locations):
            if items is None:
                continue
            block = self.embeddings[items]
            for start in range(0, len(rows), 128):
                chunk = rows[start : start + 128]
                scores = vectors[chunk] @ block.T
                for row, values in zip(chunk, scores):
                    limit = min(local_limit, len(values))
                    top = np.argpartition(-values, limit - 1)[:limit]
                    top = top[np.lexsort((items[top], -values[top]))]
                    local_top[row] = items[top]
        return DenseSearch(vectors, locations, global_top, global_scores, local_top)

    def candidates(self, search: DenseSearch) -> list[np.ndarray]:
        global_limit = int(self.config["global_candidates"])
        return [
            np.union1d(local, search.global_top[row, :global_limit])
            for row, local in enumerate(search.local_top)
        ]

    def features(self, search: DenseSearch, positions_by_row: list[np.ndarray]) -> list[np.ndarray]:
        """Five features per candidate, in ``DENSE_FEATURE_NAMES`` order."""
        depth = search.global_top.shape[1]
        local_sorted: list[np.ndarray | None] = [None] * len(positions_by_row)
        for rows, items in self._location_groups(search.search_locations):
            if items is None:
                continue
            block = self.embeddings[items]
            for start in range(0, len(rows), 128):
                chunk = rows[start : start + 128]
                scores = search.queries[chunk] @ block.T
                scores.sort(axis=1)
                for row, values in zip(chunk, scores):
                    local_sorted[row] = values
        output: list[np.ndarray] = []
        for row, positions in enumerate(positions_by_row):
            cosine = self.embeddings[positions] @ search.queries[row]
            lookup = {int(pos): rank for rank, pos in enumerate(search.global_top[row])}
            global_rank = np.array([lookup.get(int(pos), depth) for pos in positions], dtype=np.float32)
            local = local_sorted[row]
            if local is None:
                local_rank = np.full(len(positions), 1e5, dtype=np.float32)
                local_top1 = np.float32(-1.0)
            else:
                local_rank = (len(local) - np.searchsorted(local, cosine, "right")).astype(np.float32)
                local_top1 = local[-1]
            output.append(
                np.column_stack(
                    [
                        cosine,
                        cosine - search.global_top_scores[row, 0],
                        np.log1p(global_rank),
                        cosine - local_top1,
                        np.log1p(local_rank),
                    ]
                ).astype(np.float32)
            )
        return output
