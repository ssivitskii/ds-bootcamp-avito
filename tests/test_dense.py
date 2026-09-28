from __future__ import annotations

import copy
import json
import pickle
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from avito_retrieval.constants import ITEM_COLUMNS, QUERY_COLUMNS
from avito_retrieval.dense import DENSE_FEATURE_NAMES, DenseIndex, item_texts, query_texts
from avito_retrieval.improvement import pack_feature_rows
from avito_retrieval.model import HybridRetriever
from avito_retrieval.reranker import (
    DENSE_RERANK_FEATURE_NAMES,
    EXTENDED_RERANK_FEATURE_NAMES,
    ExtraFeatures,
    feature_names_for_config,
    validate_reranker_config,
    with_dense_features,
)

from test_improvement import model_config
from test_pipeline import click, item


def dense_config() -> dict[str, object]:
    return json.loads(Path("reranker_dense_config.json").read_text(encoding="utf-8"))


class FixedQueryIndex(DenseIndex):
    """Dense index with fixed query vectors, so no encoder is loaded."""

    def __init__(self, vectors: np.ndarray, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.vectors = vectors

    def encode_queries(self, queries: pd.DataFrame) -> np.ndarray:
        return self.vectors[: len(queries)]


def unit(vector: list[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    return array / np.linalg.norm(array)


class DenseIndexTest(unittest.TestCase):
    def index(self) -> FixedQueryIndex:
        embeddings = np.stack(
            [unit([1, 0]), unit([0.9, 0.1]), unit([0, 1]), unit([0.8, 0.2]), unit([0.5, 0.5])]
        )
        return FixedQueryIndex(
            np.stack([unit([1, 0])]),
            encoder_dir="unused",
            item_ids=np.array([f"{index:016x}" for index in range(5)]),
            embeddings=embeddings,
            item_locations=np.array([7, 7, 8, 8, 7]),
            config={"local_candidates": 2, "global_candidates": 1, "global_rank_depth": 3, "max_seq_length": 8},
        )

    def test_candidates_union_local_and_global_tops(self) -> None:
        index = self.index()
        queries = pd.DataFrame([["ремонт", 8, 0, "", 114]], columns=QUERY_COLUMNS)
        search = index.search(queries)
        np.testing.assert_array_equal(search.global_top[0], [0, 1, 3])
        # Location 8 holds items 2 and 3; the global top-1 (item 0) is added.
        np.testing.assert_array_equal(search.local_top[0], [3, 2])
        np.testing.assert_array_equal(index.candidates(search)[0], [0, 2, 3])

    def test_features_use_global_and_same_location_ranks(self) -> None:
        index = self.index()
        queries = pd.DataFrame([["ремонт", 8, 0, "", 114]], columns=QUERY_COLUMNS)
        search = index.search(queries)
        positions = np.array([0, 2, 3, 4])
        (features,) = index.features(search, [positions])
        self.assertEqual(features.shape, (4, len(DENSE_FEATURE_NAMES)))
        cosine = index.embeddings[positions] @ search.queries[0]
        np.testing.assert_allclose(features[:, 0], cosine, rtol=1e-6)
        np.testing.assert_allclose(features[:, 1], cosine - cosine[0], rtol=1e-6)
        # Items 0 and 3 are ranked 0 and 2 globally; the others are beyond depth 3.
        np.testing.assert_allclose(features[:, 2], np.log1p([0, 3, 2, 3]))
        local_top1 = cosine[2]
        np.testing.assert_allclose(features[:, 3], cosine - local_top1, rtol=1e-6)
        # Same-location rank counts location-8 items with a strictly larger score.
        np.testing.assert_allclose(features[:, 4], np.log1p([0, 1, 0, 1]))

    def test_unknown_location_has_no_local_candidates(self) -> None:
        index = self.index()
        queries = pd.DataFrame([["ремонт", 99, 0, "", 114]], columns=QUERY_COLUMNS)
        search = index.search(queries)
        self.assertEqual(len(search.local_top[0]), 0)
        np.testing.assert_array_equal(index.candidates(search)[0], [0])
        (features,) = index.features(search, [np.array([0])])
        self.assertEqual(features[0, 3], features[0, 0] + 1.0)

    def test_pickle_drops_loaded_encoder(self) -> None:
        index = self.index()
        index._encoder = object()
        restored = pickle.loads(pickle.dumps(index))
        self.assertIsNone(restored._encoder)
        np.testing.assert_array_equal(restored.embeddings, index.embeddings)

    def test_texts_handle_missing_values(self) -> None:
        queries = pd.DataFrame(
            {"search_query": ["Ремонт  Холодильника"], "search_infm_params_text": [None]}
        )
        self.assertEqual(query_texts(queries), ["ремонт холодильника"])
        items = pd.DataFrame(
            {"item_title_raw": ["Мастер"], "item_infm_params_text": [None], "item_description_raw": ["x" * 400]}
        )
        self.assertEqual(item_texts(items), ["Мастер |  | " + "x" * 300])


class DenseCandidateContractTest(unittest.TestCase):
    def test_dense_candidates_are_additive_and_always_eligible(self) -> None:
        items = pd.DataFrame(
            [
                item("0000000000000001", "ремонт холодильника", 1),
                item("0000000000000002", "ремонт квартиры", 1),
                item("0000000000000003", "маникюр", 2),
            ],
            columns=ITEM_COLUMNS,
        )
        history = pd.DataFrame(
            [click("история услуг", items.iloc[0].to_dict(), 1)],
            columns=QUERY_COLUMNS + ITEM_COLUMNS,
        )
        model = HybridRetriever(model_config()).fit(items, history).enable_improvement(history)
        queries = pd.DataFrame([["ремонт", 1, False, "", 114]], columns=QUERY_COLUMNS)
        queries["relevant_items"] = [["0000000000000003"]]
        plain = model.retrieve_improvement_features(
            queries[QUERY_COLUMNS], max_pool=3, geo_neighbor_options=(3,)
        )
        extended = model.retrieve_improvement_features(
            queries[QUERY_COLUMNS],
            max_pool=3,
            geo_neighbor_options=(3,),
            extra_candidates=[np.array([2])],
        )
        self.assertNotIn(2, plain[0])
        self.assertIn(2, extended[0])
        for pos, channels in plain[0].items():
            self.assertEqual(channels, {k: v for k, v in extended[0][pos].items()})
        packed = pack_feature_rows(model, extended, queries, history, 2, (3,))
        row = packed.rows[0]
        self.assertTrue(row.always_source[list(row.positions).index(2)])
        with self.assertRaises(ValueError):
            model.retrieve_improvement_features(
                queries[QUERY_COLUMNS],
                max_pool=3,
                geo_neighbor_options=(3,),
                extra_candidates=[],
            )


class DenseConfigTest(unittest.TestCase):
    def test_dense_schema_appends_five_features(self) -> None:
        config = dense_config()
        validate_reranker_config(config)
        names = feature_names_for_config(config)
        self.assertEqual(names, DENSE_RERANK_FEATURE_NAMES)
        self.assertEqual(names[:47], EXTENDED_RERANK_FEATURE_NAMES)
        self.assertEqual(names[47:], DENSE_FEATURE_NAMES)

    def test_encoder_folds_must_exclude_reranker_and_evaluation_folds(self) -> None:
        for folds in ([0, 2], [1, 3], [4]):
            config = copy.deepcopy(dense_config())
            config["dense"]["train_folds"] = folds
            with self.assertRaises(ValueError):
                validate_reranker_config(config)
        config = copy.deepcopy(dense_config())
        del config["dense"]["local_candidates"]
        with self.assertRaises(ValueError):
            validate_reranker_config(config)

    def test_with_dense_features_requires_alignment(self) -> None:
        extra = ExtraFeatures([np.zeros((2, 17), np.float32)], 1, 1, 1.0, 1)
        combined = with_dense_features(extra, [np.ones((2, 5), np.float32)])
        self.assertEqual(combined.rows[0].shape, (2, 22))
        with self.assertRaises(ValueError):
            with_dense_features(extra, [np.ones((3, 5), np.float32)])
        with self.assertRaises(ValueError):
            with_dense_features(extra, [])


if __name__ == "__main__":
    unittest.main()
