from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from avito_retrieval.constants import ITEM_COLUMNS, QUERY_COLUMNS
from avito_retrieval.cli import _selection
from avito_retrieval.improvement import (
    FEATURE_INDEX,
    FEATURE_NAMES,
    RANK_MISSING,
    PackedFeatureBundle,
    PackedRow,
    pack_feature_rows,
)
from avito_retrieval.model import HybridRetriever
from avito_retrieval.reranker import (
    LOCALITY_FEATURE_NAMES,
    RERANK_FEATURE_NAMES,
    LocalityFeatures,
    PreparedFeatures,
    _offline_model,
    _training_units,
    build_locality_features,
    prepare_features,
    rank_prepared,
    select_round3_units,
    validate_fold_exclusions,
    validate_reranker_config,
)
from avito_retrieval.text import stable_fold

from test_improvement import model_config
from test_pipeline import click, item


def reranker_config() -> dict[str, object]:
    return json.loads(Path("reranker_config.json").read_text(encoding="utf-8"))


def settings() -> dict[str, object]:
    return {
        "exact_rescore": True,
        "pool": 3,
        "geo_neighbors": 3,
        "weights": {
            "word": 1.0,
            "char": 0.35,
            "filter": 0.0,
            "history": 0.0,
            "category": 0.05,
            "location": 0.7,
            "popularity": 0.03,
            "title": 0.0,
            "microcat": 0.4,
        },
    }


def query_for_fold(fold: int, start: int = 0) -> str:
    number = start
    while stable_fold(f"запрос {number}", 5) != fold:
        number += 1
    return f"запрос {number}"


class ConfigAndSamplingTest(unittest.TestCase):
    def test_report_auto_enables_only_gate_passed_reranker(self) -> None:
        config = reranker_config()
        report = {
            "selected_settings": settings(),
            "model_overrides": {"max_geo_neighbors": 3},
            "reranker": {"enabled": True, "gate": {"pass": True}, "config": config},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            selected = _selection(model_config(), str(path))
            self.assertEqual(selected[3], config)
            report["reranker"]["gate"]["pass"] = False
            path.write_text(json.dumps(report), encoding="utf-8")
            self.assertIsNone(_selection(model_config(), str(path))[3])
            report["reranker"]["gate"]["pass"] = True
            report["reranker"]["enabled"] = False
            path.write_text(json.dumps(report), encoding="utf-8")
            self.assertIsNone(_selection(model_config(), str(path))[3])

    def test_config_and_sampling_helpers_execute_deterministically(self) -> None:
        config = reranker_config()
        validate_reranker_config(config)
        rows = []
        counter = 0
        for fold in range(5):
            for occurrence in range(4):
                query = query_for_fold(fold, counter)
                counter += 20
                rows.append(
                    {
                        "search_query": query,
                        "search_location_id": occurrence,
                        "search_is_delivery_search": False,
                        "search_infm_params_text": "",
                        "search_category": 114,
                        "item_id": f"{fold * 10 + occurrence:016x}",
                    }
                )
        train = pd.DataFrame(rows)
        small = copy.deepcopy(config)
        small["training"]["limit"] = 1
        small["evaluation"]["units_per_fold"] = 1
        training = _training_units(train, small)
        self.assertEqual(len(training), 1)
        first = select_round3_units(train, 3, small)
        second = select_round3_units(train, 3, small)
        self.assertEqual(first["relevant_items"].tolist(), second["relevant_items"].tolist())

        fit = train[
            train["search_query"].map(lambda value: stable_fold(value, 5)).isin({0, 1})
        ]
        self.assertEqual(
            validate_fold_exclusions(
                fit, {2, 3, 4}, 5, context="synthetic training history"
            ),
            [0, 1],
        )

    def test_uncached_offline_model_enforces_training_exclusions(self) -> None:
        config = reranker_config()
        rows = []
        items = []
        counter = 0
        for fold in range(5):
            query = query_for_fold(fold, counter)
            counter += 20
            current = item(f"{fold + 1:016x}", f"услуга {fold}", fold + 1)
            items.append(current)
            rows.append(click(query, current, fold + 1))
        train = pd.DataFrame(rows, columns=QUERY_COLUMNS + ITEM_COLUMNS)
        corpus = pd.DataFrame(items, columns=ITEM_COLUMNS)
        base_config = model_config()
        with tempfile.TemporaryDirectory() as directory:
            _, train_fit = _offline_model(
                train,
                corpus,
                base_config,
                settings(),
                config,
                Path(directory),
                rebuild=True,
            )
        self.assertEqual(
            sorted(
                {
                    stable_fold(value, 5)
                    for value in train_fit["search_query"].drop_duplicates()
                }
            ),
            [0, 1],
        )


class FeaturePreparationTest(unittest.TestCase):
    def _bundle(self) -> PackedFeatureBundle:
        values = np.zeros((4, len(FEATURE_NAMES)), dtype=np.float64)
        values[:, FEATURE_INDEX["exact_word"]] = [0.5, 0.5, 0.2, 100.0]
        values[:, FEATURE_INDEX["location"]] = [1.0, 0.8, 0.3, 1.0]
        ranks = {
            "__global_word_rank": np.array([1, 2, 3, RANK_MISSING], dtype=np.uint16),
            "__global_char_rank": np.full(4, RANK_MISSING, dtype=np.uint16),
            "__global_filter_rank": np.full(4, RANK_MISSING, dtype=np.uint16),
            "__geo3_word_rank": np.full(4, RANK_MISSING, dtype=np.uint16),
            "__geo3_char_rank": np.full(4, RANK_MISSING, dtype=np.uint16),
        }
        return PackedFeatureBundle(
            fold=2,
            item_ids=np.array([f"{value:016x}" for value in range(4)]),
            rows=[
                PackedRow(
                    positions=np.arange(4, dtype=np.int32),
                    values=values,
                    ranks=ranks,
                    always_source=np.zeros(4, dtype=bool),
                )
            ],
            relevant=[["0000000000000001", "ffffffffffffffff"]],
            slices={},
        )

    def test_normalization_uses_eligible_candidates_and_sampler_is_deterministic(self) -> None:
        bundle = self._bundle()
        locality = LocalityFeatures(
            rows=[np.zeros((4, len(LOCALITY_FEATURE_NAMES)), dtype=np.float32)],
            query_locality=np.zeros(1, dtype=np.float32),
            fit_rows=1,
            fit_text_groups=1,
            heldout_text_groups=1,
            fit_heldout_text_overlap=0,
            global_exact_location_rate=0.5,
        )
        config = reranker_config()
        config["training"]["hard_negatives_per_query"] = 1
        config["training"]["random_candidates_per_query"] = 1
        full = prepare_features(bundle, locality, settings(), config, training=False)
        max_column = RERANK_FEATURE_NAMES.index("query_max_exact_word")
        self.assertTrue(np.all(full.matrix[:, max_column] == 0.5))
        self.assertEqual(len(full.matrix), 3)

        first = prepare_features(bundle, locality, settings(), config, training=True)
        second = prepare_features(bundle, locality, settings(), config, training=True)
        np.testing.assert_array_equal(first.matrix, second.matrix)
        np.testing.assert_array_equal(first.labels, second.labels)
        np.testing.assert_array_equal(first.weights, second.weights)
        self.assertEqual(float(first.weights[first.labels.astype(bool)][0]), 50.0)

    def test_blend_zero_preserves_position_tie_and_rejects_sampled_rows(self) -> None:
        class ExplodingModel:
            def decision_function(self, _: np.ndarray) -> np.ndarray:
                raise AssertionError("blend=0 must not call the model")

        prepared = PreparedFeatures(
            matrix=np.zeros((2, len(RERANK_FEATURE_NAMES)), dtype=np.float32),
            labels=np.zeros(2, dtype=np.uint8),
            weights=np.ones(2, dtype=np.float32),
            positions=[np.array([0, 1], dtype=np.int32)],
            base_scores=[np.array([0.0, 0.0])],
            target_counts=[1],
            sampled=False,
        )
        item_ids = np.array(["0000000000000001", "0000000000000002"])
        predictions, _ = rank_prepared(
            ExplodingModel(),
            prepared,
            item_ids,
            [["0000000000000001"]],
            blend=0.0,
            top_k=1,
        )
        self.assertEqual(predictions, [["0000000000000001"]])
        prepared.sampled = True
        with self.assertRaisesRegex(ValueError, "Sampled training features"):
            rank_prepared(
                ExplodingModel(), prepared, item_ids, [[]], blend=0.0, top_k=1
            )


class LocalityLeakageTest(unittest.TestCase):
    def test_offline_locality_requires_disjoint_text(self) -> None:
        first = item("0000000000000001", "ремонт", 1)
        second = item("0000000000000002", "сантехник", 2)
        items = pd.DataFrame([first, second], columns=ITEM_COLUMNS)
        history = pd.DataFrame(
            [click("ремонт", first, 1), click("сантехник", second, 2)],
            columns=QUERY_COLUMNS + ITEM_COLUMNS,
        )
        config = model_config()
        config["max_geo_neighbors"] = 3
        model = HybridRetriever(config).fit(items, history).enable_improvement(history)
        queries = history.iloc[[0]][QUERY_COLUMNS].reset_index(drop=True)
        rows = model.retrieve_improvement_features(
            queries, max_pool=2, geo_neighbor_options=(3,)
        )
        units = queries.copy()
        units["relevant_items"] = [["0000000000000001"]]
        bundle = pack_feature_rows(model, rows, units, history, 2, (3,))
        with self.assertRaisesRegex(AssertionError, "held-out text groups"):
            build_locality_features(
                model,
                history,
                queries,
                bundle,
                reranker_config(),
                require_disjoint=True,
            )
        locality = build_locality_features(
            model,
            history,
            queries,
            bundle,
            reranker_config(),
            require_disjoint=False,
        )
        self.assertEqual(locality.fit_heldout_text_overlap, 1)


if __name__ == "__main__":
    unittest.main()
