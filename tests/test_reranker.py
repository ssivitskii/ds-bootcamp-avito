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
    BASE_FEATURE_SCHEMA,
    EXTRA_FEATURE_NAMES,
    LOCALITY_FEATURE_NAMES,
    RERANK_FEATURE_NAMES,
    ExtraFeatures,
    LocalityFeatures,
    PreparedFeatures,
    _offline_model,
    _bundle_for_units,
    _training_units,
    build_extra_features,
    build_locality_features,
    prepare_features,
    rank_prepared,
    select_round3_units,
    select_fresh_units,
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
        fresh = select_fresh_units(train, 3, small)
        self.assertEqual(fresh["relevant_items"].tolist(), select_fresh_units(train, 3, small)["relevant_items"].tolist())
        consumed = set(first["search_query"].map(str))
        self.assertFalse(consumed & set(fresh["search_query"].map(str)))

        fit = train[
            train["search_query"].map(lambda value: stable_fold(value, 5)).isin({0, 1})
        ]
        self.assertEqual(
            validate_fold_exclusions(
                fit, {2, 3, 4}, 5, context="synthetic training history"
            ),
            [0, 1],
        )

    def test_extended_config_rejects_invalid_baseline_contract(self) -> None:
        config = reranker_config()
        config["baseline"]["model"]["thread_limit"] = 1
        with self.assertRaisesRegex(ValueError, "baseline requires thread_limit=4"):
            validate_reranker_config(config)
        config = reranker_config()
        config["baseline"]["candidate_pool"] = 0
        with self.assertRaisesRegex(ValueError, "Candidate pools"):
            validate_reranker_config(config)
        config = reranker_config()
        config["training"]["bundle_batch_size"] = -1
        with self.assertRaisesRegex(ValueError, "Training bundle_batch_size"):
            validate_reranker_config(config)

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
        extra = ExtraFeatures(
            rows=[np.zeros((4, len(EXTRA_FEATURE_NAMES)), dtype=np.float32)],
            fixed_union_items=4,
            location_centers=1,
            median_word_vocab_nnz=1.0,
            unigram_vocabulary=1,
        )
        full = prepare_features(
            bundle, locality, settings(), config, training=False, extra=extra
        )
        max_column = RERANK_FEATURE_NAMES.index("query_max_exact_word")
        self.assertTrue(np.all(full.matrix[:, max_column] == 0.5))
        self.assertEqual(len(full.matrix), 3)

        first = prepare_features(
            bundle, locality, settings(), config, training=True, extra=extra
        )
        second = prepare_features(
            bundle, locality, settings(), config, training=True, extra=extra
        )
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


class ExtendedFeatureTest(unittest.TestCase):
    def test_metadata_and_coverage_follow_model_item_order(self) -> None:
        items = pd.DataFrame(
            [
                item("0000000000000002", "ремонт квартир", 2),
                item("0000000000000001", "ремонт холодильников", 1),
            ],
            columns=ITEM_COLUMNS,
        )
        history = pd.DataFrame(
            [click("ремонт", items.iloc[1].to_dict(), 1)],
            columns=QUERY_COLUMNS + ITEM_COLUMNS,
        )
        model = HybridRetriever(model_config()).fit(items, history).enable_improvement(history)
        queries = history[QUERY_COLUMNS].reset_index(drop=True)
        rows = model.retrieve_improvement_features(
            queries, max_pool=2, geo_neighbor_options=(3,)
        )
        units = queries.copy()
        units["relevant_items"] = [["0000000000000001"]]
        bundle = pack_feature_rows(model, rows, units, history, 2, (3,))
        first = build_extra_features(model, items, queries, bundle)
        second = build_extra_features(
            model, items.iloc[::-1].reset_index(drop=True), queries, bundle
        )
        self.assertEqual(first.rows[0].shape, (len(bundle.rows[0].positions), 17))
        np.testing.assert_array_equal(first.rows[0], second.rows[0])
        exact_location = EXTRA_FEATURE_NAMES.index("exact_location")
        positions = bundle.rows[0].positions
        assert model.item_locations is not None
        np.testing.assert_array_equal(
            first.rows[0][:, exact_location],
            (model.item_locations[positions] == 1).astype(np.float32),
        )
        self.assertGreater(first.median_word_vocab_nnz, 0)

    def test_fixed_union_center_includes_train_only_item(self) -> None:
        benchmark = pd.DataFrame(
            [
                item("0000000000000001", "ремонт один", 1),
                item("0000000000000002", "ремонт два", 1),
            ],
            columns=ITEM_COLUMNS,
        )
        benchmark.loc[:, "item_longitude"] = 0.0
        benchmark.loc[:, "item_latitude"] = [0.0, 10.0]
        train_only = item("0000000000000003", "ремонт три", 1)
        train_only["item_longitude"] = 0.0
        train_only["item_latitude"] = 100.0
        history = pd.DataFrame(
            [click("история", benchmark.iloc[0].to_dict(), 1)],
            columns=QUERY_COLUMNS + ITEM_COLUMNS,
        )
        model = HybridRetriever(model_config()).fit(benchmark, history).enable_improvement(history)
        assert model.item_ids is not None
        count = len(model.item_ids)
        packed = PackedRow(
            positions=np.arange(count, dtype=np.int32),
            values=np.zeros((count, len(FEATURE_NAMES)), dtype=np.float64),
            ranks={},
            always_source=np.zeros(count, dtype=bool),
        )
        bundle = PackedFeatureBundle(
            fold=-1,
            item_ids=model.item_ids.copy(),
            rows=[packed],
            relevant=[[]],
            slices={},
        )
        queries = pd.DataFrame(
            [
                {
                    "search_query": "ремонт",
                    "search_location_id": 1,
                    "search_is_delivery_search": False,
                    "search_infm_params_text": "",
                    "search_category": 114,
                }
            ],
            columns=QUERY_COLUMNS,
        )
        union = pd.concat(
            [benchmark, pd.DataFrame([train_only], columns=ITEM_COLUMNS)],
            ignore_index=True,
        )
        extra = build_extra_features(model, union, queries, bundle)
        distance_column = EXTRA_FEATURE_NAMES.index("log_distance_km")
        first_position = int(np.flatnonzero(model.item_ids == "0000000000000001")[0])
        # The union median latitude is 10 degrees. Benchmark-only would be 5.
        expected = np.log1p(6371.0 * np.radians(10.0))
        self.assertAlmostEqual(
            float(extra.rows[0][first_position, distance_column]), expected, places=5
        )
        self.assertEqual(extra.fixed_union_items, 3)

    def test_batched_bundle_and_sampler_match_full_extraction(self) -> None:
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
        queries = pd.DataFrame(
            [
                ["ремонт", 1, False, "", 114],
                ["маникюр", 2, False, "", 114],
                ["квартира", 1, False, "", 114],
            ],
            columns=QUERY_COLUMNS,
        )
        queries["relevant_items"] = [
            ["0000000000000001"],
            ["0000000000000003"],
            ["0000000000000002"],
        ]
        config = reranker_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = _bundle_for_units(
                model,
                history,
                queries,
                settings(),
                2,
                root / "full.joblib",
                rebuild=True,
                pool=3,
                batch_size=len(queries),
            )
            batched = _bundle_for_units(
                model,
                history,
                queries,
                settings(),
                2,
                root / "batched.joblib",
                rebuild=True,
                pool=3,
                batch_size=1,
            )
            batch600 = _bundle_for_units(
                model,
                history,
                queries,
                settings(),
                2,
                root / "batch600.joblib",
                rebuild=True,
                pool=3,
                batch_size=600,
            )
        for compared in (batched, batch600):
            np.testing.assert_array_equal(full.item_ids, compared.item_ids)
            self.assertEqual(full.relevant, compared.relevant)
            for name in full.slices:
                np.testing.assert_array_equal(full.slices[name], compared.slices[name])
            for left, right in zip(full.rows, compared.rows, strict=True):
                np.testing.assert_array_equal(left.positions, right.positions)
                np.testing.assert_array_equal(left.values, right.values)
                np.testing.assert_array_equal(left.always_source, right.always_source)
                self.assertEqual(set(left.ranks), set(right.ranks))
                for name in left.ranks:
                    np.testing.assert_array_equal(left.ranks[name], right.ranks[name])

        locality_full = build_locality_features(
            model, history, queries, full, config, require_disjoint=True
        )
        locality_batched = build_locality_features(
            model, history, queries, batched, config, require_disjoint=True
        )
        fixed_union = items
        extra_full = build_extra_features(model, fixed_union, queries, full)
        extra_batched = build_extra_features(model, fixed_union, queries, batched)
        prepared_full = prepare_features(
            full,
            locality_full,
            settings(),
            config,
            training=True,
            extra=extra_full,
        )
        prepared_batched = prepare_features(
            batched,
            locality_batched,
            settings(),
            config,
            training=True,
            extra=extra_batched,
        )
        np.testing.assert_array_equal(prepared_full.matrix, prepared_batched.matrix)
        np.testing.assert_array_equal(prepared_full.labels, prepared_batched.labels)
        np.testing.assert_array_equal(prepared_full.weights, prepared_batched.weights)


if __name__ == "__main__":
    unittest.main()
