from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from avito_retrieval.constants import ITEM_COLUMNS, QUERY_COLUMNS
from avito_retrieval.improvement import (
    FEATURE_INDEX,
    FEATURE_NAMES,
    RANK_MISSING,
    PackedFeatureBundle,
    PackedRow,
    baseline_settings,
    coordinate_tune,
    pack_feature_rows,
    predict_packed,
    select_fold_units,
)
from avito_retrieval.model import HybridRetriever
from avito_retrieval.text import normalize_text

from test_pipeline import click, item


def model_config() -> dict[str, object]:
    return {
        "seed": 1,
        "top_k": 2,
        "candidate_pool_per_channel": 2,
        "batch_size": 2,
        "word_max_features": 200,
        "word_min_df": 1,
        "char_max_features": 300,
        "char_min_df": 1,
        "description_chars": 100,
        "item_params_chars": 100,
        "history_neighbors": 2,
        "history_items_per_query": 5,
        "fallback_items": 2,
        "microcat_neighbors": 2,
        "weights": {
            "word": 1.0,
            "char": 0.55,
            "filter": 0.0,
            "history": 0.25,
            "category": 0.05,
            "location": 0.7,
            "popularity": 0.03,
        },
    }


def improvement_config() -> dict[str, object]:
    return {
        "baseline_weights": {
            **model_config()["weights"],
            "title": 0.0,
            "microcat": 0.0,
        }
    }


class PackedScoringTest(unittest.TestCase):
    def test_exact_rescore_recovers_score_censored_candidate(self) -> None:
        values = np.zeros((2, len(FEATURE_NAMES)), dtype=np.float64)
        values[0, FEATURE_INDEX["word"]] = 0.8
        values[:, FEATURE_INDEX["exact_word"]] = [0.8, 1.0]
        ranks = {
            "__global_word_rank": np.array([1, RANK_MISSING], dtype=np.uint16),
            "__global_char_rank": np.full(2, RANK_MISSING, dtype=np.uint16),
            "__global_filter_rank": np.full(2, RANK_MISSING, dtype=np.uint16),
            "__geo3_word_rank": np.full(2, RANK_MISSING, dtype=np.uint16),
            "__geo3_char_rank": np.full(2, RANK_MISSING, dtype=np.uint16),
        }
        row = PackedRow(
            positions=np.array([0, 1], dtype=np.int32),
            values=values,
            ranks=ranks,
            always_source=np.array([False, True]),
        )
        bundle = PackedFeatureBundle(
            fold=2,
            item_ids=np.array(["0000000000000001", "0000000000000002"]),
            rows=[row],
            relevant=[["0000000000000002"]],
            slices={},
        )
        settings = baseline_settings(improvement_config())
        legacy, _, _ = predict_packed(bundle, settings, top_k=1)
        settings["exact_rescore"] = True
        enhanced, _, _ = predict_packed(bundle, settings, top_k=1)
        self.assertEqual(legacy[0], ["0000000000000001"])
        self.assertEqual(enhanced[0], ["0000000000000002"])

    def test_zero_features_preserve_frozen_legacy_ranking(self) -> None:
        items = pd.DataFrame(
            [
                item("0000000000000001", "ремонт холодильников", 1),
                item("0000000000000002", "ремонт квартир", 1),
                item("0000000000000003", "маникюр", 2),
            ],
            columns=ITEM_COLUMNS,
        )
        train = pd.DataFrame(
            [click("ремонт холодильника", items.iloc[0].to_dict())],
            columns=QUERY_COLUMNS + ITEM_COLUMNS,
        )
        query = train[QUERY_COLUMNS]
        legacy_model = HybridRetriever(model_config()).fit(items, train)
        expected = legacy_model.predict(query, weights=model_config()["weights"])

        enhanced_config = model_config()
        enhanced_config["max_geo_neighbors"] = 5
        model = HybridRetriever(enhanced_config).fit(items, train).enable_improvement(train)
        feature_rows = model.retrieve_improvement_features(
            query, max_pool=3, geo_neighbor_options=(1, 3, 5)
        )
        units = query.copy()
        units["relevant_items"] = [["0000000000000001"]]
        bundle = pack_feature_rows(model, feature_rows, units, train, 2, (1, 3, 5))
        actual, _, _ = predict_packed(bundle, baseline_settings(improvement_config()), top_k=2)
        self.assertEqual(actual, expected)

    def test_max_geo_union_matches_final_selected_geo_union(self) -> None:
        items = pd.DataFrame(
            [
                item("0000000000000001", "ремонт холодильников", 1),
                item("0000000000000002", "ремонт квартир", 2),
                item("0000000000000003", "ремонт техники", 3),
                item("0000000000000004", "маникюр", 4),
            ],
            columns=ITEM_COLUMNS,
        )
        train = pd.DataFrame(
            [
                click("ремонт холодильника", items.iloc[0].to_dict(), 1),
                click("ремонт квартир", items.iloc[1].to_dict(), 1),
                click("ремонт техники", items.iloc[2].to_dict(), 1),
            ],
            columns=QUERY_COLUMNS + ITEM_COLUMNS,
        )
        config = model_config()
        config["max_geo_neighbors"] = 5
        model = HybridRetriever(config).fit(items, train).enable_improvement(train)
        query = train.iloc[[0]][QUERY_COLUMNS]
        units = query.copy()
        units["relevant_items"] = [["0000000000000001"]]
        settings = baseline_settings(improvement_config())
        settings.update({"exact_rescore": True, "pool": 3, "geo_neighbors": 3})
        settings["weights"].update(
            {"char": 0.35, "history": 0.0, "microcat": 0.4}
        )

        maximal_rows = model.retrieve_improvement_features(
            query, max_pool=3, geo_neighbor_options=(3, 5)
        )
        maximal = pack_feature_rows(model, maximal_rows, units, train, 2, (3, 5))
        final_rows = model.retrieve_improvement_features(
            query, max_pool=3, geo_neighbor_options=(3,)
        )
        final = pack_feature_rows(model, final_rows, units, train, -1, (3,))
        self.assertEqual(
            predict_packed(maximal, settings, top_k=3)[0],
            predict_packed(final, settings, top_k=3)[0],
        )

    def test_cached_packed_scoring_is_deterministic(self) -> None:
        values = np.zeros((2, len(FEATURE_NAMES)), dtype=np.float64)
        ranks = {
            "__global_word_rank": np.array([1, 2], dtype=np.uint16),
            "__global_char_rank": np.full(2, RANK_MISSING, dtype=np.uint16),
            "__global_filter_rank": np.full(2, RANK_MISSING, dtype=np.uint16),
            "__geo3_word_rank": np.full(2, RANK_MISSING, dtype=np.uint16),
            "__geo3_char_rank": np.full(2, RANK_MISSING, dtype=np.uint16),
        }
        bundle = PackedFeatureBundle(
            fold=2,
            item_ids=np.array(["0000000000000001", "0000000000000002"]),
            rows=[
                PackedRow(
                    np.array([0, 1], dtype=np.int32),
                    values,
                    ranks,
                    np.zeros(2, dtype=bool),
                )
            ],
            relevant=[["0000000000000001"]],
            slices={},
        )
        settings = baseline_settings(improvement_config())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.joblib"
            joblib.dump(bundle, path)
            restored = joblib.load(path)
        first = predict_packed(bundle, settings, top_k=2)[0]
        second = predict_packed(restored, settings, top_k=2)[0]
        self.assertEqual(first, second)

    def test_selection_guard_rejects_average_gain_that_harms_slice(self) -> None:
        rows = []
        relevant = []
        for truth_position in (1, 1, 0):
            values = np.zeros((2, len(FEATURE_NAMES)), dtype=np.float64)
            values[0, FEATURE_INDEX["word"]] = 0.8
            values[:, FEATURE_INDEX["exact_word"]] = [0.8, 1.0]
            ranks = {
                "__global_word_rank": np.array([1, RANK_MISSING], dtype=np.uint16),
                "__global_char_rank": np.full(2, RANK_MISSING, dtype=np.uint16),
                "__global_filter_rank": np.full(2, RANK_MISSING, dtype=np.uint16),
                "__geo3_word_rank": np.full(2, RANK_MISSING, dtype=np.uint16),
                "__geo3_char_rank": np.full(2, RANK_MISSING, dtype=np.uint16),
            }
            rows.append(
                PackedRow(
                    np.array([0, 1], dtype=np.int32),
                    values,
                    ranks,
                    np.array([False, True]),
                )
            )
            relevant.append([f"000000000000000{truth_position + 1}"])
        bundle = PackedFeatureBundle(
            fold=2,
            item_ids=np.array(["0000000000000001", "0000000000000002"]),
            rows=rows,
            relevant=relevant,
            slices={"protected": np.array([False, False, True])},
        )
        config = {
            **improvement_config(),
            "coordinate_passes": 1,
            "pool_options": [120],
            "geo_neighbor_options": [3],
            "weight_options": {
                name: [value]
                for name, value in {
                    "char": 0.55,
                    "history": 0.25,
                    "location": 0.7,
                    "microcat": 0.0,
                    "title": 0.0,
                }.items()
            },
            "selection_slice_min_queries": 1,
            "selection_slice_min_delta": 0.0,
        }
        selected, _, _, guard = coordinate_tune(bundle, config)
        self.assertFalse(selected["exact_rescore"])
        self.assertTrue(guard["selected_eligible"])


class MicrocatTransferTest(unittest.TestCase):
    def test_uses_all_passed_rows_and_drops_removed_text_groups(self) -> None:
        corpus_item = item("0000000000000001", "ремонт", 1)
        corpus = pd.DataFrame([corpus_item], columns=ITEM_COLUMNS)
        orphan = item("ffffffffffffffff", "рисование", 1)
        orphan["item_microcat_id"] = 99
        history = pd.DataFrame(
            [click("ремонт", corpus_item), click("особая услуга", orphan)],
            columns=QUERY_COLUMNS + ITEM_COLUMNS,
        )
        config = model_config()
        model = HybridRetriever(config).fit(corpus, history).enable_improvement(history)
        key = normalize_text("особая услуга")
        position = model.microcat_keys.index(key)
        self.assertEqual(model.microcat_distributions[position], {99: 1.0})

        model.refit_behavior(history[history["search_query"] != "особая услуга"])
        self.assertNotIn(key, model.microcat_keys)


class FreshSamplingTest(unittest.TestCase):
    def test_fresh_sample_excludes_all_original_text_clusters(self) -> None:
        rows = []
        number = 0
        while len({row["search_query"] for row in rows}) < 6:
            query = f"запрос {number}"
            number += 1
            from avito_retrieval.text import stable_fold

            if stable_fold(query, 5) != 3:
                continue
            item_row = item(f"{number:016x}", query, 1)
            rows.append(click(query, item_row))
        train = pd.DataFrame(rows)
        standard = select_fold_units(train, 3, 5, 2, 4, "standard-v1")
        fresh = select_fold_units(train, 3, 5, 2, 4, "fresh-text-v2")
        self.assertFalse(
            set(standard["search_query"].map(normalize_text))
            & set(fresh["search_query"].map(normalize_text))
        )


if __name__ == "__main__":
    unittest.main()
