from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from avito_retrieval.constants import ITEM_COLUMNS, QUERY_COLUMNS
from avito_retrieval.evaluation import make_query_units, training_without_folds
from avito_retrieval.model import HybridRetriever
from avito_retrieval.text import normalize_text, stable_fold
from avito_retrieval.validation import validate_answer


def item(item_id: str, title: str, location: int, category: int = 114) -> dict[str, object]:
    return {
        "item_title_raw": title,
        "item_rating_reviews_count": 1.0,
        "item_rating": 5.0,
        "item_price": 100,
        "item_microcat_id": 1,
        "item_longitude": 37.0,
        "item_location_id": location,
        "item_latitude": 55.0,
        "item_is_phone_hidden": False,
        "item_is_message_forbidden": False,
        "item_infm_params_text": "услуги",
        "item_id": item_id,
        "item_description_raw": "качественные услуги",
        "item_category_id": category,
    }


def click(query: str, item_row: dict[str, object], location: int = 1) -> dict[str, object]:
    return {
        "search_query": query,
        "search_location_id": location,
        "search_is_delivery_search": 0,
        "search_infm_params_text": "",
        "search_category": 114,
        **item_row,
    }


class TextAndSplitTest(unittest.TestCase):
    def test_missing_values_are_empty(self) -> None:
        self.assertEqual(normalize_text(None), "")
        self.assertEqual(normalize_text(pd.NA), "")
        self.assertEqual(normalize_text(float("nan")), "")

    def test_fold_is_normalization_stable(self) -> None:
        self.assertEqual(stable_fold(" Ёлка  ДОМ ", 5), stable_fold("елка дом", 5))

    def test_units_deduplicate_relevance_and_exclude_both_folds(self) -> None:
        first = item("0000000000000001", "ремонт", 1)
        second = item("0000000000000002", "ремонт", 1)
        rows = pd.DataFrame([click("ремонт", first), click("ремонт", first), click("ремонт", second)])
        fold = stable_fold("ремонт", 5)
        units = make_query_units(rows, fold, 5)
        self.assertEqual(units.iloc[0]["relevant_items"], [first["item_id"], second["item_id"]])
        self.assertTrue(training_without_folds(rows, {fold}, 5).empty)


class RetrievalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.items = pd.DataFrame(
            [
                item("0000000000000001", "ремонт холодильников", 1),
                item("0000000000000002", "маникюр", 2),
                item("0000000000000003", "ремонт квартир", 1),
            ],
            columns=ITEM_COLUMNS,
        )
        self.train = pd.DataFrame(
            [click("ремонт холодильника", self.items.iloc[0].to_dict())],
            columns=QUERY_COLUMNS + ITEM_COLUMNS,
        )
        self.config = {
            "seed": 1,
            "top_k": 2,
            "candidate_pool_per_channel": 2,
            "batch_size": 2,
            "word_max_features": 100,
            "word_min_df": 1,
            "char_max_features": 200,
            "char_min_df": 1,
            "description_chars": 100,
            "item_params_chars": 100,
            "history_neighbors": 2,
            "history_items_per_query": 5,
            "fallback_items": 2,
            "weights": {
                "word": 1.0,
                "char": 0.5,
                "filter": 0.1,
                "history": 0.5,
                "category": 0.1,
                "location": 0.1,
                "popularity": 0.01,
            },
        }

    def test_retrieval_is_reproducible(self) -> None:
        model = HybridRetriever(self.config).fit(self.items, self.train)
        query = self.train[QUERY_COLUMNS]
        first = model.predict(query)
        second = model.predict(query)
        self.assertEqual(first, second)
        self.assertEqual(first[0][0], "0000000000000001")
        self.assertEqual(len(first[0]), 2)

    def test_exact_score_tie_uses_ascending_position(self) -> None:
        scores = np.array([0.5, 0.5, 0.7, 0.5])
        np.testing.assert_array_equal(HybridRetriever._top_indices(scores, 2), [2, 0])

    def test_cache_save_is_atomic_under_concurrent_writers(self) -> None:
        model = HybridRetriever(self.config).fit(self.items, self.train)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.joblib"
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(lambda _: model.save(target), range(2)))
            loaded = HybridRetriever.load(target)
            self.assertEqual(loaded.predict(self.train[QUERY_COLUMNS]), model.predict(self.train[QUERY_COLUMNS]))
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])


class ValidationTest(unittest.TestCase):
    def test_validator_rejects_uppercase_item_id(self) -> None:
        queries = pd.DataFrame({"query_id": ["00WuFMaXSFZBxSzT"]})
        items = pd.DataFrame({"item_id": ["abcdef0123456789"]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "answer.csv"
            pd.DataFrame(
                {"query_id": ["00WuFMaXSFZBxSzT"], "answer": ["ABCDEF0123456789"]}
            ).to_csv(path, index=False)
            report = validate_answer(path, queries, items)
        self.assertFalse(report["valid"])
        self.assertTrue(any("lowercase hex" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
