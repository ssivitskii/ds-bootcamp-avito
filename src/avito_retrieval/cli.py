from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .audit import audit_data
from .evaluation import evaluate_pipeline
from .io import data_fingerprint, load_config, load_frames, source_fingerprint, write_json
from .model import HybridRetriever
from .validation import validate_answer


def _add_common(parser: argparse.ArgumentParser, *, config: bool = False) -> None:
    parser.add_argument("--data-dir", default="data", help="Directory containing the three parquet files")
    if config:
        parser.add_argument("--config", default="config.json", help="JSON configuration path")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Avito offline candidate generation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Validate and summarize input parquet files")
    _add_common(audit)
    audit.add_argument("--output", default="artifacts/audit.json")

    evaluate = subparsers.add_parser("evaluate", help="Tune and confirm on cold text-group folds")
    _add_common(evaluate, config=True)
    evaluate.add_argument("--output", default="artifacts/evaluation.json")
    evaluate.add_argument("--max-queries", type=int, default=None, help="Override both fold limits")

    predict = subparsers.add_parser("predict", help="Fit all train data and create answer.csv")
    _add_common(predict, config=True)
    predict.add_argument("--output", default="answer.csv")
    predict.add_argument("--cache-dir", default="cache")
    predict.add_argument("--weights-json", default=None, help="Evaluation JSON containing selected_weights")
    predict.add_argument("--no-cache", action="store_true")

    validate = subparsers.add_parser("validate", help="Strictly validate a generated answer.csv")
    _add_common(validate)
    validate.add_argument("--answer", default="answer.csv")
    validate.add_argument("--output", default=None, help="Optional JSON validation report")
    return parser


def _selected_weights(config: dict[str, Any], path: str | None) -> dict[str, float]:
    if path is None:
        return dict(config["weights"])
    with Path(path).open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    return {name: float(value) for name, value in report["selected_weights"].items()}


def _predict(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    weights = _selected_weights(config, args.weights_json)
    logging.info("Loading parquet inputs from %s", args.data_dir)
    train, queries, items = load_frames(args.data_dir)
    root = Path(args.data_dir)
    fingerprint = data_fingerprint(
        [root / "train.parquet", root / "benchmark_queries.parquet", root / "benchmark_items.parquet"],
        {
            "cache_format_version": 1,
            "model_config": config,
            "weights": weights,
            "source_sha256": source_fingerprint(Path(__file__).parent),
        },
    )
    cache_path = Path(args.cache_dir) / fingerprint / "model.joblib"
    if cache_path.exists() and not args.no_cache:
        logging.info("Loading fitted index from %s", cache_path)
        model = HybridRetriever.load(cache_path)
    else:
        model = HybridRetriever(config).fit(items, train)
        if not args.no_cache:
            logging.info("Saving fitted index to %s", cache_path)
            model.save(cache_path)
    predictions = model.predict(queries, weights=weights)
    answer = pd.DataFrame(
        {
            "query_id": queries["query_id"].astype(str),
            "answer": [" ".join(item_ids) for item_ids in predictions],
        }
    )
    answer.to_csv(args.output, index=False, encoding="utf-8", lineterminator="\n")
    report = validate_answer(args.output, queries, items)
    if not report["valid"]:
        raise ValueError(f"Generated answer failed validation: {report['errors']}")
    return {**report, "output": str(args.output), "fingerprint": fingerprint, "weights": weights}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        logging.info("Auditing parquet inputs from %s", args.data_dir)
        report = audit_data(args.data_dir)
        write_json(report, args.output)
    elif args.command == "evaluate":
        config = load_config(args.config)
        logging.info("Loading parquet inputs from %s", args.data_dir)
        train, _, items = load_frames(args.data_dir)
        report = evaluate_pipeline(train, items, config, max_queries=args.max_queries)
        report["config"] = config
        write_json(report, args.output)
    elif args.command == "predict":
        report = _predict(args)
    elif args.command == "validate":
        root = Path(args.data_dir)
        queries = pd.read_parquet(root / "benchmark_queries.parquet")
        items = pd.read_parquet(root / "benchmark_items.parquet", columns=["item_id"])
        report = validate_answer(args.answer, queries, items)
        if args.output:
            write_json(report, args.output)
        if not report["valid"]:
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 1
    else:  # pragma: no cover - argparse prevents this branch
        raise AssertionError(args.command)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0
