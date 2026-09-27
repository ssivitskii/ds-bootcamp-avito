from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .audit import audit_data
from .evaluation import evaluate_pipeline
from .improvement import predict_improved, run_improvement
from .io import data_fingerprint, load_config, load_frames, source_fingerprint, write_json
from .model import HybridRetriever
from .reranker import (
    predict_reranked,
    run_reranker_evaluation,
    train_or_load_reranker,
    validate_reranker_config,
)
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

    improve = subparsers.add_parser(
        "improve", help="Run cached three-fold candidate and reranking experiments"
    )
    _add_common(improve, config=True)
    improve.add_argument("--improvement-config", default="improvement_config.json")
    improve.add_argument("--output", default="artifacts/improvement/evaluation.json")
    improve.add_argument("--cache-dir", default="cache")
    improve.add_argument("--max-queries", type=int, default=None)
    improve.add_argument("--rebuild-cache", action="store_true")

    reranker = subparsers.add_parser(
        "evaluate-reranker", help="Train and gate the frozen HGB reranker"
    )
    _add_common(reranker, config=True)
    reranker.add_argument("--reranker-config", default="reranker_config.json")
    reranker.add_argument("--weights-json", default="artifacts/evaluation.json")
    reranker.add_argument(
        "--output", default="artifacts/improvement/round4/evaluation.reranker.json"
    )
    reranker.add_argument("--cache-dir", default="cache")
    reranker.add_argument("--rebuild-cache", action="store_true")

    predict = subparsers.add_parser("predict", help="Fit all train data and create answer.csv")
    _add_common(predict, config=True)
    predict.add_argument("--output", default="answer.csv")
    predict.add_argument("--cache-dir", default="cache")
    predict.add_argument("--weights-json", default=None, help="Evaluation JSON containing selected_weights")
    predict.add_argument(
        "--reranker-config",
        default=None,
        help="Optional reranker config override; otherwise read it from the evaluation report",
    )
    predict.add_argument("--no-cache", action="store_true")

    validate = subparsers.add_parser("validate", help="Strictly validate a generated answer.csv")
    _add_common(validate)
    validate.add_argument("--answer", default="answer.csv")
    validate.add_argument("--output", default=None, help="Optional JSON validation report")
    return parser


def _selection(
    config: dict[str, Any], path: str | None
) -> tuple[
    dict[str, float],
    dict[str, Any] | None,
    dict[str, Any],
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    if path is None:
        return dict(config["weights"]), None, {}, None, None
    with Path(path).open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    if "selected_settings" in report:
        settings = report["selected_settings"]
        weights = {name: float(value) for name, value in settings["weights"].items()}
        reranker = report.get("reranker")
        reranker_config = None
        if (
            isinstance(reranker, dict)
            and bool(reranker.get("enabled"))
            and bool(reranker.get("gate", {}).get("pass"))
        ):
            reranker_config = dict(reranker["config"])
        return (
            weights,
            settings,
            dict(report.get("model_overrides", {})),
            reranker_config,
            report,
        )
    return (
        {name: float(value) for name, value in report["selected_weights"].items()},
        None,
        {},
        None,
        report,
    )


def _predict(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    weights, settings, model_overrides, report_reranker, _ = _selection(
        config, args.weights_json
    )
    reranker_config = (
        load_config(args.reranker_config) if args.reranker_config else report_reranker
    )
    if reranker_config is not None:
        validate_reranker_config(reranker_config)
        if settings is None:
            raise ValueError("Reranking requires selected_settings from an evaluation report")
    config.update(model_overrides)
    logging.info("Loading parquet inputs from %s", args.data_dir)
    train, queries, items = load_frames(args.data_dir)
    reranker = None
    reranker_fingerprint = None
    if reranker_config is not None:
        reranker, _, reranker_fingerprint = train_or_load_reranker(
            train,
            items,
            config,
            settings,
            reranker_config,
            data_dir=args.data_dir,
            cache_dir=args.cache_dir,
            rebuild=args.no_cache,
        )
    root = Path(args.data_dir)
    fingerprint = data_fingerprint(
        [root / "train.parquet", root / "benchmark_queries.parquet", root / "benchmark_items.parquet"],
        {
            "cache_format_version": 1,
            "model_config": config,
            "weights": weights,
            "improvement_settings": settings,
            "reranker_config": reranker_config,
            "source_sha256": source_fingerprint(Path(__file__).parent),
        },
    )
    cache_path = Path(args.cache_dir) / fingerprint / "model.joblib"
    if cache_path.exists() and not args.no_cache:
        logging.info("Loading fitted index from %s", cache_path)
        model = HybridRetriever.load(cache_path)
    else:
        model = HybridRetriever(config).fit(items, train)
        if settings is not None:
            model.enable_improvement(train)
        if not args.no_cache:
            logging.info("Saving fitted index to %s", cache_path)
            model.save(cache_path)
    if settings is None:
        predictions = model.predict(queries, weights=weights)
    else:
        if not model.improvement_enabled:
            raise RuntimeError("Cached model lacks required improvement indexes")
        if reranker is None:
            predictions = predict_improved(
                model, queries, settings, top_k=int(config["top_k"])
            )
        else:
            predictions = predict_reranked(
                model,
                train,
                items,
                queries,
                settings,
                reranker,
                top_k=int(config["top_k"]),
            )
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
    return {
        **report,
        "output": str(args.output),
        "fingerprint": fingerprint,
        "weights": weights,
        "improvement_settings": settings,
        "reranker_fingerprint": reranker_fingerprint,
    }


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
    elif args.command == "improve":
        base_config = load_config(args.config)
        improvement_config = load_config(args.improvement_config)
        logging.info("Loading parquet inputs from %s", args.data_dir)
        train, _, items = load_frames(args.data_dir)
        report = run_improvement(
            train,
            items,
            base_config,
            improvement_config,
            data_dir=args.data_dir,
            cache_dir=args.cache_dir,
            max_queries=args.max_queries,
            rebuild_cache=args.rebuild_cache,
        )
        report["improvement_config"] = improvement_config
        write_json(report, args.output)
    elif args.command == "evaluate-reranker":
        base_config = load_config(args.config)
        reranker_config = load_config(args.reranker_config)
        validate_reranker_config(reranker_config)
        _, settings, model_overrides, _, base_report = _selection(
            base_config, args.weights_json
        )
        if settings is None or base_report is None:
            raise ValueError("evaluate-reranker requires a report with selected_settings")
        base_config.update(model_overrides)
        logging.info("Loading parquet inputs from %s", args.data_dir)
        train, _, items = load_frames(args.data_dir)
        reranker_report = run_reranker_evaluation(
            train,
            items,
            base_config,
            settings,
            reranker_config,
            data_dir=args.data_dir,
            cache_dir=args.cache_dir,
            rebuild=args.rebuild_cache,
        )
        report = dict(base_report)
        report["reranker"] = reranker_report
        write_json(report, args.output)
        if not bool(reranker_report["gate"]["pass"]):
            raise RuntimeError(
                f"Frozen reranker failed the preregistered gate; report: {args.output}"
            )
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
