# Marketplace Search & Ranking

**Hybrid retrieval and learned reranking for Russian service listings.**

A local search pipeline that returns up to 50 listings per query. It combines
lexical matching, a fine-tuned RuBERT bi-encoder, behavioral signals and gradient
boosting. This repository is a personal bootcamp project on organizer-provided
data; it does not describe Avito's internal search system.

## Pipeline

```text
Query + filters + location
  -> word/character TF-IDF + dense retrieval + train-only history
  -> merged candidate pool + metadata and lexical/dense features
  -> gradient-boosted reranker -> top 50 listings
```

The dense encoder uses in-batch negatives and mined hard negatives. Evaluation
groups normalized query text to keep held-out query history out of training.
The offline catalog contains 515,895 listings; it differs from the submission
catalog, so offline and platform scores are reported separately.

## Results and evidence

Metric: macro **Recall@50** over aggregated query units. These are previously
recorded repository results, not new runs performed for this README update.

| Comparison | Protocol | Recall@50 | Interpretation |
| --- | --- | ---: | --- |
| HGB47 | Reserved round-6 holdout, 1,200 queries | 0.803833 | Lexical/behavioral reference |
| First dense HGB52 | Same holdout, first evaluation | 0.860736 | Delta +0.056903; cluster-bootstrap 95% CI [0.040253, 0.073896] |
| Hard negatives + larger HGB52 | Same holdout reused during development | 0.867569 | Adaptive development result; not a new independent confirmation |

The final encoder additionally uses folds 3–4. Its independent holdout effect
and latest platform score are unknown. The published platform score 0.878067
belongs to the **first** dense version. It must not be attributed to the final
hard-negative model. Click labels are incomplete and reflect historical exposure.

## Reproduce

Python 3.12. Download the organizer archive linked in the
[full protocol](docs/experiment-log.md#установка-и-данные), verify the saved input
hashes, and place its three Parquet files in `data/`.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
.venv/bin/avito-retrieval audit --data-dir data --output artifacts/audit.json

PYTHONHASHSEED=2027 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
.venv/bin/avito-retrieval predict --data-dir data --config config.json \
  --weights-json artifacts/evaluation.json \
  --reranker-config reranker_dense_config.json --cache-dir cache --output answer.csv

PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Dense training on MPS is not bitwise deterministic. The standard
`evaluate-reranker` command currently reproduces the lexical comparison, **not**
the dense holdout tables. This remains a reproducibility limitation.

## Read the implementation

- [Dense training and hard-negative mining](src/avito_retrieval/dense.py)
- [Candidate features and reranker](src/avito_retrieval/reranker.py)
- [Grouped evaluation](src/avito_retrieval/evaluation.py)
- [Full experiment history, ablations, data provenance and limitations](docs/experiment-log.md)

The experiment log retains all earlier results, failed ideas and commands.
