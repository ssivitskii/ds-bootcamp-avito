# Поиск кандидатов для услуг Авито

Решение полностью локально строит до 50 объявлений из `benchmark_items.parquet` для каждого поискового запроса. В коде нет ответов, привязанных к `query_id`, внешних API и ручных исправлений результата.

## Данные и признаки

Первый этап формирует фиксированный candidate set:

- word TF-IDF с русским Snowball stemming ищет по заголовку, параметрам и ограниченному началу описания;
- char TF-IDF по заголовку помогает с опечатками и словоформами, не индексируя длинные параметры и описания;
- отдельный filter-канал сопоставляет поисковые фильтры с параметрами объявления;
- global pool объединяется с текстовыми результатами для трёх наиболее совместимых локаций, до 300 кандидатов на канал;
- word, char и filter similarity точно пересчитываются для каждого объявления из объединения, поэтому кандидат не получает искусственный ноль только из-за канала, через который он был найден;
- train-only признаки включают мягкую совместимость категории и локации, популярность, историю и перенос `item_microcat_id` от 16 ближайших нормализованных train-запросов.

Базовый score — взвешенная сумма этих сигналов. Второй этап применяет `HistGradientBoostingClassifier` к тому же candidate set и смешивает его logit с базовым score в пропорции `base + 0.4 × logit`. Модель использует 30 признаков:

- девять исходных каналов, базовый score и шесть значений относительно максимума внутри запроса;
- четыре взаимодействия текста, microcategory и локации;
- шесть максимумов признаков внутри запроса;
- четыре locality-признака: перенесённая склонность запроса к точной локации, её взаимодействие с location prior, сглаженный locality prior микрокатегории и их взаимодействие.

У линейного base score веса filter и history равны нулю, но их исходные значения входят в HGB. Поэтому они не считаются удалёнными признаками. Candidate IDs до и после HGB строго одинаковы; меняется только порядок. При равных scores используется возрастающий `item_id`.

Зафиксированная модель: 7 листьев, 200 итераций, learning rate `0.07`, `min_samples_leaf=50`, L2 `3.0`, seed `20260928`. Она обучается только на 2 400 query units из fold 2 при behavioral history из folds 0 и 1. Для каждого запроса используются первые 200 кандидатов базового ранжирования, 50 детерминированно выбранных кандидатов и все доступные positives; вес positive равен `100 / число всех relevant items`. После проверки HGB не переобучается на full train. Для финального поиска на full train обновляются только retrieval, behavioral и locality-признаки.

Текстовый поиск основан на [TfidfVectorizer из scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html), stemming — на [RussianSnowballStemmer из NLTK](https://www.nltk.org/api/nltk.stem.snowball.html).

## Установка и данные

Нужен Python 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

В каталоге `data/` должны лежать:

```text
data/train.parquet
data/benchmark_queries.parquet
data/benchmark_items.parquet
```

Исходный архив доступен по [ссылке организаторов](https://disk.yandex.ru/d/sNhfo0YOjGtufg), SHA-256 архива: `8dd3cba59201bae333c11db89c5111198fa10cc52a70c70bdc57bd6a248fb777`. Хеши и размеры распакованных файлов сохранены в `artifacts/input_manifest.json`. Сырые описания могут содержать контакты, поэтому они не выводятся в отчёты.

## Воспроизведение

Проверить входы:

```bash
.venv/bin/avito-retrieval audit \
  --data-dir data \
  --output artifacts/audit.json
```

С нуля обучить frozen HGB и воспроизвести независимую проверку:

```bash
PYTHONHASHSEED=2027 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
.venv/bin/avito-retrieval evaluate-reranker \
  --data-dir data \
  --config config.json \
  --reranker-config reranker_config.json \
  --weights-json artifacts/evaluation.json \
  --cache-dir cache \
  --output artifacts/improvement/round3/evaluation.reproduced.json
```

Команда проверяет preregistered hashes query units, отсутствие text leakage, неизменность candidate IDs, 10 000 cluster-bootstrap повторов и все gate-условия. Результат пишется в исключённый из Git каталог и не затирает канонический отчёт. Первый запуск строит fixed offline-каталог и feature bundles; повторный использует content-addressed cache. `--rebuild-cache` принудительно пересчитывает его.

Построить ответ. `predict` автоматически читает прошедшую gate конфигурацию reranker из канонического отчёта; если HGB-кэша нет, модель детерминированно обучается по frozen protocol:

```bash
PYTHONHASHSEED=2027 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
.venv/bin/avito-retrieval predict \
  --data-dir data \
  --config config.json \
  --weights-json artifacts/evaluation.json \
  --cache-dir cache \
  --output answer.csv
```

Для полного пересчёта без кэша добавьте `--no-cache`. Проверить итоговый CSV:

```bash
.venv/bin/avito-retrieval validate \
  --data-dir data \
  --answer answer.csv \
  --output artifacts/validation.json
```

Validator проверяет две точные колонки, полный набор уникальных `query_id`, строчный hex для `item_id`, принадлежность benchmark-корпусу, один пробел между ID, отсутствие повторов и лимит 50. Идентификаторы всегда читаются и записываются как строки, CSV сохраняется без индекса.

Без editable install команды можно запускать как `PYTHONPATH=src .venv/bin/python -m avito_retrieval ...`.

## Проверка качества

Нормализованный текст запроса назначается в fold через SHA-256. Все строки одной текстовой группы исключаются из соответствующей behavioral history. Offline-корпус фиксирован до выбора query units и содержит объединение benchmark-корпуса со всеми уникальными train-объявлениями — 515 895 объектов.

Параметры HGB были зафиксированы до чтения финальных проверочных выборок. Reserved fold 3 исключает все текстовые кластеры двух ранее использованных fold-3 samples; reserved fold 4 исключает все кластеры ранее использованного fold-4 sample. На каждом fold остаётся по 600 новых query units. Fold 3 использует history без folds 3 и 4, fold 4 — history без fold 4.

| Fold | Baseline Recall@50 | HGB Recall@50 | Δ | Cluster-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| reserved 3 | 0.765139 | 0.788194 | +0.023056 | [0.004443; 0.041876] |
| reserved 4 | 0.799843 | 0.812426 | +0.012583 | [-0.001887; 0.027674] |
| pooled | 0.782491 | 0.800310 | +0.017819 | [0.006231; 0.029925] |

Для точного воспроизведения bootstrap кластеры сортируются лексикографически, а выборка генерируется последовательно для каждого повтора зафиксированным RNG. Точечный прирост положителен на обоих folds; интервал fold 4 пересекает ноль, а объединённый интервал — нет. Все заранее определённые slices размером от 100 запросов имеют положительную delta; минимальный прирост — `+0.002564` на cross-location slice fold 4. Candidate oracle и candidate IDs совпадают для каждой query unit, поэтому измеренный прирост относится только к reranking.

Предыдущий отправленный вариант получил оценку платформы `0.801159`. Оценка платформы для нового файла пока неизвестна. Offline Recall и результат платформы рассчитаны на разных выборках и в разных условиях, поэтому их нельзя сравнивать напрямую.

Для проверки заново построен финальный benchmark-индекс и заново обучен HGB на сохранённых offline-признаках. Результат побайтно совпал с frozen prototype; полный запуск всех этапов с `--no-cache` отдельно не выполнялся. Итоговый `answer.csv` содержит 2 452 строки данных и ровно 50 уникальных кандидатов в каждой строке. SHA-256:

```text
0cf13dcba9049ed88a4edd94ec15912a5639bcd3d6fbc80ac7a292c93ed24d28
```

## Тесты

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Тесты проверяют нормализацию missing text, cold folds, строгий validator, стабильные tie-breaks, score censoring, train-only microcategory/locality transfer, deterministic sampling и weights, eligible-only query maxima, blend=0, preregistered sampling helpers, fold exclusions, candidate-mask equality и включение HGB только после успешного gate.

## Анализ ошибок и ограничения

Наиболее заметный остаточный разрыв baseline был у relevant items из другой локации. Усиление одного общего location-веса ухудшало часть таких запросов. Query-specific locality и взаимодействия HGB дали положительный прирост на cross-location slices, сохранив exact-location качество.

Точный пересчёт similarity исправляет score censoring между retrieval-каналами. Microcategory transfer помогает различать смысл коротких запросов. HGB улучшает порядок внутри неизменного pool, но не может вернуть relevant item, отсутствующий среди кандидатов; неизменный oracle делает эту границу явной.

Offline-каталог больше benchmark-корпуса из 189 212 объявлений, поэтому сложность retrieval отличается. Train почти не содержит запросов с `search_category=0`, тогда как в benchmark их 222, и этот срез оценён недостаточно. Клики неполны и отражают прошлое поведение. Истинная benchmark-разметка недоступна, а новый файл ещё не оценён платформой.
