# Поиск кандидатов для услуг Авито

Решение полностью локально строит до 50 объявлений из `benchmark_items.parquet` для каждого поискового запроса. В коде нет ответов, привязанных к `query_id`, внешних API и ручных исправлений результата.

## Данные, retrieval и ранжирование

Первый этап собирает кандидатов:

- word TF-IDF с русским Snowball stemming ищет по заголовку, параметрам и ограниченному началу описания;
- char TF-IDF по заголовку помогает с опечатками и словоформами;
- отдельный filter-канал сопоставляет поисковые фильтры с параметрами объявления;
- global pool объединяется с текстовыми результатами для трёх наиболее совместимых локаций;
- word, char и filter similarity точно пересчитываются для каждого объявления из объединения;
- train-only сигналы включают мягкую совместимость категории и локации, популярность, историю и перенос `item_microcat_id` от 16 ближайших нормализованных train-запросов.

Базовый score — взвешенная сумма этих сигналов. Финальный reranker — `HistGradientBoostingClassifier`: 7 листьев, 200 итераций, learning rate `0.07`, `min_samples_leaf=50`, L2 `3.0`, seed `20260928`. Он использует 47 признаков:

- 30 retrieval/locality-признаков: девять исходных каналов, base score, относительные значения и максимумы внутри запроса, четыре взаимодействия и четыре train-only locality-признака;
- 13 metadata-признаков: цена, рейтинг, число отзывов, ограничения связи, совпадения локации и категории, расстояние до медианного центра локации, delivery/global-category flags и длины запроса и заголовка;
- четыре lexical coverage-признака: доля query-unigrams в полном тексте и заголовке, IDF-взвешенное покрытие и нормированная длина документа в словаре.

Центры локаций всегда считаются по фиксированному объединению train и benchmark, а признаки кандидатов переиндексируются в порядок текущего каталога. Нормирователь длины считается по текстовой матрице текущего индекса. На inference HGB ранжирует объединение с pool 700 и смешивается с базовым score как `base + 0.2 × logit`. При равных scores используется возрастающий `item_id`.

Модель обучается на 6 000 query units из fold 2 при behavioral history из folds 0 и 1. Retrieval для обучения остаётся зафиксированным на pool 300 и строится пакетами по 600 запросов. Для каждого запроса sampler сохраняет первые 200 кандидатов базового ранжирования, 50 детерминированно выбранных кандидатов и все доступные positives; вес positive равен `100 / число всех relevant items`. После проверки HGB не переобучается на full train. Для финального поиска обновляются только retrieval, behavioral и locality-признаки.

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

С нуля обучить frozen HGB и повторить проверку:

```bash
PYTHONHASHSEED=2027 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
.venv/bin/avito-retrieval evaluate-reranker \
  --data-dir data \
  --config config.json \
  --reranker-config reranker_config.json \
  --weights-json artifacts/evaluation.json \
  --cache-dir cache \
  --output artifacts/improvement/round4/evaluation.reproduced.json
```

Команда проверяет preregistered hashes query units, отсутствие text leakage, отдельный CURRENT baseline HGB30 с pool 300, additive candidate contract для pool 700, candidate oracle, 10 000 cluster-bootstrap повторов и все gate-условия. Результат пишется в исключённый из Git каталог и не затирает канонический отчёт. Первый запуск строит fixed offline-каталог и feature bundles; повторный использует content-addressed cache. `--rebuild-cache` принудительно пересчитывает кэш.

Построить ответ. `predict` автоматически читает прошедшую gate конфигурацию из канонического отчёта; если HGB-кэша нет, обе frozen-модели детерминированно обучаются по сохранённому протоколу:

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

CURRENT baseline — ранее отправленный HGB30: train2400, pool 300, blend `0.4`. Его конфигурация, unit hash и historical artifact provenance сохранены отдельно внутри `reranker_config.json`. Новый HGB47 зафиксирован на четырёх уже потреблённых dev-наборах до чтения round-4 holdout.

Round-4 fold 3 исключает все текстовые кластеры трёх ранее использованных fold-3 samples; fold 4 исключает кластеры двух fold-4 samples. На каждом fold остаётся по 600 новых query units. Fold 3 использует history без folds 3 и 4, fold 4 — history без fold 4.

| Fold | CURRENT HGB30 | HGB47 | Δ | Cluster-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| round-4 reserved 3 | 0.751944 | 0.797361 | +0.045417 | [0.027304; 0.065057] |
| round-4 reserved 4 | 0.818333 | 0.838194 | +0.019861 | [0.005042; 0.035957] |
| pooled | 0.785139 | 0.817778 | +0.032639 | [0.020694; 0.045102] |

Для точного воспроизведения bootstrap кластеры сортируются лексикографически, а выборка генерируется последовательно для каждого повтора зафиксированным RNG. Точечный прирост и доверительный интервал положительны на обоих folds и в объединении. Все заранее определённые slices размером от 100 запросов имеют положительную delta; минимальный прирост — `+0.001012`. Baseline candidate IDs входят в расширенный candidate set, а candidate oracle не уменьшается для каждой query unit.

Предыдущий отправленный вариант получил оценку платформы `0.814074`. Оценка платформы для нового файла пока неизвестна. Offline Recall и результат платформы рассчитаны на разных выборках и в разных условиях, поэтому их нельзя сравнивать напрямую.

Штатный CLI заново обучил HGB30 и HGB47, используя независимо проверенные неизменяемые lexical index и train candidate bundle. Деревья обеих моделей совпали с frozen references, а prediction при другом `PYTHONHASHSEED` побайтно совпал с prototype. Полный запуск всех этапов с `--no-cache` отдельно не выполнялся. Итоговый `answer.csv` содержит 2 452 строки данных и ровно 50 уникальных кандидатов в каждой строке. SHA-256:

```text
41eaef84135070089ff87c9f85d98dc07574c4e27a3fb0fd331e43d9abcb75ba
```

## Тесты

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Тесты проверяют нормализацию missing text, cold folds, строгий validator, стабильные tie-breaks, score censoring, train-only transfer, deterministic sampling и weights, eligible-only maxima, раздельные candidate masks, fixed-union metadata, эквивалентность пакетной и цельной сборки кандидатов, preregistered sampling helpers и включение HGB только после успешного gate.

## Анализ ошибок и ограничения

Наиболее заметный разрыв прежнего решения был у relevant items из другой локации. Усиление одного общего location-веса ухудшало часть таких запросов. Query-specific locality, расстояние до центра локации и взаимодействия HGB улучшили этот срез без общего жёсткого location-фильтра.

Расширение inference pool с 300 до 700 повышает candidate oracle, а metadata и lexical coverage помогают упорядочить новые кандидаты. HGB всё ещё не может вернуть relevant item, отсутствующий в pool 700. На защищённых срезах gate пройден, но малый one-token slice fold 4 (`n=43`) ухудшился на `−0.023256`; этот результат описательный и остаётся риском.

Offline-каталог больше benchmark-корпуса из 189 212 объявлений, поэтому сложность retrieval отличается. Train почти не содержит запросов с `search_category=0`, тогда как в benchmark их 222, и этот срез оценён недостаточно. Клики неполны и отражают прошлое поведение. Истинная benchmark-разметка недоступна, а новый файл ещё не оценён платформой.
