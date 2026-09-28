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

### Dense-канал (текущий ответ)

Поверх описанного пайплайна добавлен дообученный bi-encoder [`sergeyzh/rubert-tiny-turbo`](https://huggingface.co/sergeyzh/rubert-tiny-turbo). Конфигурация — `reranker_dense_config.json`, код — `src/avito_retrieval/dense.py`.

- Энкодер дообучается одну эпоху с `MultipleNegativesRankingLoss` (in-batch negatives, batch 256, lr `1e-4`, `max_seq_length=128`) на 183 777 уникальных парах «запрос → кликнутое объявление» только из folds 0 и 1. Fold 2 (обучение HGB) и folds 3–4 (оценка) энкодер не видит; конфиг это проверяет. Текст запроса: `search_query | search_infm_params_text`, текст объявления: `title | params[:200] | description[:300]`.
- Кандидаты: к пулу 700 аддитивно добавляются dense top-200 среди объявлений локации запроса и dense top-50 по всему каталогу. Они всегда eligible, как history-кандидаты; исходные кандидаты и их признаки не меняются.
- К 47 признакам HGB добавлены пять dense-признаков: косинус, отставание от глобального top-1, log глобального dense-ранга (глубина 1000), отставание от top-1 в локации запроса и log ранга внутри этой локации.
- HGB52 обучается на 12 000 query units fold 2 (history без folds 2–4, pool 300 + dense-кандидаты): 15 листьев, 400 итераций, L2 `10`, остальное как у HGB47; blend `0.2`.

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

Для полного пересчёта без кэша добавьте `--no-cache`.

Текущий `answer.csv` построен dense-конфигурацией. Если кэша нет, команда дообучит энкодер, закодирует каталоги, соберёт кандидатов и обучит HGB52:

```bash
PYTHONHASHSEED=2027 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
.venv/bin/avito-retrieval predict \
  --data-dir data \
  --config config.json \
  --weights-json artifacts/evaluation.json \
  --reranker-config reranker_dense_config.json \
  --cache-dir cache \
  --output answer.csv
```

Энкодер кэшируется в `cache/dense-encoder/<fingerprint>` (ключ — `train.parquet` и секция `dense`), эмбеддинги и кандидаты — рядом с HGB в `cache/reranker/<fingerprint>`. Fine-tuning на MPS не побитово детерминирован, поэтому повторное обучение с нуля даёт близкий, но не байт-в-байт одинаковый ответ. Текущий `answer.csv` построен штатным `predict` из кэша, прогретого артефактами оценки: энкодером, эмбеддингами offline-каталога и обучающими кандидатами. Благодаря этому отправляемая модель совпадает с оценённой. Полный запуск с нуля для dense-варианта не выполнялся.

Проверить итоговый CSV:

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

Вариант до HGB47 получил оценку платформы `0.814074`, HGB47 — `0.840951`. Offline Recall и результат платформы рассчитаны на разных выборках и в разных условиях, поэтому их нельзя сравнивать напрямую.

Штатный CLI заново обучил HGB30 и HGB47, используя независимо проверенные неизменяемые lexical index и train candidate bundle. Деревья обеих моделей совпали с frozen references, а prediction при другом `PYTHONHASHSEED` побайтно совпал с prototype. Полный запуск всех этапов с `--no-cache` отдельно не выполнялся. Ответ HGB47 содержал 2 452 строки и ровно 50 уникальных кандидатов в каждой; его SHA-256 — `41eaef84135070089ff87c9f85d98dc07574c4e27a3fb0fd331e43d9abcb75ba`.

### Dense HGB52

Приведённые ниже метрики получены экспериментальными скриптами оценки. Штатная команда `evaluate-reranker` пока не поддерживает корректное сравнение HGB47 с dense-конфигурацией и не воспроизводит эти таблицы. Команда `predict` с `--reranker-config reranker_dense_config.json` поддерживает dense-канал.

Сравнение с CURRENT HGB47 (pool 700, blend 0.2) на том же offline-каталоге из 515 895 объявлений. Сначала — четыре уже использованных dev-набора по 600 запросов; на них выбирались только глубина dense-кандидатов и blend:

| Набор | HGB47 | Dense HGB52 | Δ | Candidate oracle |
|---|---:|---:|---:|---:|
| round-3 fold 3 | 0.824861 | 0.874861 | +0.050000 | 0.9375 → 0.9708 |
| round-3 fold 4 | 0.842287 | 0.899931 | +0.057644 | 0.9261 → 0.9698 |
| round-4 fold 3 | 0.797361 | 0.856389 | +0.059028 | 0.8974 → 0.9529 |
| round-4 fold 4 | 0.838194 | 0.870250 | +0.032056 | 0.9365 → 0.9632 |

Затем замороженная модель один раз проверена на reserved holdout round 6 (по 600 запросов, текстовые кластеры не пересекаются с прежними выборками; в round 7 не использовался):

| Fold | HGB47 | Dense HGB52 | Δ | Cluster-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| 3 | 0.798778 | 0.863417 | +0.064639 | [0.041412; 0.088435] |
| 4 | 0.808889 | 0.858056 | +0.049167 | [0.026182; 0.073761] |
| pooled | 0.803833 | 0.860736 | +0.056903 | [0.040253; 0.073896] |

Все срезы размером от 100 запросов положительны, минимум — `+0.0188` (cross-location, fold 4). Отрицателен только малый срез `relevant_multiple` fold 4 (`n=26`, `−0.0577`). Без dense-признаков те же 12 000 обучающих запросов дают `−0.0025` на dev, а только dense-признаки без новых кандидатов — `+0.0259`. Оставшаяся часть прироста приходится на dense-кандидатов, которые поднимают candidate oracle на 3–6 п.п.

`answer.csv` содержит 2 452 строки по 50 уникальных кандидатов; в среднем 69% top-50 совпадает с прежним ответом. SHA-256:

```text
54243ed49e8ea8e1abf29ec0cae2258c4a92aed54f43dd84138fe6fc7fb147c9
```

Оценка платформы для нового файла пока неизвестна.

## Тесты

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Тесты проверяют нормализацию missing text, cold folds, строгий validator, стабильные tie-breaks, score censoring, train-only transfer, deterministic sampling и weights, eligible-only maxima, раздельные candidate masks, fixed-union metadata, эквивалентность пакетной и цельной сборки кандидатов, preregistered sampling helpers и включение HGB только после успешного gate. Для dense-канала проверяются: объединение локальных и глобальных кандидатов, все пять признаков, аддитивность и always-eligible dense-кандидатов, запрет пересечения folds энкодера с folds обучения и оценки HGB, выравнивание признаков.

## Анализ ошибок и ограничения

Наиболее заметный разрыв прежнего решения был у relevant items из другой локации. Усиление одного общего location-веса ухудшало часть таких запросов. Query-specific locality, расстояние до центра локации и взаимодействия HGB улучшили этот срез без общего жёсткого location-фильтра.

Расширение inference pool с 300 до 700 повышает candidate oracle, а metadata и lexical coverage помогают упорядочить новые кандидаты. HGB всё ещё не может вернуть relevant item, отсутствующий в pool 700. На защищённых срезах gate пройден, но малый one-token slice fold 4 (`n=43`) ухудшился на `−0.023256`; этот результат описательный и остаётся риском.

Offline-каталог больше benchmark-корпуса из 189 212 объявлений, поэтому сложность retrieval отличается. Train почти не содержит запросов с `search_category=0`, тогда как в benchmark их 222, и этот срез оценён недостаточно. Клики неполны и отражают прошлое поведение. Истинная benchmark-разметка недоступна, а новый файл ещё не оценён платформой.

Dense-канал хорошо справляется с опечатками и перефразированием («ремонт болкона» → остекление балконов), но маленький энкодер иногда путает морфологически близкие слова: по запросу «виниры» в top-6 попадают «винный шкаф» и «винное казино». Dense-ранги считаются по текущему каталогу, а benchmark-корпус в 2.7 раза меньше offline-каталога, поэтому распределения рангов на платформе сдвинуты. Косинусные признаки от размера каталога не зависят.
