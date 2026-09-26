# Кандидатогенерация объявлений для поиска услуг Авито

Решение строит до 50 кандидатов из `benchmark_items.parquet` для каждого запроса и сохраняет строго валидный `answer.csv`. Оно полностью работает локально, не использует внешние API и не содержит ответов, привязанных к конкретным `query_id`.

## Подход

Основной сигнал — разреженный информационный поиск по нескольким полям:

- word TF-IDF с русским Snowball stemming и униграммами/биграммами сопоставляет `search_query` с заголовком, параметрами и ограниченным началом описания;
- char TF-IDF по заголовку устойчив к опечаткам и словоформам; длинные параметры и описания в этот индекс не входят, чтобы не раздувать память и не усиливать keyword stuffing;
- текст поисковых фильтров образует отдельный слабый канал. Фильтры иногда шумные, поэтому они не смешиваются с основным запросом с большим весом;
- global lexical pool объединяется с location-aware lexical pool. Это позволяет короткому массовому запросу сохранить локальные объявления до reranking;
- клики train дают transfer от точного и близкого текста запроса, общую популярность и сглаженные совместимости `search_category → item_category_id` и `search_location_id → item_location_id`;
- категория и география — только мягкие признаки. Они никогда не отсекают объявление: глобальная категория `0`, соседний город или редкая корректная пара остаются доступны.

Окончательный score — взвешенная сумма каналов. При равном score используется возрастающий `item_id`, поэтому результат детерминирован. Реализация основана на [TfidfVectorizer из scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) и [RussianSnowballStemmer из NLTK](https://www.nltk.org/api/nltk.stem.snowball.html).

## Установка

Нужен Python 3.12. Версии библиотек закреплены:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

В каталоге `data/` должны находиться:

```text
data/train.parquet
data/benchmark_queries.parquet
data/benchmark_items.parquet
```

Исходный архив доступен по [ссылке организаторов](https://disk.yandex.ru/d/sNhfo0YOjGtufg), его SHA-256: `8dd3cba59201bae333c11db89c5111198fa10cc52a70c70bdc57bd6a248fb777`. `artifacts/input_manifest.json` содержит зафиксированные размеры, число строк и SHA-256 распакованных файлов. Parquet-файлы, кэш и окружение исключены из Git; сырые описания могут содержать контакты и нигде в отчёты не выводятся.

## Запуск

Проверить входы без вывода сырых текстов:

```bash
.venv/bin/avito-retrieval audit \
  --data-dir data \
  --output artifacts/audit.json
```

Настроить веса на cold split и один раз проверить их на независимом confirm fold:

```bash
.venv/bin/avito-retrieval evaluate \
  --data-dir data \
  --config config.json \
  --output artifacts/evaluation.json
```

`search_query` нормализуется и целиком назначается в fold через SHA-256. Все строки одной текстовой группы исключаются из обучающей истории. Во время tune исключены и tune, и confirm folds; при confirm используются все остальные строки, включая уже использованный tune fold. Полные наборы `search_*` агрегируются в query units, а релевантные `item_id` дедуплицируются.

Offline-корпус фиксирован до выбора query units: это объединение benchmark-корпуса и всех уникальных train-объявлений. Он больше тестового корпуса, содержит все offline targets и много дополнительных distractors. По умолчанию стабильно выбирается до 600 query units на каждый fold. `--max-queries 50` полезен только для короткой инженерной проверки; итоговые веса следует получать с настройками по умолчанию.

Отчёт содержит tune grid, confirm Recall@50, сравнение чистого lexical и hybrid вариантов, абляции history/metadata, а также slices по наличию фильтра, длине запроса, категории, частоте локации и наличию relevant item в обучающей истории. Отдельный seen/unseen slice важен, потому что большая часть benchmark-корпуса не встречается среди train-кликов.

### Полученные offline-результаты

Зафиксированный прогон из `artifacts/evaluation.json` дал Recall@50 `0.761963` на 600 tune query units и `0.719188` на 600 независимых confirm units. Tune fit вместе с retrieval занял 516 секунд, confirm refit и retrieval — 103 секунды. Выбраны веса:

```json
{
  "word": 1.0,
  "char": 0.55,
  "filter": 0.0,
  "history": 0.25,
  "category": 0.05,
  "location": 0.7,
  "popularity": 0.03
}
```

Абляции на confirm показывают, откуда берётся результат:

| Вариант | Recall@50 |
|---|---:|
| word TF-IDF | 0.171667 |
| word + char | 0.288162 |
| hybrid без metadata | 0.291218 |
| hybrid без history | 0.718355 |
| полный hybrid | 0.719188 |

Главный прирост даёт география до и после candidate pooling. Tuning отключил шумный filter-канал. History добавил лишь `0.000833` Recall@50 на confirm, поэтому этот эффект нельзя считать устойчивым без дополнительных folds. Для 423 query units с хотя бы одним unseen relevant item recall составил `0.722253`, для 177 полностью seen units — `0.711864`: итог не держится только на запоминании train item ID.

У offline-оценки есть три ограничения. Каталог из 515 895 объявлений больше финального benchmark-корпуса из 189 212, поэтому абсолютная сложность retrieval отличается. В confirm не оказалось запросов с `search_category=0`, хотя в benchmark их 222, и это создаёт distribution shift для global-category slice. Истинная разметка и score платформы недоступны, поэтому offline Recall@50 не является оценкой будущего leaderboard score.

Построить финальный ответ на всём train с весами из offline-отчёта:

```bash
.venv/bin/avito-retrieval predict \
  --data-dir data \
  --config config.json \
  --weights-json artifacts/evaluation.json \
  --cache-dir cache \
  --output answer.csv
```

Зафиксированный `answer.csv` содержит 2 452 строки данных и 50 уникальных кандидатов в каждой строке. Его SHA-256:

```text
3cf35dc7fa80cc879fd953fcd8a73559cf31806049e8b915d218740d3a6f7931
```

Независимый повтор команды с пустым кэшем создал побайтно идентичный файл с тем же SHA-256.

Первый запуск сохраняет индекс в `cache/<fingerprint>/model.joblib`. Fingerprint учитывает полные SHA-256 parquet-файлов, их схемы, конфигурацию, веса, версию формата кэша и SHA-256 исходных модулей модели. Повторный запуск переиспользует только точно соответствующий кэш. `--no-cache` принудительно перестраивает индекс.

Независимая проверка перед отправкой:

```bash
.venv/bin/avito-retrieval validate \
  --data-dir data \
  --answer answer.csv \
  --output artifacts/validation.json
```

Validator проверяет ровно две колонки, полный набор уникальных `query_id`, формат 16-символьных ID, строчный hex для `item_id`, принадлежность benchmark-корпусу, один пробел между ID, отсутствие повторов и лимит 50. Все ID читаются и записываются как строки; CSV сохраняется без индекса.

Без editable install те же команды запускаются как `PYTHONPATH=src .venv/bin/python -m avito_retrieval ...`.

## Проверка кода

Детерминированные тесты используют только стандартный `unittest`:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Они покрывают missing text, стабильный cold fold, агрегацию релевантности, tie-break, повторяемость retrieval и ошибки регистра `item_id`. План с зависимостями и checkpoint-ами находится в `WORK_PLAN.md`.

## Что показал анализ ошибок

Решение явно проверяет четыре ожидаемых источника промахов:

- короткие общие запросы теряют локальные объявления в global top-k — добавлен location-aware pool до reranking;
- длинные описания и параметры содержат повторения — длина ограничена, char-канал оставлен title-only;
- фильтры могут противоречить основному тексту — это отдельный настраиваемый канал, который tuning в фактическом прогоне отключил;
- история может переоценивать warm validation — её confirm-прирост оказался минимальным, а seen/unseen slices проверены отдельно.

Численные выводы о качестве следует брать из воспроизводимого `artifacts/evaluation.json`, а не из train overlap или отдельных просмотренных примеров.
