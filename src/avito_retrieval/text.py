from __future__ import annotations

import functools
import hashlib
import re
import unicodedata
from typing import Iterable

from nltk.stem.snowball import RussianStemmer
from pandas.api.types import is_scalar
import pandas as pd

_NON_WORD = re.compile(r"[^0-9a-zа-яё]+", re.IGNORECASE)
_TOKEN = re.compile(r"[0-9a-zа-яё]+", re.IGNORECASE)
_STEMMER = RussianStemmer()


def normalize_text(value: object) -> str:
    """Normalize text without changing identifiers or relying on external models."""
    if value is None or (is_scalar(value) and bool(pd.isna(value))):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower().replace("ё", "е")
    return " ".join(_NON_WORD.sub(" ", text).split())


@functools.lru_cache(maxsize=300_000)
def _stem_token(token: str) -> str:
    if token.isdigit() or not re.search(r"[а-я]", token):
        return token
    return _STEMMER.stem(token)


def stem_tokenize(text: str) -> list[str]:
    """Tokenizer for TfidfVectorizer; kept at module level so caches are serializable."""
    normalized = normalize_text(text)
    return [_stem_token(token) for token in _TOKEN.findall(normalized)]


def stable_fold(search_query: object, modulo: int) -> int:
    normalized = normalize_text(search_query)
    digest = hashlib.sha256(normalized.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulo


def stable_rank(parts: Iterable[object], seed: int) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(f"{seed}\x1e{payload}".encode("utf-8")).hexdigest()


def query_document(row: object) -> str:
    query = normalize_text(getattr(row, "search_query", ""))
    return query


def query_filter_document(row: object) -> str:
    return normalize_text(getattr(row, "search_infm_params_text", ""))


def item_word_document(row: object, description_chars: int, params_chars: int) -> str:
    title = normalize_text(getattr(row, "item_title_raw", ""))
    params = normalize_text(getattr(row, "item_infm_params_text", ""))[:params_chars]
    description = normalize_text(getattr(row, "item_description_raw", ""))[:description_chars]
    # Repetition is a transparent field weight before TF-IDF normalization.
    return " ".join([title, title, title, params, params, description]).strip()


def item_char_document(row: object) -> str:
    title = normalize_text(getattr(row, "item_title_raw", ""))
    # Character n-grams intentionally use title only: item params may exceed 19k
    # characters and otherwise dominate memory as well as the lexical signal.
    return title


def item_filter_document(row: object, params_chars: int) -> str:
    return normalize_text(getattr(row, "item_infm_params_text", ""))[:params_chars]
