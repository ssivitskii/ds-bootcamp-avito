from __future__ import annotations

QUERY_COLUMNS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]

ITEM_COLUMNS = [
    "item_title_raw",
    "item_rating_reviews_count",
    "item_rating",
    "item_price",
    "item_microcat_id",
    "item_longitude",
    "item_location_id",
    "item_latitude",
    "item_is_phone_hidden",
    "item_is_message_forbidden",
    "item_infm_params_text",
    "item_id",
    "item_description_raw",
    "item_category_id",
]

TRAIN_COLUMNS = QUERY_COLUMNS + ITEM_COLUMNS

QUERY_ID_PATTERN = r"^[0-9A-Za-z]{16}$"
ITEM_ID_PATTERN = r"^[0-9a-f]{16}$"
