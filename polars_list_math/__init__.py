from __future__ import annotations

from ._dfs_to_lazy_df import dfs_to_lazy_df
from ._expected_value_of_game import expected_value_of_game
from ._json_array_values import install as _install_json_array_values
from ._json_array_values import json_array_values
from ._json_object_items import install as _install_json_object_items
from ._json_object_items import json_object_items
from ._list_combinations import install as _install_list_combinations
from ._list_combinations import list_combinations, list_combinations_to
from ._list_mean_similarity import install as _install_list_mean_similarity
from ._list_mean_similarity import list_mean_similarity, list_mean_similarity_to
from ._list_similarity import (
    install as _install_list_similarity,
)
from ._list_similarity import (
    list_similarity,
    py_list_similarity,
)
from ._list_zip import install as _install_list_zip
from ._list_zip import list_zip
from ._url_build import url_build
from ._url_query_encode import url_query_encode

__all__ = [
    "install",
    "dfs_to_lazy_df",
    "expected_value_of_game",
    "json_array_values",
    "json_object_items",
    "list_combinations",
    "list_combinations_to",
    "list_mean_similarity",
    "list_mean_similarity_to",
    "list_similarity",
    "list_zip",
    "py_list_similarity",
    "url_build",
    "url_query_encode",
]


def install(*, overwrite: bool = False) -> None:
    _install_json_array_values(overwrite=overwrite)
    _install_json_object_items(overwrite=overwrite)
    _install_list_combinations(overwrite=overwrite)
    _install_list_zip(overwrite=overwrite)
    _install_list_similarity(overwrite=overwrite)
    _install_list_mean_similarity(overwrite=overwrite)


install()
