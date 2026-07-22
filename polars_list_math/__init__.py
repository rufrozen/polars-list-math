from __future__ import annotations

from ._list_combinations import install as _install_list_combinations
from ._list_mean_similarity import install as _install_list_mean_similarity
from ._list_similarity import (
    install as _install_list_similarity,
)
from ._list_similarity import (
    py_list_similarity,
)
from ._list_zip import install as _install_list_zip

__all__ = [
    "install",
    "py_list_similarity",
]


def install(*, overwrite: bool = False) -> None:
    _install_list_combinations(overwrite=overwrite)
    _install_list_zip(overwrite=overwrite)
    _install_list_similarity(overwrite=overwrite)
    _install_list_mean_similarity(overwrite=overwrite)


install()
