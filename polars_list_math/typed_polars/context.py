"""Runtime bindings for schema fields with dynamic physical columns."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, Self, runtime_checkable


@runtime_checkable
class ContextFieldProtocol(Protocol):
    """Minimal field identity required for runtime context bindings."""

    @property
    def context_source(self) -> object: ...

    @property
    def context_path(self) -> tuple[object, ...]: ...


class Context:
    """Bind dynamic schema fields to ordered runtime keys."""

    def __init__(self) -> None:
        self._bindings: dict[tuple[object, ...], tuple[str, ...]] = {}

    def bind(self, field: ContextFieldProtocol, keys: Iterable[str]) -> Self:
        """Bind a dynamic field to its complete ordered key set."""
        if not isinstance(field, ContextFieldProtocol):
            raise TypeError("field must implement ContextFieldProtocol")
        self._bindings[field.context_path] = _validate_keys(keys)
        return self

    def _bind_path(
        self,
        field: ContextFieldProtocol,
        path: tuple[object, ...],
        keys: Iterable[str],
    ) -> None:
        self._bindings[path] = _validate_keys(keys)

    def keys_for(
        self,
        field: ContextFieldProtocol,
        path: tuple[object, ...] | None = None,
    ) -> tuple[str, ...]:
        """Return bound keys, treating an absent binding as an empty set."""
        if path is None:
            path = field.context_path
        exact = self._bindings.get(path)
        if exact is not None:
            return exact
        return self._bindings.get((field.context_source,), ())

    def signature(
        self,
    ) -> frozenset[tuple[tuple[object, ...], tuple[str, ...]]]:
        """Return an immutable cache key for the current bindings."""
        return frozenset(self._bindings.items())


def _validate_keys(keys: Iterable[str]) -> tuple[str, ...]:
    if isinstance(keys, (str, bytes)):
        raise TypeError("FlatDict keys must be an iterable of strings")
    materialized = tuple(keys)
    if any(not isinstance(key, str) or not key for key in materialized):
        raise TypeError("FlatDict keys must be non-empty strings")
    if len(set(materialized)) != len(materialized):
        raise TypeError("FlatDict keys must be unique")
    return materialized


def context_signature(
    context: Context | None,
) -> frozenset[tuple[tuple[object, ...], tuple[str, ...]]]:
    """Return a stable empty signature when no context was supplied."""
    return frozenset() if context is None else context.signature()


def context_keys(
    context: Context | None,
    field: ContextFieldProtocol,
    path: tuple[object, ...] | None = None,
) -> tuple[str, ...]:
    """Resolve field keys with absent context meaning no dynamic fields."""
    return () if context is None else context.keys_for(field, path)
