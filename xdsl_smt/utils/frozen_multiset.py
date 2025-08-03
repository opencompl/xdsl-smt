from __future__ import annotations
from typing import Generic, Iterable, TypeVar
from dataclasses import dataclass

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class FrozenMultiset(Generic[T]):
    """A frozen multiset."""

    _contents: frozenset[tuple[T, int]]

    @staticmethod
    def from_iterable(values: Iterable[T]) -> FrozenMultiset[T]:
        items = dict[T, int]()
        for value in values:
            if value in items:
                items[value] += 1
            else:
                items[value] = 1
        return FrozenMultiset[T](frozenset(items.items()))

    def __repr__(self) -> str:
        items: list[T] = []
        for item, count in self._contents:
            items.extend([item] * count)
        return f"FrozenMultiset({items!r})"

    def __hash__(self) -> int:
        return self._contents.__hash__()
