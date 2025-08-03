from __future__ import annotations
from typing import Generic, Iterable, TypeVar
from dataclasses import dataclass

T = TypeVar("T")


@dataclass(frozen=True)
class FrozenMultiset(Generic[T]):
    """A frozen multiset."""

    __slots__ = ("_contents",)

    _contents: frozenset[tuple[T, int]]

    def __init__(self, values: Iterable[T]):
        items = dict[T, int]()
        for value in values:
            if value in items:
                items[value] += 1
            else:
                items[value] = 1
        self.__setattr__("_contents", frozenset(items.items()))

    def __repr__(self) -> str:
        items: list[T] = []
        for item, count in self._contents:
            items.extend([item] * count)
        return f"FrozenMultiset({items!r})"

    def __hash__(self) -> int:
        return self._contents.__hash__()
