import random
from typing import Sequence, TypeVar

T = TypeVar("T")


class Random:
    from_file: bool
    file_rands: list[int]
    rands_len: int
    index: int

    def __init__(self, seed: int | None = None):
        if seed is not None:
            random.seed(seed)
        self.from_file = False
        self.file_rands = []
        self.rands_len = 0
        self.index = 0

    def __get_rand__(self) -> int:
        result = self.file_rands[self.index]
        self.index += 1
        self.index %= self.rands_len
        return result

    def random(self) -> float:
        if self.from_file:
            result = self.__get_rand__()
            return result / 100
        return random.random()

    def choice(self, lst: Sequence[T]) -> T:
        if self.from_file:
            cur_index = self.__get_rand__() % len(lst)
            return lst[cur_index]
        return random.choice(lst)

    def choice2(self, lst: Sequence[T]) -> list[T]:
        if self.from_file:
            # assume the file provides 2 different numbers
            cur_index1 = self.__get_rand__() % len(lst)
            cur_index2 = self.__get_rand__() % len(lst)
            return [lst[cur_index1], lst[cur_index2]]
        return random.sample(lst, 2)

    def randint(self, a: int, b: int) -> int:
        if self.from_file:
            # first get the number in range [0, b-a+1)
            rand = self.__get_rand__() % (b - a + 1)
            return rand + a
        return random.randint(a, b)

    def read_from_file(self, rand_file: str):
        lst: list[int] = []
        with open(rand_file) as f:
            for line in f.readlines():
                lst += [int(x) for x in line.split()]
        self.file_rands = lst
        self.rands_len = len(lst)
        self.from_file = True
        assert self.rands_len != 0
