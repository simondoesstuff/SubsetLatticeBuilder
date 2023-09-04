from typing import AbstractSet, Iterable, Self


class node(frozenset):
    def __init__(self, initial: str | Iterable = None) -> None:
        if isinstance(initial, str):
            initial = [int(x) for x in initial.split()]
        super().__new__(node, initial)

    def __repr__(self) -> str:
        return f'({",".join(str(x) for x in sorted(self))})'
    
    def copy(self):
        return self
    
    def difference(self, other: Iterable) -> Self:
        return node(super().difference(other))
    
    def intersection(self, other: Iterable) -> Self:
        return node(super().intersection(other))
    
    def union(self, other: Iterable) -> Self:
        return node(super().union(other))
    
    def symmetric_difference(self, other: Iterable) -> Self:
        return node(super().symmetric_difference(other))
    
    def __and__(self, other: AbstractSet) -> Self:
        return self.intersection(other)
    
    def __or__(self, other: AbstractSet) -> Self:
        return self.union(other)
    
    def __xor__(self, other: AbstractSet) -> Self:
        return self.symmetric_difference(other)
    
    def __sub__(self, other: AbstractSet) -> Self:
        return self.difference(other)
    
    def __ge__(self, other) -> bool:
        if isinstance(other, int):
            return len(self) >= other
        return super().__ge__(other)
    
    def __gt__(self, other) -> bool:
        if isinstance(other, int):
            return len(self) > other
        return super().__gt__(other)
    
    def __le__(self, other) -> bool:
        if isinstance(other, int):
            return len(self) <= other
        return super().__le__(other)
    
    def __lt__(self, other) -> bool:
        if isinstance(other, int):
            return len(self) < other
        return super().__lt__(other)
    
    def __bool__(self) -> bool:
        return len(self) > 0


def parse_nodes(path: str) -> list[node]:
    contents = open(path).read().splitlines()
    nodes = [[int(x) for x in line.split()] for line in contents]
    nodes = [ node(x) for x in nodes ]
    return nodes


def write_nodes(path: str, nodes: list[node]) -> None:
    with open(path, 'w') as f:
        for node in nodes:
            node = sorted(node)
            f.write(' '.join([str(x) for x in node]))
            f.write('\n')

