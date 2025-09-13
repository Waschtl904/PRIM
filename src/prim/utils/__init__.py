# src/prim/utils/__init__.py

from typing import List, Tuple


def find_pairs(data: List[int]) -> List[Tuple[int, int]]:
    """Ermittelt alle Zweier-Paare in `data`."""
    return [(data[i], data[i + 1]) for i in range(len(data) - 1)]


def chunk_list(data: List[int], size: int) -> List[List[int]]:
    """Teilt eine Liste in Chunks der Größe `size`."""
    return [data[i : i + size] for i in range(0, len(data), size)]
