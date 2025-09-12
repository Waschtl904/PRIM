from prim.core.forisek_jancina import forisek_jancina_test

__all__ = ["is_prime", "batch_primality_test"]

is_prime = forisek_jancina_test


def batch_primality_test(numbers):
    """
    Vereinfacht: Für eine Liste von Zahlen forisek_jancina_test anwenden.
    """
    return [forisek_jancina_test(n) for n in numbers]
