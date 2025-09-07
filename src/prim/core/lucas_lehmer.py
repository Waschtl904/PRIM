def is_mersenne_prime(p: int) -> bool:
    """
    Lucas–Lehmer-Test für Mersenne-Primzahlen.

    Prüft, ob 2**p - 1 eine Primzahl ist (nur gültig für p > 2).
    Rückgabe: True, wenn M(p) = 2**p - 1 prim ist, sonst False.
    """
    if p < 2:
        return False
    # Mersenne-Zahl
    M = (1 << p) - 1
    # Spezielle Fälle: kleine p
    if p == 2:
        return True  # 2**2 - 1 = 3

    # Lucas-Lehmer-Folge: s_0 = 4, s_{n+1} = s_n^2 - 2 mod M
    s = 4
    for _ in range(p - 2):
        s = (s * s - 2) % M
    return s == 0
