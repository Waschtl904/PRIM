# prim/optimized_cython.py


def optimized_primality_test(n: int) -> bool:
    """
    Python-Implementierung eines Miller-Rabin-Tests als Platzhalter
    später durch Cython ersetzen.
    """
    if n < 2:
        return False
    # feste Basen für 32-bit
    bases = [2, 7, 61]
    # schreibe n-1 = d * 2^s
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    def check(a):
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            return True
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                return True
        return False

    for a in bases:
        if a % n == 0:
            return True
        if not check(a):
            return False
    return True
