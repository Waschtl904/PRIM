from sympy.ntheory.primetest import isprime as is_strong_lucas_prp


def miller_rabin(n: int) -> bool:
    """
    Deterministischer Miller–Rabin-Test für 32-Bit-Zahlen
    mit Basen 2, 7, 61.
    """
    n = int(n)
    if n < 2 or n % 2 == 0:
        return n == 2
    d, s = n - 1, 0
    while d & 1 == 0:
        d //= 2
        s += 1
    for a in (2, 7, 61):
        if a >= n:
            continue
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def lucas_prp(n: int) -> bool:
    """
    Lucas-PRP-Test via SymPy.
    Gibt True zurück, wenn n Lucas wahrscheinliche Primzahl ist.
    """
    return is_strong_lucas_prp(int(n))


def baillie_psw(n: int) -> bool:
    """
    Kombination aus Miller-Rabin und Lucas-PRP (Baillie-PSW).
    """
    return miller_rabin(n) and lucas_prp(n)
