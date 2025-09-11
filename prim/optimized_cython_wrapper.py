# prim/optimized_cython_wrapper.py
try:
    from prim.optimized_cython import optimized_primality_test

    def optimized_primality_test_wrapper(n: int) -> bool:
        """Wrapper für die Cython-optimierte Miller-Rabin-Funktion."""
        return bool(optimized_primality_test(n))

except ImportError:

    def optimized_primality_test_wrapper(n: int) -> bool:
        # Fallback Python-Implementierung (Minimum)
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0:
            return False

        bases = [2, 7, 61]
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1

        for a in bases:
            if a % n == 0:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True
