#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

// Export-Makros für DLL-Build unter MSVC und andere Compiler
#ifdef _MSC_VER
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    // mul_mod für n < 2^32
    static uint64_t mul_mod(uint64_t a, uint64_t b, uint64_t mod)
    {
        return (uint64_t)(a * b) % mod;
    }

    // Optimierte deterministische Basen für verschiedene Bereiche
    static const uint32_t FJ_BASES_SMALL[] = {2, 7, 61}; // Für Zahlen bis 2^32
    static const int FJ_BASES_SMALL_COUNT = 3;

    // Schnelle Modular-Exponentiation
    static uint64_t mod_exp(uint64_t base, uint64_t exp, uint64_t mod)
    {
        uint64_t result = 1;
        base %= mod;
        while (exp)
        {
            if (exp & 1)
                result = mul_mod(result, base, mod);
            base = mul_mod(base, base, mod);
            exp >>= 1;
        }
        return result;
    }

    // Miller-Rabin-Test mit einer spezifischen Basis
    static bool miller_rabin_witness(uint32_t n, uint32_t a)
    {
        if (a % n == 0)
            return true; // a ≡ 0 (mod n), behandle als "wahrscheinlich prim"

        uint32_t d = n - 1;
        int s = 0;
        while ((d & 1) == 0)
        {
            d >>= 1;
            s++;
        }

        uint64_t x = mod_exp(a, d, n);
        if (x == 1 || x == n - 1)
            return true;

        for (int i = 0; i < s - 1; i++)
        {
            x = mul_mod(x, x, n);
            if (x == n - 1)
                return true;
        }
        return false;
    }

    // Optimierter Forisek-Jancina Test mit mehreren deterministischen Basen
    EXPORT bool forisek_jancina_test(uint32_t n)
    {
        if (n < 2)
            return false;

        // Spezialfälle für kleine Primzahlen
        if (n == 2 || n == 3 || n == 5 || n == 7)
            return true;

        // Schnelle Eliminierung für gerade Zahlen und kleine Faktoren
        if (n % 2 == 0 || n % 3 == 0 || n % 5 == 0 || n % 7 == 0)
            return false;

        // Quadratzahlen eliminieren (z.B. 121 = 11^2)
        uint32_t r = (uint32_t)sqrt((double)n);
        if (r * r == n)
            return false;

        // Teste alle deterministischen Basen
        for (int i = 0; i < FJ_BASES_SMALL_COUNT; i++)
        {
            if (!miller_rabin_witness(n, FJ_BASES_SMALL[i]))
                return false; // Definitiv zusammengesetzt
        }

        return true; // Wahrscheinlich prim
    }

    // Benchmark-Funktion (unverändert)
    EXPORT void benchmark_fj_test(uint32_t start, uint32_t count)
    {
        printf("Benchmarking Forisek-Jancina Test...\n");
        int primes = 0;
        for (uint32_t i = start; i < start + count; i++)
        {
            if (forisek_jancina_test(i))
                primes++;
        }
        printf("Found %d primes in [%u, %u]\n",
               primes, start, start + count - 1);
    }

#ifdef __cplusplus
}
#endif
