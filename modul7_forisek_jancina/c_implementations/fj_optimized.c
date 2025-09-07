#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

// mul_mod für n < 2^32
static uint64_t mul_mod(uint64_t a, uint64_t b, uint64_t mod)
{
    return (uint64_t)(a * b) % mod;
}

// Forisek-Jancina optimierte Hash-Tabelle für 32-bit
static const uint32_t FJ_BASES_32[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47};

// Schnelle Modular-Exponentiation
uint64_t mod_exp(uint64_t base, uint64_t exp, uint64_t mod)
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

// Hash-Funktion für Basis-Auswahl
uint32_t fj_hash(uint32_t n)
{
    return (n ^ (n >> 16)) % 15;
}

// Optimierter Forisek-Jancina Test
bool forisek_jancina_test(uint32_t n)
{
    if (n < 2)
        return false;
    // Quadratzahlen eliminieren (z.B. 121 = 11^2)
    uint32_t r = (uint32_t)sqrt((double)n);
    if (r * r == n)
        return false;
    if (n == 2 || n == 3 || n == 5 || n == 7)
        return true;
    if (n % 2 == 0 || n % 3 == 0 || n % 5 == 0 || n % 7 == 0)
        return false;

    uint32_t hash = fj_hash(n);
    uint32_t base = FJ_BASES_32[hash];

    uint32_t d = n - 1;
    int cnt = 0;
    while ((d & 1) == 0)
    {
        d >>= 1;
        cnt++;
    }

    uint64_t x = mod_exp(base, d, n);
    if (x == 1 || x == n - 1)
        return true;
    for (int i = 0; i < cnt - 1; i++)
    {
        x = mul_mod(x, x, n);
        if (x == n - 1)
            return true;
    }
    return false;
}

// Benchmark-Funktion
void benchmark_fj_test(uint32_t start, uint32_t count)
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
