#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

// Simple mod_mul (Achtung: nur für Demonstrationswerte < 2^32 sicher)
static uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t m)
{
    return (a * b) % m;
}

static uint64_t mod_pow(uint64_t a, uint64_t d, uint64_t m)
{
    uint64_t r = 1;
    a %= m;
    while (d)
    {
        if (d & 1)
            r = mod_mul(r, a, m);
        a = mod_mul(a, a, m);
        d >>= 1;
    }
    return r;
}

// Deterministische Miller-Rabin-Basen für 64-Bit
static const uint64_t MR_BASES_64[] = {
    2ULL, 325ULL, 9375ULL, 28178ULL, 450775ULL, 9780504ULL, 1795265022ULL};

bool baillie_psw(uint64_t n)
{
    if (n < 2)
        return false;
    if (n % 2 == 0)
        return n == 2;

    // Schreibe n-1 = d * 2^s
    uint64_t d = n - 1;
    int s = 0;
    while ((d & 1) == 0)
    {
        d >>= 1;
        s++;
    }

    // Deterministische Runden für 64-Bit
    for (int i = 0; i < 7; i++)
    {
        uint64_t a = MR_BASES_64[i] % (n - 2) + 2;
        uint64_t x = mod_pow(a, d, n);
        if (x == 1 || x == n - 1)
            continue;
        bool composite = true;
        for (int r = 1; r < s; r++)
        {
            x = mod_mul(x, x, n);
            if (x == n - 1)
            {
                composite = false;
                break;
            }
        }
        if (composite)
            return false;
    }
    return true;
}
