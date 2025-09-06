#include <pybind11/pybind11.h>
#include <cstdint>
namespace py = pybind11;

uint32_t fj_hash_c(uint32_t x)
{
    uint32_t h = x;
    h = (((h >> 16) ^ h) * 0x45D9F3B);
    h = (((h >> 16) ^ h) * 0x45D9F3B);
    return ((h >> 16) ^ h) & 0xFF;
}

bool sprp_c(uint32_t n, uint32_t a)
{
    if (n < 2)
        return false;
    if ((n & 1) == 0)
        return n == 2;
    uint64_t d = n - 1, s = 0;
    while ((d & 1) == 0)
    {
        d >>= 1;
        ++s;
    }
    auto modpow = [&](uint64_t base, uint64_t exp)
    {
        uint64_t result = 1;
        while (exp)
        {
            if (exp & 1)
                result = (result * base) % n;
            base = (base * base) % n;
            exp >>= 1;
        }
        return result;
    };
    uint64_t x = modpow(a, d);
    if (x == 1 || x == n - 1)
        return true;
    for (uint64_t i = 1; i < s; ++i)
    {
        x = (x * x) % n;
        if (x == n - 1)
            return true;
    }
    return false;
}

bool fj32_c(uint32_t n)
{
    if (n < 2)
        return false;
    for (uint32_t p : {2u, 3u, 5u, 7u})
    {
        if (n == p)
            return true;
        if (n % p == 0)
            return false;
    }
    uint32_t h = fj_hash_c(n);
    uint32_t bases[3] = {2, 7, 61};
    return sprp_c(n, bases[h % 3]);
}

PYBIND11_MODULE(primetest, m)
{
    m.def("fj_hash_c", &fj_hash_c);
    m.def("fj32_c", &fj32_c);
}
