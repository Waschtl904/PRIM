#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

// Deklarationen aus bisherigen Modulen
extern bool forisek_jancina_test(uint32_t n);
extern bool baillie_psw(uint64_t n);

bool hybrid_test(uint64_t n)
{
    // Für 32-bit Zahlen Forisek-Jancina, sonst Baillie-PSW
    if (n <= UINT32_MAX)
        return forisek_jancina_test((uint32_t)n);
    return baillie_psw(n);
}

int main()
{
    uint64_t tests[] = {
        97,           // PRIME (32-bit)
        121,          // COMPOSITE (32-bit)
        104729,       // PRIME (32-bit)
        2147483647ULL // PRIME (32-bit Grenze)
    };
    for (int i = 0; i < 4; i++)
    {
        printf("%llu: %s\n",
               tests[i],
               hybrid_test(tests[i]) ? "PRIME" : "COMPOSITE");
    }
    return 0;
}
