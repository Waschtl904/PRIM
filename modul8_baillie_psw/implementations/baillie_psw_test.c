// baillie_psw_test.c
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

// Deklaration
bool baillie_psw(uint64_t n);

int main()
{
    uint64_t tests[] = {
        121,   // COMPOSITE
        343,   // COMPOSITE
        1021,  // PRIME
        2047,  // COMPOSITE
        3250,  // COMPOSITE
        1321,  // PRIME
        104729 // PRIME
    };
    for (int i = 0; i < 7; i++)
    {
        printf("%llu: %s\n",
               tests[i],
               baillie_psw(tests[i]) ? "PRIME" : "COMPOSITE");
    }
    return 0;
}
