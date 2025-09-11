#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

// Deklariere die DLL-Funktion
#ifdef _WIN32
#define DLLIMPORT __declspec(dllimport)
#else
#define DLLIMPORT
#endif

DLLIMPORT bool baillie_psw_complete(uint64_t n);

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Usage: %s <number>\\n", argv[0]);
        return 1;
    }
    uint64_t n = strtoull(argv[1], NULL, 10);
    bool result = baillie_psw_complete(n);
    printf("%llu is %sprime\\n",
           (unsigned long long)n,
           result ? "" : "not ");
    return 0;
}
