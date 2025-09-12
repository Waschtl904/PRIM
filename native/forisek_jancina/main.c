// main.c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

extern uint64_t fj32_c(uint64_t n);

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <n>\n", argv[0]);
        return 1;
    }
    uint64_t n = strtoull(argv[1], NULL, 10);
    uint64_t result = fj32_c(n);
    printf("Result for %llu: %llu\n", (unsigned long long)n, (unsigned long long)result);
    return 0;
}
