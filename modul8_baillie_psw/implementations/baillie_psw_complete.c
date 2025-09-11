#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

// Export-Makros für DLL-Build
#ifdef _MSC_VER
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    // ===== SICHERE 64-BIT ARITHMETIK =====

    // Sichere Modular-Multiplikation ohne Overflow
    static uint64_t safe_mod_mul(uint64_t a, uint64_t b, uint64_t mod)
    {
        if (mod == 0)
            return 0;

        uint64_t result = 0;
        a %= mod;

        while (b > 0)
        {
            if (b & 1)
            {
                // Prüfe auf Overflow vor Addition
                if (result > mod - a)
                {
                    result = result - (mod - a);
                }
                else
                {
                    result += a;
                }
            }

            if (b > 1)
            {
                // Verdopple a, prüfe Overflow
                if (a > mod - a)
                {
                    a = 2 * a - mod;
                }
                else
                {
                    a = 2 * a;
                }
            }
            b >>= 1;
        }
        return result;
    }

    // Modulare Exponentiation
    static uint64_t mod_pow(uint64_t base, uint64_t exp, uint64_t mod)
    {
        if (mod == 1)
            return 0;

        uint64_t result = 1;
        base %= mod;

        while (exp > 0)
        {
            if (exp & 1)
            {
                result = safe_mod_mul(result, base, mod);
            }
            base = safe_mod_mul(base, base, mod);
            exp >>= 1;
        }
        return result;
    }

    // ===== JACOBI-SYMBOL BERECHNUNG =====

    static int jacobi_symbol(int64_t a, uint64_t n)
    {
        if (n <= 1 || (n & 1) == 0)
            return 0; // n muss ungerade und > 1 sein

        int result = 1;
        uint64_t ua = (a < 0) ? (n - ((-a) % n)) : (a % n);

        while (ua != 0)
        {
            // Faktor 2 entfernen
            while ((ua & 1) == 0)
            {
                ua >>= 1;
                // Jacobi(2|n) = (-1)^((n²-1)/8)
                if ((n & 7) == 3 || (n & 7) == 5)
                {
                    result = -result;
                }
            }

            // Swap ua und n
            uint64_t temp = ua;
            ua = n;
            n = temp;

            // Quadratisches Reziprozitätsgesetz
            if ((ua & 3) == 3 && (n & 3) == 3)
            {
                result = -result;
            }

            ua %= n;
        }

        return (n == 1) ? result : 0;
    }

    // ===== LUCAS-SEQUENZEN BERECHNUNG =====

    // Berechnet Lucas-Sequenzen U_k, V_k und Q^k mod n
    static void lucas_sequence(uint64_t n, uint64_t k, int64_t P, int64_t Q,
                               uint64_t *U_k, uint64_t *V_k, uint64_t *Q_k)
    {
        if (k == 0)
        {
            *U_k = 0;
            *V_k = 2 % n;
            *Q_k = 1;
            return;
        }

        if (k == 1)
        {
            *U_k = 1;
            *V_k = P % n;
            *Q_k = Q % n;
            return;
        }

        // Binäre Methode - finde höchstes gesetztes Bit
        uint64_t mask = 1ULL << 63;
        while ((k & mask) == 0)
            mask >>= 1;
        mask >>= 1; // Überspringe das höchste Bit

        // Initialisiere für k=1
        uint64_t U = 1;
        uint64_t V = P % n;
        uint64_t Qpow = Q % n;

        while (mask > 0)
        {
            // Verdopple: U_{2m} = U_m * V_m
            //           V_{2m} = V_m² - 2*Q^m
            //           Q^{2m} = Q^m²
            uint64_t U_new = safe_mod_mul(U, V, n);
            uint64_t V_new = safe_mod_mul(V, V, n);
            if (V_new >= 2 * Qpow % n)
            {
                V_new = (V_new - 2 * Qpow % n) % n;
            }
            else
            {
                V_new = (V_new + n - 2 * Qpow % n) % n;
            }
            uint64_t Q_new = safe_mod_mul(Qpow, Qpow, n);

            U = U_new;
            V = V_new;
            Qpow = Q_new;

            if (k & mask)
            {
                // Addiere 1: U_{m+1} = (P*U_m + V_m)/2
                //           V_{m+1} = (D*U_m + P*V_m)/2
                //           Q^{m+1} = Q*Q^m
                int64_t D = P * P - 4 * Q; // Diskriminante

                uint64_t PU = safe_mod_mul(P % n, U, n);
                uint64_t U_temp = (PU + V) % n;
                if (U_temp & 1)
                    U_temp += n; // Mache gerade für Division durch 2
                U = (U_temp >> 1) % n;

                uint64_t DU = safe_mod_mul((D % n + n) % n, U, n);
                uint64_t PV = safe_mod_mul(P % n, V, n);
                uint64_t V_temp = (DU + PV) % n;
                if (V_temp & 1)
                    V_temp += n; // Mache gerade für Division durch 2
                V = (V_temp >> 1) % n;

                Qpow = safe_mod_mul(Q % n, Qpow, n);
            }

            mask >>= 1;
        }

        *U_k = U;
        *V_k = V;
        *Q_k = Qpow;
    }

    // ===== MILLER-RABIN TEST =====

    static bool miller_rabin_base2(uint64_t n)
    {
        if (n < 2)
            return false;
        if (n == 2)
            return true;
        if ((n & 1) == 0)
            return false;

        // Schreibe n-1 = d * 2^s
        uint64_t d = n - 1;
        int s = 0;
        while ((d & 1) == 0)
        {
            d >>= 1;
            s++;
        }

        // Teste Basis a = 2
        uint64_t x = mod_pow(2, d, n);
        if (x == 1 || x == n - 1)
        {
            return true;
        }

        for (int i = 0; i < s - 1; i++)
        {
            x = safe_mod_mul(x, x, n);
            if (x == n - 1)
            {
                return true;
            }
            if (x == 1)
            {
                return false; // Frühzeitige Zusammengesetztheit
            }
        }

        return false;
    }

    // ===== STRONG LUCAS PSEUDOPRIME TEST =====

    static bool strong_lucas_test(uint64_t n)
    {
        if (n < 2)
            return false;
        if (n == 2)
            return true;
        if ((n & 1) == 0)
            return false;

        // Selfridge Parameter-Auswahl
        int64_t D = 5;
        int sign = 1;

        for (int attempts = 0; attempts < 20; attempts++)
        {
            int jac = jacobi_symbol(D, n);
            if (jac == -1)
            {
                break; // Gefunden: Jacobi(D|n) = -1
            }
            if (jac == 0)
            {
                return false; // D und n haben gemeinsamen Faktor > 1
            }

            // Nächstes D probieren
            D = sign * (abs((int)D) + 2);
            sign = -sign;
        }

        int64_t P = 1;
        int64_t Q = (1 - D) / 4;

        // Schreibe n+1 = d * 2^s
        uint64_t delta = n + 1;
        uint64_t d = delta;
        int s = 0;
        while ((d & 1) == 0)
        {
            d >>= 1;
            s++;
        }

        // Berechne U_d, V_d, Q^d
        uint64_t U_d, V_d, Q_d;
        lucas_sequence(n, d, P, Q, &U_d, &V_d, &Q_d);

        // Teste U_d ≡ 0 (mod n)
        if (U_d == 0)
        {
            return true;
        }

        // Teste V_{d*2^r} ≡ 0 (mod n) für r = 0, 1, ..., s-1
        uint64_t V_curr = V_d;

        if (V_curr == 0)
        {
            return true;
        }

        for (int r = 1; r < s; r++)
        {
            // V_{2k} = V_k² - 2*Q^k
            V_curr = safe_mod_mul(V_curr, V_curr, n);
            if (V_curr >= 2 * Q_d % n)
            {
                V_curr = (V_curr - 2 * Q_d % n) % n;
            }
            else
            {
                V_curr = (V_curr + n - 2 * Q_d % n) % n;
            }

            // Aktualisiere Q^{d*2^r}
            Q_d = safe_mod_mul(Q_d, Q_d, n);

            if (V_curr == 0)
            {
                return true;
            }
        }

        return false;
    }

    // ===== VOLLSTÄNDIGER BAILLIE-PSW TEST =====

    EXPORT bool baillie_psw_complete(uint64_t n)
    {
        // Triviale Fälle
        if (n < 2)
            return false;
        if (n == 2)
            return true;
        if ((n & 1) == 0)
            return false;

        // Kleine Primzahlen
        if (n < 100)
        {
            const uint64_t small_primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97};
            for (int i = 0; i < 25; i++)
            {
                if (n == small_primes[i])
                    return true;
            }
            return false;
        }

        // Schnelle Teilbarkeits-Checks
        if (n % 3 == 0 || n % 5 == 0 || n % 7 == 0 || n % 11 == 0 ||
            n % 13 == 0 || n % 17 == 0 || n % 19 == 0 || n % 23 == 0)
        {
            return false;
        }

        // Miller-Rabin Test zur Basis 2
        if (!miller_rabin_base2(n))
        {
            return false;
        }

        // Strong Lucas Pseudoprime Test
        if (!strong_lucas_test(n))
        {
            return false;
        }

        return true;
    }

    // Alias für Kompatibilität
    EXPORT bool baillie_psw(uint64_t n)
    {
        return baillie_psw_complete(n);
    }

#ifdef __cplusplus
}
#endif
