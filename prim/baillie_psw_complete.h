#ifndef BAILLIE_PSW_COMPLETE_H
#define BAILLIE_PSW_COMPLETE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * Vollständiger Baillie-PSW Primzahltest
     * @param n Zu testende Zahl
     * @return true wenn wahrscheinlich prim, false wenn definitiv zusammengesetzt
     */
    bool baillie_psw_complete(uint64_t n);

    /**
     * Alias für Kompatibilität
     */
    bool baillie_psw(uint64_t n);

#ifdef __cplusplus
}
#endif

#endif /* BAILLIE_PSW_COMPLETE_H */
