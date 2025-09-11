from prim.fj_wrapper import forisek_jancina_test
from prim.baillie_psw_wrapper import baillie_psw


def hybrid_test(n: int) -> bool:
    """
    Für 32-bit-Zahlen Forisek-Jancina, sonst Baillie-PSW.
    Direkter C-Aufruf via ctypes.
    """
    # 2**32-1 = 0xFFFFFFFF
    if n <= 0xFFFFFFFF:
        return forisek_jancina_test(n)
    return baillie_psw(n)
