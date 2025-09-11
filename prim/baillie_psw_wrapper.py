# mypy: disable-error-code="misc"

import ctypes
import os

# Verwende den vorhandenen DLL-Namen
_lib_name = "baillie_psw_complete"  # statt "baillie_psw"
_lib_path = os.path.join(os.path.dirname(__file__), f"{_lib_name}.dll")

_lib = ctypes.CDLL(_lib_path)
_lib.baillie_psw.argtypes = [ctypes.c_uint64]
_lib.baillie_psw.restype = ctypes.c_int


def baillie_psw(n: int) -> bool:
    if n < 0 or n > 0xFFFFFFFFFFFFFFFF:
        raise ValueError("n muss im Bereich einer 64-bit unsigned Integer liegen")
    return bool(_lib.baillie_psw(ctypes.c_uint64(n)))
