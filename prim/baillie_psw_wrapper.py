import os
from ctypes import CDLL, c_uint64, c_bool

lib_name = "baillie_psw.dll" if os.name == "nt" else "baillie_psw.so"
lib_path = os.path.join(
    os.path.dirname(__file__), "..", "modul8_baillie_psw", "implementations", lib_name
)
bp_lib = CDLL(lib_path)
bp_lib.baillie_psw.argtypes = [c_uint64]
bp_lib.baillie_psw.restype = c_bool


def baillie_psw(n: int) -> bool:
    return bool(bp_lib.baillie_psw(n))
