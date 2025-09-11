import os
from ctypes import CDLL, c_uint32, c_bool

# Unter Windows DLL, unter Linux/macOS .so
lib_name = "fj_test.dll" if os.name == "nt" else "fj_test.so"
lib_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "modul7_forisek_jancina",
    "c_implementations",
    lib_name,
)
fj_lib = CDLL(lib_path)

fj_lib.forisek_jancina_test.argtypes = [c_uint32]
fj_lib.forisek_jancina_test.restype = c_bool


def forisek_jancina_test(n: int) -> bool:
    return bool(fj_lib.forisek_jancina_test(n))
