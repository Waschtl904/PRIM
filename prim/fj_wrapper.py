# prim/fj_wrapper.py
import subprocess
import os


def forisek_jancina_test(n: int) -> bool:
    """
    Wrapper für die Forisek-Jancina-Executable.
    Erwartet: fj_test.exe liegt hier:
      ../modul7_forisek_jancina/c_implementations/fj_test.exe
    """
    exe_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "modul7_forisek_jancina",
        "c_implementations",
        "fj_test.exe",
    )
    proc = subprocess.run([exe_path, str(n)], capture_output=True, text=True)
    return "PRIME" in proc.stdout
