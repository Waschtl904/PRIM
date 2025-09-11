# prim/fj_wrapper.py
import subprocess
import os


def forisek_jancina_test(n: int) -> bool:
    """
    Wrapper für die Forisek-Jancina-Executable.
    """
    exe_path = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "modul7_forisek_jancina",
            "c_implementations",
            "fj_test.exe",
        )
    )

    try:
        proc = subprocess.run(
            [exe_path, str(n)], capture_output=True, text=True, timeout=5
        )
        return "PRIME" in proc.stdout.upper()
    except Exception:
        return False
