# prim/baillie_psw_wrapper.py
import subprocess
import os


def baillie_psw(n: int) -> bool:
    """
    Wrapper für die Baillie-PSW-Executable.
    """
    exe_path = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "modul8_baillie_psw",
            "implementations",
            "baillie_psw.exe",
        )
    )

    try:
        proc = subprocess.run(
            [exe_path, str(n)], capture_output=True, text=True, timeout=5
        )
        return "PRIME" in proc.stdout.upper()
    except Exception:
        return False
