# prim/baillie_psw_wrapper.py
import subprocess
import os


def baillie_psw(n: int) -> bool:
    """
    Wrapper für die Baillie-PSW-Executable.
    Erwartet: baillie_psw.exe liegt hier:
      ../modul8_baillie_psw/implementations/baillie_psw.exe
    """
    exe_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "modul8_baillie_psw",
        "implementations",
        "baillie_psw.exe",
    )
    proc = subprocess.run([exe_path, str(n)], capture_output=True, text=True)
    return "PRIME" in proc.stdout
