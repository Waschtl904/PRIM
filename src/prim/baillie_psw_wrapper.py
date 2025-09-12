# -*- coding: utf-8 -*-
"""
Baillie-PSW Test Wrapper
"""

from .core.baillie_psw import baillie_psw as _baillie_psw_core


def baillie_psw(n: int) -> bool:
    """Baillie-PSW Primzahltest wrapper"""
    return _baillie_psw_core(n)


__all__ = ["baillie_psw"]
