import os

import pandas as pd

from prim.modul6_baillie_psw import baillie_psw


def test_modul6_files_exist():
    assert os.path.exists(
        "modul6_data/baillie_psw_results.csv"
    ), "CSV-Datei wurde nicht erstellt"
    assert os.path.exists(
        "modul6_plots/performance_compare.png"
    ), "Performance-Plot wurde nicht erstellt"
    assert os.path.exists(
        "modul6_plots/discrepancies.png"
    ), "Discrepancy-Plot wurde nicht erstellt"


def test_baillie_psw_basic():
    assert baillie_psw(2) is True
    assert baillie_psw(3) is True
    assert baillie_psw(17) is True
    assert baillie_psw(4) is False
    assert baillie_psw(561) is False


def test_csv_structure():
    df = pd.read_csv("modul6_data/baillie_psw_results.csv")
    expected_cols = ["n", "is_prime_fj", "time_fj", "is_prime_bp", "time_bp"]
    assert all(col in df.columns for col in expected_cols), "Spalten fehlen"
    assert len(df) > 0, "CSV ist leer"
