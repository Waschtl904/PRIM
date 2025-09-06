def test_import_prim_module():
    import prim

    assert hasattr(prim, "__version__")
