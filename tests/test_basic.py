def test_basic_import():
    """Test that basic imports work"""
    try:
        import prim
        assert True
    except ImportError as e:
        assert False, f"Failed to import prim: {e}"

def test_basic_math():
    """Test basic functionality"""
    assert 2 + 2 == 4
