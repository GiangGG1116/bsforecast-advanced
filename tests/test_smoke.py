def test_import():
    import bsfa
    assert hasattr(bsfa, "__all__")
