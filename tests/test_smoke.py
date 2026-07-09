def test_import_package() -> None:
    import admet_platform

    assert admet_platform.__version__ == "0.1.0"
