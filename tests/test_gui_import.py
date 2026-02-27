def test_gui_modules_import_without_qt_runtime():
    import venting.gui.app as app
    import venting.gui.main as main

    assert hasattr(app, "create_main_window")
    assert hasattr(main, "main")
