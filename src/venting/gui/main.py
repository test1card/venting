from __future__ import annotations


def main() -> int:
    from .app import create_main_window, load_gui_deps

    deps = load_gui_deps()
    app = deps.QtWidgets.QApplication.instance() or deps.QtWidgets.QApplication([])
    win = create_main_window()
    win.show()
    return app.exec()
