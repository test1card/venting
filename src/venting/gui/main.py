from __future__ import annotations

from .app import create_main_window, load_gui_deps


def main() -> int:
    QtCore, QtWidgets, pg = load_gui_deps()  # tuple -> unpack
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    win = create_main_window()
    win.show()
    return app.exec()
