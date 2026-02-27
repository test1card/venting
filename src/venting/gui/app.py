from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from venting.diagnostics import summarize_result
from venting.graph import build_branching_network
from venting.presets import get_default_panel_preset_v9
from venting.profiles import (
    make_profile_exponential,
    make_profile_from_table,
    make_profile_linear,
    make_profile_step,
)
from venting.run import export_case_artifacts, make_case_output_dir
from venting.solver import solve_case_stream

from .config import GuiCaseConfig


def load_gui_deps():
    import pyqtgraph as pg
    from PySide6 import QtCore, QtWidgets

    return QtCore, QtWidgets, pg


def _profile_from_cfg(cfg: GuiCaseConfig):
    if cfg.external_model == "dynamic_pump":
        return make_profile_step(101325.0, 1e9)
    if cfg.profile_kind == "linear":
        return make_profile_linear(101325.0, cfg.rate_mmhg_per_s)
    if cfg.profile_kind == "step":
        return make_profile_step(101325.0, cfg.step_time_s)
    if cfg.profile_kind == "barometric":
        return make_profile_exponential(101325.0, cfg.rate_mmhg_per_s, p_floor=10.0)
    return make_profile_from_table(
        "gui_table",
        Path(cfg.profile_file),
        pressure_unit=cfg.profile_pressure_unit,
    )


def create_main_window():
    QtCore, QtWidgets, pg = load_gui_deps()

    class SolverWorker(QtCore.QThread):
        progress = QtCore.Signal(object)
        finished_result = QtCore.Signal(object)
        failed = QtCore.Signal(str)

        def __init__(self, cfg: GuiCaseConfig):
            super().__init__()
            self.cfg = cfg
            self.stop_requested = False

        def request_stop(self):
            self.stop_requested = True

        def run(self):
            try:
                profile = _profile_from_cfg(self.cfg)
                net = self.cfg.to_network_config()
                case = self.cfg.to_case_config()
                nodes, edges, bcs = build_branching_network(net, profile)

                def on_chunk(payload):
                    if self.stop_requested:
                        return
                    self.progress.emit(payload)

                sol = solve_case_stream(
                    nodes, edges, bcs, case, callback=on_chunk, n_chunks=20
                )
                if self.stop_requested or not sol.success:
                    return
                res = summarize_result(nodes, edges, bcs, case, sol)
                self.finished_result.emit(
                    {"res": res, "cfg": self.cfg, "profile": profile}
                )
            except Exception as exc:  # pragma: no cover - UI path
                self.failed.emit(str(exc))

    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Venting v10 GUI")
            self.resize(1400, 840)
            self.worker = None
            self.latest_res = None

            preset = get_default_panel_preset_v9()
            self.cfg = GuiCaseConfig(
                V_cell_m3=preset.V_cell,
                V_vest_m3=preset.V_vest,
                A_wall_cell_m2=preset.A_wall_cell,
                A_wall_vest_m2=preset.A_wall_vest,
            )

            root = QtWidgets.QWidget()
            root_layout = QtWidgets.QVBoxLayout(root)
            self.setCentralWidget(root)

            toolbar = QtWidgets.QHBoxLayout()
            self.btn_run = QtWidgets.QPushButton("Run")
            self.btn_stop = QtWidgets.QPushButton("Stop")
            self.btn_open = QtWidgets.QPushButton("Open config")
            self.btn_save = QtWidgets.QPushButton("Save config")
            self.status = QtWidgets.QLabel("Ready")
            for w in [
                self.btn_run,
                self.btn_stop,
                self.btn_open,
                self.btn_save,
                self.status,
            ]:
                toolbar.addWidget(w)
            toolbar.addStretch(1)
            root_layout.addLayout(toolbar)

            split = QtWidgets.QSplitter()
            root_layout.addWidget(split, 1)

            # left config panel
            left_scroll = QtWidgets.QScrollArea()
            left_scroll.setWidgetResizable(True)
            left_body = QtWidgets.QWidget()
            form = QtWidgets.QFormLayout(left_body)
            left_scroll.setWidget(left_body)
            split.addWidget(left_scroll)

            self.fields = {}

            def add_line(key, value):
                w = QtWidgets.QLineEdit(str(value))
                self.fields[key] = w
                form.addRow(key, w)

            def add_combo(key, items, value):
                w = QtWidgets.QComboBox()
                w.addItems(items)
                w.setCurrentText(value)
                self.fields[key] = w
                form.addRow(key, w)

            # required controls
            add_combo("int_model", ["orifice", "short_tube"], self.cfg.int_model)
            add_combo("exit_model", ["orifice", "short_tube"], self.cfg.exit_model)
            add_line("L_int_mm", self.cfg.L_int_mm)
            add_line("L_exit_mm", self.cfg.L_exit_mm)
            add_line("eps_int_um", self.cfg.eps_int_um)
            add_line("eps_exit_um", self.cfg.eps_exit_um)
            add_line("K_in_int", self.cfg.K_in_int)
            add_line("K_out_int", self.cfg.K_out_int)
            add_line("K_in_exit", self.cfg.K_in_exit)
            add_line("K_out_exit", self.cfg.K_out_exit)
            add_combo(
                "topology", ["single_chain", "two_chain_shared_vest"], self.cfg.topology
            )
            add_line("N_chain", self.cfg.N_chain)
            add_line("N_chain_b", self.cfg.N_chain_b)
            add_line("N_par", self.cfg.N_par)
            add_line("V_cell_m3", self.cfg.V_cell_m3)
            add_line("V_vest_m3", self.cfg.V_vest_m3)
            add_line("A_wall_cell_m2", self.cfg.A_wall_cell_m2)
            add_line("A_wall_vest_m2", self.cfg.A_wall_vest_m2)
            add_line("d_int_mm", self.cfg.d_int_mm)
            add_line("n_int_per_interface", self.cfg.n_int_per_interface)
            add_line("d_exit_mm", self.cfg.d_exit_mm)
            add_line("n_exit", self.cfg.n_exit)
            add_line("Cd_int", self.cfg.Cd_int)
            add_line("Cd_exit", self.cfg.Cd_exit)
            add_combo(
                "external_model", ["profile", "dynamic_pump"], self.cfg.external_model
            )
            add_combo(
                "profile_kind",
                ["linear", "step", "barometric", "table"],
                self.cfg.profile_kind,
            )
            add_line("profile_file", self.cfg.profile_file)
            add_combo(
                "profile_pressure_unit", ["Pa", "mmHg"], self.cfg.profile_pressure_unit
            )
            add_line("rate_mmhg_per_s", self.cfg.rate_mmhg_per_s)
            add_line("step_time_s", self.cfg.step_time_s)
            add_combo(
                "thermo", ["isothermal", "intermediate", "variable"], self.cfg.thermo
            )
            add_combo("wall_model", ["fixed", "lumped"], self.cfg.wall_model)
            add_line("h_conv_W_m2K", self.cfg.h_conv_W_m2K)
            add_line("wall_C_per_area_J_m2K", self.cfg.wall_C_per_area_J_m2K)
            add_line("wall_h_out_W_m2K", self.cfg.wall_h_out_W_m2K)
            add_line("wall_T_inf_K", self.cfg.wall_T_inf_K)
            add_line("wall_emissivity", self.cfg.wall_emissivity)
            add_line("wall_T_sur_K", self.cfg.wall_T_sur_K)
            add_line("wall_q_flux_W_m2", self.cfg.wall_q_flux_W_m2)
            add_line("duration_s", self.cfg.duration_s)
            add_line("n_pts", self.cfg.n_pts)
            add_line("V_ext_m3", self.cfg.V_ext_m3)
            add_line("pump_speed_m3s", self.cfg.pump_speed_m3s)
            add_line("P_ult_Pa", self.cfg.P_ult_Pa)
            add_line("T_ext_K", self.cfg.T_ext_K)
            add_line("output_case_name", self.cfg.output_case_name)

            # right tabs/plots
            tabs = QtWidgets.QTabWidget()
            split.addWidget(tabs)

            self.plot_p = pg.PlotWidget(title="Pressure")
            self.plot_t = pg.PlotWidget(title="Temperature")
            self.plot_m = pg.PlotWidget(title="Mass")
            self.table_peaks = QtWidgets.QTableWidget(0, 4)
            self.table_peaks.setHorizontalHeaderLabels(
                ["edge", "|ΔP|max", "t_peak", "regime"]
            )
            self.table_valid = QtWidgets.QTableWidget(0, 3)
            self.table_valid.setHorizontalHeaderLabels(["flag", "status", "message"])

            tabs.addTab(self.plot_p, "P(t)")
            tabs.addTab(self.plot_t, "T(t)")
            tabs.addTab(self.plot_m, "m(t)")
            tabs.addTab(self.table_peaks, "ΔP/peaks")
            tabs.addTab(self.table_valid, "Validity")

            self.btn_run.clicked.connect(self.on_run)
            self.btn_stop.clicked.connect(self.on_stop)
            self.btn_open.clicked.connect(self.on_open)
            self.btn_save.clicked.connect(self.on_save)

        def _read_cfg(self) -> GuiCaseConfig:
            kwargs = {}
            for key, widget in self.fields.items():
                if hasattr(widget, "currentText"):
                    value = widget.currentText()
                else:
                    value = widget.text()
                kwargs[key] = value

            int_fields = {
                "N_chain",
                "N_chain_b",
                "N_par",
                "n_int_per_interface",
                "n_exit",
                "n_pts",
            }
            float_fields = (
                set(kwargs)
                - int_fields
                - {
                    "int_model",
                    "exit_model",
                    "external_model",
                    "profile_kind",
                    "profile_pressure_unit",
                    "thermo",
                    "wall_model",
                    "profile_file",
                    "output_case_name",
                }
            )
            for k in int_fields:
                kwargs[k] = int(kwargs[k])
            for k in float_fields:
                kwargs[k] = float(kwargs[k])
            cfg = GuiCaseConfig(**kwargs)
            cfg.validate()
            return cfg

        def on_run(self):
            try:
                cfg = self._read_cfg()
            except Exception as exc:
                self.status.setText(f"Invalid config: {exc}")
                return
            self.status.setText("Running...")
            self.plot_p.clear()
            self.plot_t.clear()
            self.plot_m.clear()
            self.worker = SolverWorker(cfg)
            self.worker.progress.connect(self.on_progress)
            self.worker.finished_result.connect(self.on_finished)
            self.worker.failed.connect(
                lambda msg: self.status.setText(f"Failed: {msg}")
            )
            self.worker.start()

        def on_stop(self):
            if self.worker is not None:
                self.worker.request_stop()
                self.status.setText("Stopping requested...")

        def on_progress(self, payload):
            t = payload["t"]
            y = payload["y"]
            n_nodes = int(payload.get("n_nodes", 1))
            thermo = payload.get("thermo", "isothermal")

            m = y[:n_nodes, :]
            self.plot_m.clear()
            self.plot_m.plot(t, m[0], pen="y")

            if thermo == "isothermal":
                self.plot_t.clear()
            else:
                self.plot_t.clear()
                self.plot_t.plot(t, y[n_nodes, :], pen="c")
            self.status.setText(f"Running... {100 * payload['progress']:.0f}%")

        def on_finished(self, payload):
            self.latest_res = payload["res"]
            cfg = payload["cfg"]
            res = payload["res"]
            self.plot_p.clear()
            self.plot_p.plot(res.t, res.P[0], pen="g")
            self.plot_t.clear()
            self.plot_t.plot(res.t, res.T[0], pen="c")
            self.plot_m.clear()
            self.plot_m.plot(res.t, res.m[0], pen="y")

            self._fill_tables(res)

            out = make_case_output_dir(cfg.output_case_name)
            run_params = asdict(cfg)
            stem = f"v1000_gui_{cfg.external_model}_{cfg.profile_kind}_{cfg.thermo}"
            export_case_artifacts(out, stem, res, run_params)
            self.status.setText(f"Done. Saved to {out}")

        def _fill_tables(self, res):
            peaks = res.peak_diag
            self.table_peaks.setRowCount(len(peaks))
            for i, (edge, peak) in enumerate(peaks.items()):
                self.table_peaks.setItem(i, 0, QtWidgets.QTableWidgetItem(edge))
                self.table_peaks.setItem(
                    i, 1, QtWidgets.QTableWidgetItem(f"{res.max_dP.get(edge, 0.0):.2f}")
                )
                self.table_peaks.setItem(
                    i, 2, QtWidgets.QTableWidgetItem(f"{peak.get('t_peak', 0.0):.4g}")
                )
                self.table_peaks.setItem(
                    i, 3, QtWidgets.QTableWidgetItem(str(peak.get("regime", "")))
                )

            flags = res.meta.get("validity_flags", {})
            self.table_valid.setRowCount(len(flags))
            for i, (name, flag) in enumerate(flags.items()):
                self.table_valid.setItem(i, 0, QtWidgets.QTableWidgetItem(name))
                self.table_valid.setItem(
                    i, 1, QtWidgets.QTableWidgetItem(str(flag.get("status", "")))
                )
                self.table_valid.setItem(
                    i, 2, QtWidgets.QTableWidgetItem(str(flag.get("message", "")))
                )

        def on_open(self):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open config", "", "JSON (*.json)"
            )
            if not path:
                return
            cfg = GuiCaseConfig.load_json(path)
            for key, widget in self.fields.items():
                val = getattr(cfg, key)
                if hasattr(widget, "setCurrentText"):
                    widget.setCurrentText(str(val))
                else:
                    widget.setText(str(val))
            self.status.setText(f"Loaded {path}")

        def on_save(self):
            try:
                cfg = self._read_cfg()
            except Exception as exc:
                self.status.setText(f"Cannot save invalid config: {exc}")
                return
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save config", "case.json", "JSON (*.json)"
            )
            if not path:
                return
            cfg.save_json(path)
            self.status.setText(f"Saved {path}")

    return MainWindow()
