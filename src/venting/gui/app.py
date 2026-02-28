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
from .state_layout import infer_layout_from_modes


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


FIELD_TOOLTIPS = {
    "N_chain": "Число ячеек в цепочке [-]. Типично: 5..20",
    "N_chain_b": "Число ячеек во второй цепочке (two_chain) [-]",
    "N_par": "Число параллельных цепочек [-]. Типично: 1..10",
    "V_cell_m3": "Объём одной ячейки [м³]. Типично: 1e-5..1e-2",
    "V_vest_m3": "Объём вестибюля [м³]. Типично: 1e-4..1e-1",
    "A_wall_cell_m2": "Площадь стенки ячейки [м²] — для теплообмена",
    "A_wall_vest_m2": "Площадь стенки вестибюля [м²]",
    "d_int_mm": "Диаметр внутреннего отверстия [мм]. Типично: 0.5..10",
    "n_int_per_interface": "Число отверстий на интерфейс [-]",
    "d_exit_mm": "Диаметр выходного отверстия [мм]",
    "n_exit": "Число выходных отверстий [-]",
    "Cd_int": "Коэффициент расхода внутреннего отверстия [-]. Типично: 0.6..0.8",
    "Cd_exit": "Коэффициент расхода выходного отверстия [-]. Типично: 0.6..0.8",
    "L_int_mm": "Длина внутреннего канала [мм]. 0 = тонкое отверстие",
    "L_exit_mm": "Длина выходного канала [мм]",
    "eps_int_um": "Шероховатость внутреннего канала [мкм]. 0 = гладкий",
    "eps_exit_um": "Шероховатость выходного канала [мкм]",
    "K_in_int": "Коэффициент потерь на входе в канал [-]. Типично: 0.5",
    "K_out_int": "Коэффициент потерь на выходе из канала [-]. Типично: 1.0",
    "K_in_exit": "Коэффициент потерь на входе в выходной канал [-]",
    "K_out_exit": "Коэффициент потерь на выходе из выходного канала [-]",
    "rate_mmhg_per_s": "Скорость изменения давления [мм рт. ст. / с]. Типично: 10..50",
    "step_time_s": "Момент ступенчатого сброса давления [с]",
    "h_conv_W_m2K": "Коэффициент теплообмена газ-стенка [Вт/(м²·К)]. 0=адиабатный",
    "wall_C_per_area_J_m2K": "Теплоёмкость стенки на единицу площади [Дж/(м²·К)]",
    "wall_h_out_W_m2K": "Теплоотдача с внешней стороны стенки [Вт/(м²·К)]",
    "wall_T_inf_K": "Температура окружающей среды [К]",
    "wall_emissivity": "Степень черноты стенки [-]. 0..1",
    "wall_T_sur_K": "Температура окружающих поверхностей для радиации [К]",
    "wall_q_flux_W_m2": "Тепловой поток через стенку [Вт/м²]",
    "duration_s": "Длительность симуляции [с]",
    "n_pts": "Число точек вывода [-]. Типично: 500..2000",
    "V_ext_m3": "Объём внешней камеры (dynamic_pump) [м³]",
    "pump_speed_m3s": "Быстрота откачки насоса [м³/с]",
    "P_ult_Pa": "Предельное давление насоса [Па]",
    "T_ext_K": "Температура внешней среды [К]",
}


def create_main_window():
    QtCore, QtWidgets, pg = load_gui_deps()

    NODE_COLORS = [
        (100, 200, 100),
        (100, 180, 255),
        (255, 200, 80),
        (255, 100, 100),
        (200, 100, 255),
        (255, 165, 50),
        (100, 220, 220),
        (220, 220, 100),
    ]

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
                    nodes,
                    edges,
                    bcs,
                    case,
                    callback=on_chunk,
                    dt_chunk_s=1.0,
                    should_stop=lambda: self.stop_requested,
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
            cfg = GuiCaseConfig(
                V_cell_m3=preset.V_cell,
                V_vest_m3=preset.V_vest,
                A_wall_cell_m2=preset.A_wall_cell,
                A_wall_vest_m2=preset.A_wall_vest,
            )
            self.cfg = cfg

            root = QtWidgets.QWidget()
            root_layout = QtWidgets.QVBoxLayout(root)
            self.setCentralWidget(root)

            # Toolbar
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

            self.progress_bar = QtWidgets.QProgressBar()
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_bar.setMaximumWidth(200)
            self.progress_bar.setVisible(False)
            toolbar.addWidget(self.progress_bar)
            toolbar.addStretch(1)
            root_layout.addLayout(toolbar)

            split = QtWidgets.QSplitter()
            root_layout.addWidget(split, 1)

            # Left config panel
            left_scroll = QtWidgets.QScrollArea()
            left_scroll.setWidgetResizable(True)
            left_body = QtWidgets.QWidget()
            left_layout = QtWidgets.QVBoxLayout(left_body)
            left_scroll.setWidget(left_body)
            split.addWidget(left_scroll)

            self.fields = {}

            GROUP_STYLE = (
                "QGroupBox { font-weight: bold; margin-top: 6px; } "
                "QGroupBox::title { subcontrol-position: top left; padding: 0 4px; }"
            )

            def add_line(key, value, form_layout):
                w = QtWidgets.QLineEdit(str(value))
                tip = FIELD_TOOLTIPS.get(key, "")
                if tip:
                    w.setToolTip(tip)
                self.fields[key] = w
                form_layout.addRow(key, w)

            def add_combo(key, items, value, form_layout):
                w = QtWidgets.QComboBox()
                w.addItems(items)
                w.setCurrentText(value)
                self.fields[key] = w
                form_layout.addRow(key, w)

            # Group 1: Network topology
            box1 = QtWidgets.QGroupBox("Топология сети")
            box1.setStyleSheet(GROUP_STYLE)
            form1 = QtWidgets.QFormLayout(box1)
            add_combo(
                "topology",
                ["single_chain", "two_chain_shared_vest"],
                cfg.topology,
                form1,
            )
            add_line("N_chain", cfg.N_chain, form1)
            add_line("N_chain_b", cfg.N_chain_b, form1)
            add_line("N_par", cfg.N_par, form1)
            add_line("V_cell_m3", cfg.V_cell_m3, form1)
            add_line("V_vest_m3", cfg.V_vest_m3, form1)
            add_line("A_wall_cell_m2", cfg.A_wall_cell_m2, form1)
            add_line("A_wall_vest_m2", cfg.A_wall_vest_m2, form1)
            left_layout.addWidget(box1)

            # Group 2: Internal orifices
            box2 = QtWidgets.QGroupBox("Внутренние отверстия")
            box2.setStyleSheet(GROUP_STYLE)
            form2 = QtWidgets.QFormLayout(box2)
            add_combo(
                "int_model", ["orifice", "short_tube", "fanno"], cfg.int_model, form2
            )
            add_line("d_int_mm", cfg.d_int_mm, form2)
            add_line("n_int_per_interface", cfg.n_int_per_interface, form2)
            add_line("Cd_int", cfg.Cd_int, form2)
            add_line("L_int_mm", cfg.L_int_mm, form2)
            add_line("eps_int_um", cfg.eps_int_um, form2)
            add_line("K_in_int", cfg.K_in_int, form2)
            add_line("K_out_int", cfg.K_out_int, form2)
            left_layout.addWidget(box2)

            # Group 3: Exit orifice
            box3 = QtWidgets.QGroupBox("Выходное отверстие")
            box3.setStyleSheet(GROUP_STYLE)
            form3 = QtWidgets.QFormLayout(box3)
            add_combo(
                "exit_model", ["orifice", "short_tube", "fanno"], cfg.exit_model, form3
            )
            add_line("d_exit_mm", cfg.d_exit_mm, form3)
            add_line("n_exit", cfg.n_exit, form3)
            add_line("Cd_exit", cfg.Cd_exit, form3)
            add_line("L_exit_mm", cfg.L_exit_mm, form3)
            add_line("eps_exit_um", cfg.eps_exit_um, form3)
            add_line("K_in_exit", cfg.K_in_exit, form3)
            add_line("K_out_exit", cfg.K_out_exit, form3)
            left_layout.addWidget(box3)

            # Group 4: External environment / profile
            box4 = QtWidgets.QGroupBox("Внешняя среда / профиль")
            box4.setStyleSheet(GROUP_STYLE)
            form4 = QtWidgets.QFormLayout(box4)
            add_combo(
                "external_model",
                ["profile", "dynamic_pump"],
                cfg.external_model,
                form4,
            )
            add_combo(
                "profile_kind",
                ["linear", "step", "barometric", "table"],
                cfg.profile_kind,
                form4,
            )

            # Special profile_file row with Browse button and drag-and-drop
            file_row = QtWidgets.QWidget()
            file_layout = QtWidgets.QHBoxLayout(file_row)
            file_layout.setContentsMargins(0, 0, 0, 0)
            file_layout.setSpacing(4)

            profile_file_edit = QtWidgets.QLineEdit(str(cfg.profile_file))
            profile_file_edit.setAcceptDrops(True)
            profile_file_edit.setPlaceholderText("Путь к CSV (t_s, P_Pa)...")

            def dragEnterEvent(event):
                if event.mimeData().hasUrls():
                    event.acceptProposedAction()

            def dropEvent(event):
                urls = event.mimeData().urls()
                if urls:
                    profile_file_edit.setText(urls[0].toLocalFile())

            profile_file_edit.dragEnterEvent = dragEnterEvent
            profile_file_edit.dropEvent = dropEvent

            browse_btn = QtWidgets.QPushButton("…")
            browse_btn.setMaximumWidth(28)
            browse_btn.setToolTip("Выбрать CSV файл")

            def on_browse():
                path, _ = QtWidgets.QFileDialog.getOpenFileName(
                    None,
                    "Файл профиля давления",
                    "",
                    "CSV файлы (*.csv);;Все файлы (*)",
                )
                if path:
                    profile_file_edit.setText(path)

            browse_btn.clicked.connect(on_browse)
            file_layout.addWidget(profile_file_edit)
            file_layout.addWidget(browse_btn)

            self.fields["profile_file"] = profile_file_edit
            form4.addRow("profile_file", file_row)

            add_combo(
                "profile_pressure_unit",
                ["Pa", "mmHg"],
                cfg.profile_pressure_unit,
                form4,
            )
            add_line("rate_mmhg_per_s", cfg.rate_mmhg_per_s, form4)
            add_line("step_time_s", cfg.step_time_s, form4)
            add_line("V_ext_m3", cfg.V_ext_m3, form4)
            add_line("pump_speed_m3s", cfg.pump_speed_m3s, form4)
            add_line("P_ult_Pa", cfg.P_ult_Pa, form4)
            add_line("T_ext_K", cfg.T_ext_K, form4)
            left_layout.addWidget(box4)

            # Group 5: Thermodynamics
            box5 = QtWidgets.QGroupBox("Термодинамика")
            box5.setStyleSheet(GROUP_STYLE)
            form5 = QtWidgets.QFormLayout(box5)
            add_combo(
                "thermo",
                ["isothermal", "intermediate", "variable"],
                cfg.thermo,
                form5,
            )
            add_combo("wall_model", ["fixed", "lumped"], cfg.wall_model, form5)
            add_line("h_conv_W_m2K", cfg.h_conv_W_m2K, form5)
            add_line("wall_C_per_area_J_m2K", cfg.wall_C_per_area_J_m2K, form5)
            add_line("wall_h_out_W_m2K", cfg.wall_h_out_W_m2K, form5)
            add_line("wall_T_inf_K", cfg.wall_T_inf_K, form5)
            add_line("wall_emissivity", cfg.wall_emissivity, form5)
            add_line("wall_T_sur_K", cfg.wall_T_sur_K, form5)
            add_line("wall_q_flux_W_m2", cfg.wall_q_flux_W_m2, form5)
            left_layout.addWidget(box5)

            # Group 6: Calculation parameters
            box6 = QtWidgets.QGroupBox("Параметры расчёта")
            box6.setStyleSheet(GROUP_STYLE)
            form6 = QtWidgets.QFormLayout(box6)
            add_line("duration_s", cfg.duration_s, form6)
            add_line("n_pts", cfg.n_pts, form6)
            add_line("output_case_name", cfg.output_case_name, form6)
            left_layout.addWidget(box6)

            left_layout.addStretch(1)

            # Right: tabs with plots and tables
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

            # Run history
            self.history: list[GuiCaseConfig] = []
            self.history_list = QtWidgets.QListWidget()
            self.history_list.setMaximumHeight(72)
            self.history_list.setToolTip(
                "Последние запуски. Двойной клик — загрузить параметры."
            )
            self.history_list.itemDoubleClicked.connect(self._on_history_clicked)
            root_layout.addWidget(QtWidgets.QLabel("История запусков:"))
            root_layout.addWidget(self.history_list)

            self._connect_visibility_logic()

        def _read_cfg(self) -> GuiCaseConfig:
            import dataclasses

            field_map: dict[str, type] = {}
            for f in dataclasses.fields(GuiCaseConfig):
                ftype = f.type
                if isinstance(ftype, str):
                    ftype = {"int": int, "float": float, "str": str, "bool": bool}.get(
                        ftype, str
                    )
                field_map[f.name] = ftype

            kwargs = {}
            error_fields = []

            for key, widget in self.fields.items():
                raw = (
                    widget.currentText()
                    if hasattr(widget, "currentText")
                    else widget.text()
                )
                ftype = field_map.get(key, str)
                try:
                    if ftype is int:
                        kwargs[key] = int(raw)
                    elif ftype is float:
                        kwargs[key] = float(raw)
                    elif ftype is bool:
                        kwargs[key] = raw.lower() in ("true", "1", "yes")
                    else:
                        kwargs[key] = raw
                    widget.setStyleSheet("")
                except (ValueError, TypeError):
                    widget.setStyleSheet(
                        "background-color: #5a1a1a; border: 1px solid #cc0000;"
                    )
                    error_fields.append(key)

            if error_fields:
                raise ValueError(
                    f"Некорректные значения в полях: {', '.join(error_fields)}"
                )

            cfg = GuiCaseConfig(**kwargs)
            cfg.validate()
            return cfg

        def _connect_visibility_logic(self):
            INT_TUBE = ["L_int_mm", "eps_int_um", "K_in_int", "K_out_int"]
            EXIT_TUBE = ["L_exit_mm", "eps_exit_um", "K_in_exit", "K_out_exit"]
            PUMP = ["V_ext_m3", "pump_speed_m3s", "P_ult_Pa", "T_ext_K"]
            PROFILE = [
                "profile_kind",
                "rate_mmhg_per_s",
                "step_time_s",
                "profile_file",
                "profile_pressure_unit",
            ]
            RATE = ["rate_mmhg_per_s"]
            STEP = ["step_time_s"]
            FILE = ["profile_file", "profile_pressure_unit"]
            LUMPED = [
                "wall_C_per_area_J_m2K",
                "wall_h_out_W_m2K",
                "wall_T_inf_K",
                "wall_emissivity",
                "wall_T_sur_K",
                "wall_q_flux_W_m2",
            ]
            CHAIN_B = ["N_chain_b"]

            def set_visible(keys, visible):
                for key in keys:
                    w = self.fields.get(key)
                    if not w:
                        continue
                    w.setVisible(visible)
                    form = w.parent().layout() if w.parent() else None
                    if isinstance(form, QtWidgets.QFormLayout):
                        idx = form.indexOf(w)
                        if idx >= 0:
                            row, _ = form.getItemPosition(idx)
                            label = form.itemAt(row, QtWidgets.QFormLayout.LabelRole)
                            if label and label.widget():
                                label.widget().setVisible(visible)

            self.fields["int_model"].currentTextChanged.connect(
                lambda v: set_visible(INT_TUBE, v != "orifice")
            )
            self.fields["exit_model"].currentTextChanged.connect(
                lambda v: set_visible(EXIT_TUBE, v != "orifice")
            )
            self.fields["external_model"].currentTextChanged.connect(
                lambda v: (
                    set_visible(PUMP, v == "dynamic_pump"),
                    set_visible(PROFILE, v != "dynamic_pump"),
                )
            )
            self.fields["profile_kind"].currentTextChanged.connect(
                lambda v: (
                    set_visible(RATE, v in ("linear", "barometric")),
                    set_visible(STEP, v == "step"),
                    set_visible(FILE, v == "table"),
                )
            )
            self.fields["wall_model"].currentTextChanged.connect(
                lambda v: set_visible(LUMPED, v == "lumped")
            )
            self.fields["topology"].currentTextChanged.connect(
                lambda v: set_visible(CHAIN_B, v == "two_chain_shared_vest")
            )

            # Apply initial state
            for key in [
                "int_model",
                "exit_model",
                "external_model",
                "profile_kind",
                "wall_model",
                "topology",
            ]:
                self.fields[key].currentTextChanged.emit(self.fields[key].currentText())

        def on_run(self):
            try:
                cfg = self._read_cfg()
            except Exception as exc:
                self.status.setText(f"Invalid config: {exc}")
                return
            self.status.setText("Расчёт...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.plot_p.clear()
            self.plot_t.clear()
            self.plot_m.clear()
            self.worker = SolverWorker(cfg)
            self.worker.progress.connect(self.on_progress)
            self.worker.finished_result.connect(self.on_finished)
            self.worker.failed.connect(
                lambda msg: (
                    self.status.setText(f"Failed: {msg}"),
                    self.progress_bar.setVisible(False),
                )
            )
            self.worker.start()

        def on_stop(self):
            if self.worker is not None:
                self.worker.request_stop()
                self.status.setText("Stopping requested...")

        def on_progress(self, payload):
            t = payload["t"]
            y = payload["y"]
            n_nodes = int(payload.get("node_count", payload.get("n_nodes", 1)))
            layout = infer_layout_from_modes(
                payload.get("thermo", "isothermal"),
                payload.get("wall_model", "fixed"),
                n_nodes,
            )
            n_show = min(n_nodes, 8)
            m = y[layout.m_slice, :]
            self.plot_m.clear()
            for i in range(n_show):
                self.plot_m.plot(
                    t,
                    m[i],
                    pen=pg.mkPen(color=NODE_COLORS[i % len(NODE_COLORS)], width=1.5),
                )
            self.plot_t.clear()
            if layout.t_slice is not None:
                t_arr = y[layout.t_slice, :]
                for i in range(min(n_show, t_arr.shape[0])):
                    self.plot_t.plot(
                        t,
                        t_arr[i],
                        pen=pg.mkPen(
                            color=NODE_COLORS[i % len(NODE_COLORS)], width=1.5
                        ),
                    )
            self.progress_bar.setValue(int(100 * payload["progress"]))
            self.status.setText(f"Расчёт... {int(100 * payload['progress'])}%")

        def on_finished(self, payload):
            self.latest_res = payload["res"]
            cfg = payload["cfg"]
            res = payload["res"]
            n_show = min(res.P.shape[0], 8)

            self.plot_p.clear()
            self.plot_p.addLegend()
            for i in range(n_show):
                self.plot_p.plot(
                    res.t,
                    res.P[i],
                    pen=pg.mkPen(color=NODE_COLORS[i % len(NODE_COLORS)], width=2),
                    name=f"node {i}",
                )
            self.plot_p.plot(
                res.t,
                res.P_ext,
                pen=pg.mkPen(
                    color=(160, 160, 160),
                    width=1.5,
                    style=QtCore.Qt.DashLine,
                ),
                name="P_ext",
            )

            self.plot_t.clear()
            self.plot_t.addLegend()
            for i in range(min(res.T.shape[0], n_show)):
                self.plot_t.plot(
                    res.t,
                    res.T[i],
                    pen=pg.mkPen(color=NODE_COLORS[i % len(NODE_COLORS)], width=2),
                    name=f"node {i}",
                )

            self.plot_m.clear()
            self.plot_m.addLegend()
            for i in range(n_show):
                self.plot_m.plot(
                    res.t,
                    res.m[i],
                    pen=pg.mkPen(color=NODE_COLORS[i % len(NODE_COLORS)], width=2),
                    name=f"node {i}",
                )

            self._fill_tables(res)
            self.progress_bar.setValue(100)
            self.progress_bar.setVisible(False)
            out = make_case_output_dir(cfg.output_case_name)
            run_params = asdict(cfg)
            stem = f"v1000_gui_{cfg.external_model}_{cfg.profile_kind}_{cfg.thermo}"
            export_case_artifacts(out, stem, res, run_params)
            self.status.setText(f"Готово. Результаты: {out}")

            self.history.insert(0, payload["cfg"])
            self.history = self.history[:10]
            self.history_list.clear()
            for i, h in enumerate(self.history):
                self.history_list.addItem(
                    f"#{i + 1}  {h.profile_kind}  "
                    f"d_int={h.d_int_mm}мм  d_exit={h.d_exit_mm}мм  {h.thermo}"
                )

        def _fill_tables(self, res):
            from PySide6 import QtGui

            STATUS_BG = {
                "ok": QtGui.QColor(30, 80, 30),
                "warning": QtGui.QColor(90, 65, 0),
                "fail": QtGui.QColor(100, 15, 15),
            }
            STATUS_FG = {
                "ok": QtGui.QColor(120, 240, 120),
                "warning": QtGui.QColor(255, 210, 60),
                "fail": QtGui.QColor(255, 80, 80),
            }

            # Peaks table
            peaks = res.peak_diag
            self.table_peaks.setRowCount(len(peaks))
            for i, (edge, peak) in enumerate(peaks.items()):
                regime = str(peak.get("regime", ""))
                items = [
                    QtWidgets.QTableWidgetItem(edge),
                    QtWidgets.QTableWidgetItem(f"{res.max_dP.get(edge, 0.0):.1f} Pa"),
                    QtWidgets.QTableWidgetItem(f"{peak.get('t_peak', 0.0):.4g} s"),
                    QtWidgets.QTableWidgetItem(regime),
                ]
                for j, item in enumerate(items):
                    if "CHOKED" in regime and j in (1, 3):
                        item.setForeground(QtGui.QColor(255, 140, 0))
                    self.table_peaks.setItem(i, j, item)
            self.table_peaks.resizeColumnsToContents()

            # Validity table
            flags = res.meta.get("validity_flags", {})
            self.table_valid.setRowCount(len(flags))
            for i, (name, flag) in enumerate(flags.items()):
                status = str(flag.get("status", "ok"))
                name_item = QtWidgets.QTableWidgetItem(name)
                status_item = QtWidgets.QTableWidgetItem(status.upper())
                msg_item = QtWidgets.QTableWidgetItem(str(flag.get("message", "")))
                status_item.setBackground(
                    STATUS_BG.get(status, QtGui.QColor(50, 50, 50))
                )
                status_item.setForeground(
                    STATUS_FG.get(status, QtGui.QColor(200, 200, 200))
                )
                bold_font = QtGui.QFont()
                bold_font.setBold(True)
                status_item.setFont(bold_font)
                self.table_valid.setItem(i, 0, name_item)
                self.table_valid.setItem(i, 1, status_item)
                self.table_valid.setItem(i, 2, msg_item)
            self.table_valid.resizeColumnsToContents()
            self.table_valid.horizontalHeader().setStretchLastSection(True)

        def _on_history_clicked(self, item):
            idx = self.history_list.row(item)
            if 0 <= idx < len(self.history):
                cfg = self.history[idx]
                for key, widget in self.fields.items():
                    val = getattr(cfg, key, None)
                    if val is None:
                        continue
                    if hasattr(widget, "setCurrentText"):
                        widget.setCurrentText(str(val))
                    else:
                        widget.setText(str(val))
                self.status.setText(f"Загружены параметры запуска #{idx + 1}")

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
