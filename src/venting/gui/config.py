from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from venting.cases import CaseConfig, NetworkConfig

ALLOWED_THERMO = {"isothermal", "intermediate", "variable"}
ALLOWED_WALL = {"fixed", "lumped"}
ALLOWED_EXTERNAL = {"profile", "dynamic_pump"}
ALLOWED_EDGE_MODEL = {"orifice", "short_tube"}
ALLOWED_PROFILE = {"linear", "step", "barometric", "table"}


@dataclass
class GuiCaseConfig:
    # geometry/network (SI unless suffix says otherwise)
    topology: str = "single_chain"
    N_chain: int = 10
    N_chain_b: int = 10
    N_par: int = 2
    V_cell_m3: float = 1.0e-4
    V_vest_m3: float = 1.0e-4
    A_wall_cell_m2: float = 0.01
    A_wall_vest_m2: float = 0.01
    d_int_mm: float = 2.0
    n_int_per_interface: int = 1
    d_exit_mm: float = 2.0
    n_exit: int = 1
    Cd_int: float = 0.62
    Cd_exit: float = 0.62
    int_model: str = "orifice"
    exit_model: str = "orifice"
    L_int_mm: float = 0.0
    L_exit_mm: float = 0.0
    K_in_int: float = 0.5
    K_out_int: float = 1.0
    eps_int_um: float = 0.0
    K_in_exit: float = 0.5
    K_out_exit: float = 1.0
    eps_exit_um: float = 0.0

    # boundary/profile
    external_model: str = "profile"
    profile_kind: str = "linear"
    profile_file: str = ""
    profile_pressure_unit: str = "Pa"
    rate_mmhg_per_s: float = 20.0
    step_time_s: float = 0.01

    # thermal/case
    thermo: str = "isothermal"
    duration_s: float = 150.0
    n_pts: int = 800
    h_conv_W_m2K: float = 0.0
    T_wall_K: float = 300.0
    wall_model: str = "fixed"
    wall_C_per_area_J_m2K: float = 1e9
    wall_h_out_W_m2K: float = 0.0
    wall_T_inf_K: float = 300.0
    wall_emissivity: float = 0.0
    wall_T_sur_K: float = 300.0
    wall_q_flux_W_m2: float = 0.0

    # dynamic pump
    V_ext_m3: float = 0.1
    T_ext_K: float = 300.0
    pump_speed_m3s: float = 0.0
    P_ult_Pa: float = 0.0

    # output
    output_case_name: str = "gui"

    def validate(self) -> None:
        if self.thermo not in ALLOWED_THERMO:
            raise ValueError("thermo is invalid")
        if self.wall_model not in ALLOWED_WALL:
            raise ValueError("wall_model is invalid")
        if self.external_model not in ALLOWED_EXTERNAL:
            raise ValueError("external_model is invalid")
        if self.topology not in {"single_chain", "two_chain_shared_vest"}:
            raise ValueError("topology is invalid")
        if (
            self.int_model not in ALLOWED_EDGE_MODEL
            or self.exit_model not in ALLOWED_EDGE_MODEL
        ):
            raise ValueError("edge model must be orifice|short_tube")
        if self.profile_kind not in ALLOWED_PROFILE:
            raise ValueError("profile_kind is invalid")
        for name in [
            "N_chain",
            "N_par",
            "N_chain_b",
            "n_int_per_interface",
            "n_exit",
            "n_pts",
        ]:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        for name in [
            "V_cell_m3",
            "V_vest_m3",
            "A_wall_cell_m2",
            "A_wall_vest_m2",
            "d_int_mm",
            "d_exit_mm",
            "duration_s",
        ]:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be > 0")

    def to_network_config(self) -> NetworkConfig:
        self.validate()
        return NetworkConfig(
            N_chain=self.N_chain,
            N_chain_b=self.N_chain_b,
            topology=self.topology,
            N_par=self.N_par,
            V_cell=self.V_cell_m3,
            V_vest=self.V_vest_m3,
            A_wall_cell=self.A_wall_cell_m2,
            A_wall_vest=self.A_wall_vest_m2,
            d_int_mm=self.d_int_mm,
            n_int_per_interface=self.n_int_per_interface,
            d_exit_mm=self.d_exit_mm,
            n_exit=self.n_exit,
            Cd_int=self.Cd_int,
            Cd_exit=self.Cd_exit,
            int_model=self.int_model,
            exit_model=self.exit_model,
            L_int_mm=self.L_int_mm,
            L_exit_mm=self.L_exit_mm,
            K_in_int=self.K_in_int,
            K_out_int=self.K_out_int,
            eps_int_um=self.eps_int_um,
            K_in_exit=self.K_in_exit,
            K_out_exit=self.K_out_exit,
            eps_exit_um=self.eps_exit_um,
        )

    def to_case_config(self) -> CaseConfig:
        self.validate()
        return CaseConfig(
            thermo=self.thermo,
            h_conv=self.h_conv_W_m2K,
            T_wall=self.T_wall_K,
            duration=self.duration_s,
            n_pts=self.n_pts,
            wall_model=self.wall_model,
            wall_C_per_area=self.wall_C_per_area_J_m2K,
            wall_h_out=self.wall_h_out_W_m2K,
            wall_T_inf=self.wall_T_inf_K,
            wall_emissivity=self.wall_emissivity,
            wall_T_sur=self.wall_T_sur_K,
            wall_q_flux=self.wall_q_flux_W_m2,
            external_model=self.external_model,
            V_ext=self.V_ext_m3,
            T_ext=self.T_ext_K,
            pump_speed_m3s=self.pump_speed_m3s,
            P_ult_Pa=self.P_ult_Pa,
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, payload: str) -> GuiCaseConfig:
        return cls(**json.loads(payload))

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> GuiCaseConfig:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
