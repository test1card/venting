from dataclasses import dataclass

import numpy as np

from .constants import P_STOP, T0


@dataclass(frozen=True)
class CaseConfig:
    thermo: str
    h_conv: float
    T_wall: float
    duration: float
    n_pts: int
    p_stop: float = P_STOP
    p_rms_tol: float = 1.0
    l_char_m: float = 0.1
    wall_model: str = "fixed"  # fixed|lumped
    wall_C_per_area: float = 1e9
    wall_h_out: float = 0.0
    wall_T_inf: float = T0
    wall_emissivity: float = 0.0
    wall_T_sur: float = T0
    wall_q_flux: float = 0.0
    external_model: str = "profile"  # profile|dynamic_pump
    V_ext: float = 0.1
    T_ext: float = T0
    pump_speed_m3s: float = 0.0
    P_ult_Pa: float = 0.0


NumberOrList = float | int | list[float] | tuple[float, ...]


@dataclass(frozen=True)
class NetworkConfig:
    N_chain: int
    N_par: int
    V_cell: NumberOrList
    V_vest: float
    A_wall_cell: NumberOrList
    A_wall_vest: float
    d_int_mm: NumberOrList
    n_int_per_interface: NumberOrList
    d_exit_mm: float
    n_exit: int
    Cd_int: NumberOrList
    Cd_exit: float
    int_model: str = "orifice"
    exit_model: str = "orifice"
    L_int_mm: NumberOrList = 0.0
    L_exit_mm: float = 0.0
    eps_um: float = 0.0
    K_in: float = 0.5
    K_out: float = 1.0
    K_in_int: NumberOrList | None = None
    K_out_int: NumberOrList | None = None
    eps_int_um: NumberOrList | None = None
    K_in_exit: float | None = None
    K_out_exit: float | None = None
    eps_exit_um: float | None = None
    topology: str = "single_chain"  # single_chain|two_chain_shared_vest
    N_chain_b: int | None = None
    use_gap: bool = False
    V_gap: float = 0.0
    A_wall_gap: float = 0.0
    gap_w: float = 0.0
    gap_delta: float = 0.0
    gap_L: float = 0.0


@dataclass
class SolveResult:
    t: np.ndarray
    m: np.ndarray
    T: np.ndarray
    P: np.ndarray
    P_ext: np.ndarray
    peak_diag: dict[str, dict]
    max_dP: dict[str, float]
    tau_exit: float
    meta: dict
