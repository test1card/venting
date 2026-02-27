from dataclasses import dataclass

import numpy as np

from .constants import P_STOP


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


@dataclass(frozen=True)
class NetworkConfig:
    N_chain: int
    N_par: int
    V_cell: float
    V_vest: float
    A_wall_cell: float
    A_wall_vest: float
    d_int_mm: float
    n_int_per_interface: int
    d_exit_mm: float
    n_exit: int
    Cd_int: float
    Cd_exit: float
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
