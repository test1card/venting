#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
venting_v84.py — v8.4: near-production 0D venting solver (edge-based graph)

Core upgrades vs v8.3:
  A) State = (m_i, T_i) for intermediate; state = (m_i) for isothermal
     => P_i = m_i*R*T_i/V_i  is always >= 0 if m_i >= 0 (constructive positivity).
  B) No state clipping. Only denominator protection (T >= T_SAFE) and event-based stop.
  C) Symmetry handled by aggregation (V, A_wall, A_or_total), not by multipliers.
  D) Profile = object with callable P(t) + explicit event times (breakpoints).
     Envelope MUST be provided as a table to avoid invented parameters.
  E) Edge types:
       - OrificeEdge: compressible isentropic (choked/subsonic)
       - SlotChannelEdge: viscous laminar slot/channel using (P_up^2 - P_dn^2)/(R*T)
  F) Reproducibility: save NPZ + JSON metadata; optional plots.
  G) Gate tests: single-node analytic; two-node mass+energy; network smoke.

This is still a 0D model. Its “10/10” is about internal consistency, rigor,
and reproducibility, not about eliminating model-form uncertainty (Cd, geometry, etc.).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Dict, Union

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix

# matplotlib is optional (plots)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


# =============================================================================
# CONSTANTS (Air, ideal gas)
# =============================================================================
GAMMA   = 1.4
R_GAS   = 287.05  # J/(kg*K)
C_V     = R_GAS / (GAMMA - 1.0)
C_P     = GAMMA * C_V

T0      = 300.0       # K
P0      = 101325.0    # Pa

# Numerical safety (NOT state clipping; only denominators + stop conditions)
T_SAFE  = 1.0         # K (protect sqrt(T), viscosity, etc.)
M_SAFE  = 1e-18       # kg (protect division in dT)
P_STOP  = 5.0         # Pa stop when all nodes effectively evacuated (configurable)

# Critical pressure ratio for choked flow
PI_C = (2.0/(GAMMA+1.0))**(GAMMA/(GAMMA-1.0))
C_CHOKED = math.sqrt(GAMMA * (2.0/(GAMMA+1.0))**((GAMMA+1.0)/(GAMMA-1.0)))


# =============================================================================
# Utility: Units and geometry
# =============================================================================
def mm_to_m(x_mm: float) -> float:
    return x_mm * 1e-3

def mm2_to_m2(x_mm2: float) -> float:
    return x_mm2 * 1e-6

def mm3_to_m3(x_mm3: float) -> float:
    return x_mm3 * 1e-9

def circle_area_from_d_mm(d_mm: float) -> float:
    d_m = mm_to_m(d_mm)
    return math.pi * (d_m/2.0)**2

def assert_pos(name: str, val: float) -> None:
    if not (val > 0.0):
        raise ValueError(f"{name} must be > 0, got {val}")

def assert_nonneg(name: str, val: float) -> None:
    if not (val >= 0.0):
        raise ValueError(f"{name} must be >= 0, got {val}")


# =============================================================================
# Viscosity model (Sutherland) for air — needed for viscous channel edges
# =============================================================================
def mu_air_sutherland(T: float) -> float:
    """
    Dynamic viscosity of air [Pa*s] using Sutherland's law.
    Typical constants:
      mu0 = 1.716e-5 Pa*s at T0=273.15 K
      S   = 111 K
    """
    T_eff = max(T, T_SAFE)
    mu0 = 1.716e-5
    Tref = 273.15
    S = 111.0
    return mu0 * (T_eff/Tref)**1.5 * (Tref + S)/(T_eff + S)


# =============================================================================
# Profile object with explicit event times
# =============================================================================
@dataclass(frozen=True)
class Profile:
    name: str
    P: Callable[[float], float]
    events: Tuple[Tuple[float, str], ...]  # (t_event, label)

    def P_array(self, t: np.ndarray) -> np.ndarray:
        return np.array([self.P(float(tt)) for tt in t], dtype=float)

    def classify_peak(self, t_peak: float, t_end: float, tol_s: float = 0.5) -> str:
        for te, label in self.events:
            if abs(t_peak - te) <= tol_s:
                return f"boundary({label})"
        if t_peak <= tol_s:
            return "boundary(sim_start)"
        if t_peak >= (t_end - tol_s):
            return "boundary(sim_end)"
        return "internal"


def make_profile_linear(p0: float, rate_mmhg_per_s: float) -> Profile:
    rate = rate_mmhg_per_s * 133.322  # Pa/s
    t_zero = p0 / rate

    def P(t: float) -> float:
        return max(p0 - rate*t, 0.0)

    return Profile("linear", P, events=((t_zero, "P_ext=0"),))


def make_profile_step(p0: float, step_time_s: float) -> Profile:
    def P(t: float) -> float:
        return 0.0 if t >= step_time_s else p0

    return Profile("step", P, events=((step_time_s, "step_to_vacuum"),))


def make_profile_exponential(p0: float, rate0_mmhg_per_s: float, p_floor: float = 10.0) -> Profile:
    """
    Exponential 'barometric-like' decay:
      P(t) = p0 * exp(-t/tau)
    Choose tau such that initial slope matches -rate0.
      dP/dt|0 = -p0/tau = -rate0 => tau = p0/rate0
    """
    rate0 = rate0_mmhg_per_s * 133.322
    tau = p0 / rate0
    t_floor = tau * math.log(p0 / p_floor)

    def P(t: float) -> float:
        return p0 * math.exp(-t/tau)

    return Profile("barometric_exp", P, events=((0.0, "start_max_slope"), (t_floor, f"P_ext={p_floor:.0f}Pa"),))


def make_profile_from_table(name: str, table_path: Path) -> Profile:
    """
    Table format: CSV with two columns: t_s, P_Pa
    Piecewise-linear interpolation. Events = all breakpoints.
    """
    arr = np.loadtxt(str(table_path), delimiter=",")
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("Profile table must be CSV with columns: t_s, P_Pa")
    t_tab = np.array(arr[:, 0], dtype=float)
    p_tab = np.array(arr[:, 1], dtype=float)
    if not np.all(np.diff(t_tab) > 0):
        raise ValueError("Profile table time must be strictly increasing")

    def P(t: float) -> float:
        if t <= t_tab[0]:
            return float(p_tab[0])
        if t >= t_tab[-1]:
            return float(p_tab[-1])
        return float(np.interp(t, t_tab, p_tab))

    events = tuple((float(tt), f"bp{i}") for i, tt in enumerate(t_tab))
    return Profile(name, P, events=events)


# =============================================================================
# Nodes and edges
# =============================================================================
@dataclass(frozen=True)
class GasNode:
    name: str
    V: float       # m^3
    A_wall: float  # m^2 for convection (if enabled)

@dataclass(frozen=True)
class ExternalBC:
    """
    Boundary condition on a node: external pressure P_ext(t).
    T_ext is only used if flow reverses (external -> node).
    """
    node: int
    profile: Profile
    T_ext: float = T0

# --- Cd model (minimal but explicit) ---
@dataclass(frozen=True)
class CdConst:
    Cd: float
    def __call__(self, Re: Optional[float] = None, r: Optional[float] = None) -> float:
        return float(self.Cd)

# You can add CdTable later if you have calibration data:
# interpolate Cd(Re) from file. Not included here to avoid fake correlations.


@dataclass(frozen=True)
class OrificeEdge:
    """
    Compressible orifice edge between nodes a <-> b. If b == EXT_NODE, uses BC.
    Area is TOTAL area (already includes N_holes multiplier).
    """
    a: int
    b: int
    A_total: float          # m^2
    Cd_model: CdConst
    label: str = ""

EXT_NODE = -1


@dataclass(frozen=True)
class SlotChannelEdge:
    """
    Viscous laminar slot/channel edge (for 'gap' / pocket / restrictive path).
    Uses compressible isothermal-like laminar formula:

      mdot = K * (P_up^2 - P_dn^2) / (R * T_up)

    where K = w * delta^3 / (12 * mu(T) * L)

    This is a *model-form* choice. It is at least dimensionally consistent and
    avoids mixing a linear conductance with a choked conductance improperly.
    """
    a: int
    b: int
    w: float       # m
    delta: float   # m
    L: float       # m
    label: str = ""


Edge = Union[OrificeEdge, SlotChannelEdge]


# =============================================================================
# Physics: mass flow for OrificeEdge
# =============================================================================
def mdot_orifice_pos(P_up: float, T_up: float, P_dn: float, Cd: float, A: float) -> float:
    """
    Returns mdot >= 0 from upstream to downstream.
    Isentropic orifice with choked/subsonic transition. Ideal gas.
    """
    if P_up <= 0.0 or A <= 0.0:
        return 0.0
    T_eff = max(T_up, T_SAFE)
    r = max(P_dn, 0.0) / P_up
    if r >= 1.0:
        return 0.0

    if r <= PI_C:
        return Cd * A * P_up * C_CHOKED / math.sqrt(R_GAS * T_eff)

    bracket = r**(2.0/GAMMA) - r**((GAMMA+1.0)/GAMMA)
    if bracket <= 0.0:
        return 0.0
    return Cd * A * P_up * math.sqrt(2.0*GAMMA/((GAMMA-1.0)*R_GAS*T_eff) * bracket)


def mdot_slot_pos(P_up: float, T_up: float, P_dn: float, w: float, delta: float, L: float) -> float:
    """
    Compressible laminar slot: mdot >= 0 from up to dn.
    mdot = K*(P_up^2 - P_dn^2)/(R*T_up), K = w*delta^3/(12*mu*L)
    """
    if P_up <= 0.0:
        return 0.0
    T_eff = max(T_up, T_SAFE)
    mu = mu_air_sutherland(T_eff)
    K = w * (delta**3) / (12.0 * mu * L)
    dp2 = max(P_up**2 - max(P_dn, 0.0)**2, 0.0)
    return K * dp2 / (R_GAS * T_eff)


# =============================================================================
# Model configuration and results containers
# =============================================================================
@dataclass(frozen=True)
class CaseConfig:
    thermo: str                  # "isothermal" or "intermediate"
    h_conv: float                # W/(m^2*K), used only if thermo="intermediate"
    T_wall: float                # K (constant wall temperature model)

    duration: float              # s
    n_pts: int                   # time samples for output

    # stop / tolerances
    p_stop: float = P_STOP       # Pa
    p_rms_tol: float = 1.0       # Pa (equilibrium RMS tolerance)

@dataclass(frozen=True)
class NetworkConfig:
    # Aggregated branching chain:
    N_chain: int
    N_par: int

    # Geometry per cell / vestibule (single-branch cell geometry)
    V_cell: float
    V_vest: float
    A_wall_cell: float
    A_wall_vest: float

    # Orifice parameters (diameters + counts)
    d_int_mm: float
    n_int_per_interface: int     # holes per interface per cell
    d_exit_mm: float
    n_exit: int                  # holes at vestibule exit

    # Cd models
    Cd_int: float
    Cd_exit: float

    # Optional downstream gap (slot channel)
    use_gap: bool = False
    V_gap: float = 0.0
    A_wall_gap: float = 0.0
    gap_w: float = 0.0
    gap_delta: float = 0.0
    gap_L: float = 0.0


@dataclass
class SolveResult:
    t: np.ndarray
    m: np.ndarray           # shape (N_nodes, N_t)
    T: np.ndarray           # shape (N_nodes, N_t)
    P: np.ndarray           # shape (N_nodes, N_t)
    P_ext: np.ndarray       # shape (N_t,)
    peak_diag: Dict[str, dict]
    max_dP: Dict[str, float]
    tau_exit: float
    meta: dict


# =============================================================================
# Build aggregated network graph
# =============================================================================
def build_branching_network(cfg: NetworkConfig, profile: Profile) -> Tuple[List[GasNode], List[Edge], List[ExternalBC]]:
    assert_pos("N_chain", cfg.N_chain)
    assert_pos("N_par", cfg.N_par)
    assert_pos("V_cell", cfg.V_cell)
    assert_pos("V_vest", cfg.V_vest)

    nodes: List[GasNode] = []
    edges: List[Edge] = []

    # Node 0: vestibule (single, not multiplied by N_par)
    nodes.append(GasNode("vest", cfg.V_vest, cfg.A_wall_vest))

    # Optional: downstream gap node (between vest and external)
    gap_idx = None
    if cfg.use_gap:
        assert_pos("V_gap", cfg.V_gap)
        nodes.append(GasNode("gap", cfg.V_gap, cfg.A_wall_gap))
        gap_idx = 1  # after vest

    # Chain nodes: aggregated across N_par identical chains
    # Stage i node index depends on presence of gap.
    base = 1 if not cfg.use_gap else 2
    for i in range(cfg.N_chain):
        nodes.append(GasNode(f"cell{i+1}", cfg.N_par * cfg.V_cell, cfg.N_par * cfg.A_wall_cell))

    # Areas
    A_int_single = circle_area_from_d_mm(cfg.d_int_mm)
    A_exit_single = circle_area_from_d_mm(cfg.d_exit_mm)

    # Total parallel areas
    # Between each adjacent stage: N_par chains * n_int holes per interface per cell
    A_int_total = cfg.N_par * cfg.n_int_per_interface * A_int_single
    # Vestibule exit has n_exit holes total
    A_exit_total = cfg.n_exit * A_exit_single

    Cd_int_model = CdConst(cfg.Cd_int)
    Cd_exit_model = CdConst(cfg.Cd_exit)

    # Build edges:
    # Exit path: either vest->ext directly, or vest->gap->ext
    if not cfg.use_gap:
        edges.append(OrificeEdge(0, EXT_NODE, A_exit_total, Cd_exit_model, label="exit"))
    else:
        # vest -> gap: slot channel (gap restriction)
        edges.append(SlotChannelEdge(0, gap_idx, cfg.gap_w, cfg.gap_delta, cfg.gap_L, label="vest→gap(slot)"))
        # gap -> ext: exit orifice (through panel skin etc.)
        edges.append(OrificeEdge(gap_idx, EXT_NODE, A_exit_total, Cd_exit_model, label="gap→ext(exit)"))

    # Internal chain edges: cell1 <-> vest, cell(i+1) <-> cell(i)
    # NOTE: with aggregation, areas already include N_par; no multipliers required.
    # cell1 index:
    cell1 = base + 0
    edges.append(OrificeEdge(cell1, 0, A_int_total, Cd_int_model, label="A(cell1↔vest)"))
    for i in range(1, cfg.N_chain):
        a = base + i       # cell(i+1)
        b = base + (i-1)   # cell(i)
        edges.append(OrificeEdge(a, b, A_int_total, Cd_int_model, label=f"{chr(65+i)}(cell{i+1}↔cell{i})"))

    # External BC is attached to the last exit-connected node:
    bc_node = 0 if not cfg.use_gap else gap_idx
    bcs = [ExternalBC(node=bc_node, profile=profile, T_ext=T0)]

    return nodes, edges, bcs


# =============================================================================
# RHS assembly for isothermal and intermediate
# =============================================================================
def build_rhs(nodes: Sequence[GasNode],
              edges: Sequence[Edge],
              bcs: Sequence[ExternalBC],
              case: CaseConfig) -> Tuple[Callable, int]:
    """
    Returns rhs(t,y) and n_vars.

    Isothermal state: y = [m_0..m_{N-1}]  (T=T0 const)
      dm/dt from edges.
      P computed for postprocessing only.

    Intermediate state: y = [m_0..m_{N-1}, T_0..T_{N-1}]
      dm/dt from edges.
      m*c_v*dT/dt = sum inflows/outflows + Q_wall
      Q_wall = h*A_wall*(T_wall - T)
    """
    N = len(nodes)
    V = np.array([n.V for n in nodes], dtype=float)
    A_w = np.array([n.A_wall for n in nodes], dtype=float)

    # Map BCs by node
    bc_map: Dict[int, ExternalBC] = {bc.node: bc for bc in bcs}

    def P_from_mT(m: np.ndarray, T: np.ndarray) -> np.ndarray:
        return m * R_GAS * T / V

    if case.thermo == "isothermal":
        n_vars = N

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            m = y[:N]
            # enforce nonnegative mass physically: if solver steps slightly negative, stop via events.
            T = np.full(N, T0, dtype=float)
            P = P_from_mT(np.maximum(m, 0.0), T)  # only for flow direction

            dm = np.zeros(N, dtype=float)

            # Edge-wise assembly
            for e in edges:
                if isinstance(e, OrificeEdge):
                    a, b = e.a, e.b
                    # Determine downstream pressure and upstream state
                    if b == EXT_NODE:
                        bc = bc_map[a]
                        Pext = bc.profile.P(float(t))
                        # flow direction by P[a] vs Pext
                        if P[a] >= Pext:
                            Cd = e.Cd_model()
                            md = mdot_orifice_pos(P[a], T[a], Pext, Cd, e.A_total)
                            dm[a] -= md
                        else:
                            Cd = e.Cd_model()
                            md = mdot_orifice_pos(Pext, bc.T_ext, P[a], Cd, e.A_total)
                            dm[a] += md
                    else:
                        Pa, Pb = P[a], P[b]
                        if Pa >= Pb:
                            Cd = e.Cd_model()
                            md = mdot_orifice_pos(Pa, T[a], Pb, Cd, e.A_total)
                            dm[a] -= md
                            dm[b] += md
                        else:
                            Cd = e.Cd_model()
                            md = mdot_orifice_pos(Pb, T[b], Pa, Cd, e.A_total)
                            dm[b] -= md
                            dm[a] += md

                elif isinstance(e, SlotChannelEdge):
                    a, b = e.a, e.b
                    Pa, Pb = P[a], P[b]
                    if Pa >= Pb:
                        md = mdot_slot_pos(Pa, T[a], Pb, e.w, e.delta, e.L)
                        dm[a] -= md
                        dm[b] += md
                    else:
                        md = mdot_slot_pos(Pb, T[b], Pa, e.w, e.delta, e.L)
                        dm[b] -= md
                        dm[a] += md
                else:
                    raise TypeError("Unknown edge type")

            return dm

        return rhs, n_vars

    if case.thermo != "intermediate":
        raise ValueError("case.thermo must be 'isothermal' or 'intermediate'")

    n_vars = 2*N
    h = float(case.h_conv)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        m = y[:N]
        T = y[N:]
        T_eff = np.maximum(T, T_SAFE)

        # Pressures for direction decisions
        P = P_from_mT(np.maximum(m, 0.0), T_eff)

        dm = np.zeros(N, dtype=float)
        dE = np.zeros(N, dtype=float)  # contribution to m*c_v*dT/dt

        # Edge-wise assembly
        for e in edges:
            if isinstance(e, OrificeEdge):
                a, b = e.a, e.b
                if b == EXT_NODE:
                    bc = bc_map[a]
                    Pext = bc.profile.P(float(t))
                    if P[a] >= Pext:
                        Cd = e.Cd_model()
                        md = mdot_orifice_pos(P[a], T_eff[a], Pext, Cd, e.A_total)
                        # outflow from node a
                        dm[a] -= md
                        dE[a] += -md * R_GAS * T[a]
                    else:
                        Cd = e.Cd_model()
                        md = mdot_orifice_pos(Pext, bc.T_ext, P[a], Cd, e.A_total)
                        # inflow from external into a
                        dm[a] += md
                        dE[a] += md * (C_P * bc.T_ext - C_V * T[a])

                else:
                    Pa, Pb = P[a], P[b]
                    if Pa >= Pb:
                        Cd = e.Cd_model()
                        md = mdot_orifice_pos(Pa, T_eff[a], Pb, Cd, e.A_total)
                        # a upstream -> b downstream
                        dm[a] -= md
                        dm[b] += md
                        dE[a] += -md * R_GAS * T[a]
                        dE[b] += md * (C_P * T[a] - C_V * T[b])
                    else:
                        Cd = e.Cd_model()
                        md = mdot_orifice_pos(Pb, T_eff[b], Pa, Cd, e.A_total)
                        dm[b] -= md
                        dm[a] += md
                        dE[b] += -md * R_GAS * T[b]
                        dE[a] += md * (C_P * T[b] - C_V * T[a])

            elif isinstance(e, SlotChannelEdge):
                a, b = e.a, e.b
                Pa, Pb = P[a], P[b]
                if Pa >= Pb:
                    md = mdot_slot_pos(Pa, T_eff[a], Pb, e.w, e.delta, e.L)
                    dm[a] -= md
                    dm[b] += md
                    dE[a] += -md * R_GAS * T[a]
                    dE[b] += md * (C_P * T[a] - C_V * T[b])
                else:
                    md = mdot_slot_pos(Pb, T_eff[b], Pa, e.w, e.delta, e.L)
                    dm[b] -= md
                    dm[a] += md
                    dE[b] += -md * R_GAS * T[b]
                    dE[a] += md * (C_P * T[b] - C_V * T[a])
            else:
                raise TypeError("Unknown edge type")

        # Wall heat exchange (constant wall temperature model)
        Q = h * A_w * (case.T_wall - T)
        dE += Q

        dT = np.where(np.maximum(m, 0.0) > M_SAFE, dE / (np.maximum(m, 0.0) * C_V), 0.0)

        return np.concatenate([dm, dT])

    return rhs, n_vars


# =============================================================================
# Solver wrapper + events
# =============================================================================
def solve_case(nodes: Sequence[GasNode],
               edges: Sequence[Edge],
               bcs: Sequence[ExternalBC],
               case: CaseConfig) -> solve_ivp:
    N = len(nodes)
    V = np.array([n.V for n in nodes], dtype=float)

    rhs, n_vars = build_rhs(nodes, edges, bcs, case)

    # Initial state: uniform P0, T0
    # m0 = P0*V/(R*T0)
    m0 = P0 * V / (R_GAS * T0)

    if case.thermo == "isothermal":
        y0 = m0.copy()
    else:
        y0 = np.concatenate([m0, np.full(N, T0)])

    t_eval = np.linspace(0.0, case.duration, int(case.n_pts))

    # Jacobian sparsity from graph connectivity
    jac = lil_matrix((n_vars, n_vars))
    for i in range(N):
        jac[i, i] = 1
    for e in edges:
        if isinstance(e, (OrificeEdge, SlotChannelEdge)):
            a, b = e.a, e.b
            if b != EXT_NODE:
                jac[a, b] = 1
                jac[b, a] = 1
    if case.thermo == "intermediate":
        # T block
        for i in range(N):
            jac[N+i, N+i] = 1
            jac[i, N+i] = 1
            jac[N+i, i] = 1
        for e in edges:
            if isinstance(e, (OrificeEdge, SlotChannelEdge)):
                a, b = e.a, e.b
                if b != EXT_NODE:
                    jac[N+a, N+b] = 1
                    jac[N+b, N+a] = 1
                    jac[a, N+b] = 1
                    jac[b, N+a] = 1
                    jac[N+a, b] = 1
                    jac[N+b, a] = 1

    # Events: (1) nonnegative mass, (2) P RMS close to P_ext, (3) all P below p_stop
    bc0 = bcs[0]
    profile = bc0.profile

    def P_from_state(y: np.ndarray) -> np.ndarray:
        if case.thermo == "isothermal":
            m = y[:N]
            T = np.full(N, T0)
        else:
            m = y[:N]
            T = np.maximum(y[N:], T_SAFE)
        return np.maximum(m, 0.0) * R_GAS * T / V

    # (1) Mass nonnegativity: stop if any m_i < -eps (solver should avoid, but guard anyway)
    events = []
    for i in range(N):
        def ev_mi(t, y, ii=i):
            return y[ii] + 1e-15
        ev_mi.terminal = True
        ev_mi.direction = -1
        events.append(ev_mi)

    # (2) Pressure RMS equilibrium vs external
    def ev_equil_rms(t, y):
        P = P_from_state(y)
        Pe = profile.P(float(t))
        rms = float(np.sqrt(np.mean((P - Pe)**2)))
        return rms - case.p_rms_tol
    ev_equil_rms.terminal = True
    ev_equil_rms.direction = -1
    events.append(ev_equil_rms)

    # (3) Stop when max(P_i) < p_stop and P_ext already ~vacuum-ish
    def ev_p_stop(t, y):
        P = P_from_state(y)
        Pe = profile.P(float(t))
        return max(np.max(P) - case.p_stop, Pe - 10.0)
    ev_p_stop.terminal = True
    ev_p_stop.direction = -1
    events.append(ev_p_stop)

    # Step control: estimate a conservative max_step
    # Use max total area among orifices as worst-case.
    A_max = 0.0
    Cd_max = 0.0
    for e in edges:
        if isinstance(e, OrificeEdge):
            A_max = max(A_max, e.A_total)
            Cd_max = max(Cd_max, e.Cd_model())
    V_min = float(np.min(V))
    mdot_ch = Cd_max * A_max * C_CHOKED * P0 / math.sqrt(R_GAS * T0) if A_max > 0 else 0.0
    tau_min = (V_min * P0) / (R_GAS * T0 * mdot_ch) if mdot_ch > 0 else 1.0
    max_step = max(min(case.duration / 2000.0, tau_min / 10.0), 1e-4)

    rtol = 1e-7 if case.thermo == "isothermal" else 1e-6
    atol = 1e-10 if case.thermo == "isothermal" else 1e-8

    sol = solve_ivp(rhs, (0.0, case.duration), y0,
                    method="Radau",
                    t_eval=t_eval,
                    rtol=rtol, atol=atol,
                    max_step=max_step,
                    jac_sparsity=jac,
                    events=events)

    return sol


# =============================================================================
# Postprocessing: edges ΔP, peaks, regime, tau_exit
# =============================================================================
def compute_tau_exit(total_volume: float, Cd_exit: float, A_exit_total: float) -> float:
    """
    τ_exit defined from initial choked mass flow at P0,T0 (diagnostic only).
      mdot_ch = Cd*A*P0*C_choked/sqrt(R*T0)
      tau = (V_total*P0)/(R*T0*mdot_ch)
    """
    mdot_ch = Cd_exit * A_exit_total * C_CHOKED * P0 / math.sqrt(R_GAS * T0)
    if mdot_ch <= 0.0:
        return float("inf")
    return (total_volume * P0) / (R_GAS * T0 * mdot_ch)


def summarize_result(nodes: Sequence[GasNode],
                     edges: Sequence[Edge],
                     bcs: Sequence[ExternalBC],
                     case: CaseConfig,
                     sol) -> SolveResult:
    N = len(nodes)
    V = np.array([n.V for n in nodes], dtype=float)
    profile = bcs[0].profile

    t = sol.t
    if case.thermo == "isothermal":
        m = sol.y[:N]
        T = np.full_like(m, T0)
    else:
        m = sol.y[:N]
        T = sol.y[N:2*N]
    T_eff = np.maximum(T, T_SAFE)
    P = np.maximum(m, 0.0) * R_GAS * T_eff / V[:, None]
    P_ext = profile.P_array(t)

    # Build edge ΔP time series and peak diagnostics
    dP_edges: Dict[str, np.ndarray] = {}
    max_dP: Dict[str, float] = {}
    peak_diag: Dict[str, dict] = {}

    # Precompute total volume for tau_exit diagnostic
    total_volume = float(np.sum(V))

    # Determine exit area and Cd_exit for tau_exit: find the first edge labeled "exit" or containing "exit"
    A_exit_total = None
    Cd_exit = None
    for e in edges:
        if isinstance(e, OrificeEdge) and ("exit" in e.label.lower() or e.label.lower() == "exit"):
            A_exit_total = e.A_total
            Cd_exit = e.Cd_model()
            break
    if A_exit_total is None:
        # fallback: max orifice area
        A_exit_total = max((e.A_total for e in edges if isinstance(e, OrificeEdge)), default=0.0)
        Cd_exit = 0.62

    tau_exit = compute_tau_exit(total_volume, float(Cd_exit), float(A_exit_total))

    # Edge loop
    for e in edges:
        if isinstance(e, OrificeEdge):
            a, b = e.a, e.b
            label = e.label or f"orifice({a}->{b})"
            if b == EXT_NODE:
                dp = P[a, :] - P_ext
                # upstream at peak based on actual direction at that time
                pass
            else:
                dp = P[a, :] - P[b, :]
            dP_edges[label] = dp

        elif isinstance(e, SlotChannelEdge):
            a, b = e.a, e.b
            label = e.label or f"slot({a}->{b})"
            dp = P[a, :] - P[b, :]
            dP_edges[label] = dp

    for label, dp in dP_edges.items():
        idx = int(np.argmax(np.abs(dp)))
        max_dP[label] = float(np.abs(dp[idx]))

        # Determine upstream/downstream at peak for r and choked diagnostics
        # We need to locate the edge by label:
        edge_obj = None
        for e in edges:
            if (isinstance(e, OrificeEdge) and (e.label or f"orifice({e.a}->{e.b})") == label) or \
               (isinstance(e, SlotChannelEdge) and (e.label or f"slot({e.a}->{e.b})") == label):
                edge_obj = e
                break

        if edge_obj is None:
            continue

        if isinstance(edge_obj, OrificeEdge):
            a, b = edge_obj.a, edge_obj.b
            if b == EXT_NODE:
                Pa = float(P[a, idx])
                Pb = float(P_ext[idx])
                Ta = float(T_eff[a, idx])
            else:
                Pa = float(P[a, idx])
                Pb = float(P[b, idx])
                Ta = float(T_eff[a, idx])

            # upstream defined by higher pressure at peak
            if Pa >= Pb:
                P_up, P_dn, T_up = Pa, Pb, Ta
            else:
                # if reverse, upstream is "b"; approximate T_up by node b temperature
                if b == EXT_NODE:
                    P_up, P_dn, T_up = Pb, Pa, T0
                else:
                    P_up, P_dn, T_up = Pb, Pa, float(T_eff[b, idx])

            r_pk = (P_dn / P_up) if P_up > 0 else 0.0
            choked = bool(r_pk <= PI_C)

            peak_type = profile.classify_peak(float(t[idx]), float(t[-1]), tol_s=0.5)

            peak_diag[label] = {
                "t_peak": float(t[idx]),
                "t_peak_over_tau_exit": float(t[idx] / tau_exit) if np.isfinite(tau_exit) else float("nan"),
                "P_up": P_up,
                "P_down": P_dn,
                "r": float(r_pk),
                "regime": "CHOKED" if choked else "subsonic",
                "T_up": float(T_up),
                "peak_type": peak_type,
                "dP_signed": float(dp[idx]),
            }

        else:
            # slot channel: no choked criterion
            a, b = edge_obj.a, edge_obj.b
            Pa = float(P[a, idx])
            Pb = float(P[b, idx])
            peak_type = profile.classify_peak(float(t[idx]), float(t[-1]), tol_s=0.5)
            peak_diag[label] = {
                "t_peak": float(t[idx]),
                "t_peak_over_tau_exit": float(t[idx] / tau_exit) if np.isfinite(tau_exit) else float("nan"),
                "P_up": max(Pa, Pb),
                "P_down": min(Pa, Pb),
                "r": float(min(Pa, Pb)/max(Pa, Pb)) if max(Pa, Pb) > 0 else 0.0,
                "regime": "viscous_slot",
                "T_up": float(T_eff[a, idx] if Pa >= Pb else T_eff[b, idx]),
                "peak_type": peak_type,
                "dP_signed": float(dp[idx]),
            }

    meta = {
        "case": asdict(case),
        "nodes": [asdict(n) for n in nodes],
        "edges": [asdict(e) for e in edges],
        "profile": {"name": profile.name, "events": list(profile.events)},
        "solver": {
            "success": bool(sol.success),
            "message": str(sol.message),
            "t_end": float(sol.t[-1]),
            "n_steps": int(len(sol.t)),
        }
    }

    return SolveResult(
        t=t, m=m, T=T, P=P, P_ext=P_ext,
        peak_diag=peak_diag, max_dP=max_dP,
        tau_exit=float(tau_exit), meta=meta
    )


# =============================================================================
# Gate tests (single-node and two-node)
# =============================================================================
def gate_test_single_node_orifice() -> None:
    """
    Single node V, one orifice to vacuum (P_ext=0). Compare to analytical solutions
    in choked regime (r=0). PASS criteria are strict; if it fails, abort work.
    """
    # Geometry consistent with earlier examples
    V = 131.6e-6
    d_mm = 2.0
    A = circle_area_from_d_mm(d_mm)
    Cd = 0.62
    A_wall = 181.6e-4

    # Analytical alpha based on T0
    alpha = Cd * A * C_CHOKED * math.sqrt(R_GAS * T0) / V
    tau = 1.0 / alpha
    beta = (GAMMA - 1.0) / 2.0

    def P_adi(t): return P0 * (1.0 + beta*alpha*t)**(-2.0*GAMMA/(GAMMA-1.0))
    def T_adi(t): return T0 * (1.0 + beta*alpha*t)**(-2.0)
    def P_iso(t): return P0 * math.exp(-alpha*t)

    # Build model: one node, one orifice edge to ext, profile is step-to-vacuum at t=0
    prof = Profile("vacuum", lambda t: 0.0, events=((0.0, "vacuum"),))
    nodes = [GasNode("node", V, A_wall)]
    edges = [OrificeEdge(0, EXT_NODE, A, CdConst(Cd), label="exit")]
    bcs = [ExternalBC(0, prof, T_ext=T0)]

    # Run isothermal and intermediate h=0, h->inf
    case_iso = CaseConfig(thermo="isothermal", h_conv=0.0, T_wall=T0, duration=6*tau, n_pts=2000)
    sol_iso = solve_case(nodes, edges, bcs, case_iso)
    res_iso = summarize_result(nodes, edges, bcs, case_iso, sol_iso)

    case_adi = CaseConfig(thermo="intermediate", h_conv=0.0, T_wall=T0, duration=6*tau, n_pts=2000)
    sol_adi = solve_case(nodes, edges, bcs, case_adi)
    res_adi = summarize_result(nodes, edges, bcs, case_adi, sol_adi)

    case_inf = CaseConfig(thermo="intermediate", h_conv=1e6, T_wall=T0, duration=6*tau, n_pts=2000)
    sol_inf = solve_case(nodes, edges, bcs, case_inf)
    res_inf = summarize_result(nodes, edges, bcs, case_inf, sol_inf)

    # Compare in region P > 1% P0
    mask = res_adi.P[0, :] > 0.01 * P0
    t = res_adi.t[mask]

    Pn = res_adi.P[0, mask]
    Tn = res_adi.T[0, mask]
    Pa = np.array([P_adi(float(tt)) for tt in t])
    Ta = np.array([T_adi(float(tt)) for tt in t])

    errP = float(np.max(np.abs(Pn - Pa) / P0))
    errT = float(np.max(np.abs(Tn - Ta) / T0))
    if not (errP < 5e-3 and errT < 5e-3):
        raise RuntimeError(f"GATE1 FAIL (adiabatic): errP={errP:.2e}, errT={errT:.2e}")

    # h->inf should match isothermal P(t)
    mask2 = res_inf.P[0, :] > 0.01*P0
    t2 = res_inf.t[mask2]
    Pn2 = res_inf.P[0, mask2]
    Pi = np.array([P_iso(float(tt)) for tt in t2])
    errP2 = float(np.max(np.abs(Pn2 - Pi) / P0))
    if not (errP2 < 5e-3):
        raise RuntimeError(f"GATE1 FAIL (h->inf): errP={errP2:.2e}")

    # Adiabatic must differ from isothermal at 2 tau
    t2tau = 2.0 * tau
    i2 = int(np.argmin(np.abs(res_adi.t - t2tau)))
    P_ad_2 = float(res_adi.P[0, i2])
    P_is_2 = P_iso(float(res_adi.t[i2]))
    rel = abs(P_ad_2 - P_is_2)/max(P_ad_2, P_is_2)
    if not (rel > 0.2):
        raise RuntimeError(f"GATE1 FAIL (adi vs iso not separated): rel={rel:.3f}")

    # Mass conservation check (adiabatic): m_out integral vs m0 - mf
    # (numerically, we can approximate m_out from m(t) since ext is vacuum and no inflow)
    m0 = P0*V/(R_GAS*T0)
    mf = float(res_adi.m[0, -1])
    mout = m0 - mf
    if mout < -1e-8:
        raise RuntimeError("GATE1 FAIL: mass increased in vacuum blowdown")

    # If reached here: pass
    return


def gate_test_two_node_mass_energy() -> None:
    """
    Two nodes: cell -> vest -> vacuum, h=0.
    Check global mass and global internal-energy balance against integrated outflow enthalpy.
    """
    Vc = 131.6e-6
    Vv = 145.3e-6
    d_mm = 2.0
    A = circle_area_from_d_mm(d_mm)
    Cd = 0.62
    Aw = 181.6e-4

    prof = Profile("vacuum", lambda t: 0.0, events=((0.0, "vacuum"),))
    nodes = [GasNode("vest", Vv, Aw), GasNode("cell", Vc, Aw)]
    edges = [
        OrificeEdge(1, 0, A, CdConst(Cd), label="cell↔vest"),
        OrificeEdge(0, EXT_NODE, A, CdConst(Cd), label="exit"),
    ]
    bcs = [ExternalBC(0, prof, T_ext=T0)]

    case = CaseConfig(thermo="intermediate", h_conv=0.0, T_wall=T0, duration=3.0, n_pts=3000)
    sol = solve_case(nodes, edges, bcs, case)
    res = summarize_result(nodes, edges, bcs, case, sol)

    # Mass: m_total(t) + m_out(t) = const.
    m_total_0 = float(np.sum(res.m[:, 0]))
    m_total_f = float(np.sum(res.m[:, -1]))

    # Compute mdot_exit(t) using upstream node 0 vs vacuum
    # This is only for auditing, not for the solver.
    mdot = []
    for k, tt in enumerate(res.t):
        Pvest = float(res.P[0, k])
        Tvest = float(max(res.T[0, k], T_SAFE))
        md = mdot_orifice_pos(Pvest, Tvest, 0.0, Cd, A)
        mdot.append(md)
    mdot = np.array(mdot, dtype=float)
    m_out = float(np.trapezoid(mdot, res.t))

    err_m = abs((m_total_f + m_out) - m_total_0) / max(m_total_0, 1e-20)
    if not (err_m < 1e-3):
        raise RuntimeError(f"GATE2 FAIL (mass): err={err_m:.2e}")

    # Energy: E_int(t) + ∫ mdot_exit * cp * T_up dt = E0 (since rigid, Q=0)
    E0 = float(np.sum(res.m[:, 0] * C_V * res.T[:, 0]))
    Ef = float(np.sum(res.m[:, -1] * C_V * res.T[:, -1]))
    Eout = float(np.trapezoid(mdot * C_P * np.maximum(res.T[0, :], T_SAFE), res.t))
    err_E = abs((Ef + Eout) - E0) / max(E0, 1e-20)
    if not (err_E < 1e-2):
        raise RuntimeError(f"GATE2 FAIL (energy): err={err_E:.2e}")

    return


# =============================================================================
# IO: save/load results
# =============================================================================
def save_result(outdir: Path, name: str, res: SolveResult) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    npz_path = outdir / f"{name}.npz"
    json_path = outdir / f"{name}_meta.json"

    np.savez_compressed(
        npz_path,
        t=res.t, m=res.m, T=res.T, P=res.P, P_ext=res.P_ext,
        tau_exit=res.tau_exit,
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(res.meta, f, ensure_ascii=False, indent=2)


def plot_basic(outdir: Path, name: str, res: SolveResult, node_idx: int = 0) -> None:
    if not HAS_MPL:
        return
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(res.t, (res.P[node_idx] - res.P_ext)/1e3, lw=2, label=f"ΔP(node{node_idx}→ext) [kPa]")
    ax.set(xlabel="t, s", ylabel="ΔP, kPa", title=f"{name}: ΔP vs time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"{name}_dP.png", dpi=200)
    plt.close(fig)

    if res.T is not None:
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(res.t, res.T[node_idx], lw=2, label=f"T(node{node_idx}) [K]")
        ax.set(xlabel="t, s", ylabel="T, K", title=f"{name}: T vs time")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"{name}_T.png", dpi=200)
        plt.close(fig)


# =============================================================================
# Main: example run (sweeps similar to v8.3, but strict profiles)
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="./results_v84", help="Output directory")
    ap.add_argument("--profile", type=str, default="linear", choices=["linear", "step", "barometric", "table"],
                    help="External depressurization profile")
    ap.add_argument("--profile-file", type=str, default="", help="CSV file for --profile table (t_s,P_Pa)")
    ap.add_argument("--rate-mmhg", type=float, default=20.0, help="Rate for linear / barometric initial slope (mmHg/s)")
    ap.add_argument("--step-time", type=float, default=0.01, help="Step time for step profile (s)")
    ap.add_argument("--thermo", type=str, default="isothermal", choices=["isothermal", "intermediate"])
    ap.add_argument("--h", type=float, default=0.0, help="h_conv W/(m2*K), used in intermediate")
    ap.add_argument("--duration", type=float, default=150.0)
    ap.add_argument("--npts", type=int, default=2000)
    ap.add_argument("--do-plots", action="store_true")

    # Network inputs (defaults from your CAD numbers in v8.2/v8.3)
    ap.add_argument("--d-int", type=float, default=2.0)
    ap.add_argument("--d-exit", type=float, default=2.0)
    ap.add_argument("--n-int", type=int, default=1, help="holes per interface per cell")
    ap.add_argument("--n-exit", type=int, default=1, help="exit holes total")

    ap.add_argument("--Cd-int", type=float, default=0.62)
    ap.add_argument("--Cd-exit", type=float, default=0.62)

    ap.add_argument("--use-gap", action="store_true")
    ap.add_argument("--gap-V", type=float, default=0.0, help="gap volume [m3]")
    ap.add_argument("--gap-Aw", type=float, default=0.0, help="gap wall area [m2]")
    ap.add_argument("--gap-w", type=float, default=0.0, help="slot width [m]")
    ap.add_argument("--gap-delta", type=float, default=0.0, help="slot thickness [m]")
    ap.add_argument("--gap-L", type=float, default=0.0, help="slot length [m]")

    args = ap.parse_args()
    outdir = Path(args.outdir)

    # Gate tests first (hard fail on inconsistency)
    gate_test_single_node_orifice()
    gate_test_two_node_mass_energy()

    # Geometry (from your v8.2/v8.3 CAD-derived numbers)
    A_cell_mm2 = 4708.2806538
    h_mm = 27.9430913
    V_cell = mm3_to_m3(A_cell_mm2 * h_mm)
    A_vest_mm2 = 5199.9595503
    V_vest = mm3_to_m3(A_vest_mm2 * h_mm)

    P_cell_mm = 313.0
    A_wall_cell = mm2_to_m2(2*A_cell_mm2 + P_cell_mm*h_mm)
    P_vest_mm = 350.0
    A_wall_vest = mm2_to_m2(2*A_vest_mm2 + P_vest_mm*h_mm)

    # Profile selection
    if args.profile == "linear":
        prof = make_profile_linear(P0, args.rate_mmhg)
    elif args.profile == "step":
        prof = make_profile_step(P0, args.step_time)
    elif args.profile == "barometric":
        prof = make_profile_exponential(P0, args.rate_mmhg, p_floor=10.0)
    else:
        if not args.profile_file:
            raise ValueError("--profile table requires --profile-file")
        prof = make_profile_from_table("envelope_table", Path(args.profile_file))

    net_cfg = NetworkConfig(
        N_chain=10, N_par=2,
        V_cell=V_cell, V_vest=V_vest,
        A_wall_cell=A_wall_cell, A_wall_vest=A_wall_vest,
        d_int_mm=args.d_int, n_int_per_interface=args.n_int,
        d_exit_mm=args.d_exit, n_exit=args.n_exit,
        Cd_int=args.Cd_int, Cd_exit=args.Cd_exit,
        use_gap=bool(args.use_gap),
        V_gap=float(args.gap_V),
        A_wall_gap=float(args.gap_Aw),
        gap_w=float(args.gap_w),
        gap_delta=float(args.gap_delta),
        gap_L=float(args.gap_L),
    )

    nodes, edges, bcs = build_branching_network(net_cfg, prof)

    case = CaseConfig(
        thermo=args.thermo,
        h_conv=float(args.h),
        T_wall=T0,
        duration=float(args.duration),
        n_pts=int(args.npts),
        p_stop=P_STOP,
        p_rms_tol=1.0,
    )

    sol = solve_case(nodes, edges, bcs, case)
    if not sol.success:
        raise RuntimeError(f"Solver failed: {sol.message}")

    res = summarize_result(nodes, edges, bcs, case, sol)

    # Save
    name = f"v84_{prof.name}_{case.thermo}_dint{args.d_int:g}_dexit{args.d_exit:g}"
    save_result(outdir, name, res)

    # Quick console diagnostics
    # Find exit-like edge for reporting
    exit_keys = [k for k in res.max_dP.keys() if "exit" in k.lower()]
    key = exit_keys[0] if exit_keys else list(res.max_dP.keys())[0]
    pd = res.peak_diag.get(key, {})
    print("\n=== v8.4 SUMMARY ===")
    print(f"case: profile={prof.name} thermo={case.thermo} h={case.h_conv:g}")
    print(f"exit-like edge: {key}")
    print(f"  max|ΔP| = {res.max_dP[key]:.1f} Pa = {res.max_dP[key]/1e3:.3f} kPa")
    if pd:
        print(f"  t_peak = {pd['t_peak']:.3f} s, t/τ_exit={pd['t_peak_over_tau_exit']:.2f}")
        print(f"  r = {pd['r']:.4f}, regime={pd['regime']}, peak_type={pd['peak_type']}")
        print(f"  P_up={pd['P_up']/1e3:.3f} kPa, P_down={pd['P_down']/1e3:.3f} kPa, T_up={pd['T_up']:.2f} K")
    print(f"saved: {outdir / (name + '.npz')}")
    print(f"meta : {outdir / (name + '_meta.json')}")

    if args.do_plots:
        plot_basic(outdir, name, res, node_idx=0)


if __name__ == "__main__":
    main()
