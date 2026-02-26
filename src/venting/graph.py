from dataclasses import dataclass

from .constants import T0
from .geometry import assert_pos, circle_area_from_d_mm
from .cases import NetworkConfig
from .profiles import Profile


@dataclass(frozen=True)
class GasNode:
    name: str
    V: float
    A_wall: float


@dataclass(frozen=True)
class ExternalBC:
    node: int
    profile: Profile
    T_ext: float = T0


@dataclass(frozen=True)
class CdConst:
    Cd: float

    def __call__(self, Re: float | None = None, r: float | None = None) -> float:
        return float(self.Cd)


@dataclass(frozen=True)
class OrificeEdge:
    a: int
    b: int
    A_total: float
    Cd_model: CdConst
    label: str = ""


@dataclass(frozen=True)
class SlotChannelEdge:
    a: int
    b: int
    w: float
    delta: float
    L: float
    label: str = ""


EXT_NODE = -1
Edge = OrificeEdge | SlotChannelEdge


def build_branching_network(cfg: NetworkConfig, profile: Profile) -> tuple[list[GasNode], list[Edge], list[ExternalBC]]:
    assert_pos("N_chain", cfg.N_chain)
    assert_pos("N_par", cfg.N_par)
    assert_pos("V_cell", cfg.V_cell)
    assert_pos("V_vest", cfg.V_vest)

    nodes: list[GasNode] = [GasNode("vest", cfg.V_vest, cfg.A_wall_vest)]
    edges: list[Edge] = []

    gap_idx = None
    if cfg.use_gap:
        assert_pos("V_gap", cfg.V_gap)
        nodes.append(GasNode("gap", cfg.V_gap, cfg.A_wall_gap))
        gap_idx = 1

    base = 1 if not cfg.use_gap else 2
    for i in range(cfg.N_chain):
        nodes.append(GasNode(f"cell{i + 1}", cfg.N_par * cfg.V_cell, cfg.N_par * cfg.A_wall_cell))

    A_int_total = cfg.N_par * cfg.n_int_per_interface * circle_area_from_d_mm(cfg.d_int_mm)
    A_exit_total = cfg.n_exit * circle_area_from_d_mm(cfg.d_exit_mm)

    cd_int = CdConst(cfg.Cd_int)
    cd_exit = CdConst(cfg.Cd_exit)

    if not cfg.use_gap:
        edges.append(OrificeEdge(0, EXT_NODE, A_exit_total, cd_exit, label="exit"))
    else:
        edges.append(SlotChannelEdge(0, gap_idx, cfg.gap_w, cfg.gap_delta, cfg.gap_L, label="vest→gap(slot)"))
        edges.append(OrificeEdge(gap_idx, EXT_NODE, A_exit_total, cd_exit, label="gap→ext(exit)"))

    cell1 = base
    edges.append(OrificeEdge(cell1, 0, A_int_total, cd_int, label="A(cell1↔vest)"))
    for i in range(1, cfg.N_chain):
        a = base + i
        b = base + (i - 1)
        edges.append(OrificeEdge(a, b, A_int_total, cd_int, label=f"{chr(65 + i)}(cell{i + 1}↔cell{i})"))

    bc_node = 0 if not cfg.use_gap else gap_idx
    bcs = [ExternalBC(node=bc_node, profile=profile, T_ext=T0)]
    return nodes, edges, bcs
