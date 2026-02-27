from dataclasses import dataclass

from .cases import NetworkConfig
from .constants import T0
from .geometry import assert_pos, circle_area_from_d_mm
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
class ShortTubeEdge:
    a: int
    b: int
    A_total: float
    D: float
    L: float
    eps: float
    K_in: float
    K_out: float
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
Edge = OrificeEdge | SlotChannelEdge | ShortTubeEdge


def _make_exit_edge(
    cfg: NetworkConfig, a: int, A_exit_total: float, cd_exit: CdConst, label: str
) -> OrificeEdge | ShortTubeEdge:
    if cfg.exit_model == "orifice" or cfg.L_exit_mm <= 0.0:
        return OrificeEdge(a, EXT_NODE, A_exit_total, cd_exit, label=label)
    D = cfg.d_exit_mm * 1e-3
    L = cfg.L_exit_mm * 1e-3
    eps = cfg.eps_um * 1e-6
    return ShortTubeEdge(
        a=a,
        b=EXT_NODE,
        A_total=A_exit_total,
        D=D,
        L=L,
        eps=eps,
        K_in=cfg.K_in,
        K_out=cfg.K_out,
        Cd_model=cd_exit,
        label=label,
    )


def _make_int_edge(
    cfg: NetworkConfig,
    a: int,
    b: int,
    A_int_total: float,
    cd_int: CdConst,
    label: str,
) -> OrificeEdge | ShortTubeEdge:
    if cfg.int_model == "orifice" or cfg.L_int_mm <= 0.0:
        return OrificeEdge(a, b, A_int_total, cd_int, label=label)
    D = cfg.d_int_mm * 1e-3
    L = cfg.L_int_mm * 1e-3
    eps = cfg.eps_um * 1e-6
    return ShortTubeEdge(
        a=a,
        b=b,
        A_total=A_int_total,
        D=D,
        L=L,
        eps=eps,
        K_in=cfg.K_in,
        K_out=cfg.K_out,
        Cd_model=cd_int,
        label=label,
    )


def build_branching_network(
    cfg: NetworkConfig, profile: Profile
) -> tuple[list[GasNode], list[Edge], list[ExternalBC]]:
    assert_pos("N_chain", cfg.N_chain)
    assert_pos("N_par", cfg.N_par)
    assert_pos("V_cell", cfg.V_cell)
    assert_pos("V_vest", cfg.V_vest)

    if cfg.int_model not in {"orifice", "short_tube"}:
        raise ValueError("int_model must be 'orifice' or 'short_tube'")
    if cfg.exit_model not in {"orifice", "short_tube"}:
        raise ValueError("exit_model must be 'orifice' or 'short_tube'")

    nodes: list[GasNode] = [GasNode("vest", cfg.V_vest, cfg.A_wall_vest)]
    edges: list[Edge] = []

    gap_idx = None
    if cfg.use_gap:
        assert_pos("V_gap", cfg.V_gap)
        nodes.append(GasNode("gap", cfg.V_gap, cfg.A_wall_gap))
        gap_idx = 1

    base = 1 if not cfg.use_gap else 2
    for i in range(cfg.N_chain):
        nodes.append(
            GasNode(f"cell{i + 1}", cfg.N_par * cfg.V_cell, cfg.N_par * cfg.A_wall_cell)
        )

    A_int_total = (
        cfg.N_par * cfg.n_int_per_interface * circle_area_from_d_mm(cfg.d_int_mm)
    )
    A_exit_total = cfg.n_exit * circle_area_from_d_mm(cfg.d_exit_mm)

    cd_int = CdConst(cfg.Cd_int)
    cd_exit = CdConst(cfg.Cd_exit)

    if not cfg.use_gap:
        edges.append(_make_exit_edge(cfg, 0, A_exit_total, cd_exit, label="exit"))
    else:
        edges.append(
            SlotChannelEdge(
                0, gap_idx, cfg.gap_w, cfg.gap_delta, cfg.gap_L, label="vest→gap(slot)"
            )
        )
        edges.append(
            _make_exit_edge(
                cfg,
                gap_idx,
                A_exit_total,
                cd_exit,
                label="gap→ext(exit)",
            )
        )

    cell1 = base
    edges.append(
        _make_int_edge(cfg, cell1, 0, A_int_total, cd_int, label="A(cell1↔vest)")
    )
    for i in range(1, cfg.N_chain):
        a = base + i
        b = base + (i - 1)
        edges.append(
            _make_int_edge(
                cfg,
                a,
                b,
                A_int_total,
                cd_int,
                label=f"{chr(65 + i)}(cell{i + 1}↔cell{i})",
            )
        )

    bc_node = 0 if not cfg.use_gap else gap_idx
    bcs = [ExternalBC(node=bc_node, profile=profile, T_ext=T0)]
    return nodes, edges, bcs
