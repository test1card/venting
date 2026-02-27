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
    fanno: bool = False


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


def _expand_scalar_or_list(value, expected_len: int, name: str) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)] * expected_len
    seq = list(value)
    if len(seq) != expected_len:
        raise ValueError(
            f"{name} length mismatch: expected {expected_len}, got {len(seq)}"
        )
    return [float(v) for v in seq]


def _resolve_int_losses(cfg: NetworkConfig, n_ifaces: int):
    k_in_src = cfg.K_in_int if cfg.K_in_int is not None else cfg.K_in
    k_out_src = cfg.K_out_int if cfg.K_out_int is not None else cfg.K_out
    eps_src = cfg.eps_int_um if cfg.eps_int_um is not None else cfg.eps_um
    k_in = _expand_scalar_or_list(k_in_src, n_ifaces, "K_in_int")
    k_out = _expand_scalar_or_list(k_out_src, n_ifaces, "K_out_int")
    eps = [v * 1e-6 for v in _expand_scalar_or_list(eps_src, n_ifaces, "eps_int_um")]
    return k_in, k_out, eps


def _make_exit_edge(
    cfg: NetworkConfig, a: int, A_exit_total: float, cd_exit: CdConst, label: str
) -> OrificeEdge | ShortTubeEdge:
    if cfg.exit_model == "orifice" or cfg.L_exit_mm <= 0.0:
        return OrificeEdge(a, EXT_NODE, A_exit_total, cd_exit, label=label)
    fanno = cfg.exit_model == "fanno"
    d = cfg.d_exit_mm * 1e-3
    length = cfg.L_exit_mm * 1e-3
    k_in = float(cfg.K_in_exit if cfg.K_in_exit is not None else cfg.K_in)
    k_out = float(cfg.K_out_exit if cfg.K_out_exit is not None else cfg.K_out)
    eps_um = float(cfg.eps_exit_um if cfg.eps_exit_um is not None else cfg.eps_um)
    return ShortTubeEdge(
        a=a,
        b=EXT_NODE,
        A_total=A_exit_total,
        D=d,
        L=length,
        eps=eps_um * 1e-6,
        K_in=k_in,
        K_out=k_out,
        Cd_model=cd_exit,
        label=label,
        fanno=fanno,
    )


def _make_int_edge(
    cfg: NetworkConfig,
    a: int,
    b: int,
    d_int_mm: float,
    n_int_per_interface: float,
    cd_int: float,
    l_int_mm: float,
    eps_int_m: float,
    k_in: float,
    k_out: float,
    label: str,
) -> OrificeEdge | ShortTubeEdge:
    A_int_total = cfg.N_par * n_int_per_interface * circle_area_from_d_mm(d_int_mm)
    cd_model = CdConst(cd_int)
    if cfg.int_model == "orifice" or l_int_mm <= 0.0:
        return OrificeEdge(a, b, A_int_total, cd_model, label=label)
    fanno = cfg.int_model == "fanno"
    return ShortTubeEdge(
        a=a,
        b=b,
        A_total=A_int_total,
        D=d_int_mm * 1e-3,
        L=l_int_mm * 1e-3,
        eps=eps_int_m,
        K_in=k_in,
        K_out=k_out,
        Cd_model=cd_model,
        label=label,
        fanno=fanno,
    )


def _build_single_chain(
    cfg: NetworkConfig, profile: Profile
) -> tuple[list[GasNode], list[Edge], list[ExternalBC]]:
    assert_pos("N_chain", cfg.N_chain)
    assert_pos("N_par", cfg.N_par)
    assert_pos("V_vest", cfg.V_vest)

    n_ifaces = cfg.N_chain
    v_cells = _expand_scalar_or_list(cfg.V_cell, cfg.N_chain, "V_cell")
    a_wall_cells = _expand_scalar_or_list(cfg.A_wall_cell, cfg.N_chain, "A_wall_cell")
    d_ints = _expand_scalar_or_list(cfg.d_int_mm, n_ifaces, "d_int_mm")
    n_ints = _expand_scalar_or_list(
        cfg.n_int_per_interface, n_ifaces, "n_int_per_interface"
    )
    cd_ints = _expand_scalar_or_list(cfg.Cd_int, n_ifaces, "Cd_int")
    l_ints = _expand_scalar_or_list(cfg.L_int_mm, n_ifaces, "L_int_mm")
    k_in_int, k_out_int, eps_int = _resolve_int_losses(cfg, n_ifaces)

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
            GasNode(
                f"cell{i + 1}",
                cfg.N_par * v_cells[i],
                cfg.N_par * a_wall_cells[i],
            )
        )

    A_exit_total = cfg.n_exit * circle_area_from_d_mm(cfg.d_exit_mm)
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
            _make_exit_edge(cfg, gap_idx, A_exit_total, cd_exit, label="gap→ext(exit)")
        )

    cell1 = base
    edges.append(
        _make_int_edge(
            cfg,
            cell1,
            0,
            d_ints[0],
            n_ints[0],
            cd_ints[0],
            l_ints[0],
            eps_int[0],
            k_in_int[0],
            k_out_int[0],
            label="A(cell1↔vest)",
        )
    )
    for i in range(1, cfg.N_chain):
        a = base + i
        b = base + (i - 1)
        edges.append(
            _make_int_edge(
                cfg,
                a,
                b,
                d_ints[i],
                n_ints[i],
                cd_ints[i],
                l_ints[i],
                eps_int[i],
                k_in_int[i],
                k_out_int[i],
                label=f"{chr(65 + i)}(cell{i + 1}↔cell{i})",
            )
        )

    bc_node = 0 if not cfg.use_gap else gap_idx
    bcs = [ExternalBC(node=bc_node, profile=profile, T_ext=T0)]
    return nodes, edges, bcs


def _build_two_chain_shared_vest(
    cfg: NetworkConfig, profile: Profile
) -> tuple[list[GasNode], list[Edge], list[ExternalBC]]:
    n_chain_a = cfg.N_chain
    n_chain_b = cfg.N_chain_b if cfg.N_chain_b is not None else cfg.N_chain
    if n_chain_a <= 0 or n_chain_b <= 0:
        raise ValueError("N_chain and N_chain_b must be > 0")

    n_ifaces = max(n_chain_a, n_chain_b)
    v_cells = _expand_scalar_or_list(cfg.V_cell, n_ifaces, "V_cell")
    a_wall_cells = _expand_scalar_or_list(cfg.A_wall_cell, n_ifaces, "A_wall_cell")
    d_ints = _expand_scalar_or_list(cfg.d_int_mm, n_ifaces, "d_int_mm")
    n_ints = _expand_scalar_or_list(
        cfg.n_int_per_interface, n_ifaces, "n_int_per_interface"
    )
    cd_ints = _expand_scalar_or_list(cfg.Cd_int, n_ifaces, "Cd_int")
    l_ints = _expand_scalar_or_list(cfg.L_int_mm, n_ifaces, "L_int_mm")
    k_in_int, k_out_int, eps_int = _resolve_int_losses(cfg, n_ifaces)

    nodes: list[GasNode] = [GasNode("vest", cfg.V_vest, cfg.A_wall_vest)]
    edges: list[Edge] = []

    A_exit_total = cfg.n_exit * circle_area_from_d_mm(cfg.d_exit_mm)
    edges.append(
        _make_exit_edge(cfg, 0, A_exit_total, CdConst(cfg.Cd_exit), label="exit")
    )

    base_a = len(nodes)
    for i in range(n_chain_a):
        nodes.append(
            GasNode(
                f"A_cell{i + 1}",
                cfg.N_par * v_cells[i],
                cfg.N_par * a_wall_cells[i],
            )
        )
    base_b = len(nodes)
    for i in range(n_chain_b):
        nodes.append(
            GasNode(
                f"B_cell{i + 1}",
                cfg.N_par * v_cells[i],
                cfg.N_par * a_wall_cells[i],
            )
        )

    # chain A
    for i in range(n_chain_a):
        a = base_a + i
        b = 0 if i == 0 else (base_a + i - 1)
        edges.append(
            _make_int_edge(
                cfg,
                a,
                b,
                d_ints[i],
                n_ints[i],
                cd_ints[i],
                l_ints[i],
                eps_int[i],
                k_in_int[i],
                k_out_int[i],
                label=f"A{i + 1}",
            )
        )

    # chain B
    for i in range(n_chain_b):
        a = base_b + i
        b = 0 if i == 0 else (base_b + i - 1)
        edges.append(
            _make_int_edge(
                cfg,
                a,
                b,
                d_ints[i],
                n_ints[i],
                cd_ints[i],
                l_ints[i],
                eps_int[i],
                k_in_int[i],
                k_out_int[i],
                label=f"B{i + 1}",
            )
        )

    return nodes, edges, [ExternalBC(node=0, profile=profile, T_ext=T0)]


def build_branching_network(
    cfg: NetworkConfig, profile: Profile
) -> tuple[list[GasNode], list[Edge], list[ExternalBC]]:
    if cfg.int_model not in {"orifice", "short_tube", "fanno"}:
        raise ValueError("int_model must be 'orifice', 'short_tube', or 'fanno'")
    if cfg.exit_model not in {"orifice", "short_tube", "fanno"}:
        raise ValueError("exit_model must be 'orifice', 'short_tube', or 'fanno'")

    if cfg.topology == "single_chain":
        return _build_single_chain(cfg, profile)
    if cfg.topology == "two_chain_shared_vest":
        return _build_two_chain_shared_vest(cfg, profile)
    raise ValueError("topology must be 'single_chain' or 'two_chain_shared_vest'")


def build_network(
    cfg: NetworkConfig, profile: Profile
) -> tuple[list[GasNode], list[Edge], list[ExternalBC]]:
    return build_branching_network(cfg, profile)
