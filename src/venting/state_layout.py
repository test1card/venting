from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StateLayout:
    n_nodes: int
    m_slice: slice
    t_slice: slice | None
    tw_slice: slice | None


def split_state(y, node_count: int, thermo: str, wall_model: str):
    layout = infer_layout_from_modes(thermo, wall_model, node_count)
    m = y[layout.m_slice, :]
    t = y[layout.t_slice, :] if layout.t_slice is not None else None
    tw = y[layout.tw_slice, :] if layout.tw_slice is not None else None
    return m, t, tw


def infer_layout_from_modes(thermo: str, wall_model: str, n_nodes: int) -> StateLayout:
    if n_nodes <= 0:
        raise ValueError("n_nodes must be > 0")
    m_slice = slice(0, n_nodes)
    if thermo == "isothermal":
        return StateLayout(
            n_nodes=n_nodes, m_slice=m_slice, t_slice=None, tw_slice=None
        )
    t_slice = slice(n_nodes, 2 * n_nodes)
    tw_slice = slice(2 * n_nodes, 3 * n_nodes) if wall_model == "lumped" else None
    return StateLayout(
        n_nodes=n_nodes, m_slice=m_slice, t_slice=t_slice, tw_slice=tw_slice
    )
