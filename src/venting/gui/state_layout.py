from __future__ import annotations

from dataclasses import dataclass

from venting.gui.config import GuiCaseConfig


@dataclass(frozen=True)
class StateLayout:
    n_nodes: int
    m_slice: slice
    t_slice: slice | None
    tw_slice: slice | None


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


def infer_layout(cfg: GuiCaseConfig, n_nodes: int) -> StateLayout:
    return infer_layout_from_modes(cfg.thermo, cfg.wall_model, n_nodes)
