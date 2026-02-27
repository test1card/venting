from __future__ import annotations

from venting.gui.config import GuiCaseConfig
from venting.state_layout import StateLayout, infer_layout_from_modes, split_state


def infer_layout(cfg: GuiCaseConfig, n_nodes: int) -> StateLayout:
    return infer_layout_from_modes(cfg.thermo, cfg.wall_model, n_nodes)


__all__ = ["StateLayout", "infer_layout", "infer_layout_from_modes", "split_state"]
