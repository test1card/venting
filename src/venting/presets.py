from __future__ import annotations

from dataclasses import asdict, dataclass

from .geometry import mm2_to_m2, mm3_to_m3


@dataclass(frozen=True)
class PanelPresetV9:
    A_cell_mm2: float
    A_vest_mm2: float
    h_mm: float
    V_cell: float
    V_vest: float
    A_wall_cell: float
    A_wall_vest: float

    def to_dict(self) -> dict:
        return asdict(self)


def get_default_panel_preset_v9() -> PanelPresetV9:
    """Return the historic CLI default panel geometry (SI outputs)."""
    a_cell_mm2 = 4708.2806538
    a_vest_mm2 = 5199.9595503
    h_mm = 27.9430913
    return PanelPresetV9(
        A_cell_mm2=a_cell_mm2,
        A_vest_mm2=a_vest_mm2,
        h_mm=h_mm,
        V_cell=mm3_to_m3(a_cell_mm2 * h_mm),
        V_vest=mm3_to_m3(a_vest_mm2 * h_mm),
        A_wall_cell=mm2_to_m2(2 * a_cell_mm2 + 313.0 * h_mm),
        A_wall_vest=mm2_to_m2(2 * a_vest_mm2 + 350.0 * h_mm),
    )
