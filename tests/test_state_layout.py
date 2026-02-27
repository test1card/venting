import numpy as np

from venting.state_layout import infer_layout_from_modes, split_state


def _make_y(n_vars: int, n_t: int = 4):
    return np.arange(n_vars * n_t, dtype=float).reshape(n_vars, n_t)


def test_state_layout_isothermal_n1_n5():
    for n in (1, 5):
        y = _make_y(n)
        layout = infer_layout_from_modes("isothermal", "fixed", n)
        assert layout.m_slice == slice(0, n)
        assert layout.t_slice is None
        assert layout.tw_slice is None
        m, t, tw = split_state(y, n, "isothermal", "fixed")
        assert m.shape == (n, y.shape[1])
        assert t is None and tw is None


def test_state_layout_intermediate_fixed_n1_n5():
    for n in (1, 5):
        y = _make_y(2 * n)
        m, t, tw = split_state(y, n, "intermediate", "fixed")
        assert m.shape == (n, y.shape[1])
        assert t.shape == (n, y.shape[1])
        assert tw is None


def test_state_layout_variable_fixed_n1_n5():
    for n in (1, 5):
        y = _make_y(2 * n)
        m, t, tw = split_state(y, n, "variable", "fixed")
        assert m.shape == (n, y.shape[1])
        assert t.shape == (n, y.shape[1])
        assert tw is None


def test_state_layout_intermediate_lumped_n1_n5():
    for n in (1, 5):
        y = _make_y(3 * n)
        m, t, tw = split_state(y, n, "intermediate", "lumped")
        assert m.shape == (n, y.shape[1])
        assert t.shape == (n, y.shape[1])
        assert tw.shape == (n, y.shape[1])
