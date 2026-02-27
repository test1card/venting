# Model v8.5

State:
- Isothermal: `m_i`
- Intermediate: `(m_i, T_i)`

Pressure relation:
`P_i = m_i R T_i / V_i`

Mass flow:
- Orifice: compressible isentropic, with choked/subsonic branching by `r = P_dn / P_up`.
- Slot: `m_dot = K (P_up^2 - P_dn^2)/(R T_up)`, `K = w delta^3 / (12 mu L)`.

Energy equation (intermediate):
`m c_v dT/dt = Σ_in m_dot_in (c_p T_in - c_v T) - Σ_out m_dot_out R T + h A (T_wall - T)`

No state clipping is used.
