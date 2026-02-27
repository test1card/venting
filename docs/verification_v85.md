# Verification v8.5

## Governing equations used for verification

For each control volume (rigid `V`, ideal gas):

- Mass:
  - `dm/dt = Σ m_dot_in - Σ m_dot_out`
- Pressure closure:
  - `P = m R T / V`
- Intermediate energy form in solver:
  - `m c_v dT/dt = Σ_in m_dot_in (c_p T_in - c_v T) - Σ_out m_dot_out R T + h A (T_wall - T)`

Why outflow is `m_dot_out * R T`: for a rigid control volume,
`dU/dt = Σ m_dot_in h_in - Σ m_dot_out h_out + Q`, with `U = m c_v T` and
`h = c_p T`; regrouping terms for unknown node `T` yields the explicit `-m_dot_out R T`
term in the `c_v` form.

## Analytic references in tests

Single-node blowdown to vacuum (`P_ext=0`, rigid volume, ideal gas, choked branch):

- Adiabatic (`h=0`):
  - `P(t) = P0 (1 + beta alpha t)^(-2 gamma/(gamma-1))`
  - `T(t) = T0 (1 + beta alpha t)^(-2)`
  - `beta = (gamma-1)/2`
- Isothermal limit (`h -> inf`):
  - `P(t) = P0 exp(-alpha t)`

## Gate criteria

- Single-node analytic match:
  - max rel error for pressure `< 0.5%`
  - max rel error for temperature `< 0.5%`
- Isothermal-limit match:
  - max rel pressure error `< 0.5%`
- Conservation:
  - mass `< 0.1%`
  - two-node energy `< 1%`
- Monotonic vacuum blowdown pressure:
  - nonincreasing pressure (tiny numerical tolerance only)

Masks exclude the low-pressure tail (`P <= 0.01*P0`) because relative-error metrics
become numerically ill-conditioned near machine floor where absolute pressures are tiny.
