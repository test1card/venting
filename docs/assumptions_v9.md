# Assumptions and limitations (v9.0.0)

- 0D lumped compartments with perfect mixing in each node.
- Ideal-gas EOS in all modes (`P = mRT/V`).
- Short-tube thick-wall holes are modeled as **lossy nozzle via `Cd_eff`** with Darcy friction + minor losses.
- This is **not Fanno flow**; friction choking is not modeled in v9.
- Variable thermo mode uses smooth engineering fits `cp(T), cv(T), gamma(T), h(T), u(T)` for air.
- Lumped wall mode treats wall temperature as a single thermal capacitance state per node.
- Dynamic external model (`dynamic_pump`) adds a finite external volume and a pump sink law.
- Model-form uncertainty remains dominated by geometry and discharge/loss coefficients.
