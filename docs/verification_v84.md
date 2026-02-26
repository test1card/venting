# Verification v8.4

Gate tests:
1. Single-node to vacuum:
   - Compare against analytic adiabatic/isothermal references.
   - Acceptance: `<0.5%` for pressure and temperature in range `P > 1% P0`.
   - Mass consistency target: `<0.1%`.

2. Two-node (cell→vest→vacuum):
   - Mass conservation `<0.1%`.
   - Energy conservation with `h=0`: `<1%`.
