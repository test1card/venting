import math

GAMMA = 1.4
R_GAS = 287.05
C_V = R_GAS / (GAMMA - 1.0)
C_P = GAMMA * C_V

T0 = 300.0
P0 = 101325.0

T_SAFE = 1.0
M_SAFE = 1e-18
P_STOP = 5.0

PI_C = (2.0 / (GAMMA + 1.0)) ** (GAMMA / (GAMMA - 1.0))
C_CHOKED = math.sqrt(GAMMA * (2.0 / (GAMMA + 1.0)) ** ((GAMMA + 1.0) / (GAMMA - 1.0)))
