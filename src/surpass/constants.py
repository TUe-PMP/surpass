"""Physical constants and unit conversions used throughout the package."""

from scipy.constants import Boltzmann as K_B_J_K
from scipy.constants import elementary_charge as Q_C
from scipy.constants import epsilon_0 as EPS0_F_M
from scipy.constants import electron_mass as M0_KG

EPS0_F_CM = EPS0_F_M / 100.0


def thermal_voltage(temperature_k: float) -> float:
    """Return kT/q in volts; numerically it also equals kT in electron-volts."""
    return K_B_J_K * temperature_k / Q_C

