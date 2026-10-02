"""Replaceable empirical submodels: mobility and band-gap narrowing."""

from __future__ import annotations

import numpy as np

from .constants import EPS0_F_M, K_B_J_K, Q_C
from .fermi import fermi
from .materials import MobilityParameters


def caughey_thomas_mobility(
    impurity_cm3: float,
    temperature_k: float,
    parameters: MobilityParameters,
) -> float:
    """Return low-field mobility in cm^2 V^-1 s^-1."""
    n = max(float(impurity_cm3), 0.0)
    p = parameters
    mu_max_t = p.mu_max_300 * (temperature_k / 300.0) ** p.temperature_exponent
    return p.mu_min + (mu_max_t - p.mu_min) / (1.0 + (n / p.n_ref_cm3) ** p.alpha)


def palankovski_point_charge_bgn_ev(
    majority_density_cm3: float,
    majority_eta: float,
    eps_r: float,
    temperature_k: float,
) -> float:
    """Return positive band-gap narrowing magnitude in eV.

    Implements the screened point-charge expression of Palankovski,
    Kaiblinger-Grujin, and Selberherr, MSE B 66, 46 (1999),
    DOI 10.1016/S0921-5107(99)00118-X.

    This is a material-general first-order model. It does not include the
    paper's dopant-specific atomic form factor.
    """
    density = max(float(majority_density_cm3), 0.0)
    if density == 0.0:
        return 0.0
    ratio = fermi(-0.5, majority_eta) / fermi(0.5, majority_eta)
    beta_m = np.sqrt(
        density * 1e6 * Q_C**2 * ratio
        / (eps_r * EPS0_F_M * K_B_J_K * temperature_k)
    )
    delta_joule = Q_C**2 * beta_m / (4.0 * np.pi * eps_r * EPS0_F_M)
    return float(delta_joule / Q_C)

