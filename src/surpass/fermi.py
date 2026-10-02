"""Normalized complete Fermi–Dirac integrals.

We use

    F_j(eta) = 1/Gamma(j+1) * integral_0^inf x^j/[1+exp(x-eta)] dx.

The implementation favors clarity and accuracy over extreme speed. Scalar
evaluations are cached, which is sufficient for wafer reports and moderate
parameter sweeps. A later release can replace this backend with tabulation.
"""

from __future__ import annotations

from functools import lru_cache
from math import gamma

import numpy as np
from scipy.integrate import quad
from scipy.special import expit


@lru_cache(maxsize=200_000)
def _fermi_scalar(order: float, eta_rounded: float) -> float:
    eta = float(eta_rounded)
    if eta < -35.0:
        return float(np.exp(eta))

    # The Fermi factor is negligible several tens of kT above eta. Keeping a
    # minimum upper limit also resolves the nondegenerate tail accurately.
    upper = max(60.0, eta + 50.0)
    prefactor = 1.0 / gamma(order + 1.0)
    value, _ = quad(
        lambda x: prefactor * x**order * expit(eta - x),
        0.0,
        upper,
        epsabs=1e-11,
        epsrel=2e-10,
        limit=300,
    )
    return float(value)


def fermi(order: float, eta):
    """Return normalized F_j(eta) for j = -1/2, 1/2, or 3/2."""
    if order not in (-0.5, 0.5, 1.5):
        raise ValueError("Supported Fermi-integral orders are -1/2, 1/2, 3/2")
    array = np.asarray(eta, dtype=float)
    values = np.array([
        _fermi_scalar(float(order), round(float(x), 11)) for x in array.ravel()
    ]).reshape(array.shape)
    return float(values) if values.ndim == 0 else values

