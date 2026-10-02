"""Normal-incidence coherent transfer matrix and pump-generation profile.

Thin films are coherent and the terminal absorbing substrate is semi-infinite.
This is the appropriate first optical layer for a pump absorbed near the front
of a thick wafer.  It does not yet model rear reflections, diffuse scattering,
luminescence reabsorption, or photon recycling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.constants import c, h

from .optical_constants import OpticalMaterial


@dataclass(frozen=True)
class Layer:
    material: OpticalMaterial
    thickness_nm: float
    name: str | None = None

    def __post_init__(self):
        if self.thickness_nm < 0:
            raise ValueError("Layer thickness must be non-negative")


@dataclass(frozen=True)
class OpticalResult:
    wavelength_nm: float
    reflectance: float
    substrate_entry_fraction: float
    layer_absorptance: tuple[float, ...]
    balance_error: float
    substrate_alpha_cm1: float
    absorption_depth_nm: float


class OpticalStack:
    """Incident medium / coherent films / semi-infinite substrate."""

    def __init__(self, incident: OpticalMaterial, layers: list[Layer], substrate: OpticalMaterial):
        self.incident = incident
        self.layers = tuple(layers)
        self.substrate = substrate

    @staticmethod
    def _matrix(index: complex, thickness_nm: float, wavelength_nm: float):
        delta = 2.0 * np.pi * index * thickness_nm / wavelength_nm
        co, si = np.cos(delta), np.sin(delta)
        # Backward transfer for n+i*k and exp(-i*omega*t). A +i convention
        # would describe gain for positive k, not passive absorption.
        return np.array([[co, -1j * si / index], [-1j * index * si, co]], dtype=complex)

    @staticmethod
    def _flux(state: np.ndarray, incident_index: complex) -> float:
        # State is [E,H]; normalized time-averaged Poynting flux.
        return float(np.real(state[0] * np.conj(state[1])) / np.real(incident_index))

    def solve(self, wavelength_nm: float) -> OpticalResult:
        wl = float(wavelength_nm)
        n0 = self.incident.nk(wl)
        ns = self.substrate.nk(wl)
        indices = [layer.material.nk(wl) for layer in self.layers]
        matrices = [self._matrix(n, layer.thickness_nm, wl) for n, layer in zip(indices, self.layers)]

        total = np.eye(2, dtype=complex)
        for matrix in matrices:
            total = total @ matrix
        b, cc = total[0, 0] + total[0, 1] * ns, total[1, 0] + total[1, 1] * ns
        r = (n0 * b - cc) / (n0 * b + cc)
        transmission_amplitude = 2.0 * n0 / (n0 * b + cc)
        reflectance = float(abs(r) ** 2)
        substrate_entry = float(np.real(ns) / np.real(n0) * abs(transmission_amplitude) ** 2)

        state = np.array([1.0 + r, n0 * (1.0 - r)], dtype=complex)
        layer_absorption: list[float] = []
        for matrix in matrices:
            flux_left = self._flux(state, n0)
            state_right = np.linalg.solve(matrix, state)
            flux_right = self._flux(state_right, n0)
            layer_absorption.append(max(flux_left - flux_right, 0.0))
            state = state_right

        balance_error = 1.0 - reflectance - sum(layer_absorption) - substrate_entry
        alpha = self.substrate.absorption_coefficient_cm1(wl)
        depth_nm = np.inf if alpha == 0 else 1e7 / alpha
        return OpticalResult(
            wl, reflectance, substrate_entry, tuple(layer_absorption),
            float(balance_error), alpha, float(depth_nm),
        )

    def generation_profile_cm3_s(
        self, wavelength_nm: float, incident_intensity_w_cm2: float, depth_cm
    ):
        """Beer-Lambert generation in the substrate after coherent front optics."""
        if incident_intensity_w_cm2 < 0:
            raise ValueError("Incident intensity must be non-negative")
        result = self.solve(wavelength_nm)
        z = np.asarray(depth_cm, dtype=float)
        photon_energy_j = h * c / (wavelength_nm * 1e-9)
        entering_flux = incident_intensity_w_cm2 * result.substrate_entry_fraction / photon_energy_j
        return result.substrate_alpha_cm1 * entering_flux * np.exp(-result.substrate_alpha_cm1 * z)

    def generation_cell_average_cm3_s(
        self, wavelength_nm: float, incident_intensity_w_cm2: float, cell_edges_cm
    ):
        """Exact Beer-Lambert cell averages for a finite-volume grid."""
        edges = np.asarray(cell_edges_cm, dtype=float)
        if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0) or edges[0] < 0:
            raise ValueError("cell_edges_cm must be a strictly increasing non-negative grid")
        result = self.solve(wavelength_nm)
        photon_energy_j = h*c/(wavelength_nm*1e-9)
        entering_flux = incident_intensity_w_cm2*result.substrate_entry_fraction/photon_energy_j
        absorbed_fraction = (np.exp(-result.substrate_alpha_cm1*edges[:-1])
                             - np.exp(-result.substrate_alpha_cm1*edges[1:]))
        return entering_flux*absorbed_fraction/np.diff(edges)


def single_pass_escape_probability(n_inside: float, n_outside: float = 1.0, points: int = 4001) -> float:
    """Isotropic single-interface escape probability including Fresnel loss.

    The result is normalized to photons travelling in all directions. It omits
    repeated internal reflections, absorption, texture, and thin-film coatings.
    """
    if n_inside <= n_outside or n_outside <= 0:
        raise ValueError("Require n_inside > n_outside > 0")
    theta_c = np.arcsin(n_outside / n_inside)
    theta = np.linspace(0.0, theta_c, points)
    sin_i, cos_i = np.sin(theta), np.cos(theta)
    sin_t = n_inside / n_outside * sin_i
    cos_t = np.sqrt(np.maximum(1.0 - sin_t**2, 0.0))
    ts = 4 * n_inside * n_outside * cos_i * cos_t / (n_inside * cos_i + n_outside * cos_t) ** 2
    tp = 4 * n_inside * n_outside * cos_i * cos_t / (n_outside * cos_i + n_inside * cos_t) ** 2
    transmittance = 0.5 * (ts + tp)
    return float(0.5 * np.trapezoid(transmittance * np.sin(theta), theta))
