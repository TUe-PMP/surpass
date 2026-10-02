"""Bulk equilibrium, transport, recombination, diffusion, and screening."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .constants import EPS0_F_CM, K_B_J_K, Q_C, thermal_voltage
from .fermi import fermi
from .models import caughey_thomas_mobility, palankovski_point_charge_bgn_ev
from .wafer import Wafer


@dataclass(frozen=True)
class BulkState:
    eg_nominal_ev: float
    delta_eg_bgn_ev: float
    eg_effective_ev: float
    eta_n: float
    eta_p: float
    n0_cm3: float
    p0_cm3: float
    ni_effective_cm3: float
    mu_n_cm2_vs: float
    mu_p_cm2_vs: float
    resistivity_ohm_cm: float
    debye_length_nm: float
    fermi_screening_length_nm: float


@dataclass(frozen=True)
class InjectionState:
    delta_n_cm3: float
    n_cm3: float
    p_cm3: float
    eta_n: float
    eta_p: float
    d_n_cm2_s: float
    d_p_cm2_s: float
    d_amb_cm2_s: float
    tau_rad_s: float
    tau_auger_s: float
    tau_srh_s: float
    tau_total_s: float
    diffusion_length_um: float
    delta_r_rad_cm3_s: float
    delta_r_auger_cm3_s: float
    delta_r_srh_cm3_s: float


class BulkModel:
    """Calculate properties for one uniformly doped wafer.

    Parameters
    ----------
    wafer:
        Material, donor/acceptor densities, thickness, and temperature.
    enable_bgn:
        Enable Palankovski screened-Coulomb point-charge BGN. Disabled by
        default because a dopant-specific model may be preferable.
    tau_srh_s:
        Assumed bulk SRH lifetime. ``np.inf`` disables bulk SRH recombination.
    """

    def __init__(self, wafer: Wafer, enable_bgn: bool = False, tau_srh_s=np.inf):
        self.wafer = wafer
        self.material = wafer.material
        self.enable_bgn = bool(enable_bgn)
        self.tau_srh_s = float(tau_srh_s)
        if self.tau_srh_s <= 0:
            raise ValueError("tau_srh_s must be positive")
        self._bulk_state: BulkState | None = None

    def _charge_neutral_state(self, eg_ev: float):
        t = self.wafer.temperature_k
        vt = thermal_voltage(t)
        nc, nv = self.material.nc_cm3(t), self.material.nv_cm3(t)
        net = self.wafer.net_doping_cm3

        def neutrality(eta_n):
            eta_p = -eg_ev / vt - eta_n
            return nc * fermi(0.5, eta_n) - nv * fermi(0.5, eta_p) - net

        eta_n = brentq(neutrality, -180.0, 180.0)
        eta_p = -eg_ev / vt - eta_n
        n0 = nc * fermi(0.5, eta_n)
        p0 = nv * fermi(0.5, eta_p)
        return eta_n, eta_p, n0, p0

    def equilibrium(self) -> BulkState:
        if self._bulk_state is not None:
            return self._bulk_state

        t = self.wafer.temperature_k
        m = self.material
        eg0 = m.bandgap_ev(t)
        eta_n, eta_p, n0, p0 = self._charge_neutral_state(eg0)

        delta_eg = 0.0
        if self.enable_bgn:
            if self.wafer.net_doping_cm3 >= 0:
                density, eta_majority = n0, eta_n
            else:
                density, eta_majority = p0, eta_p
            delta_eg = palankovski_point_charge_bgn_ev(
                density, eta_majority, m.eps_r, t
            )
            # One update is normally adequate; repeat neutrality with narrowed gap.
            eta_n, eta_p, n0, p0 = self._charge_neutral_state(eg0 - delta_eg)

        eg = eg0 - delta_eg
        vt = thermal_voltage(t)
        nc, nv = m.nc_cm3(t), m.nv_cm3(t)
        ni = np.sqrt(nc * nv) * np.exp(-eg / (2.0 * vt))

        impurity = self.wafer.total_ionized_impurity_cm3
        mu_n = caughey_thomas_mobility(impurity, t, m.electron_mobility)
        mu_p = caughey_thomas_mobility(impurity, t, m.hole_mobility)
        conductivity = Q_C * (mu_n * n0 + mu_p * p0)
        resistivity = np.inf if conductivity == 0 else 1.0 / conductivity

        eps_f_cm = m.eps_r * EPS0_F_CM
        debye_cm = np.sqrt(
            eps_f_cm * K_B_J_K * t / (Q_C**2 * max(n0 + p0, 1e-300))
        )
        compressibility = nc * fermi(-0.5, eta_n) + nv * fermi(-0.5, eta_p)
        fermi_screen_cm = np.sqrt(
            eps_f_cm * K_B_J_K * t / (Q_C**2 * compressibility)
        )

        self._bulk_state = BulkState(
            eg_nominal_ev=eg0,
            delta_eg_bgn_ev=delta_eg,
            eg_effective_ev=eg,
            eta_n=eta_n,
            eta_p=eta_p,
            n0_cm3=n0,
            p0_cm3=p0,
            ni_effective_cm3=ni,
            mu_n_cm2_vs=mu_n,
            mu_p_cm2_vs=mu_p,
            resistivity_ohm_cm=resistivity,
            debye_length_nm=debye_cm * 1e7,
            fermi_screening_length_nm=fermi_screen_cm * 1e7,
        )
        return self._bulk_state

    def injection(self, delta_n_cm3: float) -> InjectionState:
        """Return bulk properties at a uniform electron-hole injection level."""
        delta = float(delta_n_cm3)
        if delta <= 0:
            raise ValueError("Use a positive delta_n; lifetimes are excess density / excess recombination rate")

        s = self.equilibrium()
        m, t = self.material, self.wafer.temperature_k
        nc, nv = m.nc_cm3(t), m.nv_cm3(t)
        n, p = s.n0_cm3 + delta, s.p0_cm3 + delta

        eta_n = brentq(lambda x: nc * fermi(0.5, x) - n, -180.0, 300.0)
        eta_p = brentq(lambda x: nv * fermi(0.5, x) - p, -180.0, 300.0)

        vt = thermal_voltage(t)
        # Generalized Einstein relation. It approaches D/mu = kT/q when
        # F_{1/2}/F_{-1/2} -> 1 in the nondegenerate limit.
        d_n = s.mu_n_cm2_vs * vt * fermi(0.5, eta_n) / fermi(-0.5, eta_n)
        d_p = s.mu_p_cm2_vs * vt * fermi(0.5, eta_p) / fermi(-0.5, eta_p)
        denominator = s.mu_n_cm2_vs * n + s.mu_p_cm2_vs * p
        d_amb = (
            d_n * s.mu_p_cm2_vs * p + d_p * s.mu_n_cm2_vs * n
        ) / denominator

        equilibrium_product = s.n0_cm3 * s.p0_cm3
        # Algebraically identical to n*p-n0*p0, but cancellation-free when
        # delta is much smaller than either equilibrium population.
        excess_product = delta * (s.n0_cm3 + s.p0_cm3 + delta)
        delta_r_rad = m.b_rad_cm3_s * excess_product

        equilibrium_auger_factor = (
            m.c_n_cm6_s * s.n0_cm3 + m.c_p_cm6_s * s.p0_cm3
        )
        # Subtract the equilibrium phenomenological Auger rate explicitly.
        delta_r_auger = (equilibrium_auger_factor * excess_product
                        + (m.c_n_cm6_s + m.c_p_cm6_s) * delta
                        * (equilibrium_product + excess_product))
        delta_r_srh = 0.0 if np.isinf(self.tau_srh_s) else delta / self.tau_srh_s

        tau_rad = np.inf if delta_r_rad == 0 else delta / delta_r_rad
        tau_auger = np.inf if delta_r_auger == 0 else delta / delta_r_auger
        tau_srh = self.tau_srh_s
        total_rate = delta_r_rad + delta_r_auger + delta_r_srh
        tau_total = np.inf if total_rate == 0 else delta / total_rate
        diffusion_length_um = np.sqrt(d_amb * tau_total) * 1e4

        return InjectionState(
            delta_n_cm3=delta,
            n_cm3=n,
            p_cm3=p,
            eta_n=eta_n,
            eta_p=eta_p,
            d_n_cm2_s=d_n,
            d_p_cm2_s=d_p,
            d_amb_cm2_s=d_amb,
            tau_rad_s=tau_rad,
            tau_auger_s=tau_auger,
            tau_srh_s=tau_srh,
            tau_total_s=tau_total,
            diffusion_length_um=diffusion_length_um,
            delta_r_rad_cm3_s=delta_r_rad,
            delta_r_auger_cm3_s=delta_r_auger,
            delta_r_srh_cm3_s=delta_r_srh,
        )
