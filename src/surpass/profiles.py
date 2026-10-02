"""Reconstruct a one-sided semiconductor band diagram from Poisson's first integral.

Depth is positive into a uniform, semi-infinite semiconductor. The bulk valence
edge is the energy zero; these are relative energies, not vacuum-referenced band
offsets. Quasi-Fermi levels are flat throughout this local space-charge profile.
No drift-diffusion solution, finite-wafer Poisson boundary, quantum confinement,
or spatially varying BGN is implied. See Nicollian & Brews, MOS Physics and
Technology (Wiley, 1982), for the one-dimensional Poisson first-integral method.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid

from .constants import EPS0_F_CM, Q_C
from .fermi import fermi


@dataclass(frozen=True)
class BandProfile:
    depth_cm: np.ndarray
    psi_v: np.ndarray
    electric_field_v_cm: np.ndarray
    ec_ev: np.ndarray
    ev_ev: np.ndarray
    efn_ev: np.ndarray
    efp_ev: np.ndarray
    n_cm3: np.ndarray
    p_cm3: np.ndarray
    rho_c_cm3: np.ndarray
    remaining_q_sc_c_cm2: np.ndarray
    screening_length_cm: float
    charge_90_depth_cm: float
    charge_99_depth_cm: float
    potential_99_depth_cm: float
    integrated_q_sc_c_cm2: float
    charge_integral_relative_error: float
    delta_n_cm3: float
    semi_infinite_warning: bool


def reconstruct_band_profile(model, delta_n_cm3=0.0, points=1201,
                             tail_fraction=1e-6):
    """Return the band profile associated with ``model.state(delta_n_cm3)``.

    Reparameterize the first integral by s=log(|u_s|/|u|):
    dx/ds = epsilon*V_T*|u|/|Q_sc(u)|. This avoids the logarithmic divergence
    of x as u tends to zero. The calculation ends at the explicitly specified
    potential fraction, not at an artificial finite-width depletion boundary.

    Widths denote 90/99% of the *net signed space charge*, and 99% of the
    potential relaxation. They are not sharp depletion widths. In particular,
    inversion charge and ionized-dopant charge can occupy different depths.
    Frozen-equilibrium electrostatics is rejected under injection: it does not
    solve the injected Poisson boundary and should not masquerade as a profile.
    """
    if not isinstance(points, (int, np.integer)) or points < 101:
        raise ValueError("Use at least 101 profile points")
    if not np.isfinite(tail_fraction) or not 0 < tail_fraction < 0.01:
        raise ValueError("tail_fraction must lie between zero and 0.01")
    if model.electrostatics == "equilibrium" and delta_n_cm3 != 0:
        raise ValueError("Injected profiles require self_consistent electrostatics")
    state = model.state(delta_n_cm3)
    eps = model.material.eps_r*EPS0_F_CM
    en, ep = state.eta_n_bulk, state.eta_p_bulk
    nb, pb = model.nc*fermi(0.5, en), model.nv*fermi(0.5, ep)
    compressibility = model.nc*fermi(-0.5, en)+model.nv*fermi(-0.5, ep)
    screening = np.sqrt(eps*model.vt/(Q_C*compressibility))
    us = state.psi_surface_v/model.vt
    if us == 0:
        # A flat-band profile has no space charge; zero width is meaningful.
        depth = np.linspace(0, 10*screening, points)
        u = np.zeros(points)
        qsc = np.zeros(points)
        charge90 = charge99 = potential99 = integral_error = 0.0
    else:
        # Resolve the thin accumulation/inversion sheet near s=0 as well as
        # the long screening tail. A uniform log-potential mesh under-resolves
        # the surface when |u_s| is many thermal voltages.
        s = np.linspace(0, np.sqrt(-np.log(tail_fraction)), points)**2
        u = us*np.exp(-s)
        qsc = np.array([model._space_charge_c_cm2(v, en, ep) for v in u])
        dx_ds = eps*model.vt*np.abs(u)/np.abs(qsc)
        if np.any(~np.isfinite(dx_ds)) or np.any(dx_ds <= 0):
            raise RuntimeError("Invalid Poisson first integral during reconstruction")
        depth = cumulative_trapezoid(dx_ds, s, initial=0.0)
        remaining = np.abs(qsc/qsc[0])
        charge90 = float(np.interp(0.1, remaining[::-1], depth[::-1]))
        charge99 = float(np.interp(0.01, remaining[::-1], depth[::-1]))
        potential99 = float(np.interp(np.log(100), s, depth))
    psi = u*model.vt
    n = model.nc*fermi(0.5, en+u)
    p = model.nv*fermi(0.5, ep-u)
    # Subtract bulk populations before summing to preserve near-neutral tails.
    rho = Q_C*((p-pb)-(n-nb))
    integrated = float(np.trapezoid(rho, depth))
    if us != 0:
        expected = qsc[0]-qsc[-1]
        integral_error = (integrated-expected)/abs(expected)
    return BandProfile(
        depth_cm=depth, psi_v=psi, electric_field_v_cm=-qsc/eps,
        ec_ev=model.eg-psi, ev_ev=-psi,
        efn_ev=np.full(points, model.eg+model.vt*en),
        efp_ev=np.full(points, -model.vt*ep), n_cm3=n, p_cm3=p,
        rho_c_cm3=rho, remaining_q_sc_c_cm2=qsc,
        screening_length_cm=float(screening), charge_90_depth_cm=charge90,
        charge_99_depth_cm=charge99, potential_99_depth_cm=potential99,
        integrated_q_sc_c_cm2=integrated,
        charge_integral_relative_error=float(integral_error),
        delta_n_cm3=float(delta_n_cm3),
        semi_infinite_warning=max(potential99, charge99) > 0.1*model.wafer.thickness_um*1e-4,
    )


@dataclass(frozen=True)
class TrapRelaxationSpectrum:
    energy_above_ev_ev: np.ndarray
    dit_ev1_cm2: np.ndarray
    occupancy: np.ndarray
    electron_capture_s1: float
    hole_capture_s1: float
    electron_emission_s1: np.ndarray
    hole_emission_s1: np.ndarray
    relaxation_time_s: np.ndarray

    def slow_trap_fraction(self, transient_time_s):
        """Dit-weighted fraction with tau_trap exceeding a chosen timescale.

        This counts states, not their importance to recombination or charge.
        The estimate assumes frozen carrier populations and electrostatic
        potential; feedback can change the collective response time.
        """
        if not np.isfinite(transient_time_s) or transient_time_s <= 0:
            raise ValueError("transient_time_s must be finite and positive")
        slow = (self.relaxation_time_s > transient_time_s).astype(float)
        return float(np.trapezoid(self.dit_ev1_cm2*slow, self.energy_above_ev_ev)
                     / np.trapezoid(self.dit_ev1_cm2, self.energy_above_ev_ev))


def trap_relaxation_spectrum(model, delta_n_cm3=0.0):
    """Linear occupancy relaxation for fixed local carrier populations.

    df/dt=(c_n*n+e_p)*(1-f)-(c_p*p+e_n)*f, hence
    tau^-1=c_n*(n+n1)+c_p*(p+p1). Emission rates are the same classical SRH
    rates used in release 0.5; no new fitted parameters are introduced.
    Reference: Shockley & Read, Phys. Rev. 87, 835 (1952),
    DOI 10.1103/PhysRev.87.835. Fully degenerate kinetics remains unimplemented.
    """
    state = model.state(delta_n_cm3)
    cn = model.defects.vth_n_cm_s*model.defects.sigma_n_cm2
    cp = model.defects.vth_p_cm_s*model.defects.sigma_p_cm2
    capture_n, capture_p = cn*state.n_surface_cm3, cp*state.p_surface_cm3
    emission_n, emission_p = cn*model._n1, cp*model._p1
    inverse_time = capture_n+capture_p+emission_n+emission_p
    return TrapRelaxationSpectrum(
        energy_above_ev_ev=model._energy.copy(), dit_ev1_cm2=model._dit.copy(),
        occupancy=(capture_n+emission_p)/inverse_time,
        electron_capture_s1=float(capture_n), hole_capture_s1=float(capture_p),
        electron_emission_s1=emission_n.copy(), hole_emission_s1=emission_p.copy(),
        relaxation_time_s=1/inverse_time,
    )
