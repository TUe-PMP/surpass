"""Surface electrostatics, interface charge, and distributed SRH recombination.

The electrostatics uses Fermi--Dirac carrier populations and integrates
Poisson's equation analytically through Fermi integrals. The dielectric fixed
charge can be balanced by semiconductor space charge and amphoteric
interface-state charge.  The external-charge boundary is deliberately modular:
release 0.5 provides fixed dielectric/corona charge, while a later MOS boundary
can supply a voltage-dependent charge without changing the semiconductor or
trap model.

The default SRH kinetic expression uses the excess surface carrier product
relative to equilibrium at the same potential. This preserves detailed balance,
recovers ``np-ni^2`` in the Boltzmann limit, and avoids the spurious equilibrium
rate obtained by inserting Fermi--Dirac populations into the nondegenerate SRH
formula. A quasi-Fermi-splitting option is provided only as an explicit
sensitivity bound. Fully degenerate trap kinetics, including Pauli blocking,
remains a future refinement.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from numpy.polynomial.legendre import leggauss

from .bulk import BulkModel
from .constants import EPS0_F_CM, Q_C, thermal_voltage
from .fermi import fermi


ADAMOWICZ_INP = (
    "B. Adamowicz et al., Vacuum 63, 223-227 (2001), "
    "DOI 10.1016/S0042-207X(01)00195-6. The cited chemically polished "
    "InP(100) fit is a starting point, not a POx/InP measurement."
)

# Small-u quadrature avoids catastrophic cancellation of F_3/2 differences.
_GL_NODES, _GL_WEIGHTS = leggauss(8)
_SMALL_U_T = 0.5*(_GL_NODES+1.0)
_SMALL_U_WEIGHTS = 0.5*_GL_WEIGHTS*(1.0-_SMALL_U_T)


@lru_cache(maxsize=100_000)
def _inverse_fhalf_scalar(value_rounded: float) -> float:
    value = float(value_rounded)
    return brentq(lambda eta: fermi(0.5, eta)-value, -180.0, 400.0)


@dataclass(frozen=True)
class InterfaceDefectModel:
    dit_mid_ev1_cm2: float = 1e11
    sigma_n_cm2: float = 1e-14
    sigma_p_cm2: float = 1e-13
    vth_n_cm_s: float = 4.13e7
    vth_p_cm_s: float = 1.51e7
    shape: str = "constant"
    edge_factor: float = 30.0
    energy_points: int = 401
    source: str = ADAMOWICZ_INP

    def __post_init__(self):
        positive = (self.dit_mid_ev1_cm2, self.sigma_n_cm2,
                    self.sigma_p_cm2, self.vth_n_cm_s, self.vth_p_cm_s)
        if any(x <= 0 for x in positive):
            raise ValueError("Dit, capture cross sections, and thermal velocities must be positive")
        if self.shape not in ("constant", "u_shaped"):
            raise ValueError("shape must be 'constant' or 'u_shaped'")
        if self.edge_factor < 1 or self.energy_points < 51:
            raise ValueError("Use edge_factor >= 1 and at least 51 energy points")

    def distribution(self, energy_above_ev_ev: np.ndarray, bandgap_ev: float):
        energy = np.asarray(energy_above_ev_ev, dtype=float)
        if self.shape == "constant":
            return np.full_like(energy, self.dit_mid_ev1_cm2)
        coordinate = 2.0*(energy/bandgap_ev - 0.5)
        return self.dit_mid_ev1_cm2*(1.0 + (self.edge_factor-1.0)*coordinate**2)


@dataclass(frozen=True)
class InterfaceChargeModel:
    """Amphoteric donor/acceptor partition used to calculate ``Q_it``.

    States below the charge-neutrality level (CNL) are donor-like and are
    positive when empty; states above it are acceptor-like and are negative
    when occupied.  ``transition_width_ev=0`` gives a sharp partition, while a
    finite value blends the two populations smoothly.  The CNL is a model
    parameter to be constrained experimentally, not a universal InP constant.
    """

    charge_neutrality_level_fraction: float = 0.5
    transition_width_ev: float = 0.0
    source: str = (
        "Trap occupancy follows W. Shockley and W. T. Read, Phys. Rev. 87, "
        "835-842 (1952), DOI 10.1103/PhysRev.87.835. The amphoteric "
        "donor/acceptor charge convention follows E. H. Nicollian and "
        "J. R. Brews, MOS Physics and Technology (Wiley, 1982)."
    )

    def __post_init__(self):
        if not 0.0 <= self.charge_neutrality_level_fraction <= 1.0:
            raise ValueError("charge_neutrality_level_fraction must lie in [0, 1]")
        if self.transition_width_ev < 0:
            raise ValueError("transition_width_ev must be non-negative")

    def donor_fraction(self, energy_above_ev_ev, bandgap_ev):
        energy = np.asarray(energy_above_ev_ev, dtype=float)
        cnl = self.charge_neutrality_level_fraction*bandgap_ev
        if self.transition_width_ev == 0:
            return np.where(energy < cnl, 1.0,
                            np.where(energy > cnl, 0.0, 0.5))
        argument = np.clip((energy-cnl)/self.transition_width_ev, -700, 700)
        return 1.0/(1.0+np.exp(argument))


@dataclass(frozen=True)
class FixedChargeBoundary:
    """Potential-independent dielectric plus corona sheet charge."""

    charge_number_cm2: float = 0.0

    def charge_c_cm2(self, psi_surface_v: float) -> float:
        del psi_surface_v
        return Q_C*float(self.charge_number_cm2)


@dataclass(frozen=True)
class SurfaceState:
    delta_n_cm3: float
    psi_surface_v: float
    eta_n_bulk: float
    eta_p_bulk: float
    eta_n_surface: float
    eta_p_surface: float
    n_surface_cm3: float
    p_surface_cm3: float
    recombination_cm2_s: float
    effective_s_cm_s: float
    q_external_number_cm2: float
    q_it_number_cm2: float
    q_sc_number_cm2: float
    charge_balance_number_cm2: float
    mean_trap_occupancy: float
    c_it_f_cm2: float
    c_sc_f_cm2: float
    trap_control_fraction: float


class SurfaceSRHModel:
    """Calculate U_s(delta n) for one wafer, Qf, and interface-state model."""

    def __init__(self, bulk_model: BulkModel, qf_number_cm2: float | None,
                 defects: InterfaceDefectModel,
                 electrostatics: str = "self_consistent",
                 driving_force: str = "excess_product",
                 interface_charge: InterfaceChargeModel | None = None,
                 external_boundary=None):
        if electrostatics not in ("self_consistent", "equilibrium"):
            raise ValueError("electrostatics must be 'self_consistent' or 'equilibrium'")
        if driving_force not in ("excess_product", "qfl_splitting"):
            raise ValueError("driving_force must be 'excess_product' or 'qfl_splitting'")
        self.bulk_model = bulk_model
        self.wafer = bulk_model.wafer
        self.material = bulk_model.material
        if external_boundary is not None and qf_number_cm2 not in (None, 0, 0.0):
            raise ValueError("Pass either qf_number_cm2 or external_boundary, not both")
        if external_boundary is None:
            external_boundary = FixedChargeBoundary(0.0 if qf_number_cm2 is None else qf_number_cm2)
        if not callable(getattr(external_boundary, "charge_c_cm2", None)):
            raise TypeError("external_boundary must provide charge_c_cm2(psi_surface_v)")
        self.external_boundary = external_boundary
        self.qf_number_cm2 = float(getattr(external_boundary, "charge_number_cm2", np.nan))
        self.defects = defects
        self.interface_charge = interface_charge
        self.electrostatics = electrostatics
        self.driving_force = driving_force
        self.bulk = bulk_model.equilibrium()
        self.temperature_k = self.wafer.temperature_k
        self.vt = thermal_voltage(self.temperature_k)
        self.nc = self.material.nc_cm3(self.temperature_k)
        self.nv = self.material.nv_cm3(self.temperature_k)
        self.eg = self.bulk.eg_effective_ev
        self._energy = np.linspace(0.0, self.eg, self.defects.energy_points)
        intrinsic_level = 0.5*self.eg + 0.5*self.vt*np.log(self.nv/self.nc)
        self._ni_boltzmann = np.sqrt(self.nc*self.nv)*np.exp(-self.eg/(2*self.vt))
        self._n1 = self._ni_boltzmann*np.exp(
            np.clip((self._energy-intrinsic_level)/self.vt, -700, 700))
        self._p1 = self._ni_boltzmann*np.exp(
            np.clip((intrinsic_level-self._energy)/self.vt, -700, 700))
        self._dit = self.defects.distribution(self._energy, self.eg)
        self._donor_fraction = (None if interface_charge is None else
                                interface_charge.donor_fraction(self._energy, self.eg))
        self._equilibrium_u = self._solve_u(self.bulk.eta_n, self.bulk.eta_p)

    @staticmethod
    def _inverse_fhalf(value: float) -> float:
        if value <= 0:
            raise ValueError("Carrier population must be positive")
        # Scientific-string rounding preserves relative precision for minority
        # populations far below one while still allowing cache reuse.
        return _inverse_fhalf_scalar(float(f"{float(value):.12e}"))

    def _space_charge_c_cm2(self, u: float, eta_n: float, eta_p: float) -> float:
        if abs(u) < 0.01:
            # F(u)=u^2 integral_0^1 (1-t)*[dn/deta+dp/deta](t*u) dt.
            compressibility = (self.nc*fermi(-0.5, eta_n+_SMALL_U_T*u)
                               + self.nv*fermi(-0.5, eta_p-_SMALL_U_T*u))
            integral = u*u*float(np.dot(_SMALL_U_WEIGHTS, compressibility))
        else:
            integral = (
                self.nc*(fermi(1.5, eta_n+u)-fermi(1.5, eta_n)
                         - u*fermi(0.5, eta_n))
                + self.nv*(fermi(1.5, eta_p-u)-fermi(1.5, eta_p)
                           + u*fermi(0.5, eta_p))
            )
        integral = max(float(integral), 0.0)
        if u == 0:
            return 0.0
        eps = self.material.eps_r*EPS0_F_CM
        return -np.sign(u)*np.sqrt(2*eps*Q_C*self.vt*integral)

    def _trap_occupancy(self, u: float, eta_n: float, eta_p: float):
        ns = self.nc*fermi(0.5, eta_n+u)
        ps = self.nv*fermi(0.5, eta_p-u)
        cn = self.defects.vth_n_cm_s*self.defects.sigma_n_cm2
        cp = self.defects.vth_p_cm_s*self.defects.sigma_p_cm2
        denominator = cn*(ns+self._n1) + cp*(ps+self._p1)
        return np.clip((cn*ns+cp*self._p1)/denominator, 0.0, 1.0)

    def _interface_charge_c_cm2(self, u: float, eta_n: float, eta_p: float) -> float:
        if self.interface_charge is None:
            return 0.0
        occupancy = self._trap_occupancy(u, eta_n, eta_p)
        donor = self._donor_fraction
        signed_density = self._dit*(donor*(1.0-occupancy) - (1.0-donor)*occupancy)
        return Q_C*float(np.trapezoid(signed_density, self._energy))

    def _charge_components(self, u: float, eta_n: float, eta_p: float):
        psi = u*self.vt
        qext = float(self.external_boundary.charge_c_cm2(psi))
        qit = self._interface_charge_c_cm2(u, eta_n, eta_p)
        qsc = self._space_charge_c_cm2(u, eta_n, eta_p)
        return qext, qit, qsc

    def _solve_u(self, eta_n: float, eta_p: float) -> float:
        function = lambda u: sum(self._charge_components(u, eta_n, eta_p))
        at_zero = function(0.0)
        if abs(at_zero) < Q_C*1e-6:
            return 0.0
        low, high = -8.0, 8.0
        while function(low)*function(high) > 0 and high < 512:
            low *= 2.0
            high *= 2.0
        if function(low)*function(high) > 0:
            raise RuntimeError("Could not bracket the surface charge-neutrality solution")
        return float(brentq(function, low, high, xtol=1e-12, rtol=1e-12))

    def _capacitance_diagnostics(self, u: float, eta_n: float, eta_p: float):
        du = 1e-4
        qit_plus = self._interface_charge_c_cm2(u+du, eta_n, eta_p)
        qit_minus = self._interface_charge_c_cm2(u-du, eta_n, eta_p)
        qsc_plus = self._space_charge_c_cm2(u+du, eta_n, eta_p)
        qsc_minus = self._space_charge_c_cm2(u-du, eta_n, eta_p)
        cit = max(-(qit_plus-qit_minus)/(2*du*self.vt), 0.0)
        csc = max(-(qsc_plus-qsc_minus)/(2*du*self.vt), 0.0)
        total = cit+csc
        return cit, csc, 0.0 if total == 0 else cit/total

    def state(self, delta_n_cm3: float) -> SurfaceState:
        delta = float(delta_n_cm3)
        if delta < 0:
            raise ValueError("delta_n_cm3 must be non-negative")
        if delta == 0:
            eta_n, eta_p = self.bulk.eta_n, self.bulk.eta_p
        else:
            eta_n = self._inverse_fhalf((self.bulk.n0_cm3+delta)/self.nc)
            eta_p = self._inverse_fhalf((self.bulk.p0_cm3+delta)/self.nv)
        u = self._equilibrium_u if self.electrostatics == "equilibrium" else self._solve_u(eta_n, eta_p)
        eta_ns, eta_ps = eta_n+u, eta_p-u
        ns = self.nc*fermi(0.5, eta_ns)
        ps = self.nv*fermi(0.5, eta_ps)
        qext, qit, qsc = self._charge_components(u, eta_n, eta_p)
        occupancy = self._trap_occupancy(u, eta_n, eta_p)
        mean_occupancy = float(np.trapezoid(self._dit*occupancy, self._energy)
                               / np.trapezoid(self._dit, self._energy))
        cit, csc, trap_control = self._capacitance_diagnostics(u, eta_n, eta_p)

        if delta == 0:
            rate = 0.0
        else:
            denominator = (
                (ns+self._n1)/(self.defects.vth_p_cm_s*self.defects.sigma_p_cm2*self._dit)
                + (ps+self._p1)/(self.defects.vth_n_cm_s*self.defects.sigma_n_cm2*self._dit)
            )
            if self.driving_force == "excess_product":
                ns_equilibrium = self.nc*fermi(0.5, self.bulk.eta_n+u)
                ps_equilibrium = self.nv*fermi(0.5, self.bulk.eta_p-u)
                drive_product = max(ns*ps-ns_equilibrium*ps_equilibrium, 0.0)
            else:
                qfl_drive = np.expm1(np.clip(self.eg/self.vt + eta_n + eta_p, -700, 700))
                drive_product = self._ni_boltzmann**2*qfl_drive
            rate = max(float(np.trapezoid(drive_product/denominator, self._energy)), 0.0)
        return SurfaceState(
            delta, u*self.vt, eta_n, eta_p, eta_ns, eta_ps, ns, ps,
            rate, 0.0 if delta == 0 else rate/delta,
            qext/Q_C, qit/Q_C, qsc/Q_C, (qext+qit+qsc)/Q_C,
            mean_occupancy, cit, csc, trap_control,
        )


class SurfaceBoundaryTable:
    """Fast log interpolation of a non-linear surface recombination boundary."""

    def __init__(self, delta_cm3, rate_cm2_s, psi_surface_v=None,
                 n_surface_cm3=None, p_surface_cm3=None, label=""):
        delta = np.asarray(delta_cm3, dtype=float)
        rate = np.asarray(rate_cm2_s, dtype=float)
        if (delta.ndim != 1 or delta.size < 4 or rate.shape != delta.shape
                or np.any(delta <= 0) or np.any(np.diff(delta) <= 0)
                or np.any(rate <= 0)):
            raise ValueError("Use increasing positive delta and positive rate arrays")
        self.delta_cm3 = delta
        self.delta_min_cm3 = float(delta[0])
        self.delta_max_cm3 = float(delta[-1])
        self.label = label
        self._rate = PchipInterpolator(np.log(delta), np.log(rate), extrapolate=True)
        self.psi_surface_v = None if psi_surface_v is None else np.asarray(psi_surface_v)
        self.n_surface_cm3 = None if n_surface_cm3 is None else np.asarray(n_surface_cm3)
        self.p_surface_cm3 = None if p_surface_cm3 is None else np.asarray(p_surface_cm3)

    @classmethod
    def from_surface_model(cls, model: SurfaceSRHModel, delta_min_cm3=1.0,
                           delta_max_cm3=1e21, points=181, label=""):
        delta = np.logspace(np.log10(delta_min_cm3), np.log10(delta_max_cm3), points)
        states = [model.state(value) for value in delta]
        return cls(delta, [s.recombination_cm2_s for s in states],
                   [s.psi_surface_v for s in states],
                   [s.n_surface_cm3 for s in states],
                   [s.p_surface_cm3 for s in states], label=label)

    def rate(self, delta_n_cm3):
        delta = np.asarray(delta_n_cm3, dtype=float)
        if delta.ndim == 0 and delta <= 0:
            return 0.0
        safe = np.maximum(delta, self.delta_min_cm3*1e-12)
        log_rate = np.clip(self._rate(np.log(safe)), -700, 700)
        value = np.where(delta <= 0, 0.0, np.exp(log_rate))
        return float(value) if value.ndim == 0 else value

    def effective_s_cm_s(self, delta_n_cm3):
        delta = np.asarray(delta_n_cm3, dtype=float)
        value = np.divide(self.rate(delta), delta, out=np.zeros_like(delta), where=delta > 0)
        return float(value) if value.ndim == 0 else value
