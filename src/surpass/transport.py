"""Steady-state one-dimensional excess-carrier transport.

The solver uses a cell-centred finite-volume discretization of

    dJ/dx = G(x) - R(Delta n),       J = -D_a(Delta n) d(Delta n)/dx,

with Robin surface-recombination boundary conditions at both wafer surfaces.
The local ambipolar diffusion coefficient and bulk recombination channels are
tabulated from :class:`surpass.bulk.BulkModel`, so degeneracy and nonlinear
injection dependence are retained without repeatedly solving Fermi integrals
inside the nonlinear iteration.

This release assumes local charge neutrality (Delta n = Delta p), a uniform
temperature and doping, no electric drift field in the quasi-neutral bulk, and
optional fixed local instantaneous photon recycling. Nonlinear surface-SRH
boundaries provide injection-dependent interface rates and electrostatics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq, least_squares, root
from scipy.sparse import diags

from .bulk import BulkModel


class BulkTransportTable:
    """Log-interpolated local transport and recombination coefficients."""

    CHANNELS = ("d_amb", "r_rad", "r_auger", "r_srh", "r_total")

    def __init__(self, delta_cm3, d_amb_cm2_s, r_rad_cm3_s,
                 r_auger_cm3_s, r_srh_cm3_s):
        delta = np.asarray(delta_cm3, dtype=float)
        if delta.ndim != 1 or delta.size < 4 or np.any(delta <= 0) or np.any(np.diff(delta) <= 0):
            raise ValueError("delta_cm3 must be a strictly increasing positive 1D grid")
        arrays = [np.asarray(x, dtype=float) for x in
                  (d_amb_cm2_s, r_rad_cm3_s, r_auger_cm3_s, r_srh_cm3_s)]
        if any(x.shape != delta.shape for x in arrays):
            raise ValueError("All coefficient arrays must match delta_cm3")
        if np.any(arrays[0] <= 0) or any(np.any(x < 0) for x in arrays[1:]):
            raise ValueError("Diffusion must be positive and recombination rates non-negative")

        total = arrays[1] + arrays[2] + arrays[3]
        tiny = np.finfo(float).tiny
        self.delta_cm3 = delta
        self.delta_min_cm3 = float(delta[0])
        self.delta_max_cm3 = float(delta[-1])
        self._interpolators = {}
        for name, values in zip(self.CHANNELS, [arrays[0], *arrays[1:], total]):
            self._interpolators[name] = PchipInterpolator(
                np.log(delta), np.log(np.maximum(values, tiny)), extrapolate=True
            )

    @classmethod
    def from_bulk_model(cls, model: BulkModel, delta_min_cm3=1.0,
                        delta_max_cm3=1e21, points=241):
        grid = np.logspace(np.log10(delta_min_cm3), np.log10(delta_max_cm3), points)
        states = [model.injection(value) for value in grid]
        return cls(
            grid,
            [s.d_amb_cm2_s for s in states],
            [s.delta_r_rad_cm3_s for s in states],
            [s.delta_r_auger_cm3_s for s in states],
            [s.delta_r_srh_cm3_s for s in states],
        )

    @classmethod
    def linear_lifetime(cls, diffusion_cm2_s: float, lifetime_s: float,
                        delta_min_cm3=1e2, delta_max_cm3=1e22, points=81):
        """Exact linear test model: R=Delta n/tau, assigned to SRH."""
        if diffusion_cm2_s <= 0 or lifetime_s <= 0:
            raise ValueError("diffusion_cm2_s and lifetime_s must be positive")
        grid = np.logspace(np.log10(delta_min_cm3), np.log10(delta_max_cm3), points)
        zeros = np.zeros_like(grid)
        return cls(grid, np.full_like(grid, diffusion_cm2_s), zeros, zeros, grid/lifetime_s)

    def evaluate(self, delta_cm3):
        delta = np.asarray(delta_cm3, dtype=float)
        # Low-injection recombination is asymptotically linear and D approaches
        # a constant, so log-log extrapolation below the first tabulated point is
        # preferable to imposing an artificial carrier-density floor. High-side
        # extrapolation is not allowed because Auger behavior can become steep.
        evaluated = np.minimum(np.maximum(delta, 1e-300), self.delta_max_cm3)
        log_delta = np.log(evaluated)
        return {name: np.exp(interp(log_delta)) for name, interp in self._interpolators.items()}


@dataclass(frozen=True)
class TransportResult:
    depth_cm: np.ndarray
    cell_width_cm: np.ndarray
    delta_n_cm3: np.ndarray
    generation_cm3_s: np.ndarray
    d_amb_cm2_s: np.ndarray
    r_rad_cm3_s: np.ndarray
    r_auger_cm3_s: np.ndarray
    r_srh_cm3_s: np.ndarray
    front_surface_delta_cm3: float
    rear_surface_delta_cm3: float
    front_surface_recombination_cm2_s: float
    rear_surface_recombination_cm2_s: float
    bulk_radiative_recombination_cm2_s: float
    bulk_auger_recombination_cm2_s: float
    bulk_srh_recombination_cm2_s: float
    generated_flux_cm2_s: float
    effective_lifetime_s: float
    relative_balance_error: float
    maximum_scaled_cell_residual: float
    converged: bool
    nonlinear_evaluations: int
    table_clipped_fraction: float
    internal_radiative_emission_cm2_s: float
    recycled_generation_cm2_s: float
    recycling_probability: np.ndarray

    @property
    def bulk_recombination_cm2_s(self) -> float:
        return (self.bulk_radiative_recombination_cm2_s
                + self.bulk_auger_recombination_cm2_s
                + self.bulk_srh_recombination_cm2_s)

    @property
    def total_recombination_cm2_s(self) -> float:
        return (self.bulk_recombination_cm2_s
                + self.front_surface_recombination_cm2_s
                + self.rear_surface_recombination_cm2_s)

    @property
    def average_delta_n_cm3(self) -> float:
        return float(np.sum(self.delta_n_cm3*self.cell_width_cm)/np.sum(self.cell_width_cm))

    @property
    def internal_radiative_yield(self) -> float:
        """Emission events per externally generated pair; can exceed one with recycling."""
        return self.internal_radiative_emission_cm2_s / self.generated_flux_cm2_s

    @property
    def front_effective_s_cm_s(self) -> float:
        return (0.0 if self.front_surface_delta_cm3 == 0 else
                self.front_surface_recombination_cm2_s/self.front_surface_delta_cm3)

    @property
    def rear_effective_s_cm_s(self) -> float:
        return (0.0 if self.rear_surface_delta_cm3 == 0 else
                self.rear_surface_recombination_cm2_s/self.rear_surface_delta_cm3)


class SteadyState1DSolver:
    """Nonlinear finite-volume solver for one uniformly doped wafer."""

    def __init__(self, table: BulkTransportTable):
        self.table = table

    @staticmethod
    def _boundary_rate(boundary, delta):
        if np.isscalar(boundary):
            return float(boundary)*delta
        if not hasattr(boundary, "rate"):
            raise TypeError("A surface boundary must be a non-negative velocity or expose rate(delta)")
        return boundary.rate(delta)

    @classmethod
    def _surface_state(cls, cell_delta, diffusion, boundary, half_dx):
        if np.isscalar(boundary):
            surface_velocity = float(boundary)
            if surface_velocity == 0:
                return float(cell_delta), 0.0
            surface_delta = cell_delta/(1.0 + surface_velocity*half_dx/diffusion)
            return float(surface_delta), float(surface_velocity*surface_delta)

        # Match diffusion from the first/last cell centre to the nonlinear
        # interface loss: D*(delta_cell-delta_surface)/(dx/2)=U_s(delta_surface).
        function = lambda surface_delta: (
            diffusion*(cell_delta-surface_delta)/half_dx
            - cls._boundary_rate(boundary, surface_delta)
        )
        surface_delta = brentq(function, 0.0, float(cell_delta), xtol=max(1e-12*cell_delta, 1e-6))
        return float(surface_delta), float(cls._boundary_rate(boundary, surface_delta))

    def solve(self, thickness_cm: float, generation, front_s_cm_s,
              rear_s_cm_s, cells: int = 240, initial_delta_cm3=None,
              rtol: float = 1e-8, max_nfev: int = 500,
              cell_edges_cm=None, recycling_probability=0.0) -> TransportResult:
        """Optional fixed local recycling; use identical probabilities for CW switch-off.

        Radiative result fields are net carrier losses, while
        internal_radiative_emission_cm2_s reports intrinsic photon production.
        """
        if thickness_cm <= 0 or cells < 8:
            raise ValueError("Use positive thickness_cm and at least 8 cells")
        for boundary in (front_s_cm_s, rear_s_cm_s):
            if np.isscalar(boundary) and float(boundary) < 0:
                raise ValueError("Surface recombination velocities must be non-negative")
            if not np.isscalar(boundary) and not hasattr(boundary, "rate"):
                raise TypeError("Surface boundaries must be velocities or expose rate(delta)")

        if cell_edges_cm is None:
            edges = np.linspace(0.0, thickness_cm, cells+1)
        else:
            edges = np.asarray(cell_edges_cm, dtype=float)
            if (edges.ndim != 1 or edges.size < 9 or abs(edges[0]) > 1e-15
                    or abs(edges[-1]-thickness_cm) > max(1e-15, thickness_cm*1e-12)
                    or np.any(np.diff(edges) <= 0)):
                raise ValueError("cell_edges_cm must increase from 0 to thickness_cm")
            cells = edges.size-1
        widths = np.diff(edges)
        x = 0.5*(edges[:-1] + edges[1:])
        recycling = np.asarray(recycling_probability, dtype=float)
        if recycling.shape == (): recycling = np.full(cells, float(recycling))
        if (recycling.shape != (cells,) or np.any(~np.isfinite(recycling))
                or np.any(recycling < 0) or np.any(recycling > 1)):
            raise ValueError('Recycling must be a scalar or matching cell vector in [0,1]')

        def properties(delta):
            values = self.table.evaluate(delta)
            values['r_internal'] = values['r_rad'].copy()
            values['r_rad'] = (1-recycling)*values['r_rad']
            values['r_total'] = values['r_rad']+values['r_auger']+values['r_srh']
            return values

        g = np.asarray(generation(x) if callable(generation) else generation, dtype=float)
        if g.shape == ():
            g = np.full(cells, float(g))
        if g.shape != (cells,) or np.any(g < 0) or not np.all(np.isfinite(g)):
            raise ValueError("generation must give one finite non-negative value per cell")
        generated = float(np.sum(g*widths))
        if generated <= 0:
            raise ValueError("A positive generation rate is required for steady-state excess carriers")

        if initial_delta_cm3 is None:
            candidates = self.table.delta_cm3
            values = self.table.evaluate(candidates)
            uniform_loss = ((values['r_rad']*(1-np.dot(recycling, widths)/thickness_cm)
                            + values['r_auger']+values['r_srh'])*thickness_cm
                            + self._boundary_rate(front_s_cm_s, candidates)
                            + self._boundary_rate(rear_s_cm_s, candidates))
            initial = float(candidates[np.argmin(abs(np.log(np.maximum(uniform_loss, 1e-300)/generated)))])
        else:
            initial = float(initial_delta_cm3)
        initial = np.clip(initial, self.table.delta_min_cm3, self.table.delta_max_cm3)

        scale = max(generated/cells, 1.0)

        delta_scale = initial

        def residual(scaled_delta):
            delta = delta_scale*scaled_delta
            values = properties(delta)
            d = values["d_amb"]
            transport_resistance = widths[:-1]/(2*d[:-1]) + widths[1:]/(2*d[1:])
            internal_flux = -np.diff(delta)/transport_resistance
            _, front_rate = self._surface_state(delta[0], d[0], front_s_cm_s, widths[0]/2)
            _, rear_rate = self._surface_state(delta[-1], d[-1], rear_s_cm_s, widths[-1]/2)
            left_flux = np.r_[-front_rate, internal_flux]
            right_flux = np.r_[internal_flux, rear_rate]
            return (right_flux - left_flux - widths*(g-values["r_total"]))/scale

        sparsity = diags([np.ones(cells-1), np.ones(cells), np.ones(cells-1)], [-1, 0, 1], format="csr")
        lower = max(self.table.delta_min_cm3*1e-30, 1e-300)/delta_scale
        upper = self.table.delta_max_cm3/delta_scale
        # Powell's hybrid root solve is very effective for this banded balance
        # equation. Retain a bounded least-squares fallback for unusually stiff
        # cases that attempt to leave the tabulated injection interval.
        fit = root(residual, np.ones(cells), method="hybr",
                   options={"xtol": rtol, "maxfev": max_nfev})
        root_is_physical = (
            fit.success and np.all(np.isfinite(fit.x))
            and np.all(fit.x > lower) and np.all(fit.x < upper)
        )
        if not root_is_physical:
            start = np.clip(fit.x if np.all(np.isfinite(fit.x)) else np.ones(cells),
                            lower*1.001, upper*0.999)
            fit = least_squares(
                residual, start, bounds=(lower, upper), jac_sparsity=sparsity,
                xtol=rtol, ftol=rtol, gtol=rtol, max_nfev=max_nfev,
                x_scale=1.0,
            )
        delta = delta_scale*fit.x

        values = properties(delta)
        front_delta, front_rate = self._surface_state(
            delta[0], values["d_amb"][0], front_s_cm_s, widths[0]/2
        )
        rear_delta, rear_rate = self._surface_state(
            delta[-1], values["d_amb"][-1], rear_s_cm_s, widths[-1]/2
        )
        rad = float(np.sum(values["r_rad"]*widths))
        aug = float(np.sum(values["r_auger"]*widths))
        srh = float(np.sum(values["r_srh"]*widths))
        total = rad + aug + srh + front_rate + rear_rate
        sheet_excess = float(np.sum(delta*widths))
        clipped = np.mean(delta >= 0.999*self.table.delta_max_cm3)
        max_residual = float(np.max(np.abs(residual(delta/delta_scale))))
        return TransportResult(
            depth_cm=x, cell_width_cm=widths, delta_n_cm3=delta, generation_cm3_s=g,
            d_amb_cm2_s=values["d_amb"], r_rad_cm3_s=values["r_rad"],
            r_auger_cm3_s=values["r_auger"], r_srh_cm3_s=values["r_srh"],
            front_surface_delta_cm3=front_delta,
            rear_surface_delta_cm3=rear_delta,
            front_surface_recombination_cm2_s=front_rate,
            rear_surface_recombination_cm2_s=rear_rate,
            bulk_radiative_recombination_cm2_s=rad,
            bulk_auger_recombination_cm2_s=aug,
            bulk_srh_recombination_cm2_s=srh,
            generated_flux_cm2_s=generated,
            effective_lifetime_s=sheet_excess/generated,
            relative_balance_error=(total-generated)/generated,
            maximum_scaled_cell_residual=max_residual,
            # The residual is normalized to the mean generated flux per cell.
            # 2e-4 therefore corresponds to <0.02% local imbalance on that scale;
            # the separately reported global balance is normally much tighter.
            converged=bool(max_residual < max(10*rtol, 2e-4)),
            nonlinear_evaluations=int(fit.nfev),
            table_clipped_fraction=float(clipped),
            internal_radiative_emission_cm2_s=float(values['r_internal']@widths),
            recycled_generation_cm2_s=float((values['r_internal']*recycling)@widths),
            recycling_probability=recycling.copy(),
        )
