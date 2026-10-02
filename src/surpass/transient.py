"""Transient quasi-neutral ambipolar diffusion with instantaneous surfaces.

Solve d(delta)/dt = -dJ/dx + G - R, J=-Da(delta)*d(delta)/dx,
on the same cell-centred finite-volume grid as the steady-state solver.
The semiconductor space-charge region remains a local, zero-storage boundary;
trap occupancy and potential follow injection instantaneously. Optional local
instantaneous photon recycling reduces net radiative loss; nonlocal photon
transport, dynamic trap charge and Poisson/drift-diffusion remain omitted.

Numerics: SciPy solve_ivp BDF/Radau with a tridiagonal Jacobian sparsity pattern.
See https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html.
All rates and material assumptions are inherited from BulkTransportTable and
the surface boundary objects, rather than introducing new literature values.
"""

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags
from numpy.polynomial.legendre import leggauss
from .transport import BulkTransportTable, SteadyState1DSolver


@dataclass(frozen=True)
class GaussianPulse:
    """G(t,x)=density(x)*normalized Gaussian(t), in cm^-3 s^-1.

    ``density_cm3`` is the carrier density that the full pulse would generate
    without diffusion or recombination. A finite simulation window truncates
    Gaussian tails; its actual generated dose is reported by the solver.
    """
    density_cm3: np.ndarray
    center_s: float
    sigma_s: float

    def __post_init__(self):
        density = np.asarray(self.density_cm3, dtype=float)
        if np.any(~np.isfinite(density)) or np.any(density < 0):
            raise ValueError("Pulse density must be finite and non-negative")
        if not np.isfinite(self.sigma_s) or self.sigma_s <= 0 or not np.isfinite(self.center_s):
            raise ValueError("Use finite center_s and positive finite sigma_s")
        object.__setattr__(self, 'density_cm3', density.copy())

    def __call__(self, time_s):
        return self.density_cm3*np.exp(-0.5*((time_s-self.center_s)/self.sigma_s)**2)/(np.sqrt(2*np.pi)*self.sigma_s)

    @property
    def recommended_max_step_s(self):
        return self.sigma_s/2


@dataclass(frozen=True)
class TransientResult:
    time_s: np.ndarray
    depth_cm: np.ndarray
    cell_width_cm: np.ndarray
    delta_n_cm3: np.ndarray  # shape (time, depth)
    sheet_excess_cm2: np.ndarray
    front_surface_delta_cm3: np.ndarray
    rear_surface_delta_cm3: np.ndarray
    front_surface_recombination_cm2_s: np.ndarray
    rear_surface_recombination_cm2_s: np.ndarray
    bulk_radiative_recombination_cm2_s: np.ndarray
    bulk_auger_recombination_cm2_s: np.ndarray
    bulk_srh_recombination_cm2_s: np.ndarray
    generated_flux_cm2_s: np.ndarray
    cumulative_generated_cm2: np.ndarray
    cumulative_radiative_cm2: np.ndarray
    cumulative_auger_cm2: np.ndarray
    cumulative_srh_cm2: np.ndarray
    cumulative_front_cm2: np.ndarray
    cumulative_rear_cm2: np.ndarray
    inventory_relative_error: np.ndarray
    initial_sheet_excess_cm2: float
    minimum_raw_delta_cm3: float
    integrator_steps: int
    nonlinear_evaluations: int
    converged: bool
    message: str
    # Existing radiative/cumulative fields remain NET carrier losses. These
    # additional fields distinguish intrinsic emitted photons from recycled
    # pairs, preserving v0.7 semantics when recycling_probability=0.
    internal_radiative_emission_cm2_s: np.ndarray
    recycled_generation_cm2_s: np.ndarray
    cumulative_internal_radiative_cm2: np.ndarray
    cumulative_recycled_cm2: np.ndarray
    recycling_probability: np.ndarray

    @property
    def average_delta_n_cm3(self):
        return self.sheet_excess_cm2/np.sum(self.cell_width_cm)

    @property
    def total_recombination_cm2_s(self):
        return (self.bulk_radiative_recombination_cm2_s
                + self.bulk_auger_recombination_cm2_s + self.bulk_srh_recombination_cm2_s
                + self.front_surface_recombination_cm2_s + self.rear_surface_recombination_cm2_s)

    @property
    def effective_lifetime_s(self):
        """Stored excess carriers / total instantaneous loss, not a PL fit tau."""
        return np.divide(self.sheet_excess_cm2, self.total_recombination_cm2_s,
                         out=np.full_like(self.time_s, np.nan),
                         where=self.total_recombination_cm2_s > 0)

    def normalized_radiative_flux(self):
        signal = self.internal_radiative_emission_cm2_s
        return signal/np.max(signal) if np.max(signal) > 0 else np.zeros_like(signal)

    def local_pl_decay_time_s(self, floor_fraction=1e-6):
        """-1/(d log I_rad/dt) on the falling, above-threshold trace.

        This is a sampling-dependent local slope, not an intrinsic carrier
        lifetime. In high injection I_rad may scale as delta^2, whereas in
        doped low injection it approaches proportionality to delta.
        """
        if not 0 < floor_fraction < 1:
            raise ValueError("floor_fraction must lie in (0,1)")
        signal = self.internal_radiative_emission_cm2_s
        if np.max(signal) <= 0:
            return np.full_like(signal, np.nan)
        slope = np.gradient(np.log(np.maximum(signal, np.finfo(float).tiny)), self.time_s)
        valid = (signal > np.max(signal)*floor_fraction) & (slope < 0)
        return np.divide(-1.0, slope, out=np.full_like(slope, np.nan), where=valid)


class TimeDependent1DSolver:
    """Stiff method-of-lines solver with explicit inventory diagnostics.

    ``generation`` is a scalar, one value per cell, or callable(time_s) returning
    either. Arbitrary short/discontinuous pumps require an appropriate
    ``max_step_s`` or separate solve calls at switching times. Generation
    objects exposing ``recommended_max_step_s`` automatically limit steps so
    a dark-to-pulse solve cannot skip the excitation.
    Initial densities must be cell averages (not point samples of a shallow
    exponential). Time zero may be just after an ideal instantaneous pulse.
    """

    def __init__(self, table: BulkTransportTable):
        self.table = table

    def solve(self, thickness_cm, time_s, initial_delta_cm3,
              front_s_cm_s=0.0, rear_s_cm_s=0.0, generation=0.0,
              cells=160, cell_edges_cm=None, method='BDF', rtol=1e-6,
              atol_scaled=1e-10, max_step_s=np.inf, balance_order=3,
              recycling_probability=0.0):
        """recycling_probability is a scalar or one fixed value per cell.

        The local approximation replaces Rrad(x) with (1-p_rec(x))*Rrad(x).
        Every actively reabsorbed photon returns a pair to its emission cell;
        there is no spatial redistribution, optical flight time or injection-
        dependent spectrum. Do not also scale B: that double-counts recycling.
        Result radiative-loss fields are net; internal-emission fields are raw.
        """
        times = np.asarray(time_s, dtype=float)
        if (times.ndim != 1 or times.size < 3 or np.any(~np.isfinite(times))
                or np.any(np.diff(times) <= 0)):
            raise ValueError("Use at least three finite, increasing output times")
        if not np.isfinite(thickness_cm) or thickness_cm <= 0 or cells < 8:
            raise ValueError("Use positive finite thickness and at least 8 cells")
        if method not in ('BDF', 'Radau'):
            raise ValueError("method must be BDF or Radau")
        if not np.isfinite(rtol) or rtol <= 0 or not np.isfinite(atol_scaled) or atol_scaled <= 0:
            raise ValueError("rtol and atol_scaled must be finite and positive")
        if np.isnan(max_step_s) or max_step_s <= 0 or balance_order not in (3, 5):
            raise ValueError("Use positive max_step_s and balance_order 3 or 5")
        for boundary in (front_s_cm_s, rear_s_cm_s):
            if np.isscalar(boundary):
                if not np.isfinite(boundary) or boundary < 0:
                    raise ValueError("Surface velocities must be finite and non-negative")
            elif not callable(getattr(boundary, 'rate', None)):
                raise TypeError("Surface boundaries must expose rate(delta)")
        edges = (np.linspace(0, thickness_cm, cells+1) if cell_edges_cm is None
                 else np.asarray(cell_edges_cm, dtype=float))
        if (edges.ndim != 1 or edges.size < 9 or np.any(~np.isfinite(edges))
                or abs(edges[0]) > 1e-15 or np.any(np.diff(edges) <= 0)
                or abs(edges[-1]-thickness_cm) > max(1e-15, thickness_cm*1e-12)):
            raise ValueError("Cell edges must increase from zero to wafer thickness")
        widths = np.diff(edges)
        depth = 0.5*(edges[1:]+edges[:-1])
        cells = widths.size

        def array(value, label):
            a = np.asarray(value, dtype=float)
            if a.shape == ():
                a = np.full(cells, float(a))
            if a.shape != (cells,) or np.any(~np.isfinite(a)) or np.any(a < 0):
                raise ValueError(f"{label} must be finite, non-negative, and match cells")
            return a

        initial = array(initial_delta_cm3, 'Initial density')
        recycling = array(recycling_probability, 'Recycling probability')
        if np.any(recycling > 1):
            raise ValueError('Recycling probability must lie in [0,1]')
        if np.max(initial) > self.table.delta_max_cm3:
            raise ValueError("Initial density exceeds the bulk table; extend its range")

        def pump(t):
            return array(generation(t) if callable(generation) else generation, 'Generation')

        duration = times[-1]-times[0]
        scale = max(float(np.max(initial)), float(np.max(pump(times[0])))*duration, 1.0)
        pulse_density = getattr(generation, 'density_cm3', None)
        if pulse_density is not None:
            pulse_density = np.asarray(pulse_density, dtype=float)
            if np.any(~np.isfinite(pulse_density)) or np.any(pulse_density < 0):
                raise ValueError('generation density_cm3 scaling hint must be finite and non-negative')
            scale = max(scale, float(np.max(pulse_density)))
        suggested_step = getattr(generation, 'recommended_max_step_s', None)
        if suggested_step is not None:
            if not np.isfinite(suggested_step) or suggested_step <= 0:
                raise ValueError('generation recommended_max_step_s must be positive and finite')
            max_step_s = min(max_step_s, float(suggested_step))

        def properties(delta):
            positive = np.maximum(delta, 0)
            values = self.table.evaluate(positive)
            # Original tables are positive-log interpolants; exact darkness
            # needs exact zero loss, not their tiny interpolation floor.
            for channel in ('r_rad', 'r_auger', 'r_srh'):
                values[channel] = np.where(positive > 0, values[channel], 0)
            values['r_internal'] = values['r_rad'].copy()
            values['r_recycled'] = recycling*values['r_rad']
            values['r_rad'] = (1-recycling)*values['r_rad']
            # Sum channel interpolants for consistent carrier bookkeeping.
            values['r_total'] = values['r_rad']+values['r_auger']+values['r_srh']
            return values

        def boundaries(delta, diffusion):
            front = SteadyState1DSolver._surface_state(
                max(float(delta[0]), 0), diffusion[0], front_s_cm_s, widths[0]/2)
            rear = SteadyState1DSolver._surface_state(
                max(float(delta[-1]), 0), diffusion[-1], rear_s_cm_s, widths[-1]/2)
            return front, rear

        def rhs(t, scaled_delta):
            delta = scale*scaled_delta
            values = properties(delta)
            d = values['d_amb']
            resistance = widths[:-1]/(2*d[:-1])+widths[1:]/(2*d[1:])
            flux = -np.diff(delta)/resistance
            front, rear = boundaries(delta, d)
            left, right = np.r_[-front[1], flux], np.r_[flux, rear[1]]
            return (pump(t)-values['r_total']-(right-left)/widths)/scale

        sparsity = diags([np.ones(cells-1), np.ones(cells), np.ones(cells-1)],
                         [-1, 0, 1], format='csr')
        solution = solve_ivp(rhs, (times[0], times[-1]), initial/scale,
                             method=method, rtol=rtol, atol=atol_scaled,
                             max_step=max_step_s, jac_sparsity=sparsity,
                             dense_output=True)
        if not solution.success:
            raise RuntimeError(f"Transient integration failed: {solution.message}")
        raw = solution.sol(times).T*scale
        minimum = min(float(np.min(raw)), float(np.min(solution.y))*scale)
        if minimum < -20*atol_scaled*scale:
            raise RuntimeError("Significant negative carrier density; refine tolerances/grid")
        if max(np.max(raw), np.max(solution.y)*scale) > self.table.delta_max_cm3*(1+rtol):
            raise ValueError("Transient exceeds bulk-table range; extend it and rerun")
        delta = np.maximum(raw, 0)

        def diagnostics(t, densities):
            values = properties(densities)
            rates = np.column_stack([values[name]@widths
                                     for name in ('r_rad', 'r_auger', 'r_srh')])
            fronts, rears = [], []
            for density, d in zip(densities, values['d_amb']):
                front, rear = boundaries(density, d)
                fronts.append(front)
                rears.append(rear)
            fronts, rears = np.asarray(fronts), np.asarray(rears)
            for boundary, surface in [(front_s_cm_s, fronts), (rear_s_cm_s, rears)]:
                limit = getattr(boundary, 'delta_max_cm3', np.inf)
                if np.max(surface[:, 0]) > limit*(1+rtol):
                    raise ValueError("Surface injection exceeds boundary table; extend it and rerun")
            generated = np.array([np.dot(pump(time), widths) for time in t])
            channels = np.column_stack([generated, rates, fronts[:, 1], rears[:, 1],
                                       values['r_internal']@widths, values['r_recycled']@widths])
            return channels, fronts[:, 0], rears[:, 0]

        channels, front_delta, rear_delta = diagnostics(times, delta)
        # Independent Gaussian quadrature along the integrator's dense solution.
        # Include requested output times as interval boundaries, so cumulative
        # diagnostics do not introduce a separate linear-interpolation error.
        grid = np.unique(np.r_[solution.t, times])
        mid, half = (grid[1:]+grid[:-1])/2, np.diff(grid)/2
        nodes, weights = leggauss(balance_order)
        node_times = (mid[:, None]+half[:, None]*nodes).ravel()
        raw_node_density = solution.sol(node_times).T*scale
        minimum = min(minimum, float(np.min(raw_node_density)))
        if minimum < -20*atol_scaled*scale:
            raise RuntimeError("Significant negative density in dense output; refine tolerances")
        if np.max(raw_node_density) > self.table.delta_max_cm3*(1+rtol):
            raise ValueError("Dense output exceeds bulk-table range; extend it and rerun")
        node_density = np.maximum(raw_node_density, 0)
        node_channels, _, _ = diagnostics(node_times, node_density)
        interval_integrals = half[:, None]*np.sum(
            node_channels.reshape(-1, balance_order, 8)*weights[None, :, None], axis=1)
        cumulative = np.vstack([np.zeros(8), np.cumsum(interval_integrals, axis=0)])
        cumulative = cumulative[np.searchsorted(grid, times)]
        sheet = delta@widths
        initial_sheet = float(initial@widths)
        # A single full-window dose scale avoids dividing by the exponentially
        # tiny generated inventory at the leading edge of a pulse.
        reference = max(initial_sheet+float(cumulative[-1, 0]), 1.0)
        balance = (sheet-initial_sheet-cumulative[:, 0]+np.sum(cumulative[:, 1:6], axis=1))/reference
        return TransientResult(
            time_s=times.copy(), depth_cm=depth, cell_width_cm=widths,
            delta_n_cm3=delta, sheet_excess_cm2=sheet,
            front_surface_delta_cm3=front_delta, rear_surface_delta_cm3=rear_delta,
            generated_flux_cm2_s=channels[:, 0],
            bulk_radiative_recombination_cm2_s=channels[:, 1],
            bulk_auger_recombination_cm2_s=channels[:, 2],
            bulk_srh_recombination_cm2_s=channels[:, 3],
            front_surface_recombination_cm2_s=channels[:, 4],
            rear_surface_recombination_cm2_s=channels[:, 5],
            cumulative_generated_cm2=cumulative[:, 0], cumulative_radiative_cm2=cumulative[:, 1],
            cumulative_auger_cm2=cumulative[:, 2], cumulative_srh_cm2=cumulative[:, 3],
            cumulative_front_cm2=cumulative[:, 4], cumulative_rear_cm2=cumulative[:, 5],
            inventory_relative_error=balance, initial_sheet_excess_cm2=initial_sheet,
            minimum_raw_delta_cm3=minimum, integrator_steps=solution.t.size-1,
            nonlinear_evaluations=solution.nfev,
            converged=bool(np.max(np.abs(balance)) < max(1e-3, 10*rtol)),
            message=solution.message,
            internal_radiative_emission_cm2_s=channels[:, 6],
            recycled_generation_cm2_s=channels[:, 7],
            cumulative_internal_radiative_cm2=cumulative[:, 6],
            cumulative_recycled_cm2=cumulative[:, 7], recycling_probability=recycling.copy())
