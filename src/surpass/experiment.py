"""Experimental excitation and detection operators for PL/TRPL comparison.

The classes in this module keep the semiconductor model separate from nuisance
instrument parameters. Sampled pulses are normalized as generated carrier dose;
spectral throughput converts collected photons into detected events; the IRF is
a normalized delay-probability density; bin integration converts event rate to
expected counts. Nothing here changes semiconductor recombination physics.

The numerical fitting uses ``scipy.optimize.least_squares``. See the SciPy API
documentation for algorithm details:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares


def _sampled_density(x, y, x_name, y_name):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if (x.ndim != 1 or x.size < 3 or y.shape != x.shape
            or np.any(~np.isfinite(x)) or np.any(np.diff(x) <= 0)
            or np.any(~np.isfinite(y)) or np.any(y < 0)):
        raise ValueError(f'{x_name} must increase and {y_name} must be finite, non-negative, and matching')
    # Normalize the continuous shape actually evaluated by the model, rather
    # than the piecewise-linear trapezoid through its samples.
    area = float(PchipInterpolator(x, y, extrapolate=False).integrate(x[0], x[-1]))
    if area <= 0:
        raise ValueError(f'{y_name} must have positive integral')
    return x.copy(), y/area


@dataclass(frozen=True)
class SampledPulse:
    """Measured temporal pulse shape times its full spatial carrier dose.

    ``temporal_shape`` is normalized numerically over ``relative_time_s``.
    Consequently, ``density_cm3`` is the carrier density produced by the full
    pulse without losses, exactly like :class:`GaussianPulse`. Values outside
    the sampled support are zero. PCHIP avoids oscillatory negative generation.
    Include measured baseline regions and subtract detector background first.
    """
    density_cm3: np.ndarray
    relative_time_s: np.ndarray
    temporal_shape: np.ndarray
    source: str = 'user supplied'

    def __post_init__(self):
        density = np.asarray(self.density_cm3, dtype=float)
        if np.any(~np.isfinite(density)) or np.any(density < 0):
            raise ValueError('density_cm3 must be finite and non-negative')
        time, shape = _sampled_density(self.relative_time_s, self.temporal_shape,
                                       'relative_time_s', 'temporal_shape')
        object.__setattr__(self, 'density_cm3', density.copy())
        object.__setattr__(self, 'relative_time_s', time)
        object.__setattr__(self, 'temporal_shape', shape)
        object.__setattr__(self, '_interpolator', PchipInterpolator(time, shape, extrapolate=False))

    @property
    def recommended_max_step_s(self):
        return float(np.min(np.diff(self.relative_time_s))/2)

    @property
    def support_s(self):
        return float(self.relative_time_s[0]), float(self.relative_time_s[-1])

    def __call__(self, time_s):
        time = np.asarray(time_s, dtype=float)
        shape = np.asarray(self._interpolator(time))
        shape = np.where((time >= self.relative_time_s[0])
                         & (time <= self.relative_time_s[-1]), shape, 0.)
        result = self.density_cm3*float(shape) if time.ndim == 0 else shape[..., None]*self.density_cm3
        return result


@dataclass(frozen=True)
class PulseTrain:
    """Repeat one finite-support pulse at ``first_pulse_s + k*period_s``.

    ``pulse_count=None`` means an infinite mathematical train; only pulses whose
    support contains the requested time are evaluated. Overlapping pulses add.
    The carrier dose of every pulse is the wrapped pulse's ``density_cm3``.
    """
    pulse: SampledPulse
    period_s: float
    first_pulse_s: float = 0.
    pulse_count: int | None = None

    def __post_init__(self):
        if not isinstance(self.pulse, SampledPulse):
            raise TypeError('pulse must be a SampledPulse')
        if not np.isfinite(self.period_s) or self.period_s <= 0 or not np.isfinite(self.first_pulse_s):
            raise ValueError('Use a positive finite period and finite first_pulse_s')
        if self.pulse_count is not None and (not isinstance(self.pulse_count, int) or self.pulse_count < 1):
            raise ValueError('pulse_count must be a positive integer or None')

    @property
    def recommended_max_step_s(self):
        return min(self.pulse.recommended_max_step_s, self.period_s/20)

    @property
    def density_cm3(self):
        """Per-pulse dose, exposed as a solver scaling hint."""
        return self.pulse.density_cm3

    def __call__(self, time_s):
        if np.ndim(time_s) != 0:
            return np.asarray([self(float(t)) for t in np.asarray(time_s)])
        t = float(time_s)
        lo, hi = self.pulse.support_s
        k_min = int(np.ceil((t-hi-self.first_pulse_s)/self.period_s))
        k_max = int(np.floor((t-lo-self.first_pulse_s)/self.period_s))
        k_min = max(k_min, 0)
        if self.pulse_count is not None:
            k_max = min(k_max, self.pulse_count-1)
        result = np.zeros_like(self.pulse.density_cm3, dtype=float)
        for k in range(k_min, k_max+1):
            result += self.pulse(t-self.first_pulse_s-k*self.period_s)
        return result


@dataclass(frozen=True)
class SpectralResponse:
    """Dimensionless wavelength-dependent probability/throughput.

    This may combine filters, grating, detector quantum efficiency and other
    independently calibrated losses. Values must lie in [0,1]. Extrapolation is
    refused. Geometric NA belongs in ``SlabEmissionModel.solve`` instead.
    """
    wavelength_nm: np.ndarray
    efficiency: np.ndarray
    source: str = 'user supplied'

    def __post_init__(self):
        wl, efficiency = np.asarray(self.wavelength_nm, float), np.asarray(self.efficiency, float)
        if (wl.ndim != 1 or wl.size < 2 or efficiency.shape != wl.shape
                or np.any(~np.isfinite(wl)) or np.any(np.diff(wl) <= 0)
                or np.any(~np.isfinite(efficiency)) or np.any(efficiency < 0)
                or np.any(efficiency > 1)):
            raise ValueError('Use increasing wavelengths and matching efficiencies in [0,1]')
        object.__setattr__(self, 'wavelength_nm', wl.copy())
        object.__setattr__(self, 'efficiency', efficiency.copy())

    def evaluate(self, wavelength_nm):
        wavelength = np.asarray(wavelength_nm, dtype=float)
        if np.any(wavelength < self.wavelength_nm[0]) or np.any(wavelength > self.wavelength_nm[-1]):
            raise ValueError('Requested wavelength lies outside spectral-response support')
        return np.interp(wavelength, self.wavelength_nm, self.efficiency)


@dataclass(frozen=True)
class InstrumentResponse:
    """Normalized measured impulse response as a function of relative delay."""
    delay_s: np.ndarray
    response: np.ndarray
    source: str = 'user supplied'

    def __post_init__(self):
        delay, response = _sampled_density(self.delay_s, self.response,
                                           'delay_s', 'response')
        object.__setattr__(self, 'delay_s', delay)
        object.__setattr__(self, 'response', response)
        object.__setattr__(self, '_interpolator', PchipInterpolator(delay, response, extrapolate=False))

    def evaluate(self, delay_s):
        delay = np.asarray(delay_s, dtype=float)
        value = np.asarray(self._interpolator(delay))
        return np.where((delay >= self.delay_s[0]) & (delay <= self.delay_s[-1]), value, 0.)

    def convolve(self, source_time_s, source_rate, evaluation_time_s, time_shift_s=0.):
        """Continuous convolution; source is zero outside its sampled window."""
        t, rate = np.asarray(source_time_s, float), np.asarray(source_rate, float)
        evaluation = np.asarray(evaluation_time_s, float)
        if (t.ndim != 1 or t.size < 3 or rate.shape != t.shape
                or np.any(~np.isfinite(t)) or np.any(np.diff(t) <= 0)
                or np.any(~np.isfinite(rate)) or np.any(rate < 0)
                or np.any(~np.isfinite(evaluation)) or not np.isfinite(time_shift_s)):
            raise ValueError('Use increasing source times, matching non-negative rates, and finite evaluation times')
        kernel = self.evaluate(evaluation[..., None]-t-time_shift_s)
        return np.trapezoid(kernel*rate, t, axis=-1)


@dataclass(frozen=True)
class DetectionModel:
    """Convert detected photon flux per area to binned expected counts."""
    emitting_area_cm2: float
    irf: InstrumentResponse | None = None
    electronic_gain_counts_per_event: float = 1.
    background_rate_counts_s: float = 0.

    def __post_init__(self):
        values = (self.emitting_area_cm2, self.electronic_gain_counts_per_event,
                  self.background_rate_counts_s)
        if (any(not np.isfinite(x) for x in values) or self.emitting_area_cm2 <= 0
                or self.electronic_gain_counts_per_event <= 0
                or self.background_rate_counts_s < 0):
            raise ValueError('Area/gain must be positive finite and background finite non-negative')

    def event_rate(self, source_time_s, photon_flux_cm2_s, evaluation_time_s,
                   time_shift_s=0., scale=1.):
        if not np.isfinite(scale) or scale < 0:
            raise ValueError('scale must be finite and non-negative')
        flux = np.asarray(photon_flux_cm2_s, float)
        if self.irf is None:
            t = np.asarray(source_time_s, float)
            evaluation = np.asarray(evaluation_time_s, float)
            if t.ndim != 1 or flux.shape != t.shape or np.any(np.diff(t) <= 0) or np.any(flux < 0):
                raise ValueError('Use increasing source time and matching non-negative photon flux')
            shifted = evaluation-time_shift_s
            rate = np.interp(shifted, t, flux, left=0., right=0.)
        else:
            rate = self.irf.convolve(source_time_s, flux, evaluation_time_s, time_shift_s)
        return scale*self.emitting_area_cm2*self.electronic_gain_counts_per_event*rate

    def expected_counts(self, source_time_s, photon_flux_cm2_s, bin_edges_s,
                        time_shift_s=0., scale=1., background_rate_counts_s=None,
                        quadrature_order=8):
        edges = np.asarray(bin_edges_s, float)
        if (edges.ndim != 1 or edges.size < 2 or np.any(~np.isfinite(edges))
                or np.any(np.diff(edges) <= 0) or quadrature_order < 2):
            raise ValueError('Use increasing finite bin edges and quadrature_order >=2')
        background = self.background_rate_counts_s if background_rate_counts_s is None else background_rate_counts_s
        if not np.isfinite(background) or background < 0:
            raise ValueError('Background rate must be finite and non-negative')
        nodes, weights = leggauss(quadrature_order)
        middle, half = (edges[1:]+edges[:-1])/2, np.diff(edges)/2
        times = middle[:, None]+half[:, None]*nodes
        rate = self.event_rate(source_time_s, photon_flux_cm2_s, times,
                               time_shift_s=time_shift_s, scale=scale)
        return half*np.sum(rate*weights, axis=1)+background*np.diff(edges)


@dataclass(frozen=True)
class TraceFitResult:
    scale: float
    time_shift_s: float
    background_rate_counts_s: float
    expected_counts: np.ndarray
    standardized_residual: np.ndarray
    reduced_statistic: float
    statistic_name: str
    covariance: np.ndarray
    success: bool
    message: str

    @property
    def reduced_chi_square(self):
        """Backward-compatible alias; see ``statistic_name`` for its meaning."""
        return self.reduced_statistic


def fit_detection_nuisance(detection, source_time_s, photon_flux_cm2_s,
                           bin_edges_s, measured_counts, sigma_counts=None,
                           initial_scale=1., initial_time_shift_s=0.,
                           initial_background_rate_counts_s=0.,
                           shift_bounds_s=(-np.inf, np.inf)):
    """Fit only scale, timing offset and constant background to measured bins.

    The semiconductor trace is fixed. This deliberately does not claim that a
    fitted scale is an absolute optical calibration. Provide independent
    ``sigma_counts`` for analog data (weighted least squares). If omitted, a
    signed Poisson-deviance residual is minimized, including zero-count bins.
    """
    if not isinstance(detection, DetectionModel):
        raise TypeError('detection must be a DetectionModel')
    measured = np.asarray(measured_counts, float)
    bins = np.asarray(bin_edges_s, float)
    if measured.shape != (bins.size-1,) or np.any(~np.isfinite(measured)) or np.any(measured < 0):
        raise ValueError('measured_counts must be finite, non-negative, and match bins')
    sigma = None if sigma_counts is None else np.asarray(sigma_counts, float)
    if sigma is not None and (sigma.shape != measured.shape or np.any(~np.isfinite(sigma)) or np.any(sigma <= 0)):
        raise ValueError('sigma_counts must be positive, finite, and match measured_counts')
    if initial_scale <= 0 or initial_background_rate_counts_s < 0:
        raise ValueError('Initial scale must be positive and background non-negative')
    floor = max(1., np.sum(measured)/(bins[-1]-bins[0]))*1e-15
    x0 = np.array([np.log(initial_scale), initial_time_shift_s,
                   np.log(initial_background_rate_counts_s+floor)])
    lower = [-np.inf, shift_bounds_s[0], np.log(floor)]
    upper = [np.inf, shift_bounds_s[1], np.inf]

    def unpack(x):
        return np.exp(x[0]), x[1], max(np.exp(x[2])-floor, 0.)

    def residual(x):
        scale, shift, background = unpack(x)
        predicted = detection.expected_counts(source_time_s, photon_flux_cm2_s,
            bins, time_shift_s=shift, scale=scale,
            background_rate_counts_s=background)
        if sigma is not None:
            return (predicted-measured)/sigma
        predicted = np.maximum(predicted, np.finfo(float).tiny)
        term = predicted-measured
        positive = measured > 0
        term[positive] += measured[positive]*np.log(measured[positive]/predicted[positive])
        return np.sign(predicted-measured)*np.sqrt(2*np.maximum(term, 0.))

    fit = least_squares(residual, x0, bounds=(lower, upper), method='trf',
                        x_scale='jac', max_nfev=500)
    scale, shift, background = unpack(fit.x)
    predicted = detection.expected_counts(source_time_s, photon_flux_cm2_s,
        bins, time_shift_s=shift, scale=scale, background_rate_counts_s=background)
    standardized = residual(fit.x)
    dof = max(measured.size-fit.x.size, 1)
    reduced = float(np.dot(standardized, standardized)/dof)
    covariance = np.full((3, 3), np.nan)
    if fit.jac.shape[0] >= fit.jac.shape[1]:
        try:
            covariance_x = np.linalg.inv(fit.jac.T@fit.jac)*reduced
            transform = np.diag([scale, 1., background+floor])
            covariance = transform@covariance_x@transform
        except np.linalg.LinAlgError:
            pass
    statistic = 'weighted chi-square' if sigma is not None else 'Poisson deviance'
    return TraceFitResult(scale, shift, background, predicted, standardized,
                          reduced, statistic, covariance, bool(fit.success), fit.message)


@dataclass(frozen=True)
class EmpiricalDecayFit:
    amplitudes: np.ndarray
    lifetimes_s: np.ndarray
    background: float
    fitted_signal: np.ndarray
    fit_mask: np.ndarray
    reduced_chi_square: float
    success: bool
    message: str


def fit_empirical_exponentials(time_s, signal, components=2, fit_window_s=None,
                               sigma=None, include_background=True):
    """Fit 1–2 positive exponentials as a descriptive summary only.

    The time origin is the start of the selected window. This fit does not
    identify microscopic channels; diffusion, injection dependence, recycling,
    IRF and a changing collection function can all yield multiexponential-like
    traces. Apply an IRF-aware physical observation model before interpreting
    early-time data.
    """
    t, y = np.asarray(time_s, float), np.asarray(signal, float)
    if (t.ndim != 1 or t.size < 5 or y.shape != t.shape or np.any(np.diff(t) <= 0)
            or np.any(~np.isfinite(y)) or np.any(y < 0) or components not in (1, 2)):
        raise ValueError('Use increasing time, matching finite non-negative signal, and 1 or 2 components')
    window = (t[0], t[-1]) if fit_window_s is None else fit_window_s
    mask = (t >= window[0]) & (t <= window[1])
    if np.count_nonzero(mask) < 2*components+2:
        raise ValueError('Fit window contains too few points')
    x, data = t[mask]-t[mask][0], y[mask]
    uncertainty = np.ones_like(data) if sigma is None else np.asarray(sigma, float)[mask]
    if uncertainty.shape != data.shape or np.any(~np.isfinite(uncertainty)) or np.any(uncertainty <= 0):
        raise ValueError('sigma must be positive and match signal')
    duration = max(x[-1], np.finfo(float).eps)
    background0 = max(float(np.min(data)), np.max(data)*1e-12) if include_background else 0.
    dynamic = max(float(data[0]-background0), np.max(data)*1e-12)
    if components == 1:
        initial = [np.log(dynamic), np.log(duration/3)]
    else:
        initial = [np.log(.6*dynamic), np.log(.4*dynamic),
                   np.log(duration/10), np.log(duration/2-duration/10)]
    if include_background: initial.append(np.log(max(background0, np.max(data)*1e-12)))

    def evaluate(parameters):
        if components == 1:
            amplitudes, lifetimes = np.exp(parameters[:1]), np.exp(parameters[1:2])
            offset = 2
        else:
            amplitudes = np.exp(parameters[:2])
            tau1 = np.exp(parameters[2])
            lifetimes = np.array([tau1, tau1+np.exp(parameters[3])])
            offset = 4
        background = np.exp(parameters[offset]) if include_background else 0.
        model = background+np.sum(amplitudes[:, None]*np.exp(-x[None, :]/lifetimes[:, None]), axis=0)
        return amplitudes, lifetimes, background, model

    fit = least_squares(lambda p: (evaluate(p)[3]-data)/uncertainty,
                        initial, method='trf', x_scale='jac', max_nfev=2000)
    amplitudes, lifetimes, background, model = evaluate(fit.x)
    fitted = np.full_like(y, np.nan); fitted[mask] = model
    residual = (model-data)/uncertainty
    dof = max(data.size-fit.x.size, 1)
    return EmpiricalDecayFit(amplitudes, lifetimes, background, fitted, mask,
                             float(residual@residual/dof), bool(fit.success), fit.message)
