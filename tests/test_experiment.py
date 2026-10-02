"""Analytic pulse, IRF, detection, binning and fitting checks."""
import unittest
import numpy as np
from scipy.special import erf
from surpass import (SampledPulse, PulseTrain, InstrumentResponse,
                        SpectralResponse, DetectionModel,
                        fit_detection_nuisance, fit_empirical_exponentials,
                        BulkTransportTable, TimeDependent1DSolver,
                        EmissionSpectrum, EmissionBoundary, SlabEmissionModel,
                        OpticalMaterial)


def optical(n):
    return OpticalMaterial('test', (800., 1000.), (n, n), (0., 0.), 'analytic')


class TestExperiment(unittest.TestCase):
    def test_sampled_pulse_normalization_and_solver_dose(self):
        relative = np.linspace(-4e-9, 4e-9, 81)
        shape = np.exp(-.5*(relative/1e-9)**2)
        pulse = SampledPulse(np.full(12, 1e14), relative, shape)
        dense = np.linspace(relative[0], relative[-1], 10001)
        self.assertAlmostEqual(np.trapezoid([pulse(t)[0]/1e14 for t in dense], dense), 1., places=7)
        self.assertEqual(pulse(-5e-9)[0], 0.)
        result = TimeDependent1DSolver(BulkTransportTable.linear_lifetime(10, 1.)).solve(
            .001, np.linspace(-5e-9, 5e-9, 101), 0., generation=pulse,
            cells=12, rtol=1e-8)
        self.assertAlmostEqual(result.cumulative_generated_cm2[-1]/1e11, 1., places=6)
        self.assertLess(np.max(abs(result.inventory_relative_error)), 1e-6)

    def test_pulse_train_count_overlap_and_dose(self):
        pulse = SampledPulse(np.array([2.]), [-1., 0., 1.], [0., 1., 0.])
        finite = PulseTrain(pulse, 3., first_pulse_s=2., pulse_count=2)
        self.assertEqual(finite(2.)[0], pulse(0.)[0])
        self.assertEqual(finite(5.)[0], pulse(0.)[0])
        self.assertEqual(finite(8.)[0], 0.)
        overlap = PulseTrain(pulse, .5, first_pulse_s=0., pulse_count=3)
        self.assertGreater(overlap(.5)[0], pulse(.5)[0])
        times = np.linspace(-2, 8, 20001)
        values = np.array([finite(t)[0] for t in times])
        self.assertAlmostEqual(np.trapezoid(values, times), 4., places=6)

    def test_irf_normalization_delta_and_gaussian_convolution(self):
        delay = np.linspace(-8, 8, 1601)
        sigma_h = 1.2
        irf = InstrumentResponse(delay, np.exp(-.5*(delay/sigma_h)**2))
        self.assertAlmostEqual(np.trapezoid(irf.response, delay), 1.)
        time = np.linspace(-12, 12, 2401)
        sigma_s = 1.8
        source = np.exp(-.5*(time/sigma_s)**2)/(np.sqrt(2*np.pi)*sigma_s)
        convolved = irf.convolve(time, source, time)
        sigma = np.sqrt(sigma_s**2+sigma_h**2)
        exact = np.exp(-.5*(time/sigma)**2)/(np.sqrt(2*np.pi)*sigma)
        central = abs(time) < 7
        np.testing.assert_allclose(convolved[central], exact[central], rtol=2e-6, atol=1e-10)
        shifted = irf.convolve(time, source, [2.], time_shift_s=2.)
        self.assertAlmostEqual(shifted[0], exact[len(time)//2], places=8)

    def test_detection_constant_rate_binning(self):
        model = DetectionModel(emitting_area_cm2=.02,
                               electronic_gain_counts_per_event=.5,
                               background_rate_counts_s=10.)
        time = np.linspace(0, 10, 101)
        edges = np.linspace(1, 9, 17)
        counts = model.expected_counts(time, np.full_like(time, 1000.), edges)
        np.testing.assert_allclose(counts, (1000*.02*.5+10)*np.diff(edges), rtol=1e-14)

    def test_detection_irf_preserves_integrated_events(self):
        time = np.linspace(-10, 20, 3001)
        source = np.where((time >= 0)&(time <= 2), 3., 0.)
        delay = np.linspace(-4, 4, 801)
        irf = InstrumentResponse(delay, np.exp(-.5*(delay/.5)**2))
        model = DetectionModel(1., irf)
        edges = np.linspace(-8, 12, 401)
        counts = model.expected_counts(time, source, edges)
        self.assertAlmostEqual(np.sum(counts), np.trapezoid(source, time), places=5)

    def test_spectral_response_weighting(self):
        wl = np.array([850., 900., 950.])
        spectrum = EmissionSpectrum(wl, np.ones(3))
        boundary = EmissionBoundary(optical(1.))
        fates = SlabEmissionModel(optical(1.), boundary, boundary, 0.).solve(
            .01, [.005], spectrum, angular_points=32, front_na=1.)
        response = SpectralResponse([800., 900., 1000.], [0., .5, 1.])
        rates, widths = np.array([[2e20], [1e20]]), np.array([.01])
        flux = fates.response_weighted_flux(rates, widths, response)
        # Index matched and transparent: half goes out each side; average QE=.5.
        np.testing.assert_allclose(flux, np.array([2e20, 1e20])*.01*.5*.5)
        spectral = fates.spectral_flux(rates, widths)
        np.testing.assert_allclose(np.trapezoid(spectral, wl, axis=-1),
                                   fates.emission_fluxes(rates, widths)['front_collected'])

    def test_nuisance_fit_recovers_scale_shift_background(self):
        time = np.linspace(-5e-9, 30e-9, 701)
        source = np.where(time >= 0, np.exp(-time/4e-9), 0.)*1e10
        delay = np.linspace(-2e-9, 2e-9, 161)
        irf = InstrumentResponse(delay, np.exp(-.5*(delay/.4e-9)**2))
        detector = DetectionModel(.01, irf)
        edges = np.linspace(-2e-9, 20e-9, 111)
        truth = (2.3, .7e-9, 2e8)
        measured = detector.expected_counts(time, source, edges,
            scale=truth[0], time_shift_s=truth[1], background_rate_counts_s=truth[2])
        result = fit_detection_nuisance(detector, time, source, edges, measured,
            sigma_counts=np.ones_like(measured)*.01,
            initial_scale=1., initial_time_shift_s=0.,
            initial_background_rate_counts_s=1e8,
            shift_bounds_s=(-1.5e-9, 1.5e-9))
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.scale/truth[0], 1., places=5)
        self.assertAlmostEqual(result.time_shift_s/truth[1], 1., places=4)
        self.assertAlmostEqual(result.background_rate_counts_s/truth[2], 1., places=4)

        # Without supplied uncertainties the same routine must exercise its
        # count-data (Poisson-deviance) path rather than silently assuming
        # constant Gaussian errors.
        poisson = fit_detection_nuisance(detector, time, source, edges, measured,
            initial_scale=1., initial_time_shift_s=0.,
            initial_background_rate_counts_s=1e8,
            shift_bounds_s=(-1.5e-9, 1.5e-9))
        self.assertTrue(poisson.success)
        self.assertEqual(poisson.statistic_name, "Poisson deviance")
        self.assertAlmostEqual(poisson.scale/truth[0], 1., places=4)
        self.assertAlmostEqual(poisson.time_shift_s/truth[1], 1., places=3)
        self.assertAlmostEqual(poisson.background_rate_counts_s/truth[2], 1., places=3)

    def test_empirical_exponential_fit_and_window(self):
        time = np.linspace(0, 20e-9, 201)
        signal = 5*np.exp(-time/2e-9)+2*np.exp(-time/8e-9)+.1
        fit = fit_empirical_exponentials(time, signal, components=2,
                                         fit_window_s=(1e-9, 18e-9), sigma=np.ones_like(time)*1e-4)
        self.assertTrue(fit.success)
        np.testing.assert_allclose(fit.lifetimes_s, [2e-9, 8e-9], rtol=1e-6)
        np.testing.assert_allclose(fit.amplitudes,
                                   [5*np.exp(-1e-9/2e-9), 2*np.exp(-1e-9/8e-9)], rtol=1e-6)
        self.assertAlmostEqual(fit.background, .1, places=7)
        self.assertTrue(np.all(np.isnan(fit.fitted_signal[~fit.fit_mask])))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError): SampledPulse([1], [0, 1, 1], [0, 1, 0])
        with self.assertRaises(ValueError): SpectralResponse([1, 2], [0, 1.1])
        with self.assertRaises(ValueError): InstrumentResponse([0, 1, 2], [0, 0, 0])
        with self.assertRaises(ValueError): DetectionModel(0)
        pulse = SampledPulse([1], [-1, 0, 1], [0, 1, 0])
        with self.assertRaises(ValueError): PulseTrain(pulse, 0)
        with self.assertRaises(ValueError): fit_empirical_exponentials([0, 1], [1, 0])


if __name__ == '__main__': unittest.main()
