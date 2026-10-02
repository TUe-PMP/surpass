"""Photon conservation, angular optics and local-recycling analytic limits."""
import unittest
import numpy as np
from surpass import (OpticalMaterial, Layer, EmissionSpectrum, EmissionBoundary,
                       SlabEmissionModel, BulkTransportTable, TimeDependent1DSolver,
                       BulkModel, Wafer, single_pass_escape_probability, SteadyState1DSolver,
                       OpticalStack)


def optics(n, k=0):
    return OpticalMaterial('test', (700., 1200.), (n, n), (k, k), 'analytic test')


class TestEmission(unittest.TestCase):
    def setUp(self):
        self.spectrum = EmissionSpectrum(np.array([900.]), np.array([1.]))

    def model(self, n=3.3, active=1000., passive=0., front=None, rear=None):
        boundary = EmissionBoundary(optics(1.))
        return SlabEmissionModel(optics(n), front or boundary, rear or boundary, active, passive)

    def test_normal_fresnel_and_coherent_film(self):
        boundary = EmissionBoundary(optics(1.))
        for pol in ('s', 'p'):
            r, t, a = boundary.probabilities(900., 3.3, 1., pol)
            self.assertAlmostEqual(r, ((3.3-1)/(3.3+1))**2)
            self.assertAlmostEqual(r+t+a, 1.)
        film_n = np.sqrt(3.3)
        ar = EmissionBoundary(optics(1.), [Layer(optics(film_n), 900/(4*film_n))])
        self.assertLess(ar.probabilities(900, 3.3, 1, 's')[0], 1e-20)

    def test_tir_and_absorbing_film(self):
        bare = EmissionBoundary(optics(1.))
        self.assertEqual(bare.probabilities(900, 3.3, .5, 's'), (1., 0., 0.))
        lossy = EmissionBoundary(optics(1.), [Layer(optics(2., .2), 100)])
        for mu in (.5, 1.):
            for pol in ('s', 'p'):
                r, t, a = lossy.probabilities(900, 3.3, mu, pol)
                self.assertGreater(a, 0)
                self.assertAlmostEqual(r+t+a, 1.)

    def test_pump_absorbing_film_passive_and_matches_emission_normal(self):
        air, film = optics(1.), Layer(optics(2., .2), 100)
        pump = OpticalStack(air, [film], air).solve(900)
        boundary = EmissionBoundary(air, [film])
        r, t, a = boundary.probabilities(900, 1., 1., 's')
        self.assertGreater(pump.layer_absorptance[0], 0)
        self.assertLess(abs(pump.balance_error), 1e-12)
        np.testing.assert_allclose([pump.reflectance, pump.substrate_entry_fraction, pump.layer_absorptance[0]], [r, t, a], rtol=1e-12)

    def test_transparent_slab_trapped_fraction(self):
        fates = self.model(active=0).solve(.01, [0, .005, .01], self.spectrum)
        critical_mu = np.sqrt(1-1/3.3**2)
        np.testing.assert_allclose(fates.trapped, critical_mu, rtol=1e-12)
        np.testing.assert_allclose(fates.front_escape+fates.rear_escape, 1-critical_mu, rtol=1e-12)
        np.testing.assert_allclose(fates.balance_error, 0, atol=1e-12)

    def test_single_pass_surface_limit(self):
        # Thick absorbing wafer: no light returns from rear. At z=0 half of
        # all emission sees the front immediately; Fresnel escape integral.
        fates = self.model(active=1e6).solve(.01, [0], self.spectrum, angular_points=256)
        expected = single_pass_escape_probability(3.3, points=50001)
        np.testing.assert_allclose(fates.front_escape[0, 0], expected, rtol=3e-6)

    def test_index_matched_beer_lambert(self):
        boundary = EmissionBoundary(optics(3.3))
        fates = self.model(active=100, front=boundary, rear=boundary).solve(
            .01, [.002, .008], self.spectrum)
        from scipy.special import expn
        np.testing.assert_allclose(fates.front_escape[0], .5*expn(2, 100*np.array([.002, .008])), rtol=1e-9)
        np.testing.assert_allclose(fates.rear_escape[0], fates.front_escape[0, ::-1], rtol=1e-12)

    def test_spectral_conservation_parasitic_and_na(self):
        spectrum = EmissionSpectrum(np.linspace(850, 950, 7), np.ones(7))
        film = EmissionBoundary(optics(1.), [Layer(optics(1.7, .02), 20)])
        fates = self.model(active=100, passive=20, front=film).solve(
            .01, np.linspace(0, .01, 12), spectrum, front_na=.5)
        self.assertLess(np.max(abs(fates.balance_error)), 1e-12)
        self.assertTrue(np.all(fates.front_collected < fates.front_escape))
        self.assertTrue(np.all(fates.parasitic_absorption > 0))
        avg = fates.averaged()
        np.testing.assert_allclose(sum(avg[k] for k in
            ('front_escape', 'rear_escape', 'active_reabsorption', 'parasitic_absorption', 'trapped')), 1, atol=1e-12)
        flux = fates.emission_fluxes(np.full((3, 12), 1e20), np.full(12, .01/12))
        np.testing.assert_allclose(flux['internal'], 1e18)
        np.testing.assert_allclose(flux['net_radiative_loss'], flux['front_escape']+flux['rear_escape']+flux['parasitic_absorption'])

    def test_tiny_absorption_conservation(self):
        fates = self.model(active=1e-12).solve(.01, [.005], self.spectrum)
        self.assertLess(np.max(abs(fates.balance_error)), 1e-10)
        self.assertGreater(fates.active_reabsorption[0, 0], .9)

    def test_local_recycling_analytic_lifetime_and_inventory(self):
        grid = np.logspace(2, 18, 81)
        zero = np.zeros_like(grid)
        tau, pr = 2e-9, .8
        table = BulkTransportTable(grid, np.full_like(grid, 10), grid/tau, zero, zero)
        t = np.linspace(0, 40e-9, 101)
        result = TimeDependent1DSolver(table).solve(.001, t, 1e14, cells=12,
                                                   recycling_probability=pr, rtol=1e-8)
        np.testing.assert_allclose(result.average_delta_n_cm3, 1e14*np.exp(-(1-pr)*t/tau), rtol=1e-6)
        np.testing.assert_allclose(result.bulk_radiative_recombination_cm2_s,
                                   (1-pr)*result.internal_radiative_emission_cm2_s, rtol=1e-12)
        np.testing.assert_allclose(result.cumulative_radiative_cm2,
                                   result.cumulative_internal_radiative_cm2-result.cumulative_recycled_cm2, rtol=1e-12)
        self.assertLess(np.max(abs(result.inventory_relative_error)), 1e-7)
        # Complete regeneration: no carrier loss, but nonzero internal emission.
        full = TimeDependent1DSolver(table).solve(.001, t, 1e14, cells=12, recycling_probability=1.)
        np.testing.assert_allclose(full.average_delta_n_cm3, 1e14)
        np.testing.assert_array_equal(full.bulk_radiative_recombination_cm2_s, 0)

    def test_spatial_local_recycling_inventory(self):
        grid = np.logspace(2, 18, 81)
        zero = np.zeros_like(grid)
        table = BulkTransportTable(grid, np.full_like(grid, 10), grid/2e-9, grid/50e-9, zero)
        r = TimeDependent1DSolver(table).solve(.001, np.linspace(0, 5e-9, 61), 1e14,
            cells=16, recycling_probability=np.linspace(.2, .9, 16), rtol=1e-8)
        self.assertLess(np.max(abs(r.inventory_relative_error)), 1e-7)

    def test_cw_and_transient_use_same_recycling(self):
        grid = np.logspace(2, 18, 81)
        zero = np.zeros_like(grid)
        tau, pr, g = 2e-9, .8, 1e23
        table = BulkTransportTable(grid, np.full_like(grid, 10), grid/tau, zero, zero)
        steady = SteadyState1DSolver(table).solve(.001, g, 0, 0, cells=12,
                                                 recycling_probability=pr)
        self.assertTrue(steady.converged)
        np.testing.assert_allclose(steady.delta_n_cm3, g*tau/(1-pr), rtol=1e-8)
        self.assertAlmostEqual(steady.internal_radiative_yield, 1/(1-pr))
        result = TimeDependent1DSolver(table).solve(.001, np.linspace(0, 10e-9, 101),
            steady.delta_n_cm3, cells=12, generation=g, recycling_probability=pr, rtol=1e-8)
        np.testing.assert_allclose(result.average_delta_n_cm3, steady.average_delta_n_cm3, rtol=1e-7)

    def test_bulk_cancellation_free_low_injection(self):
        # Intrinsic populations exceed the tiny probe by many orders; direct
        # n*p-n0*p0 subtraction would spuriously report zero.
        bulk = BulkModel(Wafer('Ge'))
        s = bulk.equilibrium()
        delta = 1e-5
        injection = bulk.injection(delta)
        expected = bulk.material.b_rad_cm3_s*delta*(s.n0_cm3+s.p0_cm3+delta)
        self.assertGreater(injection.delta_r_rad_cm3_s, 0)
        self.assertAlmostEqual(injection.delta_r_rad_cm3_s/expected, 1)

    def test_invalid_inputs(self):
        for density in ([0], [-1], [np.nan]):
            with self.assertRaises(ValueError): EmissionSpectrum([900], density)
        with self.assertRaises(ValueError):
            self.model(active=-1).solve(.01, [.005], self.spectrum)
        with self.assertRaises(ValueError):
            self.model().solve(.01, [.02], self.spectrum)
        with self.assertRaises(ValueError):
            self.model().solve(.01, [.005], self.spectrum, front_na=2.)
        with self.assertRaises(ValueError):
            TimeDependent1DSolver(BulkTransportTable.linear_lifetime(10, 1e-9)).solve(
                .001, [0, 1e-9, 2e-9], 1e14, recycling_probability=1.1)


if __name__ == '__main__': unittest.main()
