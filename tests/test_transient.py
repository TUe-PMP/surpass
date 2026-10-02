"""Transient analytic limits, nonlinear boundary and inventory conservation."""
import unittest
import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr
from surpass import (BulkTransportTable, TimeDependent1DSolver, GaussianPulse,
                       BulkModel, Wafer, InterfaceDefectModel, SurfaceSRHModel,
                       SurfaceBoundaryTable)


class TestTransient(unittest.TestCase):
    def test_uniform_exponential_decay(self):
        tau = 2e-9
        solver = TimeDependent1DSolver(BulkTransportTable.linear_lifetime(10, tau))
        t = np.linspace(0, 8e-9, 101)
        result = solver.solve(0.01, t, 1e14, cells=16, rtol=1e-8)
        np.testing.assert_allclose(result.average_delta_n_cm3, 1e14*np.exp(-t/tau), rtol=1e-6)
        self.assertLess(np.max(np.abs(result.inventory_relative_error)), 1e-7)

    def test_reflecting_diffusion_mode(self):
        width, diffusion, tau = 0.001, 10, 1e-8
        edges = np.linspace(0, width, 101)
        centers = (edges[:-1]+edges[1:])/2
        t = np.linspace(0, 8e-9, 61)
        initial = 1e14*(1+0.4*np.cos(np.pi*centers/width))
        result = TimeDependent1DSolver(BulkTransportTable.linear_lifetime(diffusion, tau)).solve(
            width, t, initial, cell_edges_cm=edges, rtol=1e-8)
        expected = 1e14*np.exp(-t[:, None]/tau)*(1+0.4*np.cos(np.pi*centers/width)[None, :]
                   * np.exp(-diffusion*(np.pi/width)**2*t[:, None]))
        np.testing.assert_allclose(result.delta_n_cm3, expected, rtol=3e-5)

    def test_finite_surface_eigenmode(self):
        width, diffusion, tau, surface = 0.001, 10, 1e-8, 1e5
        k = brentq(lambda k: k*np.tan(k*width/2)-surface/diffusion, 1, np.pi/width*0.999)
        edges = np.linspace(0, width, 161)
        x = (edges[:-1]+edges[1:])/2
        t = np.linspace(0, 8e-9, 61)
        initial = 1e14*np.cos(k*(x-width/2))
        result = TimeDependent1DSolver(BulkTransportTable.linear_lifetime(diffusion, tau)).solve(
            width, t, initial, surface, surface, cell_edges_cm=edges, rtol=1e-8)
        expected = initial[None, :]*np.exp(-(1/tau+diffusion*k*k)*t[:, None])
        np.testing.assert_allclose(result.delta_n_cm3, expected, rtol=2e-4)
        self.assertLess(np.max(np.abs(result.inventory_relative_error)), 1e-7)

    def test_darkness_and_continuous_generation(self):
        tau, g = 2e-9, 1e23
        solver = TimeDependent1DSolver(BulkTransportTable.linear_lifetime(10, tau))
        t = np.linspace(0, 10e-9, 61)
        dark = solver.solve(0.001, t, 0.0, cells=12)
        np.testing.assert_array_equal(dark.delta_n_cm3, 0)
        driven = solver.solve(0.001, t, 0.0, generation=g, cells=12, rtol=1e-8)
        np.testing.assert_allclose(driven.average_delta_n_cm3, g*tau*(-np.expm1(-t/tau)), rtol=1e-6)
        self.assertLess(np.max(np.abs(driven.inventory_relative_error)), 1e-7)

    def test_gaussian_pulse_dose_and_decay(self):
        tau, sigma, center, density = 2e-9, 0.5e-9, 4e-9, 1e14
        pulse = GaussianPulse(np.full(12, density), center, sigma)
        t = np.linspace(0, 10e-9, 101)
        solver = TimeDependent1DSolver(BulkTransportTable.linear_lifetime(10, tau))
        result = solver.solve(0.001, t, 0.0, generation=pulse, cells=12, rtol=1e-8)
        expected = density*np.exp(sigma*sigma/(2*tau*tau)-(t-center)/tau)*ndtr((t-center-sigma*sigma/tau)/sigma)
        np.testing.assert_allclose(result.average_delta_n_cm3, expected, rtol=2e-5, atol=density*1e-8)
        self.assertAlmostEqual(result.cumulative_generated_cm2[-1]/(density*0.001), 1, places=6)
        self.assertLess(np.max(np.abs(result.inventory_relative_error)), 1e-6)

    def test_nonlinear_radiative_decay_and_local_pl_slope(self):
        grid = np.logspace(2, 18, 81)
        b, density = 1e-8, 1e15
        zero = np.zeros_like(grid)
        table = BulkTransportTable(grid, np.full_like(grid, 10), b*grid**2, zero, zero)
        t = np.linspace(0, 200e-9, 301)
        result = TimeDependent1DSolver(table).solve(0.001, t, density, cells=12, rtol=1e-8)
        expected = density/(1+b*density*t)
        np.testing.assert_allclose(result.average_delta_n_cm3, expected, rtol=1e-6)
        np.testing.assert_allclose(result.local_pl_decay_time_s()[2:-2],
                                   0.5/(b*expected[2:-2]), rtol=2e-4)

    def test_nonlinear_surface_and_nonuniform_grid(self):
        model = SurfaceSRHModel(BulkModel(Wafer('InP', donor_cm3=8.5e18)), 4.2e12,
                                InterfaceDefectModel(energy_points=101))
        boundary = SurfaceBoundaryTable.from_surface_model(model, 1e3, 1e18, points=41)
        edges = np.r_[0, np.geomspace(1e-7, 0.001, 41)]
        table = BulkTransportTable.linear_lifetime(10, 2e-9)
        result = TimeDependent1DSolver(table).solve(0.001, np.linspace(0, 5e-9, 81),
                                                   1e14, boundary, 0.0, cell_edges_cm=edges)
        self.assertGreater(result.front_surface_recombination_cm2_s[0], 0)
        self.assertLess(np.max(np.abs(result.inventory_relative_error)), 2e-5)
        self.assertTrue(np.all(result.delta_n_cm3 >= 0))

    def test_invalid_inputs_and_table_range(self):
        table = BulkTransportTable.linear_lifetime(10, 1e-9, delta_max_cm3=1e16)
        solver = TimeDependent1DSolver(table)
        t = np.linspace(0, 1e-9, 10)
        for initial in [-1.0, np.nan, 1e17]:
            with self.assertRaises(ValueError):
                solver.solve(0.001, t, initial)
        with self.assertRaises(ValueError):
            solver.solve(0.001, t[::-1], 1e14)
        with self.assertRaises(ValueError):
            solver.solve(0.001, t, 0.0, generation=1e28, cells=12)


if __name__ == '__main__':
    unittest.main()
