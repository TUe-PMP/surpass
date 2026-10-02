"""Independent physical and numerical checks for release-0.6 band profiles."""

import unittest
import numpy as np
from surpass import (BulkModel, Wafer, SurfaceSRHModel, InterfaceDefectModel,
                       InterfaceChargeModel, reconstruct_band_profile,
                       trap_relaxation_spectrum)
from surpass.constants import EPS0_F_CM, Q_C


class TestSpatialProfiles(unittest.TestCase):
    def model(self, charge, polarity="n", doping=1e16, traps=False):
        wafer = Wafer("InP", donor_cm3=doping if polarity == "n" else 0,
                      acceptor_cm3=doping if polarity == "p" else 0)
        return SurfaceSRHModel(
            BulkModel(wafer), charge, InterfaceDefectModel(energy_points=101),
            interface_charge=InterfaceChargeModel() if traps else None)

    def test_flat_band_profile(self):
        m = self.model(0)
        p = reconstruct_band_profile(m)
        np.testing.assert_array_equal(p.psi_v, 0)
        self.assertEqual(p.charge_99_depth_cm, 0)
        self.assertEqual(p.integrated_q_sc_c_cm2, 0)
        np.testing.assert_allclose(p.efn_ev, p.efp_ev, atol=1e-12)

    def test_small_signal_exponential_screening(self):
        m = self.model(1e7)
        p = reconstruct_band_profile(m, points=401)
        expected = p.psi_v[0]*np.exp(-p.depth_cm/p.screening_length_cm)
        np.testing.assert_allclose(p.psi_v, expected, rtol=2e-4)
        self.assertAlmostEqual(p.screening_length_cm*1e7,
                               m.bulk.fermi_screening_length_nm, places=7)

    def test_gauss_charge_and_band_endpoints(self):
        for polarity, charge in [("n", -4.2e12), ("n", 4.2e12),
                                 ("p", -4.2e12), ("p", 4.2e12)]:
            m = self.model(charge, polarity, doping=5e18, traps=True)
            s = m.state(1e11)
            p = reconstruct_band_profile(m, 1e11, points=601)
            self.assertAlmostEqual(p.psi_v[0], s.psi_surface_v, places=12)
            self.assertAlmostEqual(p.n_cm3[0]/s.n_surface_cm3, 1, places=10)
            self.assertAlmostEqual(p.p_cm3[0]/s.p_surface_cm3, 1, places=10)
            np.testing.assert_allclose(p.ec_ev-p.ev_ev, m.eg, atol=1e-14)
            np.testing.assert_allclose(p.efn_ev, p.ec_ev+m.vt*(s.eta_n_bulk+p.psi_v/m.vt))
            np.testing.assert_allclose(p.efp_ev, p.ev_ev-m.vt*(s.eta_p_bulk-p.psi_v/m.vt))
            self.assertTrue(np.all(np.diff(p.depth_cm) > 0))
            self.assertLess(abs(p.psi_v[-1]/p.psi_v[0]), 1.01e-6)
            self.assertLess(abs(p.charge_integral_relative_error), 1e-3)
            eps = m.material.eps_r*EPS0_F_CM
            self.assertAlmostEqual(-eps*p.electric_field_v_cm[0]/Q_C,
                                   s.q_sc_number_cm2, delta=1.0)

    def test_poisson_differential_identity(self):
        p = reconstruct_band_profile(self.model(-1e12), points=801)
        eps = 12.5*EPS0_F_CM
        derivative = np.gradient(p.electric_field_v_cm, p.depth_cm)
        significant = np.abs(p.rho_c_cm3) > np.max(np.abs(p.rho_c_cm3))*0.02
        significant[:3] = significant[-3:] = False
        np.testing.assert_allclose(derivative[significant],
                                   p.rho_c_cm3[significant]/eps, rtol=0.005)

    def test_profile_grid_refinement(self):
        m = self.model(-4.2e12, doping=1e15)
        a = reconstruct_band_profile(m, 1e11, points=401)
        b = reconstruct_band_profile(m, 1e11, points=801)
        self.assertLess(abs(a.potential_99_depth_cm/b.potential_99_depth_cm-1), 3e-4)
        self.assertLess(abs(b.charge_integral_relative_error),
                        abs(a.charge_integral_relative_error))

    def test_trap_relaxation_occupancy_and_rate(self):
        m = self.model(-4.2e12, traps=True)
        s = m.state(1e11)
        spectrum = trap_relaxation_spectrum(m, 1e11)
        np.testing.assert_allclose(spectrum.occupancy,
                                  m._trap_occupancy(s.psi_surface_v/m.vt,
                                                    s.eta_n_bulk, s.eta_p_bulk))
        total_rate = (spectrum.electron_capture_s1+spectrum.hole_capture_s1
                      + spectrum.electron_emission_s1+spectrum.hole_emission_s1)
        np.testing.assert_allclose(spectrum.relaxation_time_s*total_rate, 1)
        self.assertGreaterEqual(spectrum.slow_trap_fraction(1e-9), 0)
        self.assertLessEqual(spectrum.slow_trap_fraction(1e-9), 1)

    def test_invalid_profile_options(self):
        m = self.model(1e12)
        for options in [{"points": 10}, {"tail_fraction": 0},
                        {"tail_fraction": 0.1}]:
            with self.assertRaises(ValueError):
                reconstruct_band_profile(m, **options)
        frozen = SurfaceSRHModel(m.bulk_model, 1e12, m.defects,
                                  electrostatics="equilibrium")
        with self.assertRaises(ValueError):
            reconstruct_band_profile(frozen, 1e11)


if __name__ == "__main__":
    unittest.main()
