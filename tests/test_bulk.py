import unittest

import numpy as np

from surpass import (
    BulkModel, Layer, OpticalStack, Wafer, get_material,
    get_optical_material, single_pass_escape_probability,
    BulkTransportTable, SteadyState1DSolver,
    InterfaceDefectModel, SurfaceSRHModel, SurfaceBoundaryTable,
    InterfaceChargeModel, FixedChargeBoundary,
)
from surpass.fermi import fermi


class TestBulkPhysics(unittest.TestCase):
    def test_fermi_boltzmann_limit(self):
        for order in (-0.5, 0.5, 1.5):
            self.assertAlmostEqual(fermi(order, -15.0) / np.exp(-15.0), 1.0, places=5)

    def test_charge_neutrality(self):
        for wafer in (
            Wafer("InP", donor_cm3=8.5e18),
            Wafer("InP", acceptor_cm3=5.1e18),
            Wafer("Si", donor_cm3=1e16),
            Wafer("Ge", acceptor_cm3=1e17),
            Wafer("GaAs", donor_cm3=1e18),
        ):
            state = BulkModel(wafer).equilibrium()
            residual = state.n0_cm3 - state.p0_cm3 - wafer.net_doping_cm3
            self.assertLess(abs(residual), max(abs(wafer.net_doping_cm3), 1) * 1e-9)

    def test_generalized_screening_is_positive(self):
        state = BulkModel(Wafer("InP", donor_cm3=1e19)).equilibrium()
        self.assertGreater(state.debye_length_nm, 0)
        self.assertGreater(state.fermi_screening_length_nm, 0)

    def test_recombination_lifetimes(self):
        state = BulkModel(
            Wafer("InP", donor_cm3=8.5e18), tau_srh_s=10e-9
        ).injection(1e15)
        self.assertTrue(np.isfinite(state.tau_total_s))
        self.assertGreater(state.diffusion_length_um, 0)
        self.assertLessEqual(state.tau_total_s, state.tau_rad_s)
        self.assertLessEqual(state.tau_total_s, state.tau_auger_s)

    def test_bgn_reduces_gap(self):
        wafer = Wafer("InP", donor_cm3=1e19)
        off = BulkModel(wafer, enable_bgn=False).equilibrium()
        on = BulkModel(wafer, enable_bgn=True).equilibrium()
        self.assertGreater(on.delta_eg_bgn_ev, 0)
        self.assertLess(on.eg_effective_ev, off.eg_effective_ev)

    def test_material_lookup(self):
        self.assertEqual(get_material("inp").symbol, "InP")

    def test_bare_interface_fresnel(self):
        air = get_optical_material("air")
        inp = get_optical_material("InP_demo")
        result = OpticalStack(air, [], inp).solve(514.0)
        index = inp.nk(514.0)
        expected = abs((1.0 - index) / (1.0 + index)) ** 2
        self.assertAlmostEqual(result.reflectance, expected, places=12)
        self.assertLess(abs(result.balance_error), 1e-12)

    def test_generation_integral(self):
        air = get_optical_material("air")
        inp = get_optical_material("InP_demo")
        stack = OpticalStack(air, [], inp)
        result = stack.solve(514.0)
        z = np.linspace(0, 20 / result.substrate_alpha_cm1, 100001)
        generation = stack.generation_profile_cm3_s(514.0, 1.0, z)
        absorbed_flux = np.trapezoid(generation, z)
        from scipy.constants import c, h
        entering_flux = result.substrate_entry_fraction / (h*c/(514e-9))
        self.assertAlmostEqual(absorbed_flux / entering_flux, 1.0, places=7)

    def test_generation_cell_average(self):
        stack = OpticalStack(get_optical_material("air"), [], get_optical_material("InP_demo"))
        edges = np.r_[0.0, np.geomspace(1e-9, 0.01, 500)]
        generation = stack.generation_cell_average_cm3_s(514.0, 1.0, edges)
        integrated = np.sum(generation*np.diff(edges))
        result = stack.solve(514.0)
        from scipy.constants import c, h
        entering = result.substrate_entry_fraction/(h*c/(514e-9))
        self.assertAlmostEqual(integrated/entering, 1.0, places=10)

    def test_escape_probability_bounds(self):
        value = single_pass_escape_probability(3.25)
        self.assertGreater(value, 0.0)
        self.assertLess(value, 0.5)

    def test_transport_uniform_no_surface(self):
        table = BulkTransportTable.linear_lifetime(10.0, 1e-6)
        result = SteadyState1DSolver(table).solve(
            thickness_cm=0.01, generation=1e20,
            front_s_cm_s=0.0, rear_s_cm_s=0.0, cells=80,
        )
        self.assertTrue(result.converged)
        self.assertLess(abs(result.average_delta_n_cm3/1e14 - 1.0), 1e-6)
        self.assertLess(abs(result.relative_balance_error), 1e-8)

    def test_transport_symmetric_analytic_solution(self):
        diffusion, lifetime = 12.0, 2e-6
        generation, width, surface = 2e20, 0.006, 3e3
        table = BulkTransportTable.linear_lifetime(diffusion, lifetime)
        result = SteadyState1DSolver(table).solve(
            width, generation, surface, surface, cells=240,
        )
        length = np.sqrt(diffusion*lifetime)
        amplitude = -(surface*generation*lifetime) / (
            diffusion/length*np.sinh(width/(2*length))
            + surface*np.cosh(width/(2*length))
        )
        analytic = generation*lifetime + amplitude*np.cosh(
            (result.depth_cm-width/2)/length
        )
        relative_rms = np.sqrt(np.mean(((result.delta_n_cm3-analytic)/analytic)**2))
        self.assertTrue(result.converged)
        self.assertLess(relative_rms, 2e-4)
        self.assertLess(abs(result.relative_balance_error), 2e-7)

    def test_transport_nonuniform_grid_conservation(self):
        width = 0.062
        edges = np.r_[0.0, np.geomspace(1e-8, 5e-4, 100),
                      np.linspace(5e-4, width, 121)[1:]]
        table = BulkTransportTable.linear_lifetime(20.0, 1e-7)
        result = SteadyState1DSolver(table).solve(
            width, 3e19, 1e3, 1e4, cell_edges_cm=edges,
        )
        self.assertTrue(result.converged)
        self.assertLess(abs(result.relative_balance_error), 1e-7)

    def test_surface_equilibrium_and_charge_sign(self):
        bulk = BulkModel(Wafer("InP", donor_cm3=8.5e18))
        defects = InterfaceDefectModel(energy_points=101)
        negative = SurfaceSRHModel(bulk, -4.2e12, defects)
        flat = SurfaceSRHModel(bulk, 0.0, defects)
        positive = SurfaceSRHModel(bulk, 4.2e12, defects)
        self.assertEqual(flat.state(0.0).recombination_cm2_s, 0.0)
        self.assertLess(negative.state(0.0).psi_surface_v, 0.0)
        self.assertGreater(positive.state(0.0).psi_surface_v, 0.0)

    def test_fixed_charge_field_effect_trend(self):
        defects = InterfaceDefectModel(energy_points=101)
        n_bulk = BulkModel(Wafer("InP", donor_cm3=8.5e18))
        p_bulk = BulkModel(Wafer("InP", acceptor_cm3=5.1e18))
        n_negative = SurfaceSRHModel(n_bulk, -4.2e12, defects).state(1e11)
        n_positive = SurfaceSRHModel(n_bulk, 4.2e12, defects).state(1e11)
        p_negative = SurfaceSRHModel(p_bulk, -4.2e12, defects).state(1e11)
        p_positive = SurfaceSRHModel(p_bulk, 4.2e12, defects).state(1e11)
        self.assertGreater(n_negative.recombination_cm2_s, n_positive.recombination_cm2_s)
        self.assertLess(p_negative.recombination_cm2_s, p_positive.recombination_cm2_s)

    def test_nonlinear_surface_boundary_transport(self):
        defects = InterfaceDefectModel(energy_points=101)
        surface = SurfaceSRHModel(
            BulkModel(Wafer("InP", donor_cm3=8.5e18)), 4.2e12, defects
        )
        boundary = SurfaceBoundaryTable.from_surface_model(
            surface, 1e6, 1e18, points=61
        )
        bulk_table = BulkTransportTable.linear_lifetime(
            10.0, 1e-6, delta_min_cm3=1e6, delta_max_cm3=1e18
        )
        result = SteadyState1DSolver(bulk_table).solve(
            0.01, 1e20, boundary, 0.0, cells=80, max_nfev=1000
        )
        self.assertTrue(result.converged)
        self.assertGreater(result.front_surface_recombination_cm2_s, 0.0)
        self.assertLess(abs(result.relative_balance_error), 1e-5)

    def test_interface_charge_disabled_recovers_fixed_charge_boundary(self):
        bulk = BulkModel(Wafer("InP", donor_cm3=8.5e18))
        defects = InterfaceDefectModel(energy_points=101)
        legacy = SurfaceSRHModel(bulk, 2.5e12, defects).state(1e12)
        modular = SurfaceSRHModel(
            bulk, None, defects,
            external_boundary=FixedChargeBoundary(2.5e12),
        ).state(1e12)
        self.assertAlmostEqual(legacy.psi_surface_v, modular.psi_surface_v, places=12)
        self.assertAlmostEqual(legacy.recombination_cm2_s,
                               modular.recombination_cm2_s, places=5)
        self.assertEqual(legacy.q_it_number_cm2, 0.0)

    def test_amphoteric_interface_charge_balance_and_pinning(self):
        bulk = BulkModel(Wafer("InP", donor_cm3=8.5e18))
        charge = InterfaceChargeModel(charge_neutrality_level_fraction=0.5)
        states = []
        for dit in (1e11, 1e15):
            model = SurfaceSRHModel(
                bulk, 4e12,
                InterfaceDefectModel(dit_mid_ev1_cm2=dit, energy_points=201),
                interface_charge=charge,
            )
            state = model.state(0.0)
            states.append((model, state))
            self.assertLess(abs(state.charge_balance_number_cm2), 100.0)
            self.assertEqual(state.recombination_cm2_s, 0.0)
        low_model, low = states[0]
        high_model, high = states[1]
        self.assertGreater(high.trap_control_fraction, low.trap_control_fraction)
        surface_fermi_ev = high_model.eg + high.eta_n_surface*high_model.vt
        cnl_ev = 0.5*high_model.eg
        self.assertLess(abs(surface_fermi_ev-cnl_ev), 0.03)

    def test_potential_dependent_external_boundary_hook(self):
        class LinearGateBoundary:
            def __init__(self, capacitance_f_cm2, voltage_v):
                self.capacitance = capacitance_f_cm2
                self.voltage = voltage_v

            def charge_c_cm2(self, psi_surface_v):
                return self.capacitance*(self.voltage-psi_surface_v)

        model = SurfaceSRHModel(
            BulkModel(Wafer("InP", donor_cm3=1e17)), None,
            InterfaceDefectModel(energy_points=101),
            external_boundary=LinearGateBoundary(1e-7, 0.25),
        )
        state = model.state(0.0)
        self.assertLess(abs(state.charge_balance_number_cm2), 100.0)
        self.assertGreater(state.psi_surface_v, 0.0)


if __name__ == "__main__":
    unittest.main()
