"""Build the release-0.7 transient validation and illustrative InP notebook."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))

def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))

md(r"""
# Time-dependent transport and internal TRPL proxy — release 0.7

This notebook solves transient quasi-neutral ambipolar diffusion with nonlinear
bulk recombination and instantaneous surface electrostatics/SRH boundaries:

$$\frac{\partial\Delta n}{\partial t}
=\frac{\partial}{\partial x}\left(D_a\frac{\partial\Delta n}{\partial x}\right)
+G(x,t)-\Delta R_\mathrm{bulk}.$$

The TRPL proxy is the **excess internal radiative emission**
$I_\mathrm{int}(t)=\int\Delta R_\mathrm{rad}(x,t)\,dx$ in photons cm$^{-2}$ s$^{-1}$.
It is not yet the detected photon flux: spectral escape, reabsorption/photon
recycling, collection optics, and the instrument response are not implemented.
No biexponential fit or experimental lifetime reinterpretation is attempted.

Trap occupancy and surface potential are assumed to follow the local bulk-side
injection instantaneously. Space-charge-region transport and carrier/trap
storage remain omitted. The release-0.6 diagnostics help assess these limits.

After extracting a new source release, restart the Jupyter kernel before Run All
to clear cached imports.
""")
code(r"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

root = Path.cwd()
if not (root/'src').exists(): root = root.parent
sys.path.insert(0, str(root/'src'))
from surpass import (
    BulkModel, BulkTransportTable, TimeDependent1DSolver, SteadyState1DSolver,
    GaussianPulse, Wafer, InterfaceDefectModel, InterfaceChargeModel,
    SurfaceSRHModel, SurfaceBoundaryTable, OpticalStack, Layer,
    get_optical_material, reconstruct_band_profile, trap_relaxation_spectrum,
)
try:
    import scienceplots
    plt.style.use(['science', 'notebook', 'no-latex'])
except ImportError:
    plt.style.use('default')
plt.rcParams.update({'figure.dpi': 120, 'font.size': 10})
""")
md('## 1. Analytic decay and conservation check')
code(r"""
test_tau_s = 2e-9
test_times = np.linspace(0, 10e-9, 151)
test_table = BulkTransportTable.linear_lifetime(10, test_tau_s)
test_result = TimeDependent1DSolver(test_table).solve(
    0.001, test_times, 1e14, cells=16, rtol=1e-8)
exact = 1e14*np.exp(-test_times/test_tau_s)
relative_error = np.max(np.abs(test_result.average_delta_n_cm3/exact-1))
print(f'Uniform exponential relative error: {relative_error:.3g}')
print(f'Maximum inventory error: {np.max(np.abs(test_result.inventory_relative_error)):.3g}')
assert relative_error < 2e-6
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
axes[0].semilogy(test_times*1e9, test_result.average_delta_n_cm3/1e14, label='solver')
axes[0].semilogy(test_times*1e9, exact/1e14, '--', label='analytic')
axes[0].set(xlabel='Time (ns)', ylabel='Normalized sheet carrier density')
axes[0].legend()
axes[1].plot(test_times*1e9, test_result.inventory_relative_error)
axes[1].set(xlabel='Time (ns)', ylabel='Integrated inventory relative error')
fig.tight_layout()
plt.show()
""")
md(r"""
## 2. Editable wafer, excitation, and interface assumptions

`initialization='pulse'` treats the laser pulse as instantaneous: the initial
cell-averaged density is the absorbed photon dose (one pair per pump photon).
Use this only when pulse duration is short compared with relevant dynamics.
`cw_switch_off` starts from the calculated CW steady state and turns the pump
off at time zero. Finite-duration pumping is demonstrated separately below.
""")
code(r"""
initialization = 'pulse'  # 'pulse' or 'cw_switch_off'
thickness_um = 620.0
thickness_cm = thickness_um*1e-4
wavelength_nm = 514.0
pulse_fluence_j_cm2 = 1e-8
cw_intensity_w_cm2 = 1.0
tau_bulk_srh_s = 100e-9
rear_s_cm_s = 1e5
enable_bgn = False
include_interface_charge = True
qf_values_cm2 = [-4.2e12, 0.0, 4.2e12]
qf_colors = {-4.2e12: 'tab:blue', 0.0: 'black', 4.2e12: 'tab:red'}
times_s = np.r_[0, np.geomspace(1e-13, 20e-9, 301)]
rtol = 1e-6
atol_scaled = 1e-10
edges_cm = np.r_[0, np.geomspace(2e-8, 5e-4, 80),
                 np.linspace(5e-4, thickness_cm, 51)[1:]]

defects = InterfaceDefectModel(dit_mid_ev1_cm2=1e11, energy_points=151)
charge_model = InterfaceChargeModel(0.5, 0.02) if include_interface_charge else None
wafers = {
    'n-InP': Wafer('InP', donor_cm3=8.5e18, thickness_um=thickness_um),
    'p-InP': Wafer('InP', acceptor_cm3=5.1e18, thickness_um=thickness_um),
}
stack = OpticalStack(get_optical_material('air'), [
    Layer(get_optical_material('POx_demo'), 10, 'POx'),
    Layer(get_optical_material('AlOx_demo'), 10, 'AlOx'),
], get_optical_material('InP_demo'))
# Cell-averaged generation at unit incident intensity. Multiplication by
# fluence/(1 W/cm2) yields carriers cm^-3 after an ideal instantaneous pulse.
unit_generation = stack.generation_cell_average_cm3_s(wavelength_nm, 1.0, edges_cm)
initial_pulse_density = unit_generation*(pulse_fluence_j_cm2/1.0)
print('Illustrative initial peak cell density:', initial_pulse_density.max(), 'cm^-3')
print('Illustrative initial absorbed sheet dose:',
      np.dot(initial_pulse_density, np.diff(edges_cm)), 'cm^-2')
""")
md('## 3. Couple optical excitation, bulk losses, and nonlinear surface boundaries')
code(r"""
tables, surface_models, boundaries, results, initial_states, summary_rows = {}, {}, {}, {}, {}, []
for label, wafer in wafers.items():
    bulk = BulkModel(wafer, enable_bgn=enable_bgn, tau_srh_s=tau_bulk_srh_s)
    table = BulkTransportTable.from_bulk_model(bulk, 1.0, 1e20, points=161)
    tables[label] = table
    solver = TimeDependent1DSolver(table)
    for qf in qf_values_cm2:
        surface = SurfaceSRHModel(bulk, qf, defects, interface_charge=charge_model)
        boundary = SurfaceBoundaryTable.from_surface_model(surface, 1.0, 1e20, points=101)
        surface_models[label, qf], boundaries[label, qf] = surface, boundary
        if initialization == 'pulse':
            initial = initial_pulse_density
        elif initialization == 'cw_switch_off':
            steady = SteadyState1DSolver(table).solve(
                thickness_cm, unit_generation*cw_intensity_w_cm2, boundary,
                rear_s_cm_s, cell_edges_cm=edges_cm, max_nfev=1500)
            if not steady.converged: raise RuntimeError('CW initializer did not converge')
            initial = steady.delta_n_cm3
        else:
            raise ValueError('Select pulse or cw_switch_off')
        initial_states[label, qf] = initial.copy()
        result = solver.solve(thickness_cm, times_s, initial, boundary, rear_s_cm_s,
                               cell_edges_cm=edges_cm, rtol=rtol, atol_scaled=atol_scaled)
        results[label, qf] = result
        dose = result.initial_sheet_excess_cm2
        summary_rows.append({
            'wafer': label, 'Qext/q (1e12)': qf/1e12,
            'initial dose (cm^-2)': dose,
            'emitted internal photons / initial dose': result.cumulative_radiative_cm2[-1]/dose,
            'front loss fraction': result.cumulative_front_cm2[-1]/dose,
            'rear loss fraction': result.cumulative_rear_cm2[-1]/dose,
            'remaining carrier fraction': result.sheet_excess_cm2[-1]/dose,
            'max inventory error': np.max(np.abs(result.inventory_relative_error)),
            'integrator steps': result.integrator_steps,
        })
        print(f'{label}, Qext/q={qf/1e12:+.1f}e12 complete')
summary = pd.DataFrame(summary_rows)
with pd.option_context('display.float_format', lambda value: f'{value:.3g}'):
    display(summary)
assert summary['max inventory error'].max() < 1e-3
""")
md(r"""
## 4. Internal emission decay versus effective carrier lifetime

$N_\mathrm{sheet}/R_\mathrm{loss}$ and $-1/(d\ln I_\mathrm{int}/dt)$ are different
quantities. Their equality requires approximately linear emission and decay.
Diffusion, nonlinear loss, surface boundary supply, and high-injection emission
can make the local PL slope time dependent without implying two independent
material lifetimes. Late signals below $10^{-6}$ of the peak are excluded from
the slope diagnostic by default.
""")
code(r"""
fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
for row, label in enumerate(wafers):
    for qf in qf_values_cm2:
        r = results[label, qf]
        color = qf_colors[qf]
        axes[row, 0].semilogy(r.time_s*1e9, r.normalized_radiative_flux(),
                              color=color, label=f'Qext/q={qf/1e12:+.1f}e12')
        axes[row, 1].plot(r.time_s*1e9, r.local_pl_decay_time_s()*1e9, color=color,
                          label='local PL slope' if qf == 0 else None)
        axes[row, 1].plot(r.time_s*1e9, r.effective_lifetime_s*1e9, '--', color=color,
                          label='N / total loss' if qf == 0 else None)
    axes[row, 0].set(title=label, ylabel='Normalized internal radiative flux',
                     ylim=(1e-6, 1.1), xlim=(0, 5))
    axes[row, 1].set(title=label, ylabel='Time constant (ns)', ylim=(0, 3), xlim=(0, 5))
axes[1, 0].set_xlabel('Time after pump off (ns)')
axes[1, 1].set_xlabel('Time after pump off (ns)')
axes[0, 0].legend(fontsize=8)
axes[0, 1].legend(fontsize=8)
fig.tight_layout()
output = root/'examples'/'output'
output.mkdir(parents=True, exist_ok=True)
fig.savefig(output/'v07_internal_trpl.png', dpi=150)
plt.show()
""")
md('## 5. Carrier redistribution and inventory balance')
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
reference = results['n-InP', 0.0]
for time_ns in [0, 0.05, 0.2, 1.0, 3.0]:
    index = np.argmin(abs(reference.time_s-time_ns*1e-9))
    axes[0].semilogy(reference.depth_cm*1e4, np.maximum(reference.delta_n_cm3[index], 1),
                      label=f'{reference.time_s[index]*1e9:.2g} ns')
axes[0].set(xlim=(0, 15), ylim=(1e5, reference.delta_n_cm3.max()*2),
             xlabel='Depth (um)', ylabel='Excess carrier density (cm$^{-3}$)')
axes[0].legend(fontsize=8)
for key, r in results.items():
    axes[1].plot(r.time_s*1e9, r.inventory_relative_error, label=f'{key[0]}, {key[1]/1e12:+.1f}')
axes[1].set(xlabel='Time (ns)', ylabel='Integrated inventory relative error')
fig.tight_layout()
plt.show()
""")
md('## 6. Tolerance and spatial-grid checks')
code(r"""
key = ('n-InP', 0.0)
reference = results[key]
fine_time = TimeDependent1DSolver(tables[key[0]]).solve(
    thickness_cm, times_s, initial_states[key], boundaries[key], rear_s_cm_s,
    cell_edges_cm=edges_cm, rtol=rtol/10, atol_scaled=atol_scaled/10, balance_order=5)
fine_edges = np.sort(np.r_[edges_cm, (edges_cm[1:]+edges_cm[:-1])/2])
fine_initial = stack.generation_cell_average_cm3_s(wavelength_nm, 1.0, fine_edges)*pulse_fluence_j_cm2
if initialization == 'cw_switch_off':
    fine_generation = stack.generation_cell_average_cm3_s(wavelength_nm, cw_intensity_w_cm2, fine_edges)
    fine_steady = SteadyState1DSolver(tables[key[0]]).solve(
        thickness_cm, fine_generation, boundaries[key], rear_s_cm_s,
        cell_edges_cm=fine_edges, max_nfev=1500)
    if not fine_steady.converged: raise RuntimeError('Fine-grid CW initializer failed')
    fine_initial = fine_steady.delta_n_cm3
fine_grid = TimeDependent1DSolver(tables[key[0]]).solve(
    thickness_cm, times_s, fine_initial, boundaries[key], rear_s_cm_s,
    cell_edges_cm=fine_edges, rtol=rtol, atol_scaled=atol_scaled)
for label, comparison in [('tighter tolerances', fine_time), ('twice as many cells', fine_grid)]:
    valid = reference.normalized_radiative_flux() > 1e-4
    deviation = np.max(np.abs(comparison.bulk_radiative_recombination_cm2_s[valid]
                             /reference.bulk_radiative_recombination_cm2_s[valid]-1))
    print(f'{label}: maximum emission-trace difference above threshold = {deviation:.3g}')
    if deviation > 0.02: print('WARNING: refine the numerical model further for quantitative use')
""")
md('## 7. Local electrostatic/trap checks along a transient')
code(r"""
key = ('n-InP', 0.0)
r = results[key]
surface = surface_models[key]
check_rows = []
for time_ns in [0, 0.2, 1.0]:
    i = np.argmin(abs(r.time_s-time_ns*1e-9))
    injection = r.front_surface_delta_cm3[i]
    profile = reconstruct_band_profile(surface, injection, points=601)
    spectrum = trap_relaxation_spectrum(surface, injection)
    width = max(profile.potential_99_depth_cm, profile.charge_99_depth_cm)
    check_rows.append({'time (ns)': r.time_s[i]*1e9,
                       'surface-side injection (cm^-3)': injection,
                       'space-charge extent (nm)': width*1e7,
                       'max local trap tau (ns)': spectrum.relaxation_time_s.max()*1e9,
                       'Dit fraction slower than 1 ns': spectrum.slow_trap_fraction(1e-9)})
with pd.option_context('display.float_format', lambda value: f'{value:.3g}'):
    display(pd.DataFrame(check_rows))
""")
md(r"""
## 8. A finite Gaussian pulse

Here the solver starts dark and the Gaussian generates a prescribed carrier
dose. `sigma_s` is the intensity-time standard deviation, not FWHM
($\mathrm{FWHM}=2\sqrt{2\ln2}\sigma$). The pulse model automatically limits
integration steps to $\sigma/2$; this remains robust but costs extra steps
over a long post-pulse window. Generic pump functions need an explicit
`max_step_s` or segmented integration at known changes.
""")
code(r"""
pulse_times = np.linspace(0, 8e-9, 301)
pulse = GaussianPulse(np.full(16, 1e14), center_s=2e-9, sigma_s=0.2e-9)
pulse_result = TimeDependent1DSolver(test_table).solve(
    0.001, pulse_times, 0.0, generation=pulse, cells=16)
fig, ax = plt.subplots(figsize=(6, 3.3))
ax.plot(pulse_times*1e9, pulse_result.average_delta_n_cm3/1e14)
ax.set(xlabel='Time (ns)', ylabel='Average excess / full pulse-generated density')
fig.tight_layout()
plt.show()
print('Generated dose / nominal full-pulse dose:', pulse_result.cumulative_generated_cm2[-1]/1e11)
print('Maximum inventory error:', np.max(np.abs(pulse_result.inventory_relative_error)))
""")
md(r"""
## Limits and next validation priorities

- Numerical convergence does not validate the assumed material/interface parameters.
- Internal radiative emission is a first TRPL proxy, not a predicted detector count.
  No photon recycling, spectral escape, collection optics, IRF, or periodic pulse train.
- Instantaneous surface charge and trap occupancy are local steady-state closures.
  Trap-time checks do not include electrostatic feedback and do not prove validity.
- The spatial band profile is not stitched into the transient bulk solution;
  storage, generation, and recombination inside the SCR are not explicitly resolved.
- A flat quasineutral bulk is not an exact complete illuminated band diagram.
- The inventory diagnostic independently integrates all loss/generation channels
  by Gaussian quadrature along accepted BDF steps. It is normalized by initial
  plus generated sheet dose, not by the vanishing late-time carrier inventory.
- Table-range violations raise errors rather than silently accepting a clipped
  high-injection solution. Small negative numerical densities are exposed in
  `minimum_raw_delta_cm3`; significant negatives raise errors.
- As a first quantitative fitting step, constrain pulse fluence/width, absorption,
  bulk B/Auger/SRH coefficients, and rear boundary before fitting Dit/Qf.

Numerical reference: SciPy `solve_ivp` BDF/Radau documentation,
https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html.
All physical parameter references are inherited from the bulk, optical and
interface models; this release introduces no new material constants.
""")

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3'},
})
path = ROOT/'notebooks'/'07_transient_transport_trpl.ipynb'
nbf.write(notebook, path)
print(path)
