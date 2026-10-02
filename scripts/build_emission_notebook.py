"""Build release-0.8 validation and transparent optical-recycling examples."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []
def md(text): cells.append(nbf.v4.new_markdown_cell(text.strip()))
def code(text): cells.append(nbf.v4.new_code_cell(text.strip()))

md(r"""
# Spectral escape and local photon recycling — v0.8

This notebook separates **intrinsic emission**, **net radiative carrier loss**,
**active reabsorption**, **parasitic absorption**, **escaped photons**, and
**NA-selected photon flux**. Coherent surface films are combined with
incoherent repeated wafer traversals and isotropic unpolarized emission.

Reabsorption returns carriers to their **emission cell**: a local instantaneous
approximation, not a nonlocal optical transport solver. Fixed internal spectrum
and absorption inputs are used throughout each transient. The optical inputs
below are synthetic; the results are demonstrations, not fitted InP predictions.
See `BULK_RECOMBINATION_AUDIT.md` for provenance and unresolved physics.

Restart your kernel after replacing the complete source package. No experimental
biexponential fit or lifetime reinterpretation is attempted.
""")
code(r"""
import sys
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
root = Path.cwd()
if not (root/'src').exists(): root = root.parent
sys.path.insert(0, str(root/'src'))
import surpass
print('surpass version:', getattr(surpass, '__version__', 'pre-0.8.1'))
print('surpass loaded from:', Path(surpass.__file__).resolve())
try:
    loaded_version = tuple(int(x) for x in surpass.__version__.split('.')[:3])
except (AttributeError, TypeError, ValueError):
    loaded_version = (0, 0, 0)
if loaded_version < (0, 8, 1):
    raise ImportError(
        'Notebook 08 requires surpass 0.8.1 or newer. The path printed above is an older source tree. '
        'Replace the project-root src folder with the one from the flat v0.8.1 archive, then restart the kernel.'
    )
from surpass import (
    BulkModel, Wafer, get_material, BulkTransportTable, TimeDependent1DSolver,
    SteadyState1DSolver, SurfaceSRHModel, InterfaceDefectModel,
    InterfaceChargeModel, SurfaceBoundaryTable, OpticalStack, Layer,
    get_optical_material, EmissionSpectrum, EmissionBoundary, SlabEmissionModel,
)
try:
    import scienceplots
    plt.style.use(['science', 'notebook', 'no-latex'])
except ImportError:
    plt.style.use('default')
plt.rcParams.update({'figure.dpi': 120, 'font.size': 10})
""")
md(r"""
## 1. Bulk checkpoint: lifetime channels and coefficient sensitivity

The InP coefficients remain representative development constants. FD carrier
statistics do not make constant B/C rates fully degenerate. Sensitivity factors
of 1/3 and 3 are illustrative brackets, **not experimental uncertainty bounds**.
Lifetimes shown are Δn/ΔR, not 1/(dR/dΔn).

Relevant n-InP study: [Semyonov et al.](https://arxiv.org/pdf/1003.6095).
It measures about 2–8e18 cm⁻³; do not transfer its fits to p-InP automatically.
""")
code(r"""
thickness_cm = 620e-4
tau_srh_s = 100e-9
wafers = {
    'n-InP': Wafer('InP', donor_cm3=8.5e18, thickness_um=620),
    'p-InP': Wafer('InP', acceptor_cm3=5.1e18, thickness_um=620),
}
probe = 1e12
rows = []
for label, wafer in wafers.items():
    model = BulkModel(wafer, tau_srh_s=tau_srh_s)
    s, inj = model.equilibrium(), model.injection(probe)
    rows.append({'wafer': label, 'B (cm3/s)': wafer.material.b_rad_cm3_s,
        'Cn (cm6/s)': wafer.material.c_n_cm6_s, 'Cp (cm6/s)': wafer.material.c_p_cm6_s,
        'tau_rad (ns)': inj.tau_rad_s*1e9, 'tau_Auger (ns)': inj.tau_auger_s*1e9,
        'tau_SRH (ns)': inj.tau_srh_s*1e9, 'tau_total (ns)': inj.tau_total_s*1e9})
display(pd.DataFrame(rows))
injections = np.logspace(10, 19, 61)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
for ax, (label, wafer) in zip(axes, wafers.items()):
    for factor in (1/3, 1., 3.):
        mat = replace(wafer.material, b_rad_cm3_s=wafer.material.b_rad_cm3_s*factor,
                      c_n_cm6_s=wafer.material.c_n_cm6_s*factor,
                      c_p_cm6_s=wafer.material.c_p_cm6_s*factor)
        bulk = BulkModel(replace(wafer, material=mat), tau_srh_s=tau_srh_s)
        states = [bulk.injection(d) for d in injections]
        ax.loglog(injections, [s.tau_total_s*1e9 for s in states], label=f'B/C × {factor:.2g}')
    ax.set(title=label, xlabel=r'$\Delta n$ (cm$^{-3}$)', ylabel='Unrecycled bulk lifetime (ns)')
    ax.legend()
fig.tight_layout(); plt.show()
""")
md('## 2. Analytic radiative-only recycling limit and carrier bookkeeping')
code(r"""
grid = np.logspace(2, 18, 81)
zeros = np.zeros_like(grid)
intrinsic_tau = 2e-9
analytic_table = BulkTransportTable(grid, np.full_like(grid, 10), grid/intrinsic_tau, zeros, zeros)
times = np.linspace(0, 40e-9, 151)
fig, ax = plt.subplots(figsize=(6, 3.4))
for p_rec in (0., .5, .9):
    r = TimeDependent1DSolver(analytic_table).solve(.001, times, 1e14, cells=12,
                                                 recycling_probability=p_rec, rtol=1e-8)
    exact = np.exp(-(1-p_rec)*times/intrinsic_tau)
    error = np.max(abs(r.average_delta_n_cm3/1e14-exact))
    assert error < 1e-6
    assert np.max(abs(r.inventory_relative_error)) < 1e-7
    print(f'p_rec={p_rec:.1f}, expected tau={intrinsic_tau/(1-p_rec)*1e9:g} ns, inventory={np.max(abs(r.inventory_relative_error)):.2g}')
    ax.semilogy(times*1e9, r.average_delta_n_cm3/1e14, label=f'p_rec={p_rec:.1f}')
ax.set(xlabel='Time (ns)', ylabel='Normalized carrier inventory'); ax.legend()
fig.tight_layout(); plt.show()
""")
md(r"""
## 3. Optical inputs: explicit internal spectrum and absorption partition

Use measured/literature doping-specific **interband** absorption separately
from parasitic/free-carrier absorption. Do not infer this partition from the
demo k table. This Gaussian photon spectrum and exponential absorption law are
SYNTHETIC. They are independent inputs, not a detailed-balance-consistent
near-edge InP dataset. Replace both before quantitative use.

Films for an emission boundary are ordered **wafer → outside**. This reverses
the layer order of an outside → wafer pump stack. NA selects a cone centered
on the surface normal; detector efficiency, illuminated area and spot geometry
are not included. Changing NA changes collected flux, not photon recycling.
""")
code(r"""
air, inp = get_optical_material('air'), get_optical_material('InP_demo')
alox, pox = get_optical_material('AlOx_demo'), get_optical_material('POx_demo')
front = EmissionBoundary(air, [Layer(pox, 10), Layer(alox, 10)])
rear = EmissionBoundary(air)
pump = OpticalStack(air, [Layer(alox, 10), Layer(pox, 10)], inp)
wl = np.linspace(875, 975, 11)
spectrum = EmissionSpectrum(wl, np.exp(-.5*((wl-920)/12)**2), source='SYNTHETIC Gaussian per nm')
alpha_active = 500*np.exp((920-wl)/12)  # synthetic cm^-1
alpha_parasitic = 20.                  # synthetic cm^-1
na = .5
optical_model = SlabEmissionModel(inp, front, rear, alpha_active, alpha_parasitic)
edges = np.r_[0, np.geomspace(2e-8, 5e-4, 50), np.linspace(5e-4, thickness_cm, 31)[1:]]
widths = np.diff(edges)
depth = (edges[1:]+edges[:-1])/2
fates = optical_model.solve(thickness_cm, depth, spectrum, angular_points=64, front_na=na)
probabilities = fates.averaged()
assert np.max(abs(fates.balance_error)) < 1e-10
assert np.all(probabilities['trapped'] < 1e-10)
print('Maximum photon-fate balance error:', np.max(abs(fates.balance_error)))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot(wl, spectrum.photon_density_per_nm, color='black')
axes[0].set(xlabel='Wavelength (nm)', ylabel='Normalized internal photons per nm')
ax2 = axes[0].twinx(); ax2.semilogy(wl, alpha_active, color='tab:red')
ax2.set_ylabel('Synthetic active absorption (cm$^{-1}$)', color='tab:red')
for key in ('front_escape', 'rear_escape', 'active_reabsorption', 'parasitic_absorption', 'front_collected'):
    axes[1].semilogx(depth*1e4, probabilities[key], label=key.replace('_', ' '))
axes[1].set(xlabel='Depth (µm)', ylabel='Spectrum-averaged photon probability', ylim=(0, 1))
axes[1].legend(fontsize=8)
fig.tight_layout(); plt.show()
""")
md(r"""
## 4. Couple spectral probabilities to n/p carrier transients

At each cell, `p_rec = <active_reabsorption>` reduces **net** radiative carrier
loss, while full intrinsic B generates emitted photons. Reabsorption is placed
back in the same cell, not at the actual absorption depth. No recycling scaling
is also applied to B. Band-gap narrowing is disabled; trap/potential response
remains instantaneous. Front Qext/q=+4.2e12 cm⁻² is an illustrative assumption,
not extracted or fitted here. Rear recombination is a separate chosen input.
""")
code(r"""
fluence = 1e-8  # J/cm2, ideal instantaneous 514-nm pulse
initial = pump.generation_cell_average_cm3_s(514., 1., edges)*fluence
times_s = np.r_[0, np.geomspace(1e-13, 50e-9, 201)]
results, fluxes, tables, boundaries = {}, {}, {}, {}
rows = []
for label, wafer in wafers.items():
    bulk = BulkModel(wafer, tau_srh_s=tau_srh_s)
    table = BulkTransportTable.from_bulk_model(bulk, 1., 1e20, points=121)
    surface = SurfaceSRHModel(bulk, 4.2e12, InterfaceDefectModel(energy_points=101),
                              interface_charge=InterfaceChargeModel(.5, .02))
    boundary = SurfaceBoundaryTable.from_surface_model(surface, 1., 1e20, points=81)
    tables[label], boundaries[label] = table, boundary
    for mode in ('off', 'local'):
        pr = 0. if mode == 'off' else probabilities['active_reabsorption']
        r = TimeDependent1DSolver(table).solve(thickness_cm, times_s, initial,
            boundary, 1e5, cell_edges_cm=edges, recycling_probability=pr)
        intrinsic = table.evaluate(r.delta_n_cm3)['r_rad']
        intrinsic = np.where(r.delta_n_cm3 > 0, intrinsic, 0)
        optical_flux = fates.emission_fluxes(intrinsic, widths)
        results[label, mode], fluxes[label, mode] = r, optical_flux
        np.testing.assert_allclose(optical_flux['internal'], r.internal_radiative_emission_cm2_s, rtol=1e-12)
        if mode == 'local':
            np.testing.assert_allclose(optical_flux['net_radiative_loss'], r.bulk_radiative_recombination_cm2_s, rtol=1e-11)
        assert r.converged
        rows.append({'wafer': label, 'recycling': mode, 'inventory error': np.max(abs(r.inventory_relative_error)),
            'initial lifetime (ns)': r.effective_lifetime_s[0]*1e9,
            'collected photons / initial pair (sampled)': np.trapezoid(optical_flux['front_collected'], times_s)/r.initial_sheet_excess_cm2})
display(pd.DataFrame(rows))
print('Maximum main-example carrier inventory error:', max(row['inventory error'] for row in rows))
print('OFF means active absorption discards carriers: an intentional no-recycling comparison, not complete photon/carrier feedback.')
""")
code(r"""
fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
for col, label in enumerate(wafers):
    for mode, color in (('off', 'tab:blue'), ('local', 'tab:red')):
        r, f = results[label, mode], fluxes[label, mode]
        def above_floor(signal):
            # Do not give solver absolute-tolerance tails physical significance.
            return np.where(signal > max(signal)*1e-6, signal, np.nan)
        axes[0, col].semilogy(times_s*1e9, above_floor(f['front_collected'])/max(f['front_collected']), color=color, label=mode)
        axes[0, col].semilogy(times_s*1e9, above_floor(r.sheet_excess_cm2)/r.initial_sheet_excess_cm2, '--', color=color)
        axes[1, col].semilogy(times_s*1e9, above_floor(f['internal']), color=color, label=f'{mode}: internal')
        axes[1, col].semilogy(times_s*1e9, above_floor(f['front_collected']), ':', color=color, label=f'{mode}: NA-collected')
    axes[0, col].set(title=label, ylabel='Normalized PL (solid), carriers (dashed)', ylim=(1e-6, 1.2))
    axes[1, col].set(xlabel='Time (ns)', ylabel='Photon flux (cm$^{-2}$ s$^{-1}$)')
    axes[0, col].legend(); axes[1, col].legend(fontsize=8)
fig.tight_layout()
output_dir = root/'examples'/'output'; output_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(output_dir/'v08_escape_recycling.png', dpi=180, bbox_inches='tight')
plt.show()
print('Displayed photon/carrier curves stop at 1e-6 of each trace peak; numerical tails are omitted.')
""")
md('## 5. Angular and solver tolerance checks; independent photon-loss inventory')
code(r"""
fine_fates = optical_model.solve(thickness_cm, depth, spectrum, angular_points=128, front_na=na)
angular_error = max(np.max(abs(fine_fates.averaged()[k]-probabilities[k])) for k in probabilities)
print('Maximum 64→128 angular probability change:', angular_error)
assert angular_error < 2e-4
label = 'n-InP'
tight = TimeDependent1DSolver(tables[label]).solve(thickness_cm, times_s, initial,
    boundaries[label], 1e5, cell_edges_cm=edges,
    recycling_probability=probabilities['active_reabsorption'], rtol=1e-7, atol_scaled=1e-11)
reference = results[label, 'local']
mask = reference.sheet_excess_cm2 > reference.initial_sheet_excess_cm2*1e-4
change = np.max(abs(tight.sheet_excess_cm2[mask]/reference.sheet_excess_cm2[mask]-1))
print('Tightened solver carrier-trace relative change:', change)
assert change < 2e-3
# Refine each finite-volume cell; pump generation remains exact cell-average.
fine_edges = np.sort(np.r_[edges, (edges[1:]+edges[:-1])/2])
fine_widths = np.diff(fine_edges)
fine_depth = (fine_edges[1:]+fine_edges[:-1])/2
spatial_fates = optical_model.solve(thickness_cm, fine_depth, spectrum,
                                   angular_points=64, front_na=na)
fine_initial = pump.generation_cell_average_cm3_s(514., 1., fine_edges)*fluence
refined = TimeDependent1DSolver(tables[label]).solve(thickness_cm, times_s, fine_initial,
    boundaries[label], 1e5, cell_edges_cm=fine_edges,
    recycling_probability=spatial_fates.averaged()['active_reabsorption'])
grid_change = np.max(abs(refined.sheet_excess_cm2[mask]/reference.sheet_excess_cm2[mask]-1))
print('Doubled spatial-grid carrier-trace relative change:', grid_change)
assert grid_change < .02
# The synthetic laws can also be evaluated on twice as many wavelengths.
fine_wl = np.linspace(wl[0], wl[-1], 21)
fine_spectrum = EmissionSpectrum(fine_wl, np.exp(-.5*((fine_wl-920)/12)**2), source='SYNTHETIC Gaussian per nm')
spectral_fates = SlabEmissionModel(inp, front, rear, 500*np.exp((920-fine_wl)/12),
    alpha_parasitic).solve(thickness_cm, depth, fine_spectrum, angular_points=64, front_na=na)
spectral_change = max(np.max(abs(spectral_fates.averaged()[k]-probabilities[k])) for k in probabilities)
print('11→21 wavelength-grid maximum probability change:', spectral_change)
assert spectral_change < .005
for label in wafers:
    r, f = results[label, 'local'], fluxes[label, 'local']
    # This check uses ordinary requested-time trapezoids, independently of the
    # solver's dense-output quadrature. Its larger sampling error is explicit.
    photon_loss = f['front_escape']+f['rear_escape']+f['parasitic_absorption']
    cumulative = np.trapezoid(photon_loss, times_s)
    print(label, 'sampled photon loss / dense quadrature net rad:', cumulative/r.cumulative_radiative_cm2[-1])
    assert abs(cumulative/r.cumulative_radiative_cm2[-1]-1) < .01
""")
md(r"""
## 6. Consistent CW switch-off initialization

Both solvers must use the same recycling probabilities. Otherwise the initial
CW carrier profile does not correspond to the transient equation being solved.
Internal emission events per external pump pair can exceed one with recycling;
that is not external quantum efficiency greater than one.
""")
code(r"""
label = 'n-InP'
generation = pump.generation_cell_average_cm3_s(514., 1., edges)
cw = SteadyState1DSolver(tables[label]).solve(thickness_cm, generation,
    boundaries[label], 1e5, cell_edges_cm=edges,
    recycling_probability=probabilities['active_reabsorption'], max_nfev=1500)
assert cw.converged
cw_off = TimeDependent1DSolver(tables[label]).solve(thickness_cm, times_s, cw.delta_n_cm3,
    boundaries[label], 1e5, cell_edges_cm=edges, recycling_probability=probabilities['active_reabsorption'])
print('CW global balance:', cw.relative_balance_error)
print('Internal emission events per pump-generated pair:', cw.internal_radiative_yield)
print('CW switch-off inventory error:', np.max(abs(cw_off.inventory_relative_error)))
fig, ax = plt.subplots(figsize=(6, 3.3))
ax.semilogy(times_s*1e9, cw_off.normalized_radiative_flux(), label='CW switch-off')
ax.semilogy(times_s*1e9, results[label, 'local'].normalized_radiative_flux(), '--', label='instantaneous pulse')
ax.set(xlabel='Time (ns)', ylabel='Normalized INTERNAL emission'); ax.legend()
fig.tight_layout(); plt.show()
""")
md(r"""
## Interpretation and next steps

* Optical escape filters emission by wavelength, angle and depth; normalized
  escaped and internal traces need not have identical shapes as carriers move.
* Recycling changes carriers and therefore emission, not just the plotting scale.
* Net radiative carrier loss is NOT intrinsic photon production. Use the
  original bulk table's `r_rad` when applying `PhotonFates.emission_fluxes`.
* A local recycling probability can conserve inventory while still missing
  important nonlocal carrier redistribution. Strong surface-localized pumping
  makes this limitation particularly relevant.
* In a perfectly transparent slab, report trapped rays separately. The static
  optical model has no photon reservoir or optical residence-time dynamics.
* Replace the synthetic spectrum/absorption and representative B/C coefficients
  with doping-specific datasets before comparing quantitatively with InP PL.

No instrument response, detector spectral sensitivity, pulse trains, dynamic
traps, spatial reabsorption kernel, lateral escape, or experimental fit yet.
""")
notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
notebook.metadata['language_info'] = {'name': 'python', 'version': '3.10+'}
nbf.write(notebook, ROOT/'notebooks'/'08_spectral_escape_recycling.ipynb')
print('Built release-0.8 notebook')
