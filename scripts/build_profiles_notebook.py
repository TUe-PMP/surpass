"""Readable source for the release-0.6 validation notebook.

Run this from the repository root. The notebook imports public package APIs;
the plots are illustrative scenarios, not calibrated predictions of POx/InP.
"""

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source):
    cells.append(nbf.v4.new_code_cell(source.strip()))


md(r"""
# Spatial band diagrams and TRPL-readiness checks — release 0.6

This notebook reconstructs the semiconductor-side spatial band diagram from
the self-consistent surface potential of release 0.5. It resolves the
space-charge region, **without** solving drift-diffusion inside it.

The profile is semi-infinite and uniformly doped, with flat quasi-Fermi levels
through the local space-charge region. Bulk $E_v=0$ is the energy reference;
vacuum energies, dielectric bands, metal work functions, and band offsets are
not supplied. Optional BGN remains uniform, set by the wafer bulk model.

The goal is to test when the existing surface-boundary approximation is
reasonable before implementing time-dependent transport/TRPL.
""")
code(r"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

root = Path.cwd()
if not (root / 'src').exists(): root = root.parent
sys.path.insert(0, str(root / 'src'))
from surpass import (
    BulkModel, Wafer, InterfaceDefectModel, InterfaceChargeModel, SurfaceSRHModel,
    reconstruct_band_profile, trap_relaxation_spectrum, OpticalStack,
    get_optical_material,
)
try:
    import scienceplots
    plt.style.use(['science', 'notebook', 'no-latex'])
except ImportError:
    plt.style.use('default')
plt.rcParams.update({'figure.dpi': 120, 'font.size': 10})
""")
md("## 1. Editable assumptions")
code(r"""
material_name = 'InP'
thickness_um = 620.0
n_doping_cm3, p_doping_cm3 = 8.5e18, 5.1e18
enable_bgn = False
include_interface_charge = True
cnl_fraction, transition_width_ev = 0.5, 0.02  # sensitivity parameters
dit_ev1_cm2 = 1e11
qf_values_cm2 = [-4.2e12, 4.2e12]  # total dielectric + corona charge
injection_values_cm3 = [0.0, 1e11]
tau_bulk_srh_s = 100e-9
profile_points = 1201
tail_fraction = 1e-6
colors = {'Ec': 'tab:blue', 'Ev': 'tab:orange',
          'EFn': 'tab:green', 'EFp': 'tab:red'}

# These capture parameters are the inherited illustrative InP starting point.
# Replace them, and the optical data, when selecting another material.
defects = InterfaceDefectModel(dit_mid_ev1_cm2=dit_ev1_cm2, energy_points=301)
charge_model = (InterfaceChargeModel(cnl_fraction, transition_width_ev)
                if include_interface_charge else None)
wafers = {
    'n': Wafer(material_name, donor_cm3=n_doping_cm3, thickness_um=thickness_um),
    'p': Wafer(material_name, acceptor_cm3=p_doping_cm3, thickness_um=thickness_um),
}
bulk_models = {key: BulkModel(wafer, enable_bgn=enable_bgn,
                             tau_srh_s=tau_bulk_srh_s)
               for key, wafer in wafers.items()}
models, profiles = {}, {}
for polarity, bulk in bulk_models.items():
    for qf in qf_values_cm2:
        model = SurfaceSRHModel(bulk, qf, defects, interface_charge=charge_model)
        models[polarity, qf] = model
        for injection in injection_values_cm3:
            profiles[polarity, qf, injection] = reconstruct_band_profile(
                model, injection, points=profile_points, tail_fraction=tail_fraction)
print(defects.source)
print(charge_model.source if charge_model else 'Interface charge disabled')
""")
md(r"""
## 2. Equilibrium and illuminated band diagrams

Positive $\psi$ shifts both band edges downward: $E_v=-\psi$ and
$E_c=E_g-\psi$ (energies in eV, potential in volts). The quasi-Fermi energies
are $E_{Fn}=E_g+kT\eta_{n,b}$ and $E_{Fp}=-kT\eta_{p,b}$.
At equilibrium they coincide; under injection they split.
""")
code(r"""
for injection in injection_values_cm3:
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    for row, polarity in enumerate(wafers):
        for column, qf in enumerate(qf_values_cm2):
            ax = axes[row, column]
            profile = profiles[polarity, qf, injection]
            z_nm = profile.depth_cm*1e7
            for values, label, style in [
                (profile.ec_ev, 'Ec', '-'), (profile.ev_ev, 'Ev', '-'),
                (profile.efn_ev, 'EFn', '--'), (profile.efp_ev, 'EFp', ':'),
            ]:
                ax.plot(z_nm, values, ls=style, color=colors[label], label=label)
            extent_nm = max(profile.potential_99_depth_cm*1e7,
                            profile.charge_99_depth_cm*1e7)
            ax.set(xlim=(0, max(2.0, 1.3*extent_nm)),
                   xlabel='Depth into semiconductor (nm)', ylabel='Energy (eV)',
                   title=f'{polarity}-{material_name}, Qext/q={qf/1e12:+.1f}e12')
    axes[0, 0].legend(ncol=2, fontsize=8)
    fig.suptitle(f'Bulk-side injection = {injection:.1e} cm$^{{-3}}$')
    fig.tight_layout()
    if injection == 0:
        output = root/'examples'/'output'
        output.mkdir(parents=True, exist_ok=True)
        fig.savefig(output/'v06_equilibrium_band_profiles.png', dpi=150)
    plt.show()
""")
md("## 3. Carrier and electric-field profiles")
code(r"""
injection = injection_values_cm3[-1]
fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
for row, polarity in enumerate(wafers):
    for qf in qf_values_cm2:
        p = profiles[polarity, qf, injection]
        label = f'Qext/q={qf/1e12:+.1f}e12'
        line, = axes[row, 0].semilogy(p.depth_cm*1e7, p.n_cm3, label='n, '+label)
        axes[row, 0].semilogy(p.depth_cm*1e7, p.p_cm3, ls='--',
                              color=line.get_color(), label='p, '+label)
        axes[row, 1].plot(p.depth_cm*1e7, p.electric_field_v_cm/1e3, label=label)
    for ax in axes[row]:
        extent = max(profiles[polarity, q, injection].potential_99_depth_cm
                     for q in qf_values_cm2)*1e7
        ax.set(xlim=(0, 1.3*extent), xlabel='Depth (nm)', title=polarity+'-'+material_name)
    axes[row, 0].set_ylabel('Carrier concentration (cm$^{-3}$)')
    axes[row, 1].set_ylabel('Electric field (kV/cm)')
axes[0, 0].legend(fontsize=7)
axes[0, 1].legend(fontsize=8)
fig.tight_layout()
plt.show()
""")
md(r"""
## 4. Length scales and conservation

The space-charge region has no sharp boundary. We report depths containing
90/99% of the **net signed semiconductor space charge**, and the depth at which
the potential has relaxed by 99%. These measures can differ in inversion.

Ratios much smaller than unity support treating the region as a local surface
boundary; they do not prove flat quasi-Fermi levels. Ratios approaching unity
motivate spatially resolved generation, recombination, and transport there.
""")
code(r"""
# The bundled InP optical data are explicitly illustrative, not measured.
pump_nm = 514.0
pump_stack = OpticalStack(get_optical_material('air'), [],
                          get_optical_material('InP_demo'))
absorption_depth_cm = pump_stack.solve(pump_nm).absorption_depth_nm*1e-7
rows = []
for (polarity, qf, injection), profile in profiles.items():
    probe = bulk_models[polarity].injection(max(injection, 1.0))
    diffusion_cm = probe.diffusion_length_um*1e-4
    width = max(profile.potential_99_depth_cm, profile.charge_99_depth_cm)
    rows.append({
        'polarity': polarity, 'Qext/q (1e12)': qf/1e12, 'injection (cm^-3)': injection,
        'screening (nm)': profile.screening_length_cm*1e7,
        'charge90 (nm)': profile.charge_90_depth_cm*1e7,
        'charge99 (nm)': profile.charge_99_depth_cm*1e7,
        'potential99 (nm)': profile.potential_99_depth_cm*1e7,
        'absorption depth (nm)': absorption_depth_cm*1e7,
        'diffusion length (um)': probe.diffusion_length_um,
        'width / absorption': width/absorption_depth_cm,
        'width / diffusion': width/diffusion_cm,
        'width / wafer': width/(thickness_um*1e-4),
        'charge integral relative error': profile.charge_integral_relative_error,
        'finite wafer warning': profile.semi_infinite_warning,
    })
diagnostics = pd.DataFrame(rows)
with pd.option_context('display.float_format', lambda value: f'{value:.3g}'):
    display(diagnostics)
assert diagnostics['charge integral relative error'].abs().max() < 1e-3
""")
md("## 5. Lower doping: when does the surface-boundary approximation become questionable?")
code(r"""
doping_grid = np.logspace(15, 19, 13)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for polarity in ['n', 'p']:
    for qf in qf_values_cm2:
        widths = []
        for doping in doping_grid:
            wafer = Wafer(material_name, donor_cm3=doping if polarity == 'n' else 0,
                          acceptor_cm3=doping if polarity == 'p' else 0,
                          thickness_um=thickness_um)
            model = SurfaceSRHModel(BulkModel(wafer), qf, defects,
                                     interface_charge=charge_model)
            p = reconstruct_band_profile(model, injection_values_cm3[-1], points=601)
            widths.append(max(p.potential_99_depth_cm, p.charge_99_depth_cm))
        label = f'{polarity}-type, {qf/1e12:+.1f}e12'
        axes[0].loglog(doping_grid, np.array(widths)*1e7, label=label)
        axes[1].loglog(doping_grid, np.array(widths)/absorption_depth_cm, label=label)
axes[0].axhline(absorption_depth_cm*1e7, color='black', ls=':', label='pump absorption depth')
axes[1].axhline(1, color='black', ls=':')
axes[0].set(xlabel='Doping (cm$^{-3}$)', ylabel='Space-charge extent (nm)')
axes[1].set(xlabel='Doping (cm$^{-3}$)', ylabel='Extent / pump absorption depth')
axes[0].legend(fontsize=7)
fig.tight_layout()
plt.show()
""")
md(r"""
## 6. Trap relaxation before TRPL

At fixed local populations and potential, the SRH occupancy equation is

$$\frac{df}{dt}=(c_n n_s+e_p)(1-f)-(c_p p_s+e_n)f,$$

so $\tau_t^{-1}=c_n(n_s+n_1)+c_p(p_s+p_1)$. This is the local, frozen-carrier
relaxation time, not the collective time including electrostatic feedback.
The same classical emission model as v0.5 is retained; fully degenerate
capture/emission kinetics remains a limitation.
""")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
trap_rows = []
for ax, polarity in zip(axes, wafers):
    for qf in qf_values_cm2:
        model = models[polarity, qf]
        spectrum = trap_relaxation_spectrum(model, injection_values_cm3[-1])
        ax.semilogy(spectrum.energy_above_ev_ev, spectrum.relaxation_time_s*1e9,
                     label=f'Qext/q={qf/1e12:+.1f}e12')
        trap_rows.append({'polarity': polarity, 'Qext/q (1e12)': qf/1e12,
                          'max local trap tau (ns)': spectrum.relaxation_time_s.max()*1e9,
                          'Dit-weighted fraction slower than 1 ns': spectrum.slow_trap_fraction(1e-9)})
    ax.axhline(1.0, color='black', ls=':')
    ax.set(title=polarity+'-'+material_name, xlabel='Trap energy above local Ev (eV)',
           ylabel='Local occupancy relaxation time (ns)')
    ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
with pd.option_context('display.float_format', lambda value: f'{value:.3g}'):
    display(pd.DataFrame(trap_rows))
""")
md(r"""
## Interpretation and next step

- This is a computed spatial band diagram, not an atomistic band-structure calculation.
- The bulk-side injection is specified for these local checks. In the coupled
  transport workflow use `result.front_surface_delta_cm3` as that input.
  Do not simply splice this local profile into the quasi-neutral transport
  profile: they use different depth domains and must be matched explicitly.
- The semiconductor potential tends to zero asymptotically. Its truncated tail
  and grid-dependent charge-integral error are exposed, not hidden.
- A width comparable to the wafer thickness invalidates the semi-infinite
  approximation; the API flags either 99% depth exceeding 10% of the thickness.
  This is a heuristic warning, not an automatically corrected finite-wafer solution.
- The next release can add transient quasi-neutral diffusion with an
  instantaneous surface boundary, provided these spatial and trap-time checks
  support that approximation. Later extensions can evolve trap occupancy and
  solve coupled Poisson/drift-diffusion within the space-charge region.

References: Nicollian & Brews, *MOS Physics and Technology* (Wiley, 1982), for
Poisson electrostatics; Shockley & Read, *Phys. Rev.* **87**, 835 (1952), DOI
10.1103/PhysRev.87.835, for capture/emission kinetics. Preset InP defect and
bulk-parameter references remain in the model objects and README.
""")

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3'},
})
destination = ROOT/'notebooks'/'06_spatial_band_profiles.ipynb'
nbf.write(notebook, destination)
print(destination)
