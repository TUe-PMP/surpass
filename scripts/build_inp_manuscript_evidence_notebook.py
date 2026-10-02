"""Build notebook 10: manuscript-oriented evidence for POx/AlOx/InP passivation."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md(r"""
# Evidence notebook: fixed-charge field effect in POx/AlOx-passivated InP

This notebook asks a deliberately limited question: **are the measured PL and
corona trends physically consistent with positive fixed charge in the
POx/AlOx stack?** It is not an absolute fit of PL intensity or biexponential
TRPL. Unknown interface-defect distributions, capture cross sections, bulk
coefficients, photon recycling, emission collection, and the TRPL pulse fluence
make such a fit non-unique.

The calculations therefore progress from directly constrained quantities to
increasingly model-dependent ones. Every plot labels the assumptions it needs.
The 1D calculations use the same Fermi--Dirac bulk, optical-generation,
interface-SRH, electrostatic, and transport functions as notebooks 01--09.
""")

code(r"""
import sys
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext
from scipy.constants import h, c
from IPython.display import display

root = Path.cwd()
if not (root/'src').exists(): root = root.parent
sys.path.insert(0, str(root/'src'))
import surpass
print('surpass version:', surpass.__version__)
print('surpass loaded from:', Path(surpass.__file__).resolve())
if tuple(map(int, surpass.__version__.split('.')[:3])) < (0, 9, 0):
    raise ImportError('Notebook 10 requires surpass 1.0.0 or newer.')

from surpass import (
    BulkModel, BulkTransportTable, InterfaceChargeModel, InterfaceDefectModel, Layer, OpticalStack,
    SteadyState1DSolver, SurfaceBoundaryTable, SurfaceSRHModel,
    Wafer, get_optical_material,
)
try:
    import scienceplots
    plt.style.use(['science', 'notebook', 'no-latex'])
except ImportError:
    plt.style.use('default')
plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 300, 'font.size': 10,
                     'axes.spines.top': True, 'axes.spines.right': True})
""")

md(r"""
## 1. Experimental constraints and editable assumptions

The CW PL conditions are known: 514 nm, approximately 1 µm spot, and a 50×,
NA 0.75 objective. Different incident intensities were intentionally used for
the two dopings, so absolute n-/p-InP PL values must **not** be compared.
TRPL used 509 nm excitation, but pulse fluence, repetition rate and spot size
are not available here; only the reported amplitude-weighted fit values are
shown. The normalized bars below summarize observations, not model outputs.
""")

code(r"""
# Wafer and illumination inputs from the manuscript/experimental notes.
wavelength_nm = 514.0
thickness_um = 620.0
thickness_cm = thickness_um*1e-4
wafer_settings = {
    'n-InP': dict(wafer=Wafer('InP', donor_cm3=8.5e18, thickness_um=thickness_um),
                  intensity=1.0, color='tab:blue'),
    'p-InP': dict(wafer=Wafer('InP', acceptor_cm3=5.1e18, thickness_um=thickness_um),
                  intensity=13.0, color='tab:red'),
}

# Measured changes (PL ratios are approximate manuscript values).
observations = pd.DataFrame([
    {'wafer':'n-InP', 'PL after stack / before':3.0,
     'TRPL before (ns)':2.55, 'TRPL after (ns)':3.40,
     'PL after -6e12 corona / before corona':0.70},
    {'wafer':'p-InP', 'PL after stack / before':0.70,
     'TRPL before (ns)':1.88, 'TRPL after (ns)':1.79,
     'PL after -6e12 corona / before corona':1.30},
])
display(observations)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
x = np.arange(2)
colors = [wafer_settings[w]['color'] for w in observations.wafer]
axes[0].bar(x, observations['PL after stack / before'], color=colors)
axes[0].axhline(1, color='.35', lw=.8)
axes[0].set(ylabel='Normalized PL', title='Deposition + anneal')
axes[1].bar(x-.18, observations['TRPL before (ns)'], .36, color=colors, alpha=.45, label='before')
axes[1].bar(x+.18, observations['TRPL after (ns)'], .36, color=colors, label='after')
axes[1].set(ylabel='Amplitude-weighted fit (ns)', title='Reported TRPL'); axes[1].legend()
axes[2].bar(x, observations['PL after -6e12 corona / before corona'], color=colors)
axes[2].axhline(1, color='.35', lw=.8)
axes[2].set(ylabel='Normalized PL', title=r'Added negative corona')
for ax in axes:
    ax.set_xticks(x, observations.wafer)
fig.tight_layout()
fig.savefig(root/'examples'/'output'/'v09_inp_evidence_observations.png', bbox_inches='tight')
""")

md(r"""
## 2. Pump generation and relevant length scales

The optical constants are the package's rounded Aspnes-guided InP
interpolation. A "bare" calculation still includes Fresnel reflection at the
air/InP boundary and Beer--Lambert absorption inside InP; it simply has no
coherent films in the transfer matrix. The second calculation explicitly uses
air/AlOx/POx/InP. Its film thicknesses and optical constants are editable below.
The bundled POx/AlOx values are illustrative placeholders, so the coated result
is an optical-sensitivity case rather than a calibrated correction.

The generation profile is strongly front-loaded. Whether that surface matters
is then controlled by the ambipolar diffusion length, not by the 620 µm wafer
thickness alone.
""")

code(r"""
air = get_optical_material('air')
inp_optical = get_optical_material('InP_demo')
pox_optical = get_optical_material('POx_demo')
alox_optical = get_optical_material('AlOx_demo')
pox_thickness_nm = 10.0   # replace by measured thickness for the PL sample
alox_thickness_nm = 10.0  # replace by measured thickness for the PL sample
bare_pump = OpticalStack(air, [], inp_optical)
coated_pump = OpticalStack(air, [Layer(alox_optical, alox_thickness_nm, 'AlOx'),
                                  Layer(pox_optical, pox_thickness_nm, 'POx')], inp_optical)
USE_COATED_TMM = True
pump = coated_pump if USE_COATED_TMM else bare_pump
bare_optical = bare_pump.solve(wavelength_nm)
coated_optical = coated_pump.solve(wavelength_nm)
optical = pump.solve(wavelength_nm)
depth_um = np.geomspace(1e-4, 100, 500)

# Bulk SRH is disabled here to show the intrinsic radiative/Auger ceiling.
injections = np.logspace(8, 19, 180)
bulk_models = {label: BulkModel(s['wafer'], enable_bgn=False, tau_srh_s=np.inf)
               for label, s in wafer_settings.items()}

fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), constrained_layout=True)
for label, settings in wafer_settings.items():
    g_bare = bare_pump.generation_profile_cm3_s(wavelength_nm, settings['intensity'], depth_um*1e-4)
    g_coated = coated_pump.generation_profile_cm3_s(wavelength_nm, settings['intensity'], depth_um*1e-4)
    axes[0].loglog(depth_um, g_bare, color=settings['color'], ls='--',
                   label=f"{label}: bare")
    axes[0].loglog(depth_um, g_coated, color=settings['color'],
                   label=f"{label}: coated")
    states = [bulk_models[label].injection(d) for d in injections]
    axes[1].loglog(injections, [s.diffusion_length_um for s in states],
                   color=settings['color'], label=label)
axes[0].axvline(optical.absorption_depth_nm/1000, color='black', ls='--', lw=1,
                label=f'1/alpha = {optical.absorption_depth_nm:.0f} nm')
axes[0].set(xlabel='Depth into InP (um)', ylabel=r'$G(z)$ (cm$^{-3}$ s$^{-1}$)',
            title='514-nm generation: bare vs coated', xlim=(1e-4, 2))
# Do not let the exactly exponential tail force hundreds of irrelevant decades
# onto the axis; twelve decades already extend far beyond any meaningful source.
g_surface_max = max(max(stack.generation_profile_cm3_s(
    wavelength_nm, s['intensity'], np.array([0.0]))[0] for s in wafer_settings.values())
                    for stack in (bare_pump, coated_pump))
axes[0].set_ylim(g_surface_max*1e-12, g_surface_max*2)
axes[1].axhline(optical.absorption_depth_nm/1000, color='black', ls='--', lw=1,
                label='absorption depth')
axes[1].axhline(thickness_um, color='.4', ls=':', lw=1, label='wafer thickness')
axes[1].set(xlabel=r'Uniform excess density, $\Delta n$ (cm$^{-3}$)',
            ylabel='Intrinsic bulk diffusion length (um)', title='Transport reach')
axes[0].legend(fontsize=8); axes[1].legend(fontsize=8); fig.tight_layout()
fig.savefig(root/'examples'/'output'/'v09_inp_evidence_generation_lengths.png', bbox_inches='tight')

length_table = []
for label, model in bulk_models.items():
    eq = model.equilibrium()
    intensity = wafer_settings[label]['intensity']
    photon_energy = h*c/(wavelength_nm*1e-9)
    length_table.append({'wafer':label, 'R at 514 nm':optical.reflectance,
        'entry fraction':optical.substrate_entry_fraction,
        'incident photon flux (cm^-2 s^-1)':intensity/photon_energy,
        'entering photon flux (cm^-2 s^-1)':intensity*optical.substrate_entry_fraction/photon_energy,
        'absorption depth (nm)':optical.absorption_depth_nm,
        'Debye length (nm)':eq.debye_length_nm,
        'FD screening length (nm)':eq.fermi_screening_length_nm,
        'bulk eta majority':eq.eta_n if label.startswith('n') else eq.eta_p})
display(pd.DataFrame(length_table))
display(pd.DataFrame([
    {'optical case':'bare air/InP', 'R':bare_optical.reflectance,
     'substrate entry':bare_optical.substrate_entry_fraction,
     'film absorption':sum(bare_optical.layer_absorptance)},
    {'optical case':'air/AlOx/POx/InP (illustrative)', 'R':coated_optical.reflectance,
     'substrate entry':coated_optical.substrate_entry_fraction,
     'film absorption':sum(coated_optical.layer_absorptance)},
]))
""")

md(r"""
## 3. Bulk recombination and lifetime versus injection

For a constant radiative coefficient, the low-injection radiative lifetime is
approximately (1/(BN_mathrm{majority})). At these dopings it is already of
order nanoseconds. The solid curves use the package's representative InP
coefficients; the shaded radiative range varies (B) by a factor three. This
is a sensitivity range, not a confidence interval.

The measured TRPL numbers are amplitude-weighted biexponential fit parameters,
not directly the local (Delta n/Delta R) plotted here. Photon recycling,
spatial redistribution, surface boundary conditions, and the unknown pulse
fluence can all change their relationship.
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
bulk_rows = []
for ax, (label, settings) in zip(axes, wafer_settings.items()):
    model = bulk_models[label]
    states = [model.injection(d) for d in injections]
    tau_rad = np.array([s.tau_rad_s for s in states])*1e9
    tau_aug = np.array([s.tau_auger_s for s in states])*1e9
    tau_total = np.array([s.tau_total_s for s in states])*1e9
    ax.fill_between(injections, tau_rad/3, tau_rad*3, color='tab:green', alpha=.16,
                    label=r'$B/3$ to $3B$ sensitivity')
    ax.loglog(injections, tau_rad, color='tab:green', label='radiative')
    ax.loglog(injections, tau_aug, color='tab:orange', label='Auger')
    ax.loglog(injections, tau_total, color='black', lw=1.5, label='combined intrinsic')
    before = float(observations.loc[observations.wafer==label, 'TRPL before (ns)'].iloc[0])
    after = float(observations.loc[observations.wafer==label, 'TRPL after (ns)'].iloc[0])
    ax.axhspan(min(before, after), max(before, after), color=settings['color'], alpha=.12,
               label='reported fit range')
    ax.set(title=label, xlabel=r'$\Delta n$ (cm$^{-3}$)', ylabel='Lifetime (ns)', ylim=(1e-3, 1e5))
    eq = model.equilibrium(); probe = model.injection(1e12)
    bulk_rows.append({'wafer':label, 'n0 (cm^-3)':eq.n0_cm3, 'p0 (cm^-3)':eq.p0_cm3,
                      'tau_rad at low injection (ns)':probe.tau_rad_s*1e9,
                      'tau_Auger at low injection (ns)':probe.tau_auger_s*1e9,
                      'intrinsic tau (ns)':probe.tau_total_s*1e9})
axes[0].legend(fontsize=7); axes[1].legend(fontsize=7); fig.tight_layout()
fig.savefig(root/'examples'/'output'/'v09_inp_evidence_bulk_lifetimes.png', bbox_inches='tight')
display(pd.DataFrame(bulk_rows))
""")

md(r"""
## 4. Surface-recombination map versus doping and fixed charge

This replaces an injection-dependent (J_{0s}) color scale by the more direct
boundary quantity (S_mathrm{eff}=U_s/\Delta n_b), evaluated at a stated
low-injection probe of (10^{11},mathrm{cm^{-3}}). The map is still a model:
it assumes the Adamowicz starting cross sections, constant (D_{it}), no
interface-state charge in the electrostatics, and no Fermi-level pinning.

The black contour is (n_s=p_s), not an arbitrarily labelled inversion line.
The experimental doping is marked in red. Positive fixed charge accumulates
electrons on n-InP but depletes and can invert p-InP; negative charge does the
opposite.
""")

code(r"""
defects = InterfaceDefectModel(dit_mid_ev1_cm2=1e11, sigma_n_cm2=1e-14,
    sigma_p_cm2=1e-13, vth_n_cm_s=4.13e7, vth_p_cm_s=1.51e7,
    shape='constant', energy_points=201)
print(defects.source)

q_map = np.linspace(-8e12, 8e12, 49)
doping_map = np.logspace(15, 20, 39)
delta_probe = 1e11
surface_maps = {}
for polarity in ('n', 'p'):
    seff = np.empty((doping_map.size, q_map.size))
    equality = np.empty_like(seff)
    for iy, doping in enumerate(doping_map):
        wafer = Wafer('InP', donor_cm3=doping if polarity=='n' else 0,
                      acceptor_cm3=doping if polarity=='p' else 0,
                      thickness_um=thickness_um)
        bulk = BulkModel(wafer)
        for ix, charge in enumerate(q_map):
            state = SurfaceSRHModel(bulk, charge, defects,
                electrostatics='self_consistent').state(delta_probe)
            seff[iy, ix] = state.effective_s_cm_s
            equality[iy, ix] = np.log10(state.n_surface_cm3/state.p_surface_cm3)
    surface_maps[polarity] = (seff, equality)

fig, axes = plt.subplots(2, 1, figsize=(7.4, 7.1), sharex=True)
all_s = np.concatenate([surface_maps[p][0].ravel() for p in ('n','p')])
vmin, vmax = np.percentile(all_s, [2, 98])
for ax, polarity, label, actual in zip(axes, ('n','p'), ('n-InP','p-InP'), (8.5e18,5.1e18)):
    seff, equality = surface_maps[polarity]
    cf = ax.contourf(q_map/1e12, doping_map, seff, levels=np.geomspace(vmin, vmax, 31),
                     norm=LogNorm(vmin=vmin, vmax=vmax), cmap='turbo', extend='both')
    ax.contour(q_map/1e12, doping_map, equality, levels=[0], colors='black', linewidths=1.2)
    ax.axhline(actual, color='red', lw=1.3)
    ax.text(-7.6, actual*1.12, f'experimental {label}', color='red', fontsize=8,
            bbox=dict(facecolor='white', alpha=.75, edgecolor='none', boxstyle='round,pad=.2'))
    ax.set_yscale('log'); ax.set(ylabel=r'Doping (cm$^{-3}$)', title=label)
axes[1].invert_yaxis()
axes[-1].set_xlabel(r'Total external charge, $Q_\mathrm{ext}/q$ ($10^{12}$ cm$^{-2}$)')
cbar = fig.colorbar(cf, ax=axes, pad=.025, fraction=.045,
                    ticks=10.0**np.arange(np.floor(np.log10(vmin)),
                                         np.ceil(np.log10(vmax))+1),
                    format=LogFormatterMathtext(),
                    label=r'$S_\mathrm{eff}=U_s/\Delta n_b$ (cm s$^{-1}$)')
fig.subplots_adjust(hspace=.22, right=.86)
fig.savefig(root/'examples'/'output'/'v09_inp_evidence_surface_map.png', bbox_inches='tight')
""")

md(r"""
## 5. Full 1D CW calculation at the experimental PL intensities

The next calculation lets the optical generation profile, nonlinear bulk
recombination, ambipolar diffusion, and nonlinear front surface boundary find
the injection profile self-consistently. The rear velocity is deliberately
large; the result itself tests whether the rear matters. Bulk SRH is disabled
in the nominal run because it is not independently known. Local instantaneous
photon recycling is also disabled, so the plotted internal radiative flux is a
conservative optical proxy rather than detected PL.

Here "coupled" (S_mathrm{eff}) means that the surface is not assigned a
constant recombination velocity. For every trial surface excess density the
model solves charge neutrality
(Q_mathrm{ext}+Q_mathrm{it}+Q_mathrm{sc}=0), obtains the band bending and
surface carrier populations, integrates surface SRH recombination, and matches
that rate to diffusive delivery from the first finite-volume cell. After the
full generation--diffusion--recombination problem converges, the reported value
is

$$S_mathrm{eff}=U_s/Delta n_s.$$

It is therefore an operating-point-dependent boundary response, not a unique
material constant. It changes with illumination, doping, charge, defects and
capture cross sections.
""")

code(r"""
q_sweep = np.linspace(-8e12, 8e12, 33)
rear_s_cm_s = 1e5
edges_cm = np.r_[0.0, np.geomspace(1e-8, 5e-4, 110),
                 np.linspace(5e-4, thickness_cm, 101)[1:]]
coupled_rows, coupled_results = [], {}

for label, settings in wafer_settings.items():
    bulk = BulkModel(settings['wafer'], tau_srh_s=np.inf)
    table = BulkTransportTable.from_bulk_model(bulk, 1.0, 1e21, points=181)
    solver = SteadyState1DSolver(table)
    generation = pump.generation_cell_average_cm3_s(
        wavelength_nm, settings['intensity'], edges_cm)
    for charge in q_sweep:
        surface = SurfaceSRHModel(bulk, charge, defects,
            electrostatics='self_consistent', driving_force='excess_product')
        boundary = SurfaceBoundaryTable.from_surface_model(surface, 1.0, 1e21, points=101)
        result = solver.solve(thickness_cm, generation, boundary, rear_s_cm_s,
                              cell_edges_cm=edges_cm, max_nfev=1800)
        coupled_results[label, charge] = result
        coupled_rows.append({'wafer':label, 'Qext/q':charge,
            'internal PL proxy':result.internal_radiative_emission_cm2_s,
            'tau_eff_ns':result.effective_lifetime_s*1e9,
            'front fraction':result.front_surface_recombination_cm2_s/result.generated_flux_cm2_s,
            'rear fraction':result.rear_surface_recombination_cm2_s/result.generated_flux_cm2_s,
            'average injection':result.average_delta_n_cm3,
            'front Seff':result.front_effective_s_cm_s,
            'balance error':result.relative_balance_error,
            'converged':result.converged})
coupled = pd.DataFrame(coupled_rows)
assert coupled.converged.all()
print('Maximum absolute global balance error:', coupled['balance error'].abs().max())
display(coupled[np.isclose(coupled['Qext/q'], 0)][[
    'wafer', 'average injection', 'tau_eff_ns', 'front fraction',
    'rear fraction', 'front Seff', 'balance error']])

fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), sharex=True)
for label, settings in wafer_settings.items():
    d = coupled[coupled.wafer==label]
    q = d['Qext/q'].to_numpy()/1e12
    pl = d['internal PL proxy'].to_numpy()
    pl0 = np.interp(0, q, pl)
    axes[0,0].plot(q, pl/pl0, color=settings['color'], label=label)
    axes[0,1].plot(q, d.tau_eff_ns, color=settings['color'], label=label)
    axes[1,0].plot(q, d['front fraction'], color=settings['color'], label=label)
    axes[1,1].semilogy(q, d['front Seff'], color=settings['color'], label=label)
for ax in axes.ravel():
    ax.axvspan(-1.8, 4.2, color='gold', alpha=.12) # trajectory if stack starts at +4.2e12
axes[0,0].set(ylabel='Internal radiative flux / value at zero charge', title='PL proxy')
axes[0,1].set(ylabel='Generation/inventory lifetime (ns)', title='Effective lifetime')
axes[1,0].set(xlabel=r'$Q_\mathrm{ext}/q$ ($10^{12}$ cm$^{-2}$)', ylabel='Front-surface loss / generation')
axes[1,1].set(xlabel=r'$Q_\mathrm{ext}/q$ ($10^{12}$ cm$^{-2}$)', ylabel=r'Coupled $S_\mathrm{eff}$ (cm s$^{-1}$)')
axes[0,0].legend(); axes[0,1].legend(); fig.tight_layout()
fig.savefig(root/'examples'/'output'/'v09_inp_evidence_cw_charge_sweep.png', bbox_inches='tight')

# For a fixed wafer and excitation condition, this plot makes the link between
# the boundary loss and the internal PL proxy explicit. It is not a universal
# calibration curve: changing doping, generation profile or recycling moves it.
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
for ax, (label, settings) in zip(axes, wafer_settings.items()):
    d = coupled[coupled.wafer==label].sort_values('Qext/q')
    pl = d['internal PL proxy'].to_numpy()
    q = d['Qext/q'].to_numpy()/1e12
    points = ax.scatter(d['front Seff'], pl/np.interp(0, q, pl), c=q,
                        cmap='coolwarm', s=28, zorder=3)
    ax.plot(d['front Seff'], pl/np.interp(0, q, pl), color='.55', lw=.8, zorder=1)
    ax.set_xscale('log')
    ax.set(title=label, xlabel=r'Coupled front $S_\mathrm{eff}$ (cm s$^{-1}$)',
           ylabel='Internal PL proxy / value at zero charge')
fig.colorbar(points, ax=axes, label=r'$Q_\mathrm{ext}/q$ ($10^{12}$ cm$^{-2}$)')
fig.savefig(root/'examples'/'output'/'v09_inp_evidence_pl_vs_seff.png', bbox_inches='tight')
""")

md(r"""
## 5b. Separate chemical passivation from field-effect passivation

The preceding graph varied charge only after choosing one (D_{it}). The
deposition/annealing experiment changes at least two things simultaneously:

1. chemical passivation lowers the electrically active interface-defect density;
2. dielectric fixed charge changes the surface carrier populations.

The illustrative comparison below now spans
(D_{it}=10^{12}) to (3\times10^{10},\mathrm{eV^{-1}cm^{-2}}). This deliberately
brackets a much stronger chemical-passivation change. It is a sensitivity
range, **not a fitted pair of InP interface-state densities**. The Adamowicz
InP surface model used for the capture cross sections had a U-shaped
distribution with a minimum near (10^{11},\mathrm{eV^{-1}cm^{-2}}); silicon
benchmarks should not be transferred quantitatively to POx/InP. The important
point is the topology: lowering
(D_{it}) at zero charge improves both doping types, whereas the experiment
improves n-InP but slightly worsens p-InP. An additional adverse field effect
on p-InP is therefore required; positive charge supplies that asymmetry.
""")

code(r"""
dit_unpassivated = 1e12
dit_passivated = 3e10
p_recycle_previous = 0.93  # upper/sensitivity case from notebook 08's synthetic spectrum

def run_charge_series(label, dit, recycling):
    settings = wafer_settings[label]
    bulk = BulkModel(settings['wafer'], tau_srh_s=np.inf)
    table = BulkTransportTable.from_bulk_model(bulk, 1.0, 1e21, points=181)
    solver = SteadyState1DSolver(table)
    generation = pump.generation_cell_average_cm3_s(
        wavelength_nm, settings['intensity'], edges_cm)
    local_defects = InterfaceDefectModel(
        dit, 1e-14, 1e-13, 4.13e7, 1.51e7,
        shape='constant', energy_points=201)
    rows = []
    for charge in q_sweep:
        boundary = SurfaceBoundaryTable.from_surface_model(
            SurfaceSRHModel(bulk, charge, local_defects,
                electrostatics='self_consistent', driving_force='excess_product'),
            1.0, 1e21, points=101)
        result = solver.solve(thickness_cm, generation, boundary, rear_s_cm_s,
            cell_edges_cm=edges_cm, max_nfev=1800,
            recycling_probability=recycling)
        rows.append({'wafer':label, 'Qext/q':charge, 'Dit':dit,
            'recycling':recycling,
            'PL proxy':result.internal_radiative_emission_cm2_s,
            'tau_eff_ns':result.effective_lifetime_s*1e9,
            'front fraction':result.front_surface_recombination_cm2_s/result.generated_flux_cm2_s,
            'front Seff':result.front_effective_s_cm_s,
            'balance error':result.relative_balance_error,
            'converged':result.converged})
    return rows

# Reuse the already-calculated low-Dit/no-recycling series. Calculate the
# less-passivated surface, then repeat both states with the previous notebook's
# high recycling sensitivity factor.
transition_rows = []
for label in wafer_settings:
    existing = coupled[coupled.wafer==label]
    transition_rows.extend([
        {'wafer':r.wafer, 'Qext/q':getattr(r, '_1'), 'Dit':dit_passivated,
         'recycling':0.0, 'PL proxy':getattr(r, '_2'), 'tau_eff_ns':r.tau_eff_ns,
         'front fraction':getattr(r, '_4'), 'balance error':getattr(r, '_8'),
         'converged':r.converged}
        for r in existing.itertuples(index=False)
    ])
    transition_rows += run_charge_series(label, dit_unpassivated, 0.0)
    transition_rows += run_charge_series(label, dit_unpassivated, p_recycle_previous)
    transition_rows += run_charge_series(label, dit_passivated, p_recycle_previous)
transition = pd.DataFrame(transition_rows)
assert transition.converged.all()

fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), sharex=True)
measured_ratios = {'n-InP':3.0, 'p-InP':0.70}
measured_tau = {'n-InP':(2.55,3.40), 'p-InP':(1.88,1.79)}
for col, (label, settings) in enumerate(wafer_settings.items()):
    for recycling, ls, rec_label in [(0.0, '-', 'no recycling'),
                                      (p_recycle_previous, '--', r'$p_{rec}=0.93$')]:
        subset = transition[(transition.wafer==label) &
                            np.isclose(transition.recycling, recycling)]
        untreated = subset[np.isclose(subset.Dit, dit_unpassivated)].sort_values('Qext/q')
        treated = subset[np.isclose(subset.Dit, dit_passivated)].sort_values('Qext/q')
        q = treated['Qext/q'].to_numpy()/1e12
        ref_pl = np.interp(0, untreated['Qext/q'], untreated['PL proxy'])
        axes[0,col].plot(q, untreated['PL proxy']/ref_pl, color='.45', ls=ls,
                         label=f'less passivated, {rec_label}')
        axes[0,col].plot(q, treated['PL proxy']/ref_pl, color=settings['color'], ls=ls,
                         label=f'passivated, {rec_label}')
        axes[1,col].plot(q, untreated.tau_eff_ns, color='.45', ls=ls)
        axes[1,col].plot(q, treated.tau_eff_ns, color=settings['color'], ls=ls)
    axes[0,col].axhline(measured_ratios[label], color='black', lw=1, ls=':',
                        label='approx. measured after/before')
    before, after = measured_tau[label]
    axes[1,col].axhline(before, color='.35', lw=1, ls=':', label='TRPL before')
    axes[1,col].axhline(after, color=settings['color'], lw=1, ls=':', label='TRPL after')
    axes[0,col].set(title=label, ylabel='Internal PL proxy / less-passivated value at Q=0')
    axes[1,col].set(xlabel=r'$Q_\mathrm{ext}/q$ ($10^{12}$ cm$^{-2}$)',
                    ylabel='Generation/inventory lifetime (ns)')
axes[0,0].legend(fontsize=6.5); axes[0,1].legend(fontsize=6.5)
axes[1,0].legend(fontsize=7); axes[1,1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(root/'examples'/'output'/'v09_inp_evidence_dit_charge_transition.png', bbox_inches='tight')

# Interpolate the no-recycling positive-charge branch at the observed ratio.
# This is a diagnostic of sign and scale for the chosen Dit pair, not a fit.
fit_rows = []
for label in wafer_settings:
    s = transition[(transition.wafer==label) & np.isclose(transition.recycling,0)]
    untreated = s[np.isclose(s.Dit,dit_unpassivated)].sort_values('Qext/q')
    treated = s[np.isclose(s.Dit,dit_passivated)].sort_values('Qext/q')
    ref = np.interp(0, untreated['Qext/q'], untreated['PL proxy'])
    ratio = treated['PL proxy'].to_numpy()/ref
    charge = treated['Qext/q'].to_numpy()
    branch = (charge >= 0) & (charge <= 4e12)
    branch_q, branch_ratio = charge[branch], ratio[branch]
    if branch_ratio[-1] >= branch_ratio[0]:
        inferred = np.interp(measured_ratios[label], branch_ratio, branch_q)
    else:
        inferred = np.interp(measured_ratios[label], branch_ratio[::-1], branch_q[::-1])
    fit_rows.append({'wafer':label, 'illustrative Q/q from PL ratio':inferred,
                     'measured ratio':measured_ratios[label]})
display(pd.DataFrame(fit_rows))
""")

md(r"""
### Coupled (D_{it})--(Q_f) maps: the chemical/field-effect link

The maps below vary the two effects that deposition and annealing can change.
Every pixel is a full one-dimensional steady-state solution at the experimental
CW intensity, rather than a surface-only calculation at an imposed injection.
The columns report the converged (S_mathrm{eff}=U_s/Delta n_s), the internal
radiative flux relative to an illustrative poor uncharged interface, and the
electron and hole concentrations directly at the semiconductor surface. The
surface populations make the physical origin of the field effect visible:
surface recombination is suppressed when one carrier type becomes scarce.

The schematic process trajectory starts explicitly at
((Q_f/q,D_{it})=(0,10^{12})) and ends at
((+2times10^{12},3times10^{10})). It is illustrative, not an extracted
deposition trajectory.

The map is the fairest way to use these simulations: it shows that chemical
passivation and field-effect passivation are partly confounded. It should not
be read as an extraction of either coordinate without independent electrical
or spectroscopic constraints.

Very large values of (U_s/Delta n_s) occur when a strong surface sink drives
the extrapolated surface excess density almost to zero. In that regime the
wafer is diffusion-limited: the ratio is an effective Robin-boundary parameter,
not a microscopic carrier speed, and need not be bounded by the thermal
velocity. The (S_mathrm{eff}) color scale is therefore saturated above
(10^8,mathrm{cm,s^{-1}}); the PL and surface-loss results remain calculated
from the unclipped recombination flux.
""")

code(r"""
# Figure orientation switch:
# True  -> landscape: n-/p-InP on rows, physical quantities on columns.
# False -> portrait:  n-/p-InP on columns, physical quantities on rows.
COUPLED_MAP_LANDSCAPE = True

# Include the two illustrative path endpoints exactly. This makes the poor,
# uncharged normalization an actual calculated grid point rather than a nearby
# logarithmic-grid approximation.
dit_grid = np.unique(np.r_[np.geomspace(1e10, 3e12, 13),
                           dit_unpassivated, dit_passivated])
q_grid_2d = np.linspace(-6e12, 6e12, 17)
coupled_maps = {}
reference_checks = []
for label, settings in wafer_settings.items():
    bulk = BulkModel(settings['wafer'], tau_srh_s=np.inf)
    table = BulkTransportTable.from_bulk_model(bulk, 1.0, 1e21, points=161)
    solver = SteadyState1DSolver(table)
    generation = pump.generation_cell_average_cm3_s(
        wavelength_nm, settings['intensity'], edges_cm)
    seff_map = np.empty((dit_grid.size, q_grid_2d.size))
    pl_map = np.empty_like(seff_map)
    ns_map = np.empty_like(seff_map)
    ps_map = np.empty_like(seff_map)
    for iy, dit in enumerate(dit_grid):
        local_defects = InterfaceDefectModel(
            dit, 1e-14, 1e-13, 4.13e7, 1.51e7,
            shape='constant', energy_points=151)
        for ix, charge in enumerate(q_grid_2d):
            surface_model = SurfaceSRHModel(
                bulk, charge, local_defects,
                electrostatics='self_consistent', driving_force='excess_product')
            boundary = SurfaceBoundaryTable.from_surface_model(
                surface_model, 1.0, 1e21, points=81)
            result = solver.solve(thickness_cm, generation, boundary, rear_s_cm_s,
                cell_edges_cm=edges_cm, max_nfev=1400)
            if not result.converged:
                raise RuntimeError(f'2D map did not converge: {label}, Dit={dit:g}, Q={charge:g}')
            seff_map[iy, ix] = result.front_effective_s_cm_s
            pl_map[iy, ix] = result.internal_radiative_emission_cm2_s
            surface_state = surface_model.state(result.front_surface_delta_cm3)
            ns_map[iy, ix] = surface_state.n_surface_cm3
            ps_map[iy, ix] = surface_state.p_surface_cm3
    # Normalize to the exact illustrative untreated state.
    iy_ref = np.flatnonzero(np.isclose(dit_grid, dit_unpassivated, rtol=1e-12))[0]
    ix_ref = np.flatnonzero(np.isclose(q_grid_2d, 0.0, atol=1.0))[0]
    relative_pl = pl_map/pl_map[iy_ref, ix_ref]
    assert np.isclose(relative_pl[iy_ref, ix_ref], 1.0)
    reference_checks.append({
        'wafer': label,
        'normalization Dit (eV^-1 cm^-2)': dit_grid[iy_ref],
        'normalization Qf/q (cm^-2)': q_grid_2d[ix_ref],
        'relative PL at reference': relative_pl[iy_ref, ix_ref],
    })
    coupled_maps[label] = (seff_map, relative_pl, ns_map, ps_map)
display(pd.DataFrame(reference_checks))

# Use one carrier-density scale for electrons and holes so the two maps can be
# compared directly. Percentile clipping keeps extreme degenerate populations
# from compressing all of the useful contrast into a few colors.
all_carriers = np.concatenate([
    coupled_maps[label][2].ravel() for label in wafer_settings] +
    [coupled_maps[label][3].ravel() for label in wafer_settings])
carrier_log_min, carrier_log_max = np.percentile(np.log10(all_carriers), [1, 99])
carrier_levels = np.linspace(carrier_log_min, carrier_log_max, 25)
carrier_tick_exponents = np.arange(np.ceil(carrier_log_min),
                                   np.floor(carrier_log_max)+1)
seff_plot_min, seff_plot_max = 1e1, 1e8
all_relative_pl = np.concatenate([coupled_maps[label][1].ravel()
                                  for label in wafer_settings])
pl_plot_min = max(0.2, np.nanmin(all_relative_pl))
pl_plot_max = np.nanmax(all_relative_pl)
pl_levels = np.linspace(pl_plot_min, pl_plot_max, 25)

quantity_order = ('ns', 'ps', 'seff', 'pl')
if COUPLED_MAP_LANDSCAPE:
    fig, axes = plt.subplots(2, 4, figsize=(20, 7.2), sharex=True, sharey=True,
                             constrained_layout=True)
    panel = lambda type_index, quantity_index: axes[type_index, quantity_index]
    colorbar_axes = lambda quantity_index: axes[:, quantity_index]
else:
    fig, axes = plt.subplots(4, 2, figsize=(10.5, 14.0), sharex=True, sharey=True,
                             constrained_layout=True)
    panel = lambda type_index, quantity_index: axes[quantity_index, type_index]
    colorbar_axes = lambda quantity_index: axes[quantity_index, :]

path_start = (0.0, dit_unpassivated)
path_end = (2.0, dit_passivated)
plot_handles = {}
for type_index, (label, settings) in enumerate(wafer_settings.items()):
    seff_map, rel_pl_map, ns_map, ps_map = coupled_maps[label]
    ax_ns = panel(type_index, 0)
    ax_ps = panel(type_index, 1)
    ax_seff = panel(type_index, 2)
    ax_pl = panel(type_index, 3)

    plot_handles['ns'] = ax_ns.contourf(q_grid_2d/1e12, dit_grid, np.log10(ns_map),
        levels=carrier_levels, cmap='magma', extend='both')
    plot_handles['ps'] = ax_ps.contourf(q_grid_2d/1e12, dit_grid, np.log10(ps_map),
        levels=carrier_levels, cmap='magma', extend='both')
    plot_handles['seff'] = ax_seff.contourf(q_grid_2d/1e12, dit_grid, seff_map,
        levels=np.geomspace(seff_plot_min, seff_plot_max, 25),
        norm=LogNorm(vmin=seff_plot_min, vmax=seff_plot_max),
        cmap='viridis', extend='both')
    plot_handles['pl'] = ax_pl.contourf(q_grid_2d/1e12, dit_grid, rel_pl_map,
        levels=pl_levels, cmap='turbo', extend='both')

    equality = np.log10(ns_map/ps_map)
    for ax in (ax_ns, ax_ps):
        equality_line = ax.contour(q_grid_2d/1e12, dit_grid, equality,
                                   levels=[0], colors='cyan', linewidths=1.2)
        ax.clabel(equality_line, fmt={0: r'$n_s=p_s$'}, inline=True,
                  fontsize=6, colors='cyan')
    # Measured deposition/anneal ratio, shown only when it falls in the map range.
    ratio = measured_ratios[label]
    if rel_pl_map.min() <= ratio <= rel_pl_map.max():
        measured_line = ax_pl.contour(q_grid_2d/1e12, dit_grid, rel_pl_map,
                                      levels=[ratio], colors='white', linewidths=1.3)
        ax_pl.clabel(measured_line, fmt={ratio: f'measured {ratio:.2g}x'},
                      inline=True, fontsize=6.5, colors='white')

    row_axes = [panel(type_index, quantity_index) for quantity_index in range(4)]
    for ax in row_axes:
        ax.set_yscale('log')
        ax.annotate('', xy=path_end, xytext=path_start,
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        ax.plot(*path_start, marker='o', ms=5, mfc='white', mec='black', zorder=5)
        ax.plot(*path_end, marker='>', ms=5, mfc='black', mec='black', zorder=5)
        ax.text(-0.3, 1.25*dit_unpassivated, 'illustrative path', color='black', fontsize=7,
                ha='center', bbox=dict(facecolor='white', alpha=.72, edgecolor='none',
                                       boxstyle='round,pad=.2'))
        ax.set_ylabel(r'$D_{it}$ (eV$^{-1}$ cm$^{-2}$)')
    ax_ns.set_title(f'{label}: surface electrons $n_s$')
    ax_ps.set_title(f'{label}: surface holes $p_s$')
    ax_seff.set_title(f'{label}: coupled $S_{{eff}}$')
    ax_seff.text(-5.7, 1.25e10, r'$\geq10^8$: diffusion-limited',
        fontsize=7, color='white',
        bbox=dict(facecolor='black', alpha=.35, edgecolor='none', boxstyle='round,pad=.2'))
    ax_pl.set_title(f'{label}: relative internal PL')
    ax_pl.text(-0.15, 7.2e11, 'own unpassivated reference = 1', fontsize=7,
        ha='center', color='black',
        bbox=dict(facecolor='white', alpha=.72, edgecolor='none', boxstyle='round,pad=.2'))

if COUPLED_MAP_LANDSCAPE:
    for quantity_index in range(4):
        axes[-1,quantity_index].set_xlabel(r'$Q_\mathrm{ext}/q$ ($10^{12}$ cm$^{-2}$)')
else:
    for type_index in range(2):
        axes[-1,type_index].set_xlabel(r'$Q_\mathrm{ext}/q$ ($10^{12}$ cm$^{-2}$)')

carrier_cbar = fig.colorbar(plot_handles['ns'], ax=np.r_[colorbar_axes(0), colorbar_axes(1)],
    ticks=carrier_tick_exponents, label=r'Surface concentration (cm$^{-3}$)', pad=.02)
carrier_cbar.ax.set_yticklabels(
    [rf'$10^{{{int(exponent)}}}$' for exponent in carrier_tick_exponents])
seff_cbar = fig.colorbar(plot_handles['seff'], ax=colorbar_axes(2),
    ticks=10.0**np.arange(int(np.log10(seff_plot_min)),
                          int(np.log10(seff_plot_max))+1),
    format=LogFormatterMathtext(),
    label=r'Coupled $S_\mathrm{eff}$ (cm s$^{-1}$)', pad=.02)
fig.colorbar(plot_handles['pl'], ax=colorbar_axes(3),
             label='PL proxy / own poor uncharged reference', pad=.02)
fig.savefig(root/'examples'/'output'/'v09_inp_evidence_coupled_dit_q_maps.png', bbox_inches='tight')
""")

md(r"""
### Photon-recycling sensitivity

Notebook 08's synthetic emission/absorption model gave a depth-dependent
active-reabsorption probability of approximately 0.927--0.943. That number is
not yet calibrated for these doped wafers or the actual stack. The comparison
below therefore treats 0.93 as a high sensitivity case and includes 0.5 as an
intermediate case. Recycling reduces **net** radiative carrier loss but leaves
the intrinsic radiative-emission rate intact. It can raise the internal
emission-event count above one per external pair; this is not external PL
quantum efficiency.
""")

code(r"""
# Only the p_rec=0.5 low-Dit case is missing from the preceding calculations.
middle_rows = []
for label in wafer_settings:
    middle_rows += run_charge_series(label, dit_passivated, 0.5)
middle = pd.DataFrame(middle_rows)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharex=True)
for ax, (label, settings) in zip(axes, wafer_settings.items()):
    for recycling, source, color in [
        (0.0, transition, 'black'), (0.5, middle, 'tab:orange'),
        (p_recycle_previous, transition, 'tab:green')]:
        s = source[(source.wafer==label) & np.isclose(source.Dit,dit_passivated)
                   & np.isclose(source.recycling,recycling)].sort_values('Qext/q')
        q = s['Qext/q'].to_numpy()/1e12
        pl = s['PL proxy'].to_numpy()
        ax.plot(q, pl/np.interp(0,q,pl), color=color, label=f'$p_{{rec}}={recycling:g}$')
    ax.set(title=label, xlabel=r'$Q_\mathrm{ext}/q$ ($10^{12}$ cm$^{-2}$)',
           ylabel='Passivated PL proxy / value at zero charge')
axes[0].legend(); axes[1].legend(); fig.tight_layout()
fig.savefig(root/'examples'/'output'/'v09_inp_evidence_recycling_sensitivity.png', bbox_inches='tight')
""")

md(r"""
## 6. What does the negative-corona trajectory constrain?

Corona adds charge to whatever fixed charge was already present. The curves
below re-express the same simulations versus added negative corona for several
assumed initial stack charges. They show why the **direction** of the measured
response supports the field-effect interpretation, while its magnitude cannot
uniquely determine (Q_f): different starting charges and interface parameters
can share the same local slope.
""")

code(r"""
corona = np.linspace(0, -6e12, 121)
initial_charges = [0.0, 2e12, 4.2e12, 6e12]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharex=True)
trajectory_rows = []
for ax, (label, settings) in zip(axes, wafer_settings.items()):
    d = coupled[coupled.wafer==label].sort_values('Qext/q')
    qgrid = d['Qext/q'].to_numpy(); plgrid = d['internal PL proxy'].to_numpy()
    for q0 in initial_charges:
        qpath = q0 + corona
        valid = (qpath >= qgrid.min()) & (qpath <= qgrid.max())
        predicted = np.interp(qpath[valid], qgrid, plgrid)
        predicted /= np.interp(q0, qgrid, plgrid)
        ax.plot(-corona[valid]/1e12, predicted, label=f'initial {q0/1e12:g}')
        trajectory_rows.append({'wafer':label, 'initial Q/q':q0,
            'PL ratio after -6e12':predicted[-1],
            'monotonic over path':bool(np.all(np.diff(predicted) <= 1e-6)) if label=='n-InP'
                                  else bool(np.all(np.diff(predicted) >= -1e-6))})
    measured = float(observations.loc[observations.wafer==label,
        'PL after -6e12 corona / before corona'].iloc[0])
    ax.axhline(measured, color=settings['color'], ls='--', lw=1.2, label='approx. measured')
    ax.axhline(1, color='.5', lw=.7)
    ax.set(title=label, xlabel=r'Magnitude of added negative corona ($10^{12}$ cm$^{-2}$)',
           ylabel='PL proxy / value before corona')
axes[0].legend(fontsize=7, ncol=2); axes[1].legend(fontsize=7, ncol=2); fig.tight_layout()
fig.savefig(root/'examples'/'output'/'v09_inp_evidence_corona_paths.png', bbox_inches='tight')
display(pd.DataFrame(trajectory_rows))
""")

md(r"""
## 7. Surface-parameter sensitivity: trend versus magnitude

The surface-only calculation below varies (D_{it}) and the electron/hole
capture-cross-section ratio. It is intentionally cheaper and clearer than
pretending every uncertain parameter combination is a complete PL fit.
Multiplying (D_{it}) mainly rescales recombination; changing
(sigma_n/sigma_p) can move and skew the recombination maximum. The field-
effect polarity is nevertheless governed primarily by the calculated surface
carrier populations.
""")

code(r"""
sensitivity_cases = {
    'literature start': (1e11, 1e-14, 1e-13),
    'Dit / 3': (1e11/3, 1e-14, 1e-13),
    'Dit x 3': (3e11, 1e-14, 1e-13),
    'equal sigma': (1e11, 3.16e-14, 3.16e-14),
    'reversed ratio': (1e11, 1e-13, 1e-14),
}
q_dense = np.linspace(-8e12, 8e12, 121)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharex=True)
for ax, (label, settings) in zip(axes, wafer_settings.items()):
    bulk = BulkModel(settings['wafer'])
    for case, (dit, sn, sp) in sensitivity_cases.items():
        local = InterfaceDefectModel(dit, sn, sp, 4.13e7, 1.51e7,
                                     shape='constant', energy_points=201)
        rates = np.array([SurfaceSRHModel(bulk, q, local).state(delta_probe).effective_s_cm_s
                          for q in q_dense])
        ax.semilogy(q_dense/1e12, rates, label=case)
    # Illustrative pinning sensitivity: the CNL is not measured for POx/InP,
    # so this is a diagnostic rather than a recommended parameter set.
    pinned_defects = InterfaceDefectModel(1e11, 1e-14, 1e-13, 4.13e7, 1.51e7,
                                          shape='constant', energy_points=201)
    pinned_rates = np.array([SurfaceSRHModel(
        bulk, q, pinned_defects,
        interface_charge=InterfaceChargeModel(0.5, 0.02)).state(delta_probe).effective_s_cm_s
        for q in q_dense])
    ax.semilogy(q_dense/1e12, pinned_rates, color='black', ls='--',
                label='illustrative amphoteric Qit')
    ax.set(title=label, xlabel=r'$Q_\mathrm{ext}/q$ ($10^{12}$ cm$^{-2}$)',
           ylabel=r'$S_\mathrm{eff}$ at $\Delta n=10^{11}$ cm$^{-3}$ (cm s$^{-1}$)')
axes[0].legend(fontsize=7); axes[1].legend(fontsize=7); fig.tight_layout()
fig.savefig(root/'examples'/'output'/'v09_inp_evidence_surface_sensitivity.png', bbox_inches='tight')
""")

md(r"""
## Conclusions that this notebook can and cannot support

**Robust qualitative evidence**

- At 514 nm the carriers are generated close to the treated surface, while the
  calculated diffusion length is much larger than the absorption depth. A
  front-surface field effect can therefore strongly affect PL even for a thick
  wafer; the rear contribution is reported separately.
- Positive external charge produces the expected favorable field effect on
  n-InP and the opposite tendency on p-InP over the relevant branch. Adding
  negative corona reverses those tendencies, matching the measured directions.
- The asymmetric n-/p-InP response after deposition and annealing is therefore
  more informative than either corona curve alone. Lowering (D_{it}) without
  charge raises the calculated PL of both wafers. Reproducing an n-InP increase
  together with a p-InP decrease requires an adverse p-type field effect,
  consistent with positive fixed charge.
- Photon recycling can materially increase effective lifetime and internal PL,
  but the calculated polarity of the charge response remains intact over the
  tested sensitivity range.

**What is not uniquely established**

- The corona trend by itself does not prove a unique nonzero (Q_f): depending
  on the uncertain interface kinetics, a trajectory starting near zero charge
  can have the same local direction. Monotonicity and the absence of a p-InP
  turning point constrain the branch but do not constitute a calibrated charge
  measurement.
- The factor-three PL increase cannot be fitted uniquely without the actual
  POx/AlOx optical constants/thicknesses, (D_{it}(E)), capture cross sections,
  bulk coefficients, photon recycling, and collection efficiency.
- The reported biexponential TRPL values should not be equated directly with a
  single simulated lifetime. A quantitative TRPL comparison additionally needs
  the 509-nm pulse fluence, repetition rate, spot size and measured IRF.

For a manuscript, the most useful main-text output is the charge-dependent
surface/PL trend at the two experimental dopings plus the corona trajectories.
The generation/length-scale, lifetime-sensitivity, and full doping--charge maps
are strong supporting-information figures.
""")

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata.kernelspec = {'display_name':'Python 3', 'language':'python', 'name':'python3'}
notebook.metadata.language_info = {'name':'python', 'version':'3'}
destination = ROOT/'notebooks'/'10_inp_pox_alox_manuscript_evidence.ipynb'
nbf.write(notebook, destination)
print(destination)
