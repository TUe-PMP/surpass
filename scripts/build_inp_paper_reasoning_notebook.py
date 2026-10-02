"""Build a paper-focused notebook for the POx/AlOx/InP passivation argument."""
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md(r"""
# From field-effect passivation to the measured InP PL response

This notebook develops the modelling argument for PO$_x$/AlO$_x$-passivated
InP in the same order as it can be presented in the paper:

1. establish the general low-injection field effect versus doping and charge;
2. show how injection moves the surface-carrier balance for the actual wafers;
3. include the experimental optical generation and one-dimensional transport;
4. separate chemical passivation ($D_{it}$ reduction) from positive fixed
   charge and compare the resulting PL trends with experiment.

The goal is **qualitative mechanistic support**, not a unique extraction of
$Q_f$ or $D_{it}$. The model consistently uses Fermi--Dirac carrier statistics,
self-consistent surface charge neutrality, distributed surface SRH
recombination, optical generation, nonlinear bulk recombination and ambipolar
transport. Unknown interface-state distributions, capture cross sections,
photon recycling and absolute optical collection prevent a unique fit.

### Sign convention and experimental hypothesis

Positive $Q_f/q$ denotes positive sheet charge in or on the dielectric. The
hypothesis tested here is that deposition and annealing both lower $D_{it}$ and
introduce or activate **positive** $Q_f$. Added negative corona charge then
partly compensates this dielectric charge.
""")

code(r"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import LogFormatterMathtext
from scipy.constants import h, c
from IPython.display import display

root = Path.cwd()
if not (root/'src').exists():
    root = root.parent
sys.path.insert(0, str(root/'src'))

import surpass
from surpass import (
    BulkModel, BulkTransportTable, InterfaceChargeModel, InterfaceDefectModel,
    Layer, OpticalStack, SteadyState1DSolver, SurfaceBoundaryTable,
    SurfaceSRHModel, Wafer, get_optical_material, reconstruct_band_profile,
)

print('surpass version:', surpass.__version__)
print('loaded from:', Path(surpass.__file__).resolve())
if tuple(map(int, surpass.__version__.split('.')[:3])) < (0, 9, 0):
    raise ImportError('This notebook requires surpass 1.0.0 or newer.')

try:
    import scienceplots
    plt.style.use(['science', 'notebook', 'no-latex'])
except ImportError:
    plt.style.use('default')

plt.rcParams.update({
    'figure.dpi': 120, 'savefig.dpi': 300, 'font.size': 10,
    'axes.spines.top': True, 'axes.spines.right': True,
})
output_dir = root/'examples'/'output'
output_dir.mkdir(parents=True, exist_ok=True)
""")

md(r"""
## 1. Experimental inputs and model switches

Absolute PL from n- and p-InP is not compared because different incident
intensities were used. Every simulated PL map is therefore normalized to the
poor, uncharged reference of the **same wafer polarity**.

The nominal electrostatics omits $Q_{it}$ because the charge-neutrality level
and amphoteric-state distribution of PO$_x$/InP are not independently known.
Setting `INCLUDE_INTERFACE_CHARGE=True` activates an explicitly labelled
sensitivity case; it must not be mistaken for a measured interface model.
""")

code(r"""
wavelength_nm = 514.0
thickness_um = 620.0
thickness_cm = thickness_um*1e-4
wafer_settings = {
    'n-InP': dict(wafer=Wafer('InP', donor_cm3=8.5e18, thickness_um=thickness_um),
                  intensity_w_cm2=1.0, color='tab:blue'),
    'p-InP': dict(wafer=Wafer('InP', acceptor_cm3=5.1e18, thickness_um=thickness_um),
                  intensity_w_cm2=13.0, color='tab:red'),
}

observations = pd.DataFrame([
    {'wafer':'n-InP', 'PL stack/bare':3.0, 'TRPL bare (ns)':2.55,
     'TRPL stack (ns)':3.40, 'PL after -6e12 corona/before':0.70},
    {'wafer':'p-InP', 'PL stack/bare':0.70, 'TRPL bare (ns)':1.88,
     'TRPL stack (ns)':1.79, 'PL after -6e12 corona/before':1.30},
])
display(observations)

# Literature-motivated starting values, not fitted POx/InP parameters.
DIT_MAP = 1e11
SIGMA_N = 1e-14
SIGMA_P = 1e-13
VTH_N = 4.13e7
VTH_P = 1.51e7
defects = InterfaceDefectModel(
    DIT_MAP, SIGMA_N, SIGMA_P, VTH_N, VTH_P,
    shape='constant', energy_points=201,
)
print(defects.source)

INCLUDE_INTERFACE_CHARGE = False
interface_charge = (InterfaceChargeModel(0.5, 0.02)
                    if INCLUDE_INTERFACE_CHARGE else None)
ENABLE_BGN = False

def surface_model(wafer, charge_cm2, local_defects=defects):
    return SurfaceSRHModel(
        BulkModel(wafer, enable_bgn=ENABLE_BGN), charge_cm2, local_defects,
        electrostatics='self_consistent', driving_force='excess_product',
        interface_charge=interface_charge,
    )
""")

md(r"""
## 2. Differential low-injection field-effect map — main-text candidate

At exact equilibrium, both $U_s$ and the excess density vanish. The physically
meaningful low-injection quantity is therefore

$$
S_{\mathrm{eff},0}=\lim_{\Delta n_b\rightarrow0}
\frac{U_s(\Delta n_b)}{\Delta n_b}.
$$

The first cell verifies numerical convergence of this ratio. The map then uses
a small perturbation rather than calling $Delta n=0$. This avoids importing a
silicon-specific $J_0$ convention into a PL/lifetime problem.
""")

code(r"""
low_injection_probes = np.array([1e6, 1e7, 1e8, 1e9, 1e10])
convergence_rows = []
for label, settings in wafer_settings.items():
    for charge in (-4e12, 0.0, 4e12):
        model = surface_model(settings['wafer'], charge)
        values = np.array([model.state(d).effective_s_cm_s
                           for d in low_injection_probes])
        convergence_rows.append({
            'wafer': label, 'Q/q (cm^-2)': charge,
            'S at 1e6': values[0], 'S at 1e8': values[2],
            'S at 1e10': values[-1],
            'relative span 1e6--1e9': np.ptp(values[:4])/np.mean(values[:4]),
        })
display(pd.DataFrame(convergence_rows))

DELTA_LOW = 1e8
""")

code(r"""
q_map = np.linspace(-1.5e13, 1.5e13, 81)
doping_map = np.logspace(15, 20, 51)
low_maps = {}

for polarity in ('n', 'p'):
    seff = np.empty((doping_map.size, q_map.size))
    balance = np.empty_like(seff)
    surface_degenerate = np.empty_like(seff, dtype=bool)
    bulk_eta = np.empty(doping_map.size)
    for iy, doping in enumerate(doping_map):
        wafer = Wafer('InP', donor_cm3=doping if polarity=='n' else 0.0,
                      acceptor_cm3=doping if polarity=='p' else 0.0,
                      thickness_um=thickness_um)
        bulk = BulkModel(wafer, enable_bgn=ENABLE_BGN).equilibrium()
        bulk_eta[iy] = bulk.eta_n if polarity == 'n' else bulk.eta_p
        for ix, charge in enumerate(q_map):
            state = surface_model(wafer, charge).state(DELTA_LOW)
            seff[iy, ix] = state.effective_s_cm_s
            balance[iy, ix] = np.log10(state.n_surface_cm3/state.p_surface_cm3)
            surface_degenerate[iy, ix] = ((state.eta_n_surface >= 0) or
                                          (state.eta_p_surface >= 0))
    low_maps[polarity] = dict(seff=seff, balance=balance,
                              surface_degenerate=surface_degenerate,
                              bulk_eta=bulk_eta)

# The surface-only ratio is mathematically allowed to become extreme when the
# extrapolated surface excess density tends to zero. Above about 1e8 cm/s the
# experiment is already transport-limited, so clip only the displayed scale.
# The underlying arrays remain unclipped for diagnostics.
vmin, vmax = 1e-1, 1e8
levels = np.geomspace(vmin, vmax, 28)

fig, axes = plt.subplots(2, 1, figsize=(7.6, 7.4), sharex=True,
                         constrained_layout=True)
for ax, polarity, label, actual in zip(
        axes, ('n','p'), ('n-InP','p-InP'), (8.5e18,5.1e18)):
    data = low_maps[polarity]
    cf = ax.contourf(q_map/1e12, doping_map,
                     np.clip(data['seff'], vmin, vmax), levels=levels,
                     norm=LogNorm(vmin=vmin, vmax=vmax), cmap='turbo', extend='both')
    balance_line = ax.contour(q_map/1e12, doping_map, data['balance'],
                              levels=[0], colors='black', linewidths=1.35)
    ax.clabel(balance_line, fmt={0:r'$n_s=p_s$'}, fontsize=7, inline=True)
    ax.contourf(q_map/1e12, doping_map, data['surface_degenerate'].astype(float),
                levels=[0.5,1.5], colors='none', hatches=['//'], alpha=0)
    # Bulk degeneracy onset eta_majority=0, interpolated in log doping.
    eta = data['bulk_eta']
    idx = np.flatnonzero(np.diff(np.signbit(eta)))
    if idx.size:
        i = idx[0]
        log_ndeg = np.interp(0, eta[i:i+2], np.log10(doping_map[i:i+2]))
        ndeg = 10**log_ndeg
        ax.axhline(ndeg, color='crimson', lw=1.3)
        ax.annotate('bulk degeneracy', xy=(-11.5, ndeg*1.7), xytext=(-7.5, ndeg/1.7),
                    color='crimson', fontsize=8,
                    arrowprops=dict(arrowstyle='->', color='crimson', lw=1.1),
                    bbox=dict(facecolor='white', alpha=.72, edgecolor='none',
                              boxstyle='round,pad=.2'))
    ax.axhline(actual, color='white', lw=1.4, ls='--')
    ax.text(-14.3, actual*1.12, f'wafer: {actual:.2g} cm$^{{-3}}$', fontsize=7,
            bbox=dict(facecolor='white', alpha=.75, edgecolor='none',
                      boxstyle='round,pad=.2'))
    ax.set_yscale('log')
    ax.set_ylabel(r'Doping density (cm$^{-3}$)')
    ax.set_title(label + r'; hatched: surface degeneracy')
axes[1].invert_yaxis()
axes[1].set_xlabel(r'Total external charge, $Q_{ext}/q$ ($10^{12}$ cm$^{-2}$)')
fig.colorbar(cf, ax=axes, pad=.02, ticks=10.0**np.arange(
    int(np.log10(vmin)), int(np.log10(vmax))+1), format=LogFormatterMathtext(),
    label=rf'Clipped $S_{{eff,0}}\approx U_s/\Delta n_b$ at $\Delta n_b={DELTA_LOW:.0e}$ cm$^{{-3}}$ (cm s$^{{-1}}$)')
fig.savefig(output_dir/'paper_inp_main_low_injection_map.png', bbox_inches='tight')
""")

md(r"""
### Interpretation

The topology is the robust result. Fixed charge suppresses recombination when
it drives one carrier type away from the surface. Heavy bulk doping shifts the
$n_s=p_s$ contour to larger $|Q_{ext}|$: much more sheet charge is required to
invert a highly doped wafer than a lightly doped wafer. The hatched regions
show where either surface reduced Fermi energy is non-negative, so the
Fermi--Dirac treatment is actually used rather than merely flagged.

The displayed scale is clipped at $10^{-1}$ and $10^8$ cm s$^{-1}$; the
unclipped values remain available in `low_maps`. The upper bound denotes a
regime in which a real wafer would be limited by diffusive carrier delivery.

The map does not establish the initial charge of the experimental stack. A
negative-corona sweep can have the observed local slope even when it starts at
$Q_f=0$.
""")

md(r"""
## 3. Injection-dependent cuts at the actual dopings — main-text candidate

For every $(Q_{ext},\Delta n_b)$ pair the surface potential is solved again
from

$$Q_{ext}+Q_{it}+Q_{sc}=0,$$

using the illuminated electron and hole quasi-Fermi levels. This is important:
the calculation does not insert illuminated carrier populations into a frozen
dark band diagram. Markers identify $n_s=p_s$ where that crossing lies inside
the plotted charge range.
""")

code(r"""
injection_levels = np.array([1e8, 1e11, 1e13, 1e15, 1e16, 1e17, 1e18])
q_cuts = np.linspace(-1.5e13, 1.5e13, 161)
cut_data = {}

def zero_crossing(x, y):
    indices = np.flatnonzero(np.signbit(y[:-1]) != np.signbit(y[1:]))
    if not indices.size:
        return np.nan
    i = indices[0]
    return float(np.interp(0, y[i:i+2], x[i:i+2]))

for label, settings in wafer_settings.items():
    curves = []
    for delta in injection_levels:
        seff, rate, ratio = [], [], []
        for charge in q_cuts:
            state = surface_model(settings['wafer'], charge).state(delta)
            seff.append(state.effective_s_cm_s)
            rate.append(state.recombination_cm2_s)
            ratio.append(np.log10(state.n_surface_cm3/state.p_surface_cm3))
        curves.append(dict(delta=delta, seff=np.array(seff), rate=np.array(rate),
                           balance=np.array(ratio),
                           q_equal=zero_crossing(q_cuts, np.array(ratio))))
    cut_data[label] = curves

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True,
                         constrained_layout=True)
norm = LogNorm(injection_levels.min(), injection_levels.max())
cmap = plt.get_cmap('viridis')
for ax, (label, settings) in zip(axes, wafer_settings.items()):
    for curve in cut_data[label]:
        color = cmap(norm(curve['delta']))
        displayed = np.clip(curve['seff'], 1e-1, 1e8)
        ax.semilogy(q_cuts/1e12, displayed, color=color, lw=1.25)
        if np.isfinite(curve['q_equal']):
            y_equal = np.clip(np.interp(curve['q_equal'], q_cuts, curve['seff']),
                              1e-1, 1e8)
            ax.plot(curve['q_equal']/1e12, y_equal, 'o', color=color,
                    mec='black', mew=.45, ms=4.5)
    ax.set_title(label + r'; circles: $n_s=p_s$')
    ax.set_xlabel(r'$Q_{ext}/q$ ($10^{12}$ cm$^{-2}$)')
    ax.set_ylabel(r'$S_{eff}=U_s/\Delta n_b$ (cm s$^{-1}$)')
    ax.set_ylim(1e-1,3e8)
    ax.text(.02,.94,r'$S_{eff}\geq10^8$: transport-limited',transform=ax.transAxes,
            fontsize=7,bbox=dict(facecolor='white',alpha=.72,edgecolor='none',
                                 boxstyle='round,pad=.2'))
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes,
             label=r'Imposed bulk injection, $\Delta n_b$ (cm$^{-3}$)')
fig.savefig(output_dir/'paper_inp_main_injection_dependent_cuts.png', bbox_inches='tight')

crossing_rows = []
for label, curves in cut_data.items():
    for curve in curves:
        crossing_rows.append({'wafer':label, 'Delta n (cm^-3)':curve['delta'],
                              'Q at ns=ps (cm^-2)':curve['q_equal'],
                              'Q at max Us (cm^-2)':q_cuts[np.argmax(curve['rate'])],
                              'Q at max Seff (cm^-2)':q_cuts[np.argmax(curve['seff'])]})
display(pd.DataFrame(crossing_rows))
""")

md(r"""
### Interpretation

The balance point and the recombination maximum are related but need not
coincide. Unequal electron/hole capture cross sections skew the SRH maximum,
while $S_{eff}=U_s/\Delta n_b$ and $U_s$ are different observables. At higher
injection the quasi-Fermi levels, interface occupancy and space charge all
change. Once injection approaches the doping density, the usual dark notions
of depletion and inversion also become less sharp.

This figure is therefore a caution against assigning the corona turning point
to a dark $n_s=p_s$ condition without specifying illumination.
""")

md(r"""
## 4. Optical generation and transport length scales — supplementary candidate

The full air/AlO$_x$/PO$_x$/InP transfer-matrix calculation is used. The film
constants and thicknesses remain editable because the bundled PO$_x$ and
AlO$_x$ data are illustrative. The comparison establishes whether the treated
front surface is sampled by the experiment; it is not used to fit $Q_f$.
""")

code(r"""
air = get_optical_material('air')
inp_optical = get_optical_material('InP_demo')
pox_optical = get_optical_material('POx_demo')
alox_optical = get_optical_material('AlOx_demo')
POX_THICKNESS_NM = 10.0
ALOX_THICKNESS_NM = 10.0
pump = OpticalStack(
    air,
    [Layer(alox_optical, ALOX_THICKNESS_NM, 'AlOx'),
     Layer(pox_optical, POX_THICKNESS_NM, 'POx')],
    inp_optical,
)
optical = pump.solve(wavelength_nm)
depth_um = np.geomspace(1e-4, 100, 500)
uniform_injections = np.logspace(8, 19, 160)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9), constrained_layout=True)
for label, settings in wafer_settings.items():
    generation = pump.generation_profile_cm3_s(
        wavelength_nm, settings['intensity_w_cm2'], depth_um*1e-4)
    axes[0].loglog(depth_um, generation, color=settings['color'], label=label)
    bulk = BulkModel(settings['wafer'], enable_bgn=ENABLE_BGN, tau_srh_s=np.inf)
    lengths = [bulk.injection(d).diffusion_length_um for d in uniform_injections]
    axes[1].loglog(uniform_injections, lengths, color=settings['color'], label=label)
axes[0].axvline(optical.absorption_depth_nm/1000, color='black', ls='--', lw=1,
                label=f'1/alpha = {optical.absorption_depth_nm:.0f} nm')
axes[0].set(xlabel='Depth into InP (um)', ylabel=r'$G(z)$ (cm$^{-3}$ s$^{-1}$)',
            title='514-nm generation in coated wafer', xlim=(1e-4,2))
axes[0].set_ylim(1e20, None)
axes[1].axhline(optical.absorption_depth_nm/1000, color='black', ls='--', lw=1,
                label='absorption depth')
axes[1].axhline(thickness_um, color='.4', ls=':', lw=1, label='wafer thickness')
axes[1].set(xlabel=r'Uniform $\Delta n$ (cm$^{-3}$)',
            ylabel='Intrinsic diffusion length (um)', title='Transport reach')
for ax in axes: ax.legend(fontsize=8)
fig.savefig(output_dir/'paper_inp_si_generation_and_lengths.png', bbox_inches='tight')

photon_energy = h*c/(wavelength_nm*1e-9)
display(pd.DataFrame([
    {'wafer':label,
     'incident photon flux (cm^-2 s^-1)':s['intensity_w_cm2']/photon_energy,
     'entering photon flux (cm^-2 s^-1)':s['intensity_w_cm2']*optical.substrate_entry_fraction/photon_energy,
     'R':optical.reflectance, 'entry fraction':optical.substrate_entry_fraction,
     'absorption depth (nm)':optical.absorption_depth_nm}
    for label, s in wafer_settings.items()
]))
""")

md(r"""
## 5. Coupled $D_{it}$--$Q_f$ maps at the experimental CW conditions

Every pixel below is a complete one-dimensional steady-state solution. Optical
generation, bulk radiative/Auger recombination, ambipolar diffusion and the
nonlinear surface boundary are solved together. At convergence,

$$S_{eff,front}=U_s/\Delta n_s.$$

This operating-point boundary response is not a microscopic velocity and may
become very large when a strong surface sink drives the extrapolated surface
excess density toward zero. Values above $10^8$ cm s$^{-1}$ are therefore
display-clipped as diffusion-limited, while the PL is calculated from the
unclipped flux.
""")

code(r"""
DIT_POOR = 1e12
DIT_GOOD = 3e10
Q_PATH_END = 2e12
rear_s_cm_s = 1e5
dit_grid = np.unique(np.r_[np.geomspace(1e10,3e12,13), DIT_POOR, DIT_GOOD])
q_grid = np.linspace(-6e12,6e12,17)
edges_cm = np.r_[0.0, np.geomspace(1e-8,5e-4,100),
                 np.linspace(5e-4,thickness_cm,91)[1:]]
coupled_maps = {}
reference_checks = []

for label, settings in wafer_settings.items():
    bulk = BulkModel(settings['wafer'], enable_bgn=ENABLE_BGN, tau_srh_s=np.inf)
    transport = BulkTransportTable.from_bulk_model(bulk, 1.0, 1e21, points=161)
    solver = SteadyState1DSolver(transport)
    generation = pump.generation_cell_average_cm3_s(
        wavelength_nm, settings['intensity_w_cm2'], edges_cm)
    arrays = {name:np.empty((dit_grid.size,q_grid.size))
              for name in ('seff','pl','ns','ps','front_delta')}
    for iy, dit in enumerate(dit_grid):
        local_defects = InterfaceDefectModel(
            dit, SIGMA_N, SIGMA_P, VTH_N, VTH_P,
            shape='constant', energy_points=151)
        for ix, charge in enumerate(q_grid):
            smodel = surface_model(settings['wafer'], charge, local_defects)
            boundary = SurfaceBoundaryTable.from_surface_model(
                smodel, 1.0, 1e21, points=81)
            result = solver.solve(
                thickness_cm, generation, boundary, rear_s_cm_s,
                cell_edges_cm=edges_cm, max_nfev=1500)
            if not result.converged:
                raise RuntimeError(f'No convergence: {label}, Dit={dit:g}, Q={charge:g}')
            state = smodel.state(result.front_surface_delta_cm3)
            arrays['seff'][iy,ix] = result.front_effective_s_cm_s
            arrays['pl'][iy,ix] = result.internal_radiative_emission_cm2_s
            arrays['ns'][iy,ix] = state.n_surface_cm3
            arrays['ps'][iy,ix] = state.p_surface_cm3
            arrays['front_delta'][iy,ix] = result.front_surface_delta_cm3
    iy0 = np.flatnonzero(np.isclose(dit_grid,DIT_POOR,rtol=1e-12))[0]
    ix0 = np.flatnonzero(np.isclose(q_grid,0,atol=1))[0]
    arrays['relative_pl'] = arrays['pl']/arrays['pl'][iy0,ix0]
    reference_checks.append({'wafer':label, 'Dit reference':dit_grid[iy0],
                             'Q reference':q_grid[ix0],
                             'relative PL reference':arrays['relative_pl'][iy0,ix0]})
    coupled_maps[label] = arrays

display(pd.DataFrame(reference_checks))
""")

md(r"""
### Compact PL consequence — main-text candidate

The arrow is an illustrative process trajectory from a poor, uncharged
interface to a chemically passivated interface with positive fixed charge. It
is not a fitted trajectory. White contours mark the approximate experimental
stack/bare PL ratios. The central claim is the direction of the response:
chemical passivation alone tends to help both wafers, whereas positive charge
adds a favourable field effect for n-InP and an adverse one for p-InP.
""")

code(r"""
measured_ratios = {'n-InP':3.0, 'p-InP':0.70}
all_pl = np.concatenate([d['relative_pl'].ravel() for d in coupled_maps.values()])
pl_min, pl_max = max(.2,all_pl.min()), all_pl.max()
pl_levels = np.linspace(pl_min,pl_max,31)

fig, axes = plt.subplots(1,2,figsize=(11,4.1),sharex=True,sharey=True,
                         constrained_layout=True)
for ax,(label,settings) in zip(axes,wafer_settings.items()):
    rel = coupled_maps[label]['relative_pl']
    cf = ax.contourf(q_grid/1e12,dit_grid,rel,levels=pl_levels,
                     cmap='turbo',extend='both')
    ratio = measured_ratios[label]
    if rel.min() <= ratio <= rel.max():
        line = ax.contour(q_grid/1e12,dit_grid,rel,levels=[ratio],
                          colors='white',linewidths=1.4)
        ax.clabel(line,fmt={ratio:f'measured {ratio:.2g}x'},colors='white',fontsize=7)
    ax.annotate('',xy=(Q_PATH_END/1e12,DIT_GOOD),xytext=(0,DIT_POOR),
                arrowprops=dict(arrowstyle='->',color='black',lw=1.7))
    ax.plot(0,DIT_POOR,'o',mfc='white',mec='black',ms=5,zorder=5)
    ax.plot(Q_PATH_END/1e12,DIT_GOOD,'>',mfc='black',mec='black',ms=5,zorder=5)
    ax.set_yscale('log')
    ax.set(title=label,xlabel=r'$Q_f/q$ ($10^{12}$ cm$^{-2}$)',
           ylabel=r'$D_{it}$ (eV$^{-1}$ cm$^{-2}$)')
fig.colorbar(cf,ax=axes,label='Internal PL proxy / own poor uncharged reference')
fig.savefig(output_dir/'paper_inp_main_coupled_pl_maps.png',bbox_inches='tight')
""")

md(r"""
### Full causal chain — supplementary candidate

The eight panels expose the mechanism behind the compact PL maps:

$$n_s,\ p_s\ \longrightarrow\ S_{eff}\ \longrightarrow\ \mathrm{PL}.$$

The cyan contour is $n_s=p_s$. The white contour in the PL panels is the
measured relative PL change. The two meanings are deliberately kept visually
distinct.
""")

code(r"""
PLOT_LANDSCAPE = True  # False gives wafer types as columns instead of rows.
all_carriers = np.concatenate([d[k].ravel() for d in coupled_maps.values()
                               for k in ('ns','ps')])
clo,chi = np.percentile(np.log10(all_carriers),[1,99])
carrier_levels = np.linspace(clo,chi,25)
carrier_ticks = np.arange(np.ceil(clo),np.floor(chi)+1)
seff_min,seff_max = 1e1,1e8

if PLOT_LANDSCAPE:
    fig,axes = plt.subplots(2,4,figsize=(20,7.2),sharex=True,sharey=True,
                            constrained_layout=True)
    panel = lambda i,j: axes[i,j]
    caxes = lambda j: axes[:,j]
else:
    fig,axes = plt.subplots(4,2,figsize=(10.5,14),sharex=True,sharey=True,
                            constrained_layout=True)
    panel = lambda i,j: axes[j,i]
    caxes = lambda j: axes[j,:]

handles = {}
for i,(label,settings) in enumerate(wafer_settings.items()):
    d = coupled_maps[label]
    axn,axp,axs,axpl = [panel(i,j) for j in range(4)]
    handles['n'] = axn.contourf(q_grid/1e12,dit_grid,np.log10(d['ns']),
                                 levels=carrier_levels,cmap='magma',extend='both')
    handles['p'] = axp.contourf(q_grid/1e12,dit_grid,np.log10(d['ps']),
                                 levels=carrier_levels,cmap='magma',extend='both')
    handles['s'] = axs.contourf(q_grid/1e12,dit_grid,d['seff'],
        levels=np.geomspace(seff_min,seff_max,25),
        norm=LogNorm(seff_min,seff_max),cmap='viridis',extend='both')
    handles['pl'] = axpl.contourf(q_grid/1e12,dit_grid,d['relative_pl'],
                                   levels=pl_levels,cmap='turbo',extend='both')
    equality = np.log10(d['ns']/d['ps'])
    for ax in (axn,axp):
        line = ax.contour(q_grid/1e12,dit_grid,equality,levels=[0],
                          colors='cyan',linewidths=1.2)
        ax.clabel(line,fmt={0:r'$n_s=p_s$'},colors='cyan',fontsize=6)
    ratio = measured_ratios[label]
    if d['relative_pl'].min() <= ratio <= d['relative_pl'].max():
        line = axpl.contour(q_grid/1e12,dit_grid,d['relative_pl'],levels=[ratio],
                            colors='white',linewidths=1.3)
        axpl.clabel(line,fmt={ratio:f'measured {ratio:.2g}x'},colors='white',fontsize=6)
    for ax in (axn,axp,axs,axpl):
        ax.set_yscale('log')
        ax.annotate('',xy=(Q_PATH_END/1e12,DIT_GOOD),xytext=(0,DIT_POOR),
                    arrowprops=dict(arrowstyle='->',color='black',lw=1.4))
        ax.set_ylabel(r'$D_{it}$ (eV$^{-1}$ cm$^{-2}$)')
    axn.set_title(label+r': surface electrons $n_s$')
    axp.set_title(label+r': surface holes $p_s$')
    axs.set_title(label+r': coupled $S_{eff}$')
    axpl.set_title(label+': relative internal PL')

if PLOT_LANDSCAPE:
    for j in range(4): axes[-1,j].set_xlabel(r'$Q_f/q$ ($10^{12}$ cm$^{-2}$)')
else:
    for i in range(2): axes[-1,i].set_xlabel(r'$Q_f/q$ ($10^{12}$ cm$^{-2}$)')

cb = fig.colorbar(handles['n'],ax=np.r_[caxes(0),caxes(1)],ticks=carrier_ticks,
                  label=r'Surface concentration (cm$^{-3}$)',pad=.02)
cb.ax.set_yticklabels([rf'$10^{{{int(t)}}}$' for t in carrier_ticks])
fig.colorbar(handles['s'],ax=caxes(2),format=LogFormatterMathtext(),
             label=r'Coupled $S_{eff}$ (cm s$^{-1}$)',pad=.02)
fig.colorbar(handles['pl'],ax=caxes(3),
             label='PL proxy / own poor uncharged reference',pad=.02)
fig.savefig(output_dir/'paper_inp_si_full_causal_maps.png',bbox_inches='tight')
""")

md(r"""
## 6. Representative steady-state band diagrams — supplementary candidate

The profiles below use the converged surface injection from the full transport
calculation at the illustrative endpoint. They are local semi-infinite Poisson
reconstructions, not full-wafer drift--diffusion band diagrams. Their purpose
is to make accumulation/depletion and the nanometre-scale electrostatic region
visible to readers who are less familiar with semiconductor surfaces.
""")

code(r"""
fig,axes = plt.subplots(2,2,figsize=(10.5,7.2),constrained_layout=True)
band_rows=[]
iy_end=np.flatnonzero(np.isclose(dit_grid,DIT_GOOD,rtol=1e-12))[0]
ix_end=np.argmin(np.abs(q_grid-Q_PATH_END))
for row,(label,settings) in enumerate(wafer_settings.items()):
    delta_surface=coupled_maps[label]['front_delta'][iy_end,ix_end]
    for col,(charge,case) in enumerate(((0.0,'flat-charge reference'),
                                        (Q_PATH_END,'positive-charge endpoint'))):
        local_defects=InterfaceDefectModel(DIT_GOOD,SIGMA_N,SIGMA_P,VTH_N,VTH_P,
                                           shape='constant',energy_points=151)
        model=surface_model(settings['wafer'],charge,local_defects)
        profile=reconstruct_band_profile(model,delta_surface,points=901)
        x_nm=profile.depth_cm*1e7
        ax=axes[row,col]
        ax.plot(x_nm,profile.ec_ev,label=r'$E_C$')
        ax.plot(x_nm,profile.ev_ev,label=r'$E_V$')
        ax.plot(x_nm,profile.efn_ev,'--',label=r'$E_{Fn}$')
        ax.plot(x_nm,profile.efp_ev,'--',label=r'$E_{Fp}$')
        ax.set(xlabel='Depth into InP (nm)',ylabel='Energy relative to bulk $E_V$ (eV)',
               title=f'{label}: {case}',xlim=(0,max(profile.potential_99_depth_cm*1e7*1.4,2)))
        band_rows.append({'wafer':label,'case':case,'surface delta (cm^-3)':delta_surface,
                          'psi_s (V)':model.state(delta_surface).psi_surface_v,
                          '99% potential depth (nm)':profile.potential_99_depth_cm*1e7})
axes[0,0].legend(fontsize=7,ncol=2)
fig.savefig(output_dir/'paper_inp_si_band_profiles.png',bbox_inches='tight')
display(pd.DataFrame(band_rows))
""")

md(r"""
## 7. What the combined evidence supports

### Supported qualitatively

- Chemical passivation alone tends to improve both polarities in this model.
- Positive dielectric charge accumulates electrons on n-InP and suppresses
  surface recombination over the relevant branch.
- The same charge depletes p-InP and can offset the benefit of lower $D_{it}$.
- Added negative corona charge therefore decreases the n-InP PL and increases
  the p-InP PL, as observed.
- The strong n-InP improvement together with the slight p-InP degradation is
  more informative about the initial polarity than either corona slope alone.

### Not uniquely extracted

- A corona slope does not locate $Q_f=0$; a neutral starting stack can show the
  same local direction.
- The magnitude of $Q_f$ is correlated with $D_{it}(E)$, capture cross
  sections, $Q_{it}$, optical constants, bulk recombination and photon
  recycling.
- The approximate PL and biexponential-TRPL ratios should therefore not be
  presented as a unique quantitative fit.

### Suggested paper allocation

- **Main text:** low-injection doping--charge map, injection-dependent cuts at
  the actual dopings, and compact coupled relative-PL maps.
- **Supplementary information:** optical generation and length scales, full
  eight-panel causal maps, representative band diagrams, $J_{0s}$ comparison,
  and sensitivity to capture cross sections, $Q_{it}$ and photon recycling.

### Key references

- R. B. M. Girisch, R. P. Mertens, and R. F. De Keersmaecker,
  *IEEE Trans. Electron Devices* **35**, 203--222 (1988),
  https://doi.org/10.1109/16.2441.
- K. R. McIntosh and L. E. Black, *J. Appl. Phys.* **116**, 014503 (2014),
  https://doi.org/10.1063/1.4886595.
- B. Adamowicz *et al.*, *Vacuum* **63**, 223--227 (2001),
  https://doi.org/10.1016/S0042-207X(01)00195-6.
- B. Macco *et al.*, *J. Appl. Phys.* **131**, 195301 (2022),
  https://doi.org/10.1063/5.0089589.
""")


notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata.kernelspec = {
    'display_name': 'Python 3', 'language': 'python', 'name': 'python3'
}
notebook.metadata.language_info = {'name': 'python', 'version': '3'}
destination = ROOT/'notebooks'/'11_inp_paper_reasoning.ipynb'
nbf.write(notebook, destination)
print(destination)
