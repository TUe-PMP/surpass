"""Build the release-0.2 notebooks from readable cell sources."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def markdown(text):
    return nbf.v4.new_markdown_cell(text.strip())


def write(name, cells):
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
    )
    nbf.write(notebook, NB_DIR / name)


write("01_bulk_model_validation.ipynb", [
    markdown(r"""
# Bulk model validation — release 0.2

This notebook consolidates the equilibrium, mobility, recombination, diffusion,
band-gap-narrowing (BGN), and screening checks used before adding surfaces and
optics. The default examples are the experimental **n-InP ($8.5\times10^{18}$
cm$^{-3}$)** and **p-InP ($5.1\times10^{18}$ cm$^{-3}$)** wafers.

The distinction between a *code check* and a *material validation* is important:
charge neutrality and limiting behavior can be tested exactly, whereas mobility,
$B$, and Auger coefficients remain literature-model choices with uncertainty.
"""),
    code(r"""
from dataclasses import replace
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

root = Path.cwd()
if not (root / "src").exists():
    root = root.parent
sys.path.insert(0, str(root / "src"))

from surpass import BulkModel, Wafer, get_material

try:
    import scienceplots
    plt.style.use(["science", "notebook", "no-latex"])
except ImportError:
    plt.style.use("default")
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})
"""),
    markdown("""
## 1. Editable assumptions

`tau_srh_s` is an assumed bulk SRH lifetime, not an intrinsic material
constant. BGN is shown both off and on because the bundled Palankovski model is
generic rather than dopant-specific.
"""),
    code(r"""
material_name = "InP"
temperature_k = 300.0
thickness_um = 620.0
n_doping_cm3 = 8.5e18
p_doping_cm3 = 5.1e18
tau_srh_s = 100e-9
injection_grid = np.logspace(11, 19, 181)

material = get_material(material_name)
wafers = {
    "n-InP": Wafer(material, donor_cm3=n_doping_cm3, thickness_um=thickness_um,
                    temperature_k=temperature_k),
    "p-InP": Wafer(material, acceptor_cm3=p_doping_cm3, thickness_um=thickness_um,
                    temperature_k=temperature_k),
}

display(pd.DataFrame({
    "quantity": ["Band structure / DOS", "Mobility", "Radiative / Auger", "BGN"],
    "implementation": ["FD neutrality; Vurgaftman/Ioffe values",
                       "compact Caughey-Thomas estimate",
                       "constant phenomenological coefficients",
                       "Palankovski point-charge model; optional"],
    "validation status": ["reference check", "provisional", "sensitivity required", "scenario model"],
}))
print("References carried by the preset:")
for ref in material.references:
    print(" -", ref)
print("Cautions:")
for note in material.cautions:
    print(" -", note)
"""),
    markdown("## 2. Equilibrium and measured-wafer sanity checks"),
    code(r"""
rows = []
for label, wafer in wafers.items():
    for bgn in (False, True):
        s = BulkModel(wafer, enable_bgn=bgn).equilibrium()
        majority_eta = s.eta_n if wafer.net_doping_cm3 > 0 else s.eta_p
        rows.append({
            "wafer": label, "BGN": bgn, "Eg (eV)": s.eg_effective_ev,
            "dEg (meV)": 1e3*s.delta_eg_bgn_ev, "majority eta": majority_eta,
            "n0 (cm^-3)": s.n0_cm3, "p0 (cm^-3)": s.p0_cm3,
            "mu_n": s.mu_n_cm2_vs, "mu_p": s.mu_p_cm2_vs,
            "rho (ohm cm)": s.resistivity_ohm_cm,
            "Debye (nm)": s.debye_length_nm,
            "FD screening (nm)": s.fermi_screening_length_nm,
        })
equilibrium_table = pd.DataFrame(rows)
display(equilibrium_table.style.format(precision=3))

# One available experimental cross-check for the n wafer.
pred = BulkModel(wafers["n-InP"]).equilibrium()
measured_mu, measured_rho = 1299.0, 6.0e-4
display(pd.DataFrame({
    "quantity": ["electron mobility (cm2/Vs)", "resistivity (ohm cm)"],
    "model": [pred.mu_n_cm2_vs, pred.resistivity_ohm_cm],
    "measured": [measured_mu, measured_rho],
    "difference (%)": [100*(pred.mu_n_cm2_vs/measured_mu-1),
                       100*(pred.resistivity_ohm_cm/measured_rho-1)],
}))
"""),
    markdown("## 3. Doping sweeps: mobility, screening, and BGN"),
    code(r"""
dopings = np.logspace(14, 20, 121)
data = {"n": [], "p": []}
for polarity in data:
    for density in dopings:
        wafer = Wafer(material, donor_cm3=density if polarity == "n" else 0,
                      acceptor_cm3=density if polarity == "p" else 0,
                      temperature_k=temperature_k)
        off = BulkModel(wafer).equilibrium()
        on = BulkModel(wafer, enable_bgn=True).equilibrium()
        data[polarity].append((off, on))

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for polarity, color in [("n", "tab:blue"), ("p", "tab:red")]:
    states = data[polarity]
    mobility = [a.mu_n_cm2_vs if polarity == "n" else a.mu_p_cm2_vs for a, _ in states]
    axes[0].loglog(dopings, mobility, color=color, label=f"{polarity}-type")
    axes[1].loglog(dopings, [a.debye_length_nm for a, _ in states], ls="--", color=color)
    axes[1].loglog(dopings, [a.fermi_screening_length_nm for a, _ in states], color=color,
                   label=f"{polarity}-type FD")
    axes[2].semilogx(dopings, [1e3*b.delta_eg_bgn_ev for _, b in states], color=color,
                     label=f"{polarity}-type")
axes[0].set(xlabel="Doping (cm$^{-3}$)", ylabel="Majority mobility (cm$^2$ V$^{-1}$ s$^{-1}$)")
axes[1].set(xlabel="Doping (cm$^{-3}$)", ylabel="Screening length (nm)")
axes[2].set(xlabel="Doping (cm$^{-3}$)", ylabel="BGN (meV)")
axes[0].legend(); axes[1].legend(); axes[2].legend()
fig.tight_layout()
"""),
    markdown(r"""
## 4. Injection-dependent recombination and transport

The plotted lifetimes are **effective excess-carrier lifetimes**
$\tau_\mathrm{eff}=\Delta n/\Delta R$. A small-signal or transient experiment
instead approaches the differential lifetime
$\tau_\mathrm{diff}=(d\Delta R/d\Delta n)^{-1}$; both are shown below.
"""),
    code(r"""
fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
for column, (label, wafer) in enumerate(wafers.items()):
    model = BulkModel(wafer, tau_srh_s=tau_srh_s)
    states = [model.injection(x) for x in injection_grid]
    rad = np.array([x.delta_r_rad_cm3_s for x in states])
    aug = np.array([x.delta_r_auger_cm3_s for x in states])
    srh = np.array([x.delta_r_srh_cm3_s for x in states])
    total = rad + aug + srh
    axes[0, column].loglog(injection_grid, injection_grid/rad*1e9, label="radiative")
    axes[0, column].loglog(injection_grid, injection_grid/aug*1e9, label="Auger")
    axes[0, column].loglog(injection_grid, injection_grid/total*1e9, lw=2, label="total effective")
    differential = 1/np.gradient(total, injection_grid)
    axes[0, column].loglog(injection_grid, differential*1e9, ls="--", label="total differential")
    axes[1, column].loglog(injection_grid, [x.diffusion_length_um for x in states], lw=2)
    axes[0, column].set_title(label)
    axes[0, column].set_ylabel("Lifetime (ns)")
    axes[1, column].set(xlabel="$\\Delta n$ (cm$^{-3}$)", ylabel="Ambipolar diffusion length ($\\mu$m)")
axes[0, 0].legend(fontsize=8)
fig.tight_layout()
"""),
    markdown("## 5. Sensitivity to uncertain recombination coefficients"),
    code(r"""
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
for ax, (label, wafer) in zip(axes, wafers.items()):
    for scale, color in [(0.3, "tab:blue"), (1.0, "black"), (3.0, "tab:red")]:
        varied = replace(material, b_rad_cm3_s=material.b_rad_cm3_s*scale,
                         c_n_cm6_s=material.c_n_cm6_s*scale,
                         c_p_cm6_s=material.c_p_cm6_s*scale)
        varied_wafer = Wafer(varied, donor_cm3=wafer.donor_cm3,
                             acceptor_cm3=wafer.acceptor_cm3,
                             thickness_um=wafer.thickness_um, temperature_k=wafer.temperature_k)
        model = BulkModel(varied_wafer, tau_srh_s=tau_srh_s)
        ax.loglog(injection_grid, [model.injection(x).tau_total_s*1e9 for x in injection_grid],
                  color=color, label=f"B, C scale = {scale:g}")
    ax.set(title=label, xlabel="$\\Delta n$ (cm$^{-3}$)")
axes[0].set_ylabel("Total effective lifetime (ns)")
axes[0].legend(fontsize=8)
fig.tight_layout()
"""),
    markdown(r"""
## Interpretation and release-0.2 decision

- FD charge neutrality, screening, generalized Einstein transport, and the
  recombination bookkeeping pass internal checks.
- The compact n-InP mobility estimate agrees reasonably with the supplied
  measured mobility/resistivity; the p-type mobility still needs a wafer-specific
  Hall cross-check.
- BGN is an optional scenario, not yet a calibrated dopant-specific InP model.
- The spread under coefficient scaling is a reminder that absolute lifetime and
  diffusion length should not be reported without uncertainty bounds.
- These outputs are suitable inputs to the optical and later drift-diffusion
  stages, but they are not yet a prediction of measured PL or TRPL.
"""),
])


write("02_optical_generation.ipynb", [
    markdown(r"""
# Front optics and pump generation — release 0.2

This notebook is the first optical layer: coherent normal-incidence propagation
through thin POx/AlOx films followed by Beer–Lambert absorption in a thick InP
substrate. It answers how much pump power enters the wafer and where carriers
are generated. It deliberately does **not** yet claim a PL collection efficiency
or photon-recycling factor.
"""),
    code(r"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

root = Path.cwd()
if not (root / "src").exists(): root = root.parent
sys.path.insert(0, str(root / "src"))
from surpass import (BulkModel, Layer, OpticalStack, Wafer,
                        get_optical_material, single_pass_escape_probability)

try:
    import scienceplots
    plt.style.use(["science", "notebook", "no-latex"])
except ImportError:
    plt.style.use("default")
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})
"""),
    markdown("""
## 1. Editable optical experiment

The film thicknesses and optical constants below are placeholders. Replace the
demo dispersions with measured ellipsometry before quantitative fitting.
"""),
    code(r"""
pump_wavelength_nm = 514.0
wafer_thickness_um = 620.0
pox_thickness_nm = 10.0
alox_thickness_nm = 10.0
intensity_n_w_cm2 = 1.0
intensity_p_w_cm2 = 13.0

air = get_optical_material("air")
inp = get_optical_material("InP_demo")
pox = get_optical_material("POx_demo")
alox = get_optical_material("AlOx_demo")

bare = OpticalStack(air, [], inp)
coated = OpticalStack(air, [Layer(pox, pox_thickness_nm, "POx"),
                            Layer(alox, alox_thickness_nm, "AlOx")], inp)

for item in (inp, pox, alox):
    print(f"{item.name}: N({pump_wavelength_nm:g} nm)={item.nk(pump_wavelength_nm)}, "
          f"status={item.status}")
    print("  source:", item.source)
    if item.note: print("  caution:", item.note)
"""),
    markdown("## 2. Energy-balance and absorption-depth checks"),
    code(r"""
rows = []
for name, stack in [("bare InP", bare), ("POx/AlOx/InP", coated)]:
    result = stack.solve(pump_wavelength_nm)
    rows.append({"stack": name, "R": result.reflectance,
                 "power entering InP": result.substrate_entry_fraction,
                 "thin-film absorption": sum(result.layer_absorptance),
                 "balance error": result.balance_error,
                 "InP alpha (cm^-1)": result.substrate_alpha_cm1,
                 "1/e depth (nm)": result.absorption_depth_nm})
display(pd.DataFrame(rows).style.format(precision=5))
"""),
    markdown("## 3. Pump-generation profiles"),
    code(r"""
depth_nm = np.linspace(0, 1500, 1501)
depth_cm = depth_nm*1e-7
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for stack_name, stack, intensity, color in [
    ("bare, 1 W cm$^{-2}$", bare, intensity_n_w_cm2, "black"),
    ("coated, 1 W cm$^{-2}$", coated, intensity_n_w_cm2, "tab:blue"),
    ("coated, 13 W cm$^{-2}$", coated, intensity_p_w_cm2, "tab:red")]:
    generation = stack.generation_profile_cm3_s(pump_wavelength_nm, intensity, depth_cm)
    axes[0].plot(depth_nm, generation, label=stack_name, color=color)
    axes[1].semilogy(depth_nm, generation, label=stack_name, color=color)
for ax in axes:
    ax.set(xlabel="Depth into InP (nm)", ylabel="$G(z)$ (cm$^{-3}$ s$^{-1}$)")
axes[0].legend(fontsize=8)
axes[0].set_title("Linear scale"); axes[1].set_title("Log scale")
fig.tight_layout()
"""),
    markdown("## 4. Optical depth versus transport length scales"),
    code(r"""
comparison = []
for label, wafer in {
    "n-InP": Wafer("InP", donor_cm3=8.5e18, thickness_um=wafer_thickness_um),
    "p-InP": Wafer("InP", acceptor_cm3=5.1e18, thickness_um=wafer_thickness_um),
}.items():
    state = BulkModel(wafer, tau_srh_s=100e-9).injection(1e15)
    comparison.append({"wafer": label, "diffusion length at dn=1e15 (um)": state.diffusion_length_um,
                       "pump 1/e depth (um)": coated.solve(pump_wavelength_nm).absorption_depth_nm/1000,
                       "wafer thickness (um)": wafer_thickness_um})
display(pd.DataFrame(comparison).style.format(precision=3))
"""),
    markdown(r"""
The pump is absorbed near the front surface, while a diffusion length can be
orders of magnitude larger. Consequently, a rear boundary condition cannot be
chosen from the optical absorption depth alone; the later transport solver must
compare wafer thickness, injection-dependent diffusion length, and both surface
recombination velocities.
"""),
    markdown("## 5. A bounded escape-cone check (not yet a PL model)"),
    code(r"""
emission_wavelength_nm = 920.0
n_emission = inp.nk(emission_wavelength_nm).real
p_escape = single_pass_escape_probability(n_emission)
display(pd.DataFrame({
    "emission wavelength (nm)": [emission_wavelength_nm],
    "demo InP index": [n_emission],
    "single-pass escape probability": [p_escape],
}))
"""),
    markdown(r"""
This escape value integrates Fresnel transmission over one planar escape cone
and normalizes to the full isotropic photon population. It excludes reabsorption,
re-emission, repeated reflections, thin-film interference at emission
wavelengths, free-carrier absorption, and collection numerical aperture.

### Next optical/transport increment

1. replace demo $n,k$ with tabulated or measured spectra;
2. add finite-wafer incoherent rear reflections;
3. solve 1D generation–diffusion–recombination with surface boundary conditions;
4. add wavelength-resolved spontaneous emission, reabsorption, and photon
   recycling; and
5. integrate the emitted flux through the actual collection optics for PL and
   linearize the time-dependent solver for TRPL.
"""),
])


write("03_steady_state_1d_transport.ipynb", [
    markdown(r"""
# 1D generation–diffusion–recombination solver — release 0.3

This notebook connects the bulk and optical modules. It solves the steady-state
excess-carrier continuity equation

$$\frac{dJ}{dx}=G(x)-R(\Delta n),\qquad
J=-D_a(\Delta n)\frac{d\Delta n}{dx}$$

with independent front and rear surface recombination velocities. The local
$D_a$, radiative, Auger, and bulk-SRH rates come from the Fermi–Dirac bulk model.

The result is a prediction of the carrier profile and **internal radiative
recombination flux**, not yet the detected PL signal: photon recycling,
reabsorption, collection optics, and surface band bending are not included.
"""),
    code(r"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

root = Path.cwd()
if not (root / "src").exists(): root = root.parent
sys.path.insert(0, str(root / "src"))
from surpass import (BulkModel, BulkTransportTable, Layer, OpticalStack,
                        SteadyState1DSolver, Wafer, get_optical_material)

try:
    import scienceplots
    plt.style.use(["science", "notebook", "no-latex"])
except ImportError:
    plt.style.use("default")
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})
"""),
    markdown(r"""
## 1. Editable experiment and numerical grid

The near-surface grid is essential: at 514 nm the demonstration InP absorption
depth is only about 0.09 µm, versus a 620 µm wafer. The optical module supplies
the **exact cell-averaged** Beer–Lambert generation, avoiding sampling error in
the steep first few cells.
"""),
    code(r"""
wavelength_nm = 514.0
thickness_um = 620.0
thickness_cm = thickness_um*1e-4
pox_nm, alox_nm = 10.0, 10.0
tau_bulk_srh_s = 100e-9       # assumption, not a material constant
enable_bgn = False
rear_s_default = 1e5          # cm/s; deliberately pessimistic starting scenario
front_s_values = [10.0, 1e3, 1e5]

# Intensities from the current experimental comparison; keep editable.
wafer_settings = {
    "n-InP": dict(wafer=Wafer("InP", donor_cm3=8.5e18, thickness_um=thickness_um),
                  intensity=1.0, color="tab:blue"),
    "p-InP": dict(wafer=Wafer("InP", acceptor_cm3=5.1e18, thickness_um=thickness_um),
                  intensity=13.0, color="tab:red"),
}

air = get_optical_material("air")
inp = get_optical_material("InP_demo")
pox = get_optical_material("POx_demo")
alox = get_optical_material("AlOx_demo")
stack = OpticalStack(air, [Layer(pox, pox_nm, "POx"), Layer(alox, alox_nm, "AlOx")], inp)

# 0-5 um: logarithmically refined; 5-620 um: progressively coarse.
edges_cm = np.r_[0.0, np.geomspace(1e-8, 5e-4, 140),
                 np.linspace(5e-4, thickness_cm, 161)[1:]]
centres_um = 0.5*(edges_cm[:-1] + edges_cm[1:])*1e4
print(f"Finite-volume cells: {len(centres_um)}; first/last widths = "
      f"{np.diff(edges_cm)[0]*1e7:.3g}/{np.diff(edges_cm)[-1]*1e4:.3g} nm/um")
print(f"Optical 1/e depth = {stack.solve(wavelength_nm).absorption_depth_nm:.2f} nm")
"""),
    markdown("## 2. Build injection-dependent transport tables"),
    code(r"""
solvers, generations = {}, {}
for label, settings in wafer_settings.items():
    bulk = BulkModel(settings["wafer"], enable_bgn=enable_bgn, tau_srh_s=tau_bulk_srh_s)
    table = BulkTransportTable.from_bulk_model(
        bulk, delta_min_cm3=1e5, delta_max_cm3=1e21, points=241
    )
    solvers[label] = SteadyState1DSolver(table)
    generations[label] = stack.generation_cell_average_cm3_s(
        wavelength_nm, settings["intensity"], edges_cm
    )

fig, ax = plt.subplots(figsize=(6, 3.5))
for label, generation in generations.items():
    ax.loglog(centres_um, generation, label=label)
ax.axvline(stack.solve(wavelength_nm).absorption_depth_nm/1000, color="0.4", ls="--",
           label="optical 1/e depth")
ax.set(xlabel="Depth (µm)", ylabel="$G$ (cm$^{-3}$ s$^{-1}$)")
ax.legend(fontsize=8); fig.tight_layout()
"""),
    markdown(r"""
## 3. Front-surface passivation sweep

Here $S_f$ is phenomenological. A later electrostatic/interface-SRH module will
calculate an injection-dependent boundary rate from $D_{it}$, capture cross
sections, and fixed charge instead of prescribing one constant velocity.
"""),
    code(r"""
results, rows = {}, []
for label, settings in wafer_settings.items():
    for front_s in front_s_values:
        result = solvers[label].solve(
            thickness_cm, generations[label], front_s, rear_s_default,
            cell_edges_cm=edges_cm, max_nfev=1200,
        )
        results[(label, front_s)] = result
        rows.append({
            "wafer": label, "intensity (W/cm2)": settings["intensity"],
            "Sf (cm/s)": front_s, "Sr (cm/s)": rear_s_default,
            "<dn> (cm^-3)": result.average_delta_n_cm3,
            "tau_eff (ns)": 1e9*result.effective_lifetime_s,
            "front fraction": result.front_surface_recombination_cm2_s/result.generated_flux_cm2_s,
            "rear fraction": result.rear_surface_recombination_cm2_s/result.generated_flux_cm2_s,
            "bulk rad fraction": result.internal_radiative_yield,
            "bulk Auger fraction": result.bulk_auger_recombination_cm2_s/result.generated_flux_cm2_s,
            "bulk SRH fraction": result.bulk_srh_recombination_cm2_s/result.generated_flux_cm2_s,
            "internal rad flux (cm^-2 s^-1)": result.bulk_radiative_recombination_cm2_s,
            "balance error": result.relative_balance_error,
            "max cell residual": result.maximum_scaled_cell_residual,
            "converged": result.converged,
        })
summary = pd.DataFrame(rows)
display(summary.style.format({
    "<dn> (cm^-3)": "{:.3e}", "tau_eff (ns)": "{:.3g}",
    "front fraction": "{:.3f}", "rear fraction": "{:.3f}",
    "bulk rad fraction": "{:.3f}", "bulk Auger fraction": "{:.3f}",
    "bulk SRH fraction": "{:.3f}", "internal rad flux (cm^-2 s^-1)": "{:.3e}",
    "balance error": "{:.2e}", "max cell residual": "{:.2e}"}))
"""),
    code(r"""
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for row, (label, settings) in enumerate(wafer_settings.items()):
    for front_s in front_s_values:
        result = results[(label, front_s)]
        axes[row, 0].semilogx(result.depth_cm*1e4, result.delta_n_cm3,
                              label=f"$S_f$={front_s:g} cm/s")
        axes[row, 1].plot(result.depth_cm*1e4, result.delta_n_cm3,
                         label=f"$S_f$={front_s:g} cm/s")
    axes[row, 0].set(title=f"{label}: near front", xlim=(1e-4, 20),
                     xlabel="Depth (µm)", ylabel="$\\Delta n$ (cm$^{-3}$)")
    axes[row, 1].set(title=f"{label}: full wafer", xlim=(0, thickness_um),
                     xlabel="Depth (µm)", ylabel="$\\Delta n$ (cm$^{-3}$)")
axes[0, 0].legend(fontsize=8)
fig.tight_layout()
"""),
    markdown("## 4. Does the rear surface matter?"),
    code(r"""
rear_s_values = [0.0, 1e2, 1e4, 1e6]
front_s_reference = 1e3
rear_rows = []
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
for ax, (label, settings) in zip(axes, wafer_settings.items()):
    reference_flux = None
    for rear_s in rear_s_values:
        result = solvers[label].solve(
            thickness_cm, generations[label], front_s_reference, rear_s,
            cell_edges_cm=edges_cm, max_nfev=1200,
        )
        if reference_flux is None: reference_flux = result.bulk_radiative_recombination_cm2_s
        rear_rows.append({"wafer": label, "Sr (cm/s)": rear_s,
                          "tau_eff (ns)": 1e9*result.effective_lifetime_s,
                          "internal rad flux / reflecting rear":
                              result.bulk_radiative_recombination_cm2_s/reference_flux,
                          "rear recombination fraction":
                              result.rear_surface_recombination_cm2_s/result.generated_flux_cm2_s})
        ax.plot(result.depth_cm*1e4, result.delta_n_cm3, label=f"$S_r$={rear_s:g}")
    ax.set(title=label, xlabel="Depth (µm)", ylabel="$\\Delta n$ (cm$^{-3}$)")
    ax.legend(fontsize=8)
fig.tight_layout()
display(pd.DataFrame(rear_rows).style.format(precision=4))
"""),
    markdown(r"""
## Interpretation and limitations

- The injection level is now an **output** of absorbed pump flux, bulk
  recombination, diffusion, wafer thickness, and the two boundary conditions;
  it is no longer chosen arbitrarily as in a $J_0$ normalization.
- The calculated internal radiative flux is the cleanest first proxy for PL.
  Multiplying it by a fixed escape factor is only qualitative because photon
  recycling makes escape and bulk recombination mutually coupled.
- Whether the rear matters is answered quantitatively by the $S_r$ sweep. A
  shallow generation profile does not by itself make the rear irrelevant when
  the diffusion length is comparable to the wafer thickness.
- Constant $S_f$ cannot yet represent field-effect passivation. The next
  release should couple the surface-potential/interface-SRH calculation to the
  boundary flux, ideally as $U_s(\Delta n_s,Q_f,D_{it},\sigma_n,\sigma_p)$.
- TRPL requires the time-dependent form of the same finite-volume system and a
  model for the detected photon flux; it should follow after the steady-state
  boundary model is validated.
"""),
])


write("04_fixed_charge_surface_coupling.ipynb", [
    markdown(r"""
# Fixed-charge/interface-SRH coupling — release 0.4

This notebook replaces the prescribed front-surface recombination velocity by
the chain

$$Q_f\rightarrow\psi_s\rightarrow(n_s,p_s)\rightarrow
U_s(\Delta n_s)\rightarrow\Delta n(x).$$

The surface electrostatics uses Fermi–Dirac statistics and can be recalculated
self-consistently at each injection level. The interface kinetics uses a
distributed SRH model and is coupled as a nonlinear boundary condition to the
steady-state 1D transport solver.

Important limitations: $Q_{it}=0$, there is no Fermi-level pinning, the
Adamowicz parameters are not measurements of POx/InP, and the reported
``PL proxy`` is internal radiative recombination rather than detected PL.
"""),
    code(r"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

root = Path.cwd()
if not (root / "src").exists(): root = root.parent
sys.path.insert(0, str(root / "src"))
from surpass import (ADAMOWICZ_INP, BulkModel, BulkTransportTable,
                        InterfaceDefectModel, Layer, OpticalStack,
                        SteadyState1DSolver, SurfaceBoundaryTable,
                        SurfaceSRHModel, Wafer, get_optical_material)

try:
    import scienceplots
    plt.style.use(["science", "notebook", "no-latex"])
except ImportError:
    plt.style.use("default")
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})
"""),
    markdown("## 1. Editable physical assumptions"),
    code(r"""
wavelength_nm = 514.0
thickness_um = 620.0
thickness_cm = thickness_um*1e-4
tau_bulk_srh_s = 100e-9
rear_s_cm_s = 1e5
enable_bgn = False
qf_values = np.linspace(-6e12, 6e12, 13)  # stack charge + applied corona charge

defects = InterfaceDefectModel(
    dit_mid_ev1_cm2=1e11,
    sigma_n_cm2=1e-14,
    sigma_p_cm2=1e-13,
    vth_n_cm_s=4.13e7,
    vth_p_cm_s=1.51e7,
    shape="constant",
    energy_points=301,
)
print(defects.source)

wafer_settings = {
    "n-InP": dict(wafer=Wafer("InP", donor_cm3=8.5e18, thickness_um=thickness_um),
                  intensity=1.0, color="tab:blue"),
    "p-InP": dict(wafer=Wafer("InP", acceptor_cm3=5.1e18, thickness_um=thickness_um),
                  intensity=13.0, color="tab:red"),
}

air, inp = get_optical_material("air"), get_optical_material("InP_demo")
pox, alox = get_optical_material("POx_demo"), get_optical_material("AlOx_demo")
stack = OpticalStack(air, [Layer(pox, 10.0, "POx"), Layer(alox, 10.0, "AlOx")], inp)
edges_cm = np.r_[0.0, np.geomspace(1e-8, 5e-4, 140),
                 np.linspace(5e-4, thickness_cm, 161)[1:]]
centres_um = 0.5*(edges_cm[:-1]+edges_cm[1:])*1e4
"""),
    markdown(r"""
## 2. Surface-only charge response

The reference injection below is used only to visualize the boundary model.
The coupled solver later determines the actual surface injection itself.
"""),
    code(r"""
delta_reference = 1e11
surface_rows = []
surface_models = {}
for label, settings in wafer_settings.items():
    bulk = BulkModel(settings["wafer"], enable_bgn=enable_bgn, tau_srh_s=tau_bulk_srh_s)
    for qf in qf_values:
        model = SurfaceSRHModel(bulk, qf, defects,
                                electrostatics="self_consistent",
                                driving_force="excess_product")
        state = model.state(delta_reference)
        surface_models[(label, qf)] = model
        surface_rows.append({"wafer": label, "Qf (cm^-2)": qf,
                             "psi_s (V)": state.psi_surface_v,
                             "n_s (cm^-3)": state.n_surface_cm3,
                             "p_s (cm^-3)": state.p_surface_cm3,
                             "U_s (cm^-2 s^-1)": state.recombination_cm2_s,
                             "S_eff=U_s/dn (cm/s)": state.effective_s_cm_s})
surface_df = pd.DataFrame(surface_rows)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
for label, settings in wafer_settings.items():
    subset = surface_df[surface_df.wafer == label]
    axes[0].plot(subset["Qf (cm^-2)"]/1e12, subset["psi_s (V)"],
                 marker="o", label=label, color=settings["color"])
    axes[1].semilogy(subset["Qf (cm^-2)"]/1e12,
                     subset["S_eff=U_s/dn (cm/s)"], marker="o",
                     label=label, color=settings["color"])
axes[0].set(xlabel="$Q_f/q$ ($10^{12}$ cm$^{-2}$)", ylabel="$\\psi_s$ (V)")
axes[1].set(xlabel="$Q_f/q$ ($10^{12}$ cm$^{-2}$)", ylabel="$U_s/\\Delta n_b$ (cm s$^{-1}$)")
axes[0].legend(); axes[1].legend(); fig.tight_layout()
"""),
    markdown(r"""
The effective velocity above is normalized to the **bulk-side** excess density.
It may greatly exceed a microscopic capture velocity in strong inversion,
because field effect concentrates the minority carrier at the surface. In the
transport solution, this does not imply an unlimited rate: the surface excess
density is depleted until diffusive supply balances $U_s$.
"""),
    markdown("## 3. Couple every fixed-charge point to optical generation and transport"),
    code(r"""
bulk_solvers, generations = {}, {}
for label, settings in wafer_settings.items():
    bulk = BulkModel(settings["wafer"], enable_bgn=enable_bgn, tau_srh_s=tau_bulk_srh_s)
    bulk_solvers[label] = SteadyState1DSolver(BulkTransportTable.from_bulk_model(
        bulk, delta_min_cm3=1.0, delta_max_cm3=1e21, points=261))
    generations[label] = stack.generation_cell_average_cm3_s(
        wavelength_nm, settings["intensity"], edges_cm)

coupled_results, coupled_rows = {}, []
for label, settings in wafer_settings.items():
    for qf in qf_values:
        boundary = SurfaceBoundaryTable.from_surface_model(
            surface_models[(label, qf)], delta_min_cm3=1.0,
            delta_max_cm3=1e21, points=181, label=f"{label}, Qf={qf:.2e}")
        result = bulk_solvers[label].solve(
            thickness_cm, generations[label], boundary, rear_s_cm_s,
            cell_edges_cm=edges_cm, max_nfev=1500)
        coupled_results[(label, qf)] = result
        coupled_rows.append({
            "wafer": label, "Qf (cm^-2)": qf,
            "<dn> (cm^-3)": result.average_delta_n_cm3,
            "dn_surface (cm^-3)": result.front_surface_delta_cm3,
            "effective Sf (cm/s)": result.front_effective_s_cm_s,
            "tau_eff (ns)": result.effective_lifetime_s*1e9,
            "surface fraction": result.front_surface_recombination_cm2_s/result.generated_flux_cm2_s,
            "bulk radiative fraction": result.internal_radiative_yield,
            "internal radiative flux": result.bulk_radiative_recombination_cm2_s,
            "balance error": result.relative_balance_error,
            "converged": result.converged,
        })
coupled_df = pd.DataFrame(coupled_rows)
if not coupled_df.converged.all():
    print("WARNING: unconverged charge points:")
    display(coupled_df.loc[~coupled_df.converged,
                           ["wafer", "Qf (cm^-2)", "balance error"]])
"""),
    code(r"""
fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
for label, settings in wafer_settings.items():
    subset = coupled_df[coupled_df.wafer == label].copy()
    q = subset["Qf (cm^-2)"]/1e12
    reference = float(subset.loc[np.isclose(subset["Qf (cm^-2)"], 0),
                                 "internal radiative flux"].iloc[0])
    axes[0, 0].plot(q, subset["internal radiative flux"]/reference,
                    marker="o", label=label, color=settings["color"])
    axes[0, 1].semilogy(q, subset["effective Sf (cm/s)"],
                        marker="o", label=label, color=settings["color"])
    axes[1, 0].plot(q, subset["tau_eff (ns)"],
                    marker="o", label=label, color=settings["color"])
    axes[1, 1].plot(q, subset["surface fraction"],
                    marker="o", label=label, color=settings["color"])
axes[0, 0].set(ylabel="Internal radiative flux / value at $Q_f=0$")
axes[0, 1].set(ylabel="Coupled effective $S_f$ (cm s$^{-1}$)")
axes[1, 0].set(xlabel="$Q_f/q$ ($10^{12}$ cm$^{-2}$)", ylabel="Effective lifetime (ns)")
axes[1, 1].set(xlabel="$Q_f/q$ ($10^{12}$ cm$^{-2}$)", ylabel="Front-surface recombination fraction")
axes[0, 0].legend(); axes[0, 1].legend(); fig.tight_layout()
"""),
    markdown("## 4. Carrier profiles at negative, zero, and positive charge"),
    code(r"""
selected_qf = [-4e12, 0.0, 4e12]
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for row, (label, settings) in enumerate(wafer_settings.items()):
    for qf in selected_qf:
        result = coupled_results[(label, qf)]
        axes[row, 0].semilogx(result.depth_cm*1e4, result.delta_n_cm3,
                              label=f"$Q_f/q$={qf/1e12:+g}")
        axes[row, 1].plot(result.depth_cm*1e4, result.delta_n_cm3,
                         label=f"$Q_f/q$={qf/1e12:+g}")
    axes[row, 0].set(title=f"{label}: near front", xlim=(1e-4, 20),
                     xlabel="Depth (µm)", ylabel="$\\Delta n$ (cm$^{-3}$)")
    axes[row, 1].set(title=f"{label}: full wafer", xlim=(0, thickness_um),
                     xlabel="Depth (µm)", ylabel="$\\Delta n$ (cm$^{-3}$)")
axes[0, 0].legend(fontsize=8); fig.tight_layout()
"""),
    markdown("## 5. Is illumination-dependent band bending important here?"),
    code(r"""
injection_sweep = np.logspace(6, 17, 80)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
comparison_rows = []
for label, settings in wafer_settings.items():
    bulk = BulkModel(settings["wafer"], enable_bgn=enable_bgn, tau_srh_s=tau_bulk_srh_s)
    for qf in (-4.2e12, 4.2e12):
        equilibrium = SurfaceSRHModel(bulk, qf, defects, electrostatics="equilibrium")
        self_consistent = SurfaceSRHModel(bulk, qf, defects, electrostatics="self_consistent")
        eq_states = [equilibrium.state(d) for d in injection_sweep]
        sc_states = [self_consistent.state(d) for d in injection_sweep]
        key = f"{label}, {qf/1e12:+.1f}e12"
        axes[0].semilogx(injection_sweep,
                         [s.psi_surface_v for s in sc_states], label=key)
        axes[0].semilogx(injection_sweep,
                         [s.psi_surface_v for s in eq_states], ls="--", color=axes[0].lines[-1].get_color())
        axes[1].loglog(injection_sweep,
                       [s.recombination_cm2_s for s in sc_states], label=key)
        axes[1].loglog(injection_sweep,
                       [s.recombination_cm2_s for s in eq_states], ls="--", color=axes[1].lines[-1].get_color())
axes[0].set(xlabel="$\\Delta n_b$ (cm$^{-3}$)", ylabel="$\\psi_s$ (V)")
axes[1].set(xlabel="$\\Delta n_b$ (cm$^{-3}$)", ylabel="$U_s$ (cm$^{-2}$ s$^{-1}$)")
axes[0].legend(fontsize=7); axes[1].legend(fontsize=7); fig.tight_layout()
"""),
    markdown(r"""
Solid curves recalculate the space charge at every injection; dashed curves
freeze the equilibrium band bending. Their agreement or divergence gives a
direct test of whether the low-injection electrostatic approximation is valid.

## Interpretation and next validation priorities

- The model now predicts the injection level instead of imposing it, and fixed
  charge enters through the physical surface boundary rather than through a
  post-processed $J_{0s}$.
- It should reproduce the qualitative corona trend: negative added charge
  increases the n-InP surface loss and decreases the p-InP loss over the
  relevant branch, while turning points remain possible.
- Absolute PL changes remain uncertain because $D_{it}(E)$, capture cross
  sections, the bulk coefficients, and photon recycling are not independently
  known.
- Interface-state charge is currently omitted. Adding amphoteric $Q_{it}$ and
  solving $Q_f+Q_{it}+Q_{sc}=0$ is the most important electrostatic extension
  if pinning is appreciable.
- The next numerical layer should be time dependent. Before fitting biexponential
  TRPL, however, bulk coefficients and the optical escape/recycling model should
  be constrained, otherwise multiple parameter combinations will be
  indistinguishable.
"""),
])

write("05_interface_charge_and_pinning.ipynb", [
    markdown(r"""
# Interface-state charge and Fermi-level pinning — release 0.5

Release 0.5 solves the surface charge balance

$$Q_\mathrm{ext}(\psi_s)+Q_{it}(\psi_s,\Delta n)+Q_{sc}(\psi_s,\Delta n)=0.$$

The same distributed defects now contribute both SRH recombination and charge.
States below an adjustable charge-neutrality level (CNL) are donor-like
(positive when empty); states above it are acceptor-like (negative when
occupied). Their steady-state occupancy is calculated from electron and hole
capture and emission rates.

The CNL and donor/acceptor partition are **sensitivity parameters**, not claimed
material constants. Degenerate carrier populations are used in Poisson's
equation, but fully degenerate trap kinetics with Pauli blocking is not yet
included.
"""),
    code(r"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

root = Path.cwd()
if not (root / "src").exists(): root = root.parent
sys.path.insert(0, str(root / "src"))
from surpass import (BulkModel, InterfaceChargeModel, InterfaceDefectModel,
                        SurfaceSRHModel, Wafer)

try:
    import scienceplots
    plt.style.use(["science", "notebook", "no-latex"])
except ImportError:
    plt.style.use("default")
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})
"""),
    markdown("## 1. Editable interface assumptions"),
    code(r"""
qf_values = np.linspace(-6e12, 6e12, 61)
dit_values = [1e9, 1e11, 1e13, 1e15]
cnl_fraction = 0.50
transition_width_ev = 0.02
delta_probe_cm3 = 1e11

wafers = {
    "n-InP": Wafer("InP", donor_cm3=8.5e18, thickness_um=620),
    "p-InP": Wafer("InP", acceptor_cm3=5.1e18, thickness_um=620),
}
charge_model = InterfaceChargeModel(cnl_fraction, transition_width_ev)

def make_model(wafer, qf, dit, charge=True):
    defects = InterfaceDefectModel(
        dit_mid_ev1_cm2=dit, sigma_n_cm2=1e-14, sigma_p_cm2=1e-13,
        shape="constant", energy_points=301)
    return SurfaceSRHModel(BulkModel(wafer), qf, defects,
                           interface_charge=charge_model if charge else None)
"""),
    markdown(r"""
## 2. Pinning and charge partition

The plotted trap-control fraction is
$C_{it}/(C_{it}+C_{sc})$, using local differential capacitances. It is a useful
electrostatic diagnostic, but it is not the classical Schottky-barrier slope
parameter.
"""),
    code(r"""
fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
rows = []
for column, (label, wafer) in enumerate(wafers.items()):
    for dit in dit_values:
        states = [make_model(wafer, qf, dit).state(0.0) for qf in qf_values]
        axes[0, column].plot(qf_values/1e12, [s.psi_surface_v for s in states],
                             label=f"$D_{{it}}={dit:.0e}$")
        axes[1, column].plot(qf_values/1e12,
                             [s.trap_control_fraction for s in states])
        for qf, state in zip(qf_values, states):
            rows.append({"wafer": label, "Dit": dit, "Qext/q": qf,
                         "psi_s": state.psi_surface_v,
                         "Qit/q": state.q_it_number_cm2,
                         "Qsc/q": state.q_sc_number_cm2,
                         "balance/q": state.charge_balance_number_cm2,
                         "trap control": state.trap_control_fraction})
    axes[0, column].set(title=label, ylabel="$\\psi_s$ (V)")
    axes[1, column].set(xlabel="$Q_{ext}/q$ ($10^{12}$ cm$^{-2}$)",
                         ylabel="$C_{it}/(C_{it}+C_{sc})$")
axes[0, 0].legend(fontsize=8)
fig.tight_layout()
charge_df = pd.DataFrame(rows)
print("Maximum absolute charge-balance error:",
      f"{charge_df['balance/q'].abs().max():.3g} elementary charges cm^-2")
"""),
    markdown("## 3. The same traps affect recombination and electrostatics"),
    code(r"""
fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
for ax, (label, wafer) in zip(axes, wafers.items()):
    for enabled, style in [(False, "--"), (True, "-")]:
        states = [make_model(wafer, qf, 1e13, charge=enabled).state(delta_probe_cm3)
                  for qf in qf_values]
        ax.semilogy(qf_values/1e12, [s.recombination_cm2_s for s in states],
                    ls=style, label="$Q_{it}$ on" if enabled else "$Q_{it}=0$")
    ax.set(title=label, xlabel="$Q_{ext}/q$ ($10^{12}$ cm$^{-2}$)",
           ylabel="$U_s$ (cm$^{-2}$ s$^{-1}$)")
    ax.legend()
fig.tight_layout()
"""),
    markdown("## 4. CNL sensitivity"),
    code(r"""
fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
for ax, (label, wafer) in zip(axes, wafers.items()):
    for cnl in (0.35, 0.50, 0.65):
        local_charge = InterfaceChargeModel(cnl, transition_width_ev)
        defects = InterfaceDefectModel(dit_mid_ev1_cm2=1e13, energy_points=301)
        states = [SurfaceSRHModel(BulkModel(wafer), qf, defects,
                                  interface_charge=local_charge).state(0.0)
                  for qf in qf_values]
        ax.plot(qf_values/1e12, [s.psi_surface_v for s in states],
                label=f"CNL = {cnl:.2f}$E_g$")
    ax.set(title=label, xlabel="$Q_{ext}/q$ ($10^{12}$ cm$^{-2}$)",
           ylabel="$\\psi_s$ (V)")
    ax.legend(fontsize=8)
fig.tight_layout()
"""),
    markdown(r"""
## 5. Architecture for a later biased gate

`SurfaceSRHModel` accepts any external boundary object that implements
`charge_c_cm2(psi_surface_v)`. The present fixed-charge boundary returns a
constant. A later metal/oxide gate can instead return, for example,

$$Q_g=C_{ox}(V_g-V_{fb}-\psi_s),$$

and the same root solve will include gate bias, work-function difference,
oxide thickness, and fixed oxide charge. That later implementation should add
finite dielectric stacks and a clear voltage/reference convention; it is not
silently approximated in release 0.5.

## Interpretation

- With small $D_{it}$, semiconductor space charge balances most external
  charge and $\psi_s$ responds strongly.
- With large $D_{it}$, trap charge absorbs more of the perturbation and the
  surface Fermi level approaches the chosen CNL: this is pinning.
- Because the same $D_{it}(E)$ and capture coefficients set both occupancy and
  recombination, changing them can shift both band bending and surface loss.
- Quantitative extraction therefore needs independent constraints on the CNL,
  donor/acceptor character, capture cross sections, and fixed charge.
"""),
])

print(f"Wrote notebooks to {NB_DIR}")
