# SURPASS — Surface and Passivation Analysis for Semiconductor Systems — release 1.0.0

Release archives are flat: extract them directly
into the project root so that `src`, `notebooks`, and `pyproject.toml` replace
the corresponding older paths. Notebooks 08–10 print both
`surpass.__version__` and `surpass.__file__`, and stop explicitly if an
older source tree takes precedence.

This is the first layer of a planned surface-passivation/PL model. It calculates
bulk equilibrium, transport, recombination, diffusion, and screening properties
and now couples nonlinear steady-state carrier transport to fixed-charge
surface electrostatics, amphoteric interface-state charge, and distributed
interface-SRH recombination, transient diffusion, spectral emission escape,
optional local photon recycling, and an experimental-observation layer for
sampled excitation, detector response, IRF convolution, and time-bin counts.
Nonlocal reabsorption and calibrated prediction of absolute detected TRPL remain
later stages.

The implementation emphasizes traceability:

- every preset carries references and validity notes;
- empirical coefficients are kept separate from equations;
- uncertain coefficients can be overridden without editing solver code;
- Fermi–Dirac statistics are used for degenerate wafers;
- both classical and generalized diffusion/screening results are reported.

## Included in release 1.0.0

- Si, Ge, InP, and GaAs 300 K presets;
- Fermi–Dirac charge neutrality with complete dopant ionization;
- optional Palankovski screened-Coulomb band-gap narrowing;
- Caughey–Thomas-like doping-dependent mobility;
- generalized Einstein relation;
- radiative, Auger, and optional bulk-SRH recombination;
- ambipolar diffusion coefficient and diffusion length;
- Debye and generalized Fermi/Thomas–Fermi screening lengths;
- an example report and injection-dependent plots for the experimental InP wafers.
- an executed bulk-validation notebook with parameter and sensitivity checks;
- coherent normal-incidence transfer matrices for thin front films;
- a semi-infinite absorbing-substrate pump-generation profile;
- a first InP optical notebook comparing bare and POx/AlOx-coated surfaces;
- a single-interface luminescence escape estimate (not yet photon recycling).
- a nonlinear, cell-centred finite-volume solution of the 1D continuity equation;
- injection-dependent ambipolar diffusion and radiative/Auger/SRH rates;
- independent Robin boundary conditions at the front and rear surfaces;
- a nonuniform grid and exact cell-averaged optical generation for the large
  514-nm absorption-depth/wafer-thickness scale separation;
- carrier profiles, effective lifetime, internal radiative flux, recombination
  partitioning, and numerical conservation diagnostics;
- analytic validation against constant-lifetime diffusion solutions.
- Fermi–Dirac surface space charge with self-consistent injection dependence;
- signed dielectric/corona charge and optional frozen-equilibrium band bending;
- constant or U-shaped interface-state distributions with separate electron
  and hole capture cross sections;
- a nonlinear surface-SRH boundary coupled directly to diffusion;
- fixed-charge sweeps for the experimental n- and p-InP wafers;
- explicit comparison of equilibrium and illumination-dependent electrostatics.
- donor-like and acceptor-like interface-state charge from the same distributed
  defects used for SRH recombination;
- self-consistent solution of `Qext + Qit + Qsc = 0` under equilibrium or
  illumination;
- adjustable charge-neutrality level and donor/acceptor transition width;
- charge-partition, interface/semiconductor differential-capacitance, mean
  occupancy, and trap-control diagnostics;
- a modular external-charge boundary that preserves the fixed-charge API and
  is ready for a later voltage-driven metal/oxide gate boundary.
- spatial semiconductor band-diagram reconstruction using the same Poisson
  first integral as the surface boundary;
- potential, electric field, carrier concentrations, band edges, and flat
  (quasi-)Fermi energies on a surface-resolving depth grid;
- explicit 90/99% net-charge depths, 99% potential-relaxation depth, tail
  truncation, and numerical charge-integral error;
- local capture/emission trap-relaxation spectra and timescale checks before TRPL;
- an executed spatial-profile notebook comparing the space-charge extent with
  pump absorption, ambipolar diffusion, and wafer thickness.
- stiff BDF/Radau transient ambipolar diffusion on nonuniform finite-volume grids;
- instantaneous nonlinear front/rear surface-SRH boundaries with Qit and Qf;
- initial pulse-generated profiles, CW switch-off, and finite Gaussian pulses;
- internal radiative-emission traces, carrier redistribution, sheet-inventory
  lifetimes, and thresholded local PL-slope diagnostics;
- independently integrated generation and radiative/Auger/SRH/front/rear losses,
  with carrier-inventory conservation and explicit high-side table-range checks;
- analytic transient tests and a transient notebook with tolerance/grid checks.
- coherent angular emission-side thin-film optics and incoherent multiple
  wafer reflections, with explicit active/passive absorption partition;
- spectral/depth-dependent front/rear escape, active reabsorption, parasitic
  absorption, lossless trapped-ray and numerical photon-balance diagnostics;
- normal-axis NA-selected photon flux, distinct from total escaped light;
- optional fixed local instantaneous recycling in both CW and transient solvers;
- separate intrinsic emission, recycled generation and net radiative-loss fields;
- a bulk-recombination provenance audit and executed optical/recycling notebook.
- corrected passive-absorption sign convention in pump film transfer matrices;
  earlier coated-wafer pump numbers may shift. An absorbing-film regression
  now checks positive absorptance, energy balance and emission-side agreement.
- normalized sampled excitation pulses evaluated with shape-preserving PCHIP;
- finite or indefinitely repeated pulse trains with per-pulse dose semantics
  and automatic transient-solver step control;
- wavelength-dependent filter/optics/detector response applied to the complete
  escaped spectrum rather than to a single effective wavelength;
- normalized sampled IRFs, continuous convolution, finite time-bin integration,
  emitting-area/gain/background conversion and event-count predictions;
- nuisance-only alignment of a fixed physical trace to measured data using
  scale, timing offset and background, with Poisson deviance for count data or
  user-provided uncertainties for weighted least squares;
- optional one- and two-exponential descriptive fits with explicit fit windows
  and cautions against microscopic interpretation;
- an executed notebook demonstrating repetition pile-up, detector broadening,
  synthetic count recovery and fit-window dependence.

## Important scope notes

1. A material preset is not a claim that all literature agrees on one value.
   III-V radiative and Auger coefficients can depend strongly on doping,
   temperature, band filling, and measurement method.
2. The initial mobility presets use a transparent Caughey–Thomas form. They are
   suitable for estimates, but future releases should transcribe and test the
   complete Sotoodeh temperature-dependent parameterization for each III-V.
3. The Palankovski point-charge BGN model is a common screening model, not a
   dopant-specific experimental fit. It is disabled by default.
4. `tau_srh_s` is an assumed bulk lifetime, not a material constant.
5. The present bulk recombination equations use phenomenological `B`, `Cn`,
   and `Cp` coefficients. Recycling is now a separate optical-feedback term;
   intrinsic coefficients remain constant, not fully degeneracy-aware rates.
   See `BULK_RECOMBINATION_AUDIT.md` before interpreting absolute lifetimes.
6. The bundled POx, AlOx, and sampled InP optical constants are explicitly
   illustrative. Replace them with process ellipsometry and a full tabulation
   of the cited InP dataset before fitting experimental data.
7. `substrate_entry_fraction` is power crossing into a semi-infinite substrate,
   not rear-side transmission through a finite wafer.
8. The quasi-neutral wafer uses local charge neutrality. Release 0.5 derives
   the front boundary rate from Qf and interface states, but does not spatially
   resolve transport inside the space-charge region.
9. The internal radiative recombination flux is not the detected PL signal.
   Release 0.8 adds spectral angular escape and local recycling, but measured
   optical inputs, nonlocal feedback and calibrated detection remain required
   before quantitative PL fitting.
10. Amphoteric interface charge is included, but its CNL and donor/acceptor
    partition are sensitivity parameters. Dielectric dipoles and mobile ionic
    charge are not included.
11. The trap occupancy uses conventional capture-emission kinetics with
    Fermi-Dirac carrier populations. Fully degenerate trap kinetics including
    Pauli blocking remains a future refinement.
12. The default degenerate-SRH driving force subtracts the equilibrium surface
    carrier product and recovers the Boltzmann expression. Fully degenerate
    trap kinetics with Pauli blocking is not yet implemented.
13. Spatial profiles are local, semi-infinite, uniformly doped semiconductor
    solutions with flat quasi-Fermi levels. They are not full Poisson–drift–
    diffusion solutions or vacuum-referenced dielectric/metal band diagrams.
    BGN is uniform when enabled; the energy zero is bulk Ev.
14. Charge and potential approach their bulk limits asymptotically, so there
    is no unique sharp space-charge width. Profile depths use stated 90/99%
    thresholds. `semi_infinite_warning` flags either 99% depth exceeding 10%
    of wafer thickness, but does not solve the resulting finite-wafer problem.
15. Trap-relaxation times assume fixed local carriers and potential. They do
    not include electrostatic feedback and are not a dynamic trap solver.
16. Transient surface potential and trap occupancy are instantaneous local
    closures. SCR storage, generation and transport are not explicitly resolved.
17. The internal emission trace is an excess photon-production rate, not a
    detector count. Spectral escape and approximate local recycling are now
    included; IRF, detector response and repeated-pulse excitation remain future stages.
18. A local PL-slope time constant is not necessarily the carrier lifetime:
    emission changes from approximately linear to quadratic with injection,
    and spatial redistribution can also change the slope.
19. Active reabsorption regenerates carriers in their emission cell. This
    approximation conserves global inventory but does not model redistribution
    by photons. Probabilities and spectrum are fixed through a solve.
20. Emission-side layers must be ordered from wafer to outside, reversing the
    outside-to-wafer pump order. Bulk attenuation is entered explicitly and
    separately from real-index Fresnel calculations. NA is a normal-axis cone,
    not a calibrated collection setup. Photon flight time is neglected.
21. Transparent TIR rays are reported as trapped, not recycled. Avoid treating
    static photon probabilities as a time-dependent high-Q photon reservoir.
22. A sampled pulse is normalized over the supplied support. Missing pulse
    tails or an unsubtracted monitor baseline therefore bias its physical dose.
23. IRF convolution assumes the modeled source is zero outside its supplied
    time window. Include sufficient pre/post history to avoid lost events.
24. Spectral response is a dimensionless probability in [0,1]. Keep geometric
    NA in the emission model and do not double-count filters or quantum efficiency.
25. `emitting_area_cm2` maps per-area collected flux to event rate. It must
    represent the imaged emitting region; laser spot, collection aperture and
    detector-active area are not generally interchangeable.
26. The nuisance fit does not infer semiconductor parameters. Its scale remains
    non-absolute unless optical throughput, area and gain are independently calibrated.
27. Empirical exponential fits summarize a selected window; their components
    are not uniquely identifiable recombination mechanisms. The present fit
    does not itself forward-convolve exponentials with an IRF.

## Install and run

From this directory:

```bash
python -m pip install -e .
python examples/inp_wafer_report.py
python -m unittest discover -s tests -v
jupyter lab notebooks/examples/01_bulk_model_validation.ipynb
jupyter lab notebooks/examples/02_optical_generation.ipynb
jupyter lab notebooks/examples/03_steady_state_1d_transport.ipynb
jupyter lab notebooks/examples/04_fixed_charge_surface_coupling.ipynb
jupyter lab notebooks/examples/05_interface_charge_and_pinning.ipynb
jupyter lab notebooks/examples/06_spatial_band_profiles.ipynb
jupyter lab notebooks/examples/07_transient_transport_trpl.ipynb
jupyter lab notebooks/examples/08_spectral_escape_recycling.ipynb
jupyter lab notebooks/examples/09_experimental_trpl_detection.ipynb
jupyter lab notebooks/paper_inp_pox_alox/10_inp_pox_alox_manuscript_evidence.ipynb
```

The example writes figures to `examples/output/`.

## Spatial profile API

```python
from surpass import reconstruct_band_profile, trap_relaxation_spectrum

# `surface_model` is a SurfaceSRHModel with self-consistent electrostatics.
# For coupled steady-state transport, use result.front_surface_delta_cm3 here.
profile = reconstruct_band_profile(surface_model, delta_n_cm3=1e11)
trap_times = trap_relaxation_spectrum(surface_model, delta_n_cm3=1e11)
print(profile.potential_99_depth_cm * 1e7, "nm")
print(profile.charge_integral_relative_error)
print(trap_times.slow_trap_fraction(1e-9))
```

Build the new notebook from readable sources with
`python scripts/build_profiles_notebook.py`. For validation where a separate
kernel cannot be launched, run `python scripts/execute_notebook_inprocess.py
notebooks/examples/06_spatial_band_profiles.ipynb`. Standard Jupyter use is unchanged.

## Transient API

```python
import numpy as np
from surpass import TimeDependent1DSolver

# Existing tabulated bulk model and nonlinear surface boundary are reused.
# initial_density must be cell averages, not samples of a shallow exponential.
result = TimeDependent1DSolver(bulk_table).solve(
    thickness_cm=0.062,
    time_s=np.r_[0, np.geomspace(1e-13, 20e-9, 301)],
    initial_delta_cm3=initial_density,
    front_s_cm_s=surface_boundary,
    rear_s_cm_s=1e5,
    cell_edges_cm=edges_cm,
)
signal = result.internal_radiative_emission_cm2_s
print(np.max(np.abs(result.inventory_relative_error)))
```

Generate the notebook with `python scripts/build_transient_notebook.py`.
After upgrading source files, restart the notebook kernel to clear cached
imports. `GaussianPulse(density_cm3, center_s, sigma_s)` can be passed as
`generation`; it automatically limits steps to sigma/2. Arbitrary short or
discontinuous pumps need suitable `max_step_s` or segmented solves.
The inventory error is normalized to initial plus full-window generated dose,
not a tiny late-time or leading-edge inventory. Quadrature uses accepted
integrator steps and dense output, independently of output sampling.

## Spectral escape and recycling API

```python
from surpass import EmissionSpectrum, EmissionBoundary, SlabEmissionModel

# Internal photon-number spectrum PER NM, not external power spectrum.
spectrum = EmissionSpectrum(wavelength_nm, photon_density_per_nm,
                            source='your internal emission dataset')
front = EmissionBoundary(air, layers_wafer_to_outside)
rear = EmissionBoundary(air)
optical = SlabEmissionModel(wafer_optics, front, rear,
                           alpha_active_cm1, alpha_parasitic_cm1)
fates = optical.solve(thickness_cm, cell_centers_cm, spectrum, front_na=0.5)
p_rec = fates.averaged()['active_reabsorption']
result = TimeDependent1DSolver(bulk_table).solve(
    thickness_cm, time_s, initial_density, cell_edges_cm=edges_cm,
    front_s_cm_s=surface_boundary, rear_s_cm_s=1e5,
    recycling_probability=p_rec,
)
# Use intrinsic rates from the ORIGINAL table, never reduced net-loss rates.
intrinsic = bulk_table.evaluate(result.delta_n_cm3)['r_rad']
intrinsic = np.where(result.delta_n_cm3 > 0, intrinsic, 0)
photons = fates.emission_fluxes(intrinsic, result.cell_width_cm)
collected = photons['front_collected']  # photons/cm2/s, before detector response
```

Both solvers default to zero recycling. Existing radiative/cumulative loss fields
report NET carrier loss when recycling is enabled; new internal-emission and
recycled-generation fields report photon production and regenerated pairs.
For CW switch-off, pass the same `recycling_probability` to both solvers.
Do not simultaneously reduce intrinsic B for the same recycling effect.
`internal_radiative_yield` now explicitly means internal emission events per
external pump pair, which can exceed one under recycling, not external EQE.

Build notebook 08 using `python scripts/build_emission_notebook.py`.
The audit is a numerical/model checkpoint, not experimental calibration.

## Experimental TRPL API

```python
from surpass import (SampledPulse, PulseTrain, InstrumentResponse,
                        SpectralResponse, DetectionModel,
                        fit_detection_nuisance)

pulse = SampledPulse(spatial_dose_cm3, relative_time_s,
                     background_subtracted_laser_monitor,
                     source='measured pulse metadata')
generation = PulseTrain(pulse, period_s=1/repetition_rate_hz,
                        first_pulse_s=0, pulse_count=number_of_pulses)

result = TimeDependent1DSolver(bulk_table).solve(
    thickness_cm, simulation_time_s, initial_density,
    generation=generation, cell_edges_cm=edges_cm,
    recycling_probability=p_recycle,
)
intrinsic = bulk_table.evaluate(result.delta_n_cm3)['r_rad']
response = SpectralResponse(wavelength_nm, total_efficiency,
                            source='filters × optics × detector QE')
detected_flux = photon_fates.response_weighted_flux(
    intrinsic, result.cell_width_cm, response, fate='front_collected')

irf = InstrumentResponse(irf_delay_s, background_subtracted_irf,
                         source='same detector/timing/spectral path')
detection = DetectionModel(emitting_area_cm2, irf,
                           electronic_gain_counts_per_event=gain,
                           background_rate_counts_s=background_rate)
expected_counts = detection.expected_counts(
    simulation_time_s, detected_flux, measured_bin_edges_s)
fit = fit_detection_nuisance(
    detection, simulation_time_s, detected_flux,
    measured_bin_edges_s, measured_counts,
    shift_bounds_s=(-1e-9, 1e-9))
```

Use `sigma_counts` for analog signals with independently estimated uncertainty.
When it is omitted, the nuisance fit minimizes signed Poisson-deviance
residuals and reports reduced Poisson deviance. Build notebook 09 with
`python scripts/build_detection_notebook.py`.

Notebook 10 is a manuscript-oriented, deliberately non-fitted assessment of
the POx/AlOx/InP experiment. It combines the measured CW illumination
conditions with generation profiles, bulk lifetime and diffusion-length
sensitivity, an InP doping--fixed-charge surface-recombination map, full 1D CW
charge sweeps, negative-corona trajectories, and interface-parameter/pinning
sensitivity. Rebuild it with
`python scripts/build_inp_manuscript_evidence_notebook.py`.

## Principal references

- I. Vurgaftman, J. R. Meyer, and L. R. Ram-Mohan, *J. Appl. Phys.* **89**, 5815–5875 (2001), DOI: 10.1063/1.1368156.
- M. Sotoodeh, A. H. Khalid, and A. A. Rezazadeh, *J. Appl. Phys.* **87**, 2890–2900 (2000), DOI: 10.1063/1.372274.
- V. Palankovski, S. Kaiblinger-Grujin, and S. Selberherr, *Mater. Sci. Eng. B* **66**, 46–49 (1999), DOI: 10.1016/S0921-5107(99)00118-X.
- A. Richter et al., *Phys. Rev. B* **86**, 165202 (2012), DOI: 10.1103/PhysRevB.86.165202.
- Ioffe Institute NSM semiconductor parameter archive (compiled 300 K values; accessed September 2026).
- D. E. Aspnes and A. A. Studna, *Phys. Rev. B* **27**, 985–1009 (1983), DOI: 10.1103/PhysRevB.27.985.
- B. Adamowicz et al., *Vacuum* **63**, 223–227 (2001), DOI: 10.1016/S0042-207X(01)00195-6.
- W. Shockley and W. T. Read, Jr., *Phys. Rev.* **87**, 835–842 (1952), DOI: 10.1103/PhysRev.87.835.
- E. H. Nicollian and J. R. Brews, *MOS Physics and Technology* (Wiley, 1982).
- O. Semyonov et al., Radiation efficiency of heavily doped bulk n-InP
  semiconductor, https://arxiv.org/pdf/1003.6095.
- S. J. Byrnes, Multilayer optical calculations, https://arxiv.org/abs/1603.02720.

Additional source notes are stored directly in `materials.py` beside the
parameters to which they apply.
