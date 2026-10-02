# SURPASS 1.0.0 release validation

Validated on 2 October 2026 with Python 3.12.

- All 58 unit tests pass after migration from the development package name to
  the public `surpass` API.
- The 56 existing numerical and physical regression tests are unchanged and
  pass without modification of the scientific implementation.
- Two release-identity tests confirm version `1.0.0` and material-independent
  lookup of the Si, Ge, InP, and GaAs presets.
- All included notebooks are valid `nbformat` JSON documents, all stored code
  cells compile, and all generated outputs and execution counts are cleared.
- A wheel for `surpass-semiconductor==1.0.0` builds successfully.
- The source/notebook release archive passes ZIP-integrity and required-file
  checks.

This release validation confirms software migration and regression behaviour.
It does not convert the model-conditioned InP parameter inference into an
independent experimental validation of the assumed material and interface
parameters.

---

# Historical release 0.9 validation record

Validated on 19 September 2026 with Python 3.12.

- 56/56 unit tests pass: all 47 release-0.8.1 checks plus 9 experimental-
  observation test methods.
- All 6 code cells of notebook 09 execute sequentially with rich outputs.
- A sampled PCHIP pulse integrates to its requested dose; a finite pulse train
  has the correct pulse count and overlapping pulses add.
- The transient solver automatically resolves sampled excitation and recovers
  pulse-generated carrier dose while closing carrier inventory.
- Gaussian source/IRF convolution recovers the analytic combined width. The
  notebook central relative error is 3.39e-6.
- IRF convolution plus bin integration preserves total events when the source,
  IRF and bin windows contain their complete tails.
- A constant event rate reproduces exact finite-bin counts including emitting
  area, gain and background.
- Spectral-response integration recovers the analytic index-matched escape and
  response-weighted flux limits.
- Nuisance alignment recovers synthetic scale, timing offset and background;
  weighted least-squares and count-data validation paths are covered.
- One- and two-exponential fits recover an exact synthetic biexponential and
  correctly reference amplitudes to the selected fitting-window origin.

Executed synthetic n-InP pulse-train example:

- Generated dose / three nominal absorbed pulse doses: 0.999999986.
- Maximum carrier-inventory error: 1.77e-6 of generated dose.
- Injected versus fitted timing shift: 0.350 versus 0.344 ns.
- Injected versus fitted scale: 104.827 versus 105.348.
- Injected versus fitted background: 2.00 versus 2.36 counts/bin.
- Reduced Poisson deviance: 1.002.
- Pre-pulse sheet excess increases from 1.20e9 before pulse one to 2.21e9
  cm^-2 before pulse three, demonstrating repetition-rate pile-up.
- The same physical trace fitted over 41–64 and 45–64 ns gives different
  descriptive biexponential results, explicitly demonstrating window dependence.

These are equation/software checks using synthetic pulse, spectrum, absorption,
response and IRF inputs—not validation against the experimental InP TRPL.
Nonlocal reabsorption, dynamic traps, wavelength-dependent IRF, PSF/finite-spot
transport, detector dead time/afterpulsing and uncertainty propagation remain omitted.

---

# Historical release 0.8 validation record

Validated on 18 September 2026 with Python 3.12, NumPy 2.3.5 and SciPy 1.17.0.

- 47/47 tests pass: 34 existing plus 13 new optical/recycling/bulk checks.
- All 8 code cells of notebook 08 execute with retained rich outputs, using
  the included in-process helper (not a separate Jupyter kernel).
- Normal-incidence Fresnel coefficients, quarter-wave AR coating, TIR,
  absorbing-film passivity and pump/emission agreement are recovered.
- Index-matched attenuation agrees with the analytic exponential-integral
  angular result; the thick-wafer front escape limit agrees with the existing
  single-interface integral within 3e-6 relative.
- Transparent TIR rays are separately accounted for as trapped. Tiny nonzero
  absorption closes photon balance without subtractive cancellation.
- Multi-wavelength active/passive absorption and NA selection conserve photons
  in all tested cases. Default notebook maximum photon-balance error: 2.89e-15.
- Local radiative-only recycling recovers tau/(1-p_rec), including zero loss
  at p_rec=1, while retaining nonzero intrinsic emission.
- Separate intrinsic emission, regenerated carriers, and net radiative loss
  close carrier inventory. CW and transient equations use the same feedback.
- Cancellation-free excess-rate algebra stays positive at extremely small
  injection in the intrinsic Ge regression; this is not Ge coefficient validation.

Executed synthetic-optics InP examples:

- Maximum carrier inventory error across four main cases: 5.01e-7 of initial dose.
- 64→128 angular nodes: maximum probability change 3.76e-8.
- 11→21 wavelengths: maximum averaged probability change 1.82e-6.
- Tightening integrator tolerances by ten: n-InP carrier trace changes by at
  most 1.05e-5 relative above the 1e-4 inventory threshold.
- Doubling spatial cells: that carrier trace changes by at most 1.46e-3 relative.
- Independent requested-time photon-loss integration agrees with dense-output
  net radiative loss within 0.072%; its finite sampling error is reported.
- CW global balance error: 2.69e-7; switch-off inventory error: 1.21e-7.
- Internal emission events per CW pump pair: about 6.79, not external EQE.
- Curves below 1e-6 of their individual peaks are omitted from example plots
  to avoid giving numerical absolute-tolerance tails physical significance.

## Corrected pump convention

The prior +i film matrix sign was inconsistent with the stated n+i*k convention.
For a 100-nm film with n=2, k=0.2 between lossless air media at 900 nm it
produced R+T>1 and clipped film absorption to zero. The corrected -i backward
transfer gives positive absorption and energy balance within 1e-12, and agrees
with the new emission-side normal-incidence calculation.

This also changes coated-wafer pump interference when the substrate is complex.
For the historical outside→POx(10 nm)→AlOx(10 nm)→demo InP stack at 514 nm,
current R=0.31487, versus historical R=0.33385; bare R=0.35570 is unchanged.
Notebook 08 instead uses the physical outside→AlOx→POx→InP order and reverses
it for emission. These remain illustrative optical datasets, not predictions.

## Physical scope

This is equation/software validation, NOT experimental validation of InP B/C,
internal spectra, interband/free-carrier absorption or the local recycling
approximation. No empirical B/C fit was silently introduced. Read
`BULK_RECOMBINATION_AUDIT.md`. Nonlocal reabsorption, photon flight-time/storage,
detector response, IRF, pulse trains and dynamic traps remain omitted.
Earlier notebooks are historical release examples and were not re-executed for
this release. Their optical outputs can reflect the previous matrix convention.

---

# Historical release 0.7 validation record

Validated on 18 September 2026 with Python 3.12.

- 34/34 unit tests pass: all 26 previous checks and 8 new transient checks.
- All code cells in the new release-0.7 notebook execute sequentially in a
  shared Python namespace without errors; rich outputs are retained. Older
  notebooks remain historical release examples, not newly revalidated outputs.
- Release 0.7 was validated using the included in-process execution helper,
  not a separate Jupyter kernel. The notebooks are standard `nbformat` 4 files.
- Optical energy balance closes to numerical precision for both the bare and
  demonstration POx/AlOx/InP stacks at 514 nm.
- The Beer-Lambert generation profile integrates to the photon flux entering
  the semi-infinite InP substrate.
- The 1D solver reproduces the analytic uniform-generation solution for zero
  surface recombination and the symmetric finite-S solution.
- A strongly nonuniform 620-µm wafer grid conserves generation and total
  recombination, while resolving the sub-micrometre pump-generation layer.
- Surface recombination is exactly zero at equilibrium by construction.
- The sign of the Fermi–Dirac surface potential follows the signed fixed charge.
- The model reproduces the expected field-effect trend: positive charge lowers
  n-InP surface recombination and raises p-InP surface recombination before the
  latter passes through strong inversion.
- A nonlinear interface-SRH boundary coupled to the transport solver closes
  global generation/recombination balance to a few parts per million in the
  full fixed-charge sweep.
- Disabling interface-state charge exactly recovers the release-0.4 fixed-charge
  solution, including its nonlinear recombination rate.
- With amphoteric interface charge enabled, `Qext + Qit + Qsc` closes to below
  100 elementary charges cm^-2 in the validation cases, negligible relative to
  the 10^12–10^15 cm^-2 component charges.
- Increasing Dit from 10^11 to 10^15 eV^-1 cm^-2 increases the local trap-control
  fraction and brings the equilibrium surface Fermi level within 0.03 eV of the
  chosen midgap CNL in the high-Dit test.
- A test voltage-dependent boundary confirms that the electrostatic root solver
  already accepts the functional form needed for a later metal/oxide gate.

Release-0.6 checks:

- Flat band produces uniform carrier populations, no semiconductor electric
  field, and coincident equilibrium Fermi energies.
- Weak band bending recovers exponential decay with the generalized FD
  screening length to within 0.02% in the small-signal test.
- Band-profile endpoints reproduce the surface-model potential and carrier
  populations, including nonzero injection and amphoteric Qit.
- Ec-Ev remains the bulk-model effective band gap and the reconstructed
  quasi-Fermi energies stay flat by construction.
- The differentiated field satisfies Poisson's equation to within 0.5% in
  significant-charge regions in the finite-difference check.
- Integrating the volume charge recovers Qsc(surface)-Qsc(tail) within 0.1%
  for the tested n/p degenerate profiles; mesh refinement reduces the error.
- Local trap occupancy agrees with the surface model, and inverse relaxation
  time equals the sum of all four capture/emission rates.
- The small-potential first integral now uses compressibility quadrature to
  avoid cancellation in the asymptotic screening tail.

These are equation/software checks, not experimental validation of the CNL,
trap spectrum, optical parameters, or degenerate capture/emission kinetics.

Release-0.7 transient checks:

- Uniform decay reproduces exp(-t/tau) to within 10^-6 relative error.
- A reflecting cosine diffusion mode agrees with analytic decay within 3e-5
  on a 100-cell grid; a finite-S symmetric eigenmode agrees within 2e-4 on
  a 160-cell grid.
- Exact darkness is preserved; constant generation reproduces G*tau*(1-exp(-t/tau)).
- A Gaussian pulse reproduces its convolution with exponential decay and full
  dose, without skipping the initial dark interval.
- Quadratic radiative loss recovers delta0/(1+B*delta0*t). The local PL slope
  recovers half the effective carrier lifetime in this limit.
- A nonlinear interface boundary on a nonuniform grid closes integrated
  carrier balance within 2e-5 of initial dose in the unit test.
- Analytic exponential/finite-S cases close inventory within 10^-7; Gaussian
  pumping closes within 10^-6 of full generated dose.
- Invalid inputs and high-side table-range violations are rejected. The
  transient equation and diagnostics sum the same channel interpolants.

No experimental TRPL fit, IRF correction, photon-recycling validation or
dynamic-trap validation is claimed.

Executed InP notebook (illustrative default parameters):

- All 9 code cells execute with retained rich outputs.
- Across the six n/p and charge cases, maximum inventory error is 7.33e-7
  of initial sheet dose.
- For n-InP at zero external charge, tightening tolerances by ten changes the
  emission trace by at most 3.19e-5 relative above the 10^-4 signal threshold.
- Doubling the spatial cell count changes that trace by at most 3.48e-4 relative.
- The separate finite-Gaussian example conserves nominal generated dose and
  has maximum inventory error 3.79e-6 at its default integration tolerances.

Representative development values at 514 nm:

- bare InP reflectance: 0.35570;
- demonstration POx/AlOx/InP reflectance: 0.33385;
- demonstration InP absorption coefficient: 1.1002e5 cm^-1;
- demonstration pump 1/e depth: 90.90 nm.

These optical numbers are software checks, not process predictions: the
bundled optical constants are explicitly illustrative and must be replaced by
full literature tables or measured ellipsometry.
