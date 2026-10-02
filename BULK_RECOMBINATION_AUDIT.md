# Bulk-recombination checkpoint for release 0.8

## What is validated, and what is not

The software recovers the expected limits of the **constant-coefficient**
model. This does not validate those coefficients for the experimental wafers.
Fermi–Dirac populations and generalized diffusion do not, by themselves, make
`B`, `Cn` and `Cp` degeneracy-aware transition rates. Band filling, nonparabolic
bands, Pauli blocking and temperature-dependent recombination remain omitted.

The intrinsic excess radiative rate is

`Rrad = B*delta*(n0+p0+delta)`.

The equilibrium-subtracted Auger rate is evaluated as

`Raug = (Cn*n0+Cp*p0)*delta*(n0+p0+delta)`
`       + (Cn+Cp)*delta*(n0*p0+delta*(n0+p0+delta))`.

These are the existing equations rewritten to avoid subtracting nearly equal
products at extremely low injection. Bulk SRH remains `delta/tau_srh`.
Reported lifetimes are **excess carrier density / excess recombination rate**,
not differential lifetimes `1/(dR/delta)`. At high injection those differ.

## Coefficient provenance and limitations

The existing InP defaults are B=1.2e-10 cm³/s, Cn=9e-31 cm⁶/s and
Cp=1e-29 cm⁶/s. They are retained for release continuity as **representative
development values**, not claimed as a verified transcription of a specified
measurement. The old material-level references do not establish a unique
source for every recombination coefficient. The notebook prints these inputs
and compares a factor-of-three sensitivity bracket; that bracket is NOT a
confidence interval. Users can override coefficients with `dataclasses.replace`.

A directly relevant primary study is O. Semyonov, A. Subashiev, Z. Chen and
S. Luryi, *Radiation efficiency of heavily doped bulk n-InP semiconductor*,
[author manuscript](https://arxiv.org/pdf/1003.6095). It investigates n-InP
at about 2–8e18 cm⁻³ over 77–330 K and separates interband absorption,
free-carrier absorption, radiative recombination and photon recycling.
Its concentration-dependent treatment demonstrates why one constant B and
one generic recycling factor should not be interpreted as measured properties
of every n/p wafer. Its n-type analysis is not a validated p-InP model, and the
8.5e18 cm⁻³ n-type example is outside its measured concentration interval.
No empirical fit from that paper is silently applied to our material presets.

GaAs, Si and especially Ge constants also remain benchmark estimates. The
constant Si coefficients are not the full injection-dependent Richter model:
[Richter et al., PRB 86, 165202 (2012)](https://doi.org/10.1103/PhysRevB.86.165202).
The Ge effective B requires a dedicated check before quantitative use.
Alloy-specific validated recombination and near-edge optical datasets are not
yet provided. Do not treat this package as a validated all-material database.

## Consequences for PL/TRPL

Recycling is implemented separately from intrinsic B. An intrinsic emission
event removes a pair; active reabsorption regenerates a pair. In the local
approximation, net radiative loss is `(1-p_rec(x))*Rrad(x)`, while emission
and escaped PL are evaluated using the full intrinsic `Rrad`. Do not reduce B
and enable recycling simultaneously for the same physical effect.

For uniform, radiative-only, low-injection decay,
`tau_effective = tau_intrinsic/(1-p_rec)`. With Auger/SRH/surface losses this
simple multiplication does not hold. Internal photon emission events per
initially generated pair can exceed one, but escaped-plus-parasitic photons
cannot exceed the available carrier dose after allowing nonradiative losses.

Near-edge absorption must separate **active interband** from **parasitic**
absorption. Doping can shift the absorption/emission spectra; the rounded demo
optical constants cannot resolve this. Release 0.8 therefore requires explicit
active/passive absorption inputs and a normalized INTERNAL photon spectrum.
The notebook's absorption and spectrum are labelled synthetic. A filtered
external spectrum is not automatically a valid internal input.

## Optical approximation and next validation targets

Release 0.8 uses coherent angular films and incoherent slab traversals, with
isotropic unpolarized emission, specular planar boundaries and no lateral
escape. NA selects a normal-axis collection cone. It models collected flux per
wafer area before instrument efficiency, not calibrated detector counts.

For a perfectly transparent, high-index slab, some rays remain trapped by
total internal reflection. Their separate `trapped` probability is reported;
it is not active reabsorption. Any nonzero bulk absorption eventually consumes
those rays in the zero-flight-time approximation, even if very weak. Actual
photon residence times, scattering and lateral escape can invalidate that
limit. Do not use this approximation for transparent high-Q optical storage.

The ray model uses real semiconductor n for interface Fresnel coefficients
and separate attenuation, valid as a weak-absorption bulk approximation. It
does not replace a full dipole-emission/LDOS calculation near a thin film.
The local regeneration approximation conserves global carrier number but
does not predict redistribution of carriers by reabsorption. For strongly
nonuniform 514-nm excitation this is a potentially important limitation.

Next: measured internal/absorption spectral inputs; nonlocal reabsorption
kernel; temperature/doping-dependent intrinsic recombination; pump/collection
spot geometry; instrument response and pulse trains. No experimental TRPL fit
or reinterpretation is performed in this release.

Angular film optics reference: [S. J. Byrnes, Multilayer optical calculations](https://arxiv.org/abs/1603.02720).
