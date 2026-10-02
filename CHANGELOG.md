# Changelog

All notable changes to SURPASS are documented here. The project follows
[Semantic Versioning](https://semver.org/).

## 1.0.0 — 2026-10-02

First public research release.

### Included

- Fermi–Dirac bulk carrier statistics and optional band-gap narrowing.
- Doping- and injection-dependent transport and generalized Einstein relations.
- SRH, radiative, and Auger bulk recombination.
- Transfer-matrix optical generation and spectral escape calculations.
- Steady-state and transient one-dimensional ambipolar transport.
- Self-consistent surface electrostatics with fixed and interface-state charge.
- Energy-resolved surface-SRH recombination.
- Spatial band-profile reconstruction and trap-relaxation diagnostics.
- PL observables, local photon recycling, and experimental-response utilities.
- Reproducible uncertainty-propagation workflows for paired n- and p-InP PL.
- Si, Ge, InP, and GaAs material presets.

### Release identity

- Renamed the development package `iii_v_bulk` to the material-independent
  public API `surpass`.
- Adopted the BSD 3-Clause license and formal citation metadata.

### Known limitations

- The quasi-neutral transport solver does not spatially resolve transport and
  recombination inside the surface space-charge region.
- Photon recycling is local and instantaneous rather than nonlocal.
- Biased metal–insulator–semiconductor gate boundaries are not implemented in
  version 1.0.0.
- Material presets and capture parameters remain model inputs whose uncertainty
  must be evaluated for quantitative inference.
