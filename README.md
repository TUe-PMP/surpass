# SURPASS

**SURface and Passivation Analysis for Semiconductor Systems**

SURPASS is a transparent Python framework for coupled semiconductor bulk,
optical, transport, surface-electrostatic, interface-recombination, and
photoluminescence calculations. It was developed to distinguish chemical and
field-effect passivation from paired measurements on differently doped
semiconductors.

Version **1.0.0** is the fixed implementation used for the accompanying study
of POx/AlOx passivation on n- and p-type InP. The framework itself is not tied
to III–V materials: the release includes Si, Ge, InP, and GaAs presets and
accepts user-defined material parameters.

Source repository: [github.com/TUe-PMP/surpass](https://github.com/TUe-PMP/surpass)

## Core capabilities

- Fermi–Dirac equilibrium carrier statistics;
- optional band-gap narrowing and doping-dependent mobility;
- SRH, radiative, and Auger bulk recombination;
- transfer-matrix optical generation;
- steady-state and transient one-dimensional ambipolar transport;
- self-consistent surface electrostatics with fixed and interface-state charge;
- constant or energy-dependent interface-state distributions;
- energy-resolved surface-SRH recombination;
- spatial band profiles and trap-relaxation diagnostics;
- spectral emission escape, local photon recycling, and detector response;
- workflows for fitting paired n- and p-type PL ratios and propagating
  experimental and model uncertainty.

SURPASS is a research model rather than a general-purpose multidimensional TCAD
package. In particular, version 1.0.0 does not implement voltage-driven gate
stacks or spatially resolved drift–diffusion through a space-charge region.

## Installation

Create a clean Python environment and install from the project root:

```bash
python -m pip install .
```

For notebooks and development tools:

```bash
python -m pip install ".[notebook,dev]"
```

Verify the installation:

```bash
python -c "import surpass; print(surpass.__version__)"
python -m pytest
```

## Minimal example

```python
from surpass import BulkModel, Wafer, get_material

silicon = get_material("Si")
wafer = Wafer(
    silicon,
    donor_cm3=1e15,
    thickness_um=280,
    temperature_k=300,
)
model = BulkModel(wafer, tau_srh_s=5e-3)
state = model.at_injection(1e14)

print(state.n_cm3, state.p_cm3)
print(state.radiative_rate_cm3_s, state.auger_rate_cm3_s)
```

## Notebooks

The notebook collection is separated into two groups:

- [`notebooks/examples`](notebooks/examples) contains general demonstrations of
  the individual physical models;
- [`notebooks/paper_inp_pox_alox`](notebooks/paper_inp_pox_alox) contains the
  workflows associated with the POx/AlOx-on-InP manuscript.

The manuscript directory is a reproducibility record, not a claim that every
experimental parameter is universally applicable. Its README identifies the
purpose and status of each notebook.

## Documentation and validation

- [`docs/MODEL_SCOPE.md`](docs/MODEL_SCOPE.md): detailed capability and scope notes;
- [`VALIDATION.md`](VALIDATION.md): numerical and limiting-case validation;
- [`BULK_RECOMBINATION_AUDIT.md`](BULK_RECOMBINATION_AUDIT.md): provenance and
  limitations of the bulk recombination parameters;
- [`CHANGELOG.md`](CHANGELOG.md): release history.

Every quantitative application should review the provenance and validity of the
material, optical, and interface parameters used. Several bundled InP thin-film
optical constants and surface-capture parameters are deliberately exposed as
replaceable assumptions rather than universal constants.

## Citation

Please cite the archived version-specific Zenodo record for the release used in
your work. The DOI will be inserted here after the GitHub `v1.0.0` release has
been archived. Machine-readable citation metadata are provided in
[`CITATION.cff`](CITATION.cff).

## License

SURPASS is distributed under the [BSD 3-Clause License](LICENSE).

Copyright © 2026 Eindhoven University of Technology.

## Author and AI-assisted development

SURPASS is authored and maintained by [Bart Macco](https://orcid.org/0000-0003-1197-441X),
Plasma & Materials Processing Group, Eindhoven University of Technology.

Portions of the implementation, documentation, testing, code review, and
refactoring were developed iteratively with assistance from OpenAI ChatGPT and
Codex using GPT-5-family models. All scientific assumptions, validation
decisions, and released code were reviewed and approved by the human author.
Further details are provided in [`AUTHORS.md`](AUTHORS.md).
