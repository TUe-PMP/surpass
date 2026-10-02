# POx/AlOx-on-InP manuscript workflows

These notebooks reproduce the computational figures in the manuscript using
SURPASS v1.0.0. The numbering follows the September 2026 manuscript supplied
with this release. Experimental schematics and figures that require instrument
source data are listed for completeness but are not reconstructed from
placeholder or digitized traces.

## Figure map

| Manuscript item | Release notebook | Status |
|---|---|---|
| Figure 1a | — | Device schematics; graphical artwork |
| Figure 1b | `figure_01b_surface_recombination_map.ipynb` | Reproducible model calculation |
| Figure 2 | — | Experimental workflow schematic |
| Figure 3 | `figure_03_relative_pl_annealing.ipynb` | Reproducible from tabulated experimental ratios |
| Figure 4 | — | SE/XPS figure; final instrument data not included |
| Figure 5 | — | Corona-response figure; final numerical data not included |
| Figure 6 | `figure_06_depth_profiles.ipynb` | Reproducible model calculation |
| Figure 7 | `figure_07_relative_pl_maps.ipynb` | Reproducible model calculation; long runtime |
| Figure 8 | `figure_08_annealing_inference.ipynb` | Reproducible from archived Monte Carlo results |
| Figure S1 | `figure_S1_corona_charge_calibration.ipynb` | Reproducible model calculation |
| Figure S2 | `figure_S2_injection_dependent_surface_recombination.ipynb` | Reproducible model calculation |

`monte_carlo_annealing_inference.ipynb` is the resumable, long-running source
analysis that generated the Figure 8 result table. Its release default does not
start a new batch. Its reconstructed configuration ID exactly matches the ID in
the archived CSV. The historical `iii_v_bulk` v0.9.0 identity is deliberately
retained inside that hash because SURPASS v1.0.0 is the renamed public release
of the same model code.

The archived Figure 8 table is
`data/inp_annealing_trajectory_batched_wide_asdep.csv` and includes batch IDs,
seeds, scientific configuration IDs, nuisance-parameter draws, fitted values,
residuals, boundary flags, and compatibility flags.

## Reproducibility conventions

- Notebook outputs are cleared in the source release.
- Figures are written to a local `output/` directory, which is excluded from
  source distributions.
- Figure 8 uses medians and central 68% empirical intervals (16th–84th
  percentiles), because the inverse-fit distributions need not be Gaussian.
- As-deposited Figure 8 points are model-mismatch diagnostics and are not
  presented as compatible quantitative inferences.
- Additional Monte Carlo batches may only be combined when their scientific
  configuration IDs match.
