# POx/AlOx-on-InP manuscript workflows

This directory contains the calculation notebooks developed for the associated
POx/AlOx passivation study. They are retained separately from the general
examples because they contain sample-specific doping levels, excitation
conditions, optical stacks, measured PL ratios, fitting ranges, and uncertainty
assumptions.

The notebooks will be mapped one-to-one to the final main-text and
supplementary figures before the manuscript release. Until the final figure
numbering is frozen, the descriptive filenames are authoritative.

| Notebook | Purpose |
|---|---|
| `10_inp_pox_alox_manuscript_evidence.ipynb` | Coupled evidence and parameter sweeps |
| `11_inp_paper_reasoning.ipynb` | Passivation-mechanism calculations |
| `11_inp_paper_reasoning_bare_reference.ipynb` | Bare-surface optical reference |
| `12_inp_passivated_depth_profiles.ipynb` | Passivated carrier and band profiles |
| `12_inp_profile_comparison_publication.ipynb` | Passivated/unpassivated publication profiles |
| `13_inp_dit_qf_monte_carlo.ipynb` | Single-state paired-polarity inference |
| `14_inp_annealing_dit_qf_trajectory.ipynb` | Monte Carlo annealing trajectories |
| `15_inp_pl_annealing_curve.ipynb` | Experimental relative-PL annealing figure |

The final archived release should contain cleared notebook outputs and separate
reference figures or tabulated results where needed. Long-running Monte Carlo
results should be archived as explicit data files rather than embedded only as
notebook output.
