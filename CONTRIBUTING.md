# Contributing

Bug reports, validation cases, documentation corrections, and focused pull
requests are welcome through the GitHub issue tracker.

Before submitting a change:

1. install the package with the `dev` optional dependencies;
2. run `python -m pytest`;
3. document changes to equations, default parameters, or numerical behaviour;
4. add a regression test for bug fixes and new physical models;
5. update `CHANGELOG.md` when the public API or scientific behaviour changes.

New material presets should include parameter provenance, units, validity
notes, and at least one independent limiting-case or literature benchmark.
