# Examples

This directory contains small, synthetic inputs and representative outputs for
the MetzgerKN helper scripts. The fitted-result JSON files are fixtures with the
same fields consumed by the plotting utilities; real values depend on the
installed `lightcurve_fitting` build and optimizer settings.

## Inputs

- `inputs/example_lightcurve.csv`: two synthetic events with `event_id`, `time`,
  `band`, `mag`, and `mag_err` columns.
- `inputs/example_lightcurve.dat`: one synthetic event in whitespace
  `datetime band mag mag_err` format.
- `inputs/features_kn_cut_1000.csv`: tiny raw feature table for
  `cut_features.py`.
- `inputs/real_lsstlike/`: real LSST-like inputs copied from the sibling
  `lightcurve-fitting` working directory:
  `lightcurve_LSSTlike_0090.csv`, `lightcurve_LSSTlike_0495.csv`,
  `lightcurve_LSSTlike_0637.csv`, and a three-row
  `features_kn_cut_1000_real_excerpt.csv`.

## Outputs

- `outputs/cut_features/features_kn_cut_1000_enriched.csv`: expected enriched
  feature table.
- `outputs/fit_parametric_lightcurve/`: representative fitter outputs,
  including `*_used_photometry.csv`, `*_parametric_result.json`, and
  `metzgerkn_boundary_hits.csv`.
- `outputs/plot_metzgerkn_boundary_lightcurves/`: generated annotated
  boundary-hit plots when the plotting example command is run.
- `outputs/plot_utils/`: generated `t0` recovery diagnostic plots when the
  plotting example command is run.
- `outputs/real_lsstlike_fitted/`: real fitted files copied from
  `lightcurve-fitting/test_n101_alex_lsstlike_fitted/` for event `0090`, plus
  an enriched output generated from the real feature excerpt.

See the top-level `README.md` for runnable commands using these paths.
