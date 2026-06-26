
## lightcurve_fitting_metzgerkn

A few python scripts to make use of the [`lightcurve-fitting`](https://github.com/boom-astro/lightcurve-fitting) repo, only with the MetzgerKN kilonova model. These are best ran from the root of this [`lightcurve-fitting`](https://github.com/boom-astro/lightcurve-fitting) crate.

## Python scripts

- `fit_parametric_lightcurve.py`: Main fitting script for running the `lightcurve_fitting` Rust/PyO3 parametric fitter on light-curve data with the `MetzgerKN` model. It accepts whitespace `.dat` files or CSV files, can fit one object or batches grouped by an ID column, and writes fitted-result JSON files, cleaned photometry CSV files, light-curve plots, editable Matplotlib pickle files, summary plots, and a `metzgerkn_boundary_hits.csv` file when fitted parameters reach their bounds.

  Example:

  ```bash
  python fit_parametric_lightcurve.py --input my_lightcurve.csv --input-format csv --model MetzgerKN
  ```

- `plot_metzgerkn_boundary_lightcurves.py`: Post-processing script for inspecting `MetzgerKN` fits that hit parameter bounds. It reads `metzgerkn_boundary_hits.csv`, finds the corresponding `*_used_photometry.csv` and `*_parametric_result.json` files, and saves annotated light-curve plots for the affected events, including model overlays when the `lightcurve_fitting` Python module is available.

  Example:

  ```bash
  python plot_metzgerkn_boundary_lightcurves.py --boundary-hits-csv lightcurve_fitting_parametric_results/metzgerkn_boundary_hits.csv
  ```

- `plot_utils.py`: Diagnostic plotting utilities for fitted parametric outputs. The command-line entry point compares recovered `t0` values against a truth CSV and produces plots showing how the `t0` recovery depends on reduced chi-squared and the number of detections per band or in total.

  Example:

  ```bash
  python plot_utils.py --output-dir lightcurve_fitting_parametric_results --truth-csv features_kn_cut_1000_enriched.csv
  ```

- `cut_features.py`: Small preprocessing script for Alex Jacquesson's FIESTA/Bulla simulation feature CSVs. It selects the columns needed by the MetzgerKN fitting analysis and adds derived columns `log10_mej_tot`, `log10_vej`, and `t0`. The input and output filenames are currently set inside the script.
