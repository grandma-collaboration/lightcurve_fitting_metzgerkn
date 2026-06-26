## lightcurve_fitting_metzgerkn

Helper scripts for fitting kilonova light curves with the [`lightcurve-fitting`](https://github.com/boom-astro/lightcurve-fitting)
Rust/PyO3 package, using the `MetzgerKN` kilonova model. This allows to plot the fitted lightcurves, the distribution of the parameters
when batch fitting and to see which lightcurves have parameters hitting bounds. In particular, they can also fit the outputs
of Alex's simulations using fiesta/Bulla, and, if he gave you the input feature csvs of his simulations (the true Bulla parameters used),
the scaling of the t0 recovery with the number of detections and with the chi^2, and scatter plots of injected vs inferred parameters (assuming a correspondance between them, but the models are not the same).

The fitting script must be run in an environment where the `lightcurve_fitting` Python module is importable. The post-processing
examples can run from this repository using the representative fixture outputs in `examples/outputs/fit_parametric_lightcurve/`.
For now this is only usable with `g`, `r` and `i` photometric bands, but adapting it should be relatively straightforward.

## Repository layout

- `fit_parametric_lightcurve.py`: fits CSV or whitespace DAT light curves,
  saves cleaned photometry, raw fit JSON, plots, and boundary-hit summaries.
- `plot_metzgerkn_boundary_lightcurves.py`: reads
  `metzgerkn_boundary_hits.csv` and produces annotated plots for events whose
  fitted parameters reached optimizer bounds.
- `plot_utils.py`: makes diagnostic plots comparing recovered `t0` with a
  truth table.
- `cut_features.py`: enriches FIESTA/Bulla feature CSVs with helper columns
  used by the `t0` diagnostics.
- `boom_objects_to_dat.py`: converts outputs from the [`lsst_boom_filter_analysis`](https://github.com/grandma-collaboration/lsst_boom_filter_analysis)
  code (the complete objects `.json` file) into the right `.dat` format for
  running `fit_parametric_lightcurve.py` on these objects.
- `examples/`: synthetic inputs plus representative outputs for every script.
  It also contains a few real LSST-like inputs and fitted outputs copied from
  the sibling `lightcurve-fitting` working directory.

## Example inputs

The example light-curve CSV has one row per photometric point:

```text
event_id,time,band,mag,mag_err
1001,60200.00,g,21.10,0.08
```

Required fitting columns are:

- `time`: numeric MJD/JD or datetime-like strings. The LSST-like
  `observationStartMJD` column is also recognized automatically.
- `band`: one of `g`, `r`, `i`, or supported aliases such as `ps1::r`
- `mag`: apparent magnitude. The LSST-like `mag_obs` column is also recognized
  automatically.
- `mag_err`: magnitude uncertainty
- optional `event_id`: used with `--id-col` for batch fitting

For Alex-style LSST-like files with columns such as
`observationStartMJD,band,m5,t_days,fiesta_filter,mag_true,snr_exp,mag_err,snr_obs,detected,mag_obs,mag_ulim`,
the fitter uses `observationStartMJD`, `band`, `mag_obs`, and `mag_err`.
Bands outside `g`, `r`, and `i` are dropped by the current MetzgerKN workflow.

The DAT format is whitespace-separated:

```text
datetime band mag mag_err
2023-09-13T00:00:00 ps1::g 21.10 0.08
```

The `examples/inputs/real_lsstlike/` directory contains real files copied from
`../lightcurve-fitting/alex_lsstlike_1000/`:

- `lightcurve_LSSTlike_0090.csv`
- `lightcurve_LSSTlike_0495.csv`
- `lightcurve_LSSTlike_0637.csv`
- `features_kn_cut_1000_real_excerpt.csv`

The matching real fitted-output examples are in
`examples/outputs/real_lsstlike_fitted/`. Event `0090` includes a real cleaned
photometry CSV, result JSON, and fitted plot copied from
`../lightcurve-fitting/test_n101_alex_lsstlike_fitted/`.

## Run the examples

From this repository:

```bash
cd path/to/lightcurve_fitting_metzgerkn
```

### 1. Enrich a truth/features table

```bash
python cut_features.py \
  --input examples/inputs/features_kn_cut_1000.csv \
  --output examples/outputs/cut_features/features_kn_cut_1000_enriched.csv
```

Expected output:

- `examples/outputs/cut_features/features_kn_cut_1000_enriched.csv`

The added columns are `log10_mej_tot`, `log10_vej`, and `t0`.

### 2. Convert BOOM object JSON files to `.dat` inputs

`boom_objects_to_dat.py` converts complete BOOM object JSON files, such as the
complete object `.json` outputs (`objs.json`) from
[`lsst_boom_filter_analysis`](https://github.com/grandma-collaboration/lsst_boom_filter_analysis),
into one whitespace `.dat` lightcurve file per object. The generated files can
then be passed directly to `fit_parametric_lightcurve.py`.

```bash
python boom_objects_to_dat.py \
  path/to/objs.json \
  examples/outputs/boom_dat_inputs \
```

Expected output:

- `examples/outputs/boom_dat_inputs/boom_lc_obj<OBJECT_ID>.dat`
- `examples/outputs/boom_dat_inputs/dat_generation_summary.json`

Each `.dat` row has the format expected by the fitter:

```text
2026-02-16T02:14:39.831Z lsst::r 23.32315445 0.11702178
```

By default, the converter reads photometry from both `prv_candidates` and
`fp_hists`. Use `--fields prv_candidates` to export only one history field, and
use `--no-clobber` if you want the script to stop rather than overwrite existing
`.dat` files.

### 3. Fit example light curves

CSV batch input:

```bash
python fit_parametric_lightcurve.py \
  --input examples/inputs/example_lightcurve.csv \
  --input-format csv \
  --id-col event_id \
  --model MetzgerKN \
  --output-dir examples/outputs/fit_parametric_lightcurve_from_run
```

Real LSST-like input copied from the original working folder:

```bash
python fit_parametric_lightcurve.py \
  --input examples/inputs/real_lsstlike/lightcurve_LSSTlike_0090.csv \
  --input-format csv \
  --model MetzgerKN \
  --output-dir examples/outputs/real_lsstlike_fitted_from_run
```

DAT input:

```bash
python fit_parametric_lightcurve.py \
  --input examples/inputs/example_lightcurve.dat \
  --input-format dat \
  --model MetzgerKN \
  --output-dir examples/outputs/fit_parametric_lightcurve_from_dat
```

Typical output files:

- `<event>_used_photometry.csv`: cleaned photometry actually passed to the fitter
- `<event>_parametric_result.json`: raw JSON-serializable fitter result
- `<event>_parametric_data.png`: data and model plot
- `<event>_parametric_data.pickle`: editable Matplotlib figure
- `metzgerkn_boundary_hits.csv`: parameter-bound hits, when any are found

### 4. Plot boundary-hit light curves

```bash
python plot_metzgerkn_boundary_lightcurves.py \
  --boundary-hits-csv examples/outputs/fit_parametric_lightcurve/metzgerkn_boundary_hits.csv \
  --photometry-dir examples/outputs/fit_parametric_lightcurve \
  --output-dir examples/outputs/plot_metzgerkn_boundary_lightcurves
```

Expected output:

- `examples/outputs/plot_metzgerkn_boundary_lightcurves/event_1001_boundary_lightcurve.png`
- `examples/outputs/plot_metzgerkn_boundary_lightcurves/event_1001_boundary_lightcurve.pickle`

If `lightcurve_fitting` is not importable, the plot is still saved, but without
model overlays.

### 5. Plot `t0` recovery diagnostics

```bash
python plot_utils.py \
  --output-dir examples/outputs/fit_parametric_lightcurve \
  --truth-csv examples/outputs/cut_features/features_kn_cut_1000_enriched.csv \
  --save-dir examples/outputs/plot_utils
```

Expected output:

- `examples/outputs/plot_utils/chi2/parametric_coverage_chi2_vs_t0_red_chi2.png`
- `examples/outputs/plot_utils/chi2/parametric_coverage_chi2_vs_t0_red_chi2.pickle`
- `examples/outputs/plot_utils/detections/parametric_coverage_detections_vs_t0.png`
- `examples/outputs/plot_utils/detections/parametric_coverage_detections_vs_t0.pickle`

## Notes

- `fit_parametric_lightcurve.py` normalizes common survey band labels to `g`,
  `r`, and `i`.
- The boundary-hit CSV is only written when at least one fitted parameter is
  within `--boundary-epsilon` of a MetzgerKN parameter bound.
- Pickle files contain editable Matplotlib figure objects and can be reopened
  in Python for further styling.
