"""
Plot light curves for sources whose MetzgerKN fits reached parameter bounds.

The script reads metzgerkn_boundary_hits.csv from a fitted output
directory of the fit_parametric_lightcurve.py script, finds the corresponding
*_used_photometry.csv files, and saves one annotated light-curve plot per event.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from textwrap import wrap
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import lightcurve_fitting as lcf
except ModuleNotFoundError:
    lcf = None


BANDS_ORDER = ["g", "r", "i"]
BAND_COLOR = {"g": "green", "r": "red", "i": "purple"}
BAND_MARKER = {"g": "x", "r": "x", "i": "x"}
BAND_LABEL = {"g": "g", "r": "r", "i": "i"}
DEFAULT_ZP = 23.90


def safe_filename(text: object) -> str:
    """Convert arbitrary text to a filesystem-safe filename fragment."""
    text = str(text)
    text = text.replace("/", "_per_")
    text = re.sub(r"[^\w.\-+]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def save_figure_pickle(fig: plt.Figure, output_path: Path) -> None:
    """Save a Matplotlib figure object as a .pickle file so it can be reopened and edited."""
    with open(output_path, "wb") as f:
        pickle.dump(fig, f)


def flux_to_mag(flux: np.ndarray, zp: float = DEFAULT_ZP) -> np.ndarray:
    """Convert positive flux values to AB magnitudes, preserving NaNs elsewhere."""
    flux = np.asarray(flux, dtype=float)
    out = np.full(flux.shape, np.nan, dtype=float)
    pos = flux > 0
    out[pos] = zp - 2.5 * np.log10(flux[pos])
    return out


def mag_and_err_to_flux_and_err(mag: np.ndarray, mag_err: np.ndarray, zp: float = DEFAULT_ZP) -> tuple[np.ndarray, np.ndarray]:
    """Convert magnitudes and magnitude errors to fluxes and flux errors."""
    mag = np.asarray(mag, dtype=float)
    mag_err = np.asarray(mag_err, dtype=float)
    flux = 10.0 ** ((zp - mag) / 2.5)
    flux_err = (np.log(10.0) / 2.5) * flux * mag_err
    return flux, flux_err


def event_id_from_filename(path: Path) -> int | None:
    """Infer an event id from a *_used_photometry.csv filename."""
    match = re.search(r"(\d+)_used_photometry\.csv$", path.name)
    if not match:
        return None
    return int(match.group(1))


def read_photometry_event_id(path: Path) -> int | None:
    """Read event_id from a photometry CSV, falling back to the filename."""
    try:
        event_ids = pd.read_csv(path, usecols=["event_id"])["event_id"].dropna().unique()
    except (FileNotFoundError, KeyError, ValueError, pd.errors.EmptyDataError):
        return event_id_from_filename(path)

    if len(event_ids) == 0:
        return event_id_from_filename(path)
    return int(event_ids[0])


def build_photometry_index(photometry_dir: Path) -> dict[int, Path]:
    """Map event ids to their corresponding *_used_photometry.csv files."""
    index = {}
    for path in sorted(photometry_dir.glob("*_used_photometry.csv")):
        event_id = event_id_from_filename(path)
        if event_id is None:
            event_id = read_photometry_event_id(path)
        if event_id is None:
            continue
        index.setdefault(event_id, path)
    return index


def result_json_for_photometry(photometry_path: Path) -> Path:
    """Return the expected parametric result JSON path for a photometry CSV."""
    return photometry_path.with_name(photometry_path.name.replace("_used_photometry.csv", "_parametric_result.json"))


def iter_fit_dicts(obj: Any) -> list[dict[str, Any]]:
    """Recursively collect per-band fit dictionaries from a saved result JSON."""
    found: list[dict[str, Any]] = []

    def rec_walk(x: Any) -> None:
        if isinstance(x, dict):
            keys = set(x)
            if ("model" in keys or "model_name" in keys) and (
                "pso_params" in keys or "svi_mu" in keys or "params" in keys
            ):
                found.append(x)
            for value in x.values():
                rec_walk(value)
        elif isinstance(x, list):
            for value in x:
                rec_walk(value)

    rec_walk(obj)
    return found


def choose_params(fit_dict: dict[str, Any], param_source: str) -> list[float] | None:
    """Select the requested parameter vector, with common fallback names."""
    if param_source in fit_dict:
        return fit_dict[param_source]
    for key in ("pso_params", "svi_mu", "params", "posterior_mean", "means"):
        if key in fit_dict:
            return fit_dict[key]
    return None


def try_eval_model_curve(fit_dict: dict[str, Any], phase_grid: np.ndarray, param_source: str) -> np.ndarray | None:
    """Evaluate a saved fitted model on a relative-time grid when possible."""
    model = fit_dict.get("model", fit_dict.get("model_name"))
    params = choose_params(fit_dict, param_source)
    if model is None or params is None or not hasattr(lcf, "eval_model"):
        return None

    try:
        y = np.asarray(lcf.eval_model(model, params, phase_grid.astype(float).tolist()), dtype=float)
    except Exception:
        return None

    if y.shape == phase_grid.shape and np.isfinite(y).any():
        return y
    return None


def is_metzgerkn_fit(fit_dict: dict[str, Any]) -> bool:
    """Return True when a fit dictionary represents the MetzgerKN model."""
    return str(fit_dict.get("model", fit_dict.get("model_name", ""))).lower() == "metzgerkn"


def metzgerkn_pso_scale(
    fit_dict: dict[str, Any],
    data_eval_times: np.ndarray,
    data_flux: np.ndarray,
    data_flux_err: np.ndarray,
    param_source: str,
) -> float:
    """Recompute the MetzgerKN PSO normalization used by the fitter."""
    if not is_metzgerkn_fit(fit_dict):
        return 1.0

    pred_at_data = try_eval_model_curve(fit_dict, data_eval_times, param_source)
    if pred_at_data is None or pred_at_data.shape != data_eval_times.shape:
        return 1.0

    snr = np.full_like(data_flux, np.inf, dtype=float)
    good_err = data_flux_err > 0.0
    snr[good_err] = data_flux[good_err] / data_flux_err[good_err]
    is_upper = good_err & (snr < 3.0)

    valid = ~is_upper & np.isfinite(pred_at_data) & np.isfinite(data_eval_times)
    if not np.any(valid):
        return 1.0

    max_pred = float(np.max(pred_at_data[valid]))
    if not np.isfinite(max_pred) or max_pred <= 1e-10:
        return 1.0

    return float(np.clip(1.0 / max_pred, 0.1, 10.0))


def apply_metzgerkn_pso_rescale(
    fit_dict: dict[str, Any],
    model_flux_norm: np.ndarray,
    data_eval_times: np.ndarray,
    data_flux: np.ndarray,
    data_flux_err: np.ndarray,
    param_source: str,
) -> np.ndarray:
    """Apply the MetzgerKN PSO normalization to a model flux curve."""
    scale = metzgerkn_pso_scale(
        fit_dict=fit_dict,
        data_eval_times=data_eval_times,
        data_flux=data_flux,
        data_flux_err=data_flux_err,
        param_source=param_source,
    )
    return np.asarray(model_flux_norm, dtype=float) * scale


def format_hit_line(row: pd.Series) -> str:
    """Format one boundary-hit row for the annotation panel."""
    parameter = row["parameter"]
    boundary = row["boundary"]
    value = float(row["value"])
    boundary_value = float(row["boundary_value"])
    distance = float(row["distance_to_boundary"])
    return f"{parameter}: {boundary} bound ({value:.4g}; bound={boundary_value:.4g}; dist={distance:.2g})"


def format_annotation_text(hits_for_event: pd.DataFrame, max_chars: int = 92) -> str:
    """Build the multi-line text block describing all boundary hits for one event."""
    sections = []
    for band, band_hits in hits_for_event.groupby("band", sort=True):
        lines = [f"{BAND_LABEL.get(str(band), band)} band:"]
        for _, row in band_hits.sort_values(["parameter", "boundary"]).iterrows():
            hit_text = format_hit_line(row)
            wrapped = wrap(hit_text, width=max_chars, subsequent_indent="  ")
            lines.extend(f"  {line}" for line in wrapped)
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def plot_event_lightcurve(
    photometry_path: Path,
    result_json_path: Path,
    hits_for_event: pd.DataFrame,
    output_path: Path,
    event_id: int,
    x_column: str,
    param_source: str,
    n_grid: int,
    zp: float,
) -> None:
    """Create and save an annotated three-band plot for one boundary-hit event."""
    # Load photometry and reconstruct phase so the model overlay uses the same
    # relative time convention as fit_parametric_lightcurve.py.
    phot = pd.read_csv(photometry_path)
    phot = phot.copy()
    if "time" in phot.columns:
        finite_times = pd.to_numeric(phot["time"], errors="coerce").to_numpy(dtype=float)
        finite_times = finite_times[np.isfinite(finite_times)]
        if len(finite_times) > 0:
            phot["phase"] = pd.to_numeric(phot["time"], errors="coerce") - float(np.min(finite_times))

    if x_column not in phot.columns:
        x_column = "phase" if "phase" in phot.columns else "time"
    if x_column not in phot.columns:
        raise ValueError(f"{photometry_path} has neither requested x column nor phase/time columns.")
    if "band" not in phot.columns or "mag" not in phot.columns:
        raise ValueError(f"{photometry_path} must contain at least band and mag columns.")

    phot["band"] = phot["band"].astype(str).str.lower()

    # Load saved fit dictionaries, indexed by band, so each panel can draw its
    # own model curve if the result JSON and lightcurve_fitting binding exist.
    fit_by_band: dict[str, dict[str, Any]] = {}
    if result_json_path.exists():
        with open(result_json_path) as f:
            result = json.load(f)
        fit_by_band = {
            str(fit.get("band", "")).lower(): fit
            for fit in iter_fit_dicts(result)
            if str(fit.get("band", "")).lower() in BANDS_ORDER
        }

    phase_values = pd.to_numeric(phot["phase"], errors="coerce").to_numpy(dtype=float) if x_column == "phase" and "phase" in phot.columns else None
    if phase_values is not None and np.isfinite(phase_values).any():
        phase_min = float(np.nanmin(phase_values[np.isfinite(phase_values)]))
        phase_max = float(np.nanmax(phase_values[np.isfinite(phase_values)]))
        phase_grid = np.linspace(phase_min, phase_max, int(n_grid))
        eval_grid = phase_grid - phase_min
    else:
        phase_min = float("nan")
        phase_grid = None
        eval_grid = None

    # The figure has one data panel per band plus a right-hand text panel that
    # explains which fitted parameters reached optimizer boundaries.
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(21, 5),
        sharex=True,
        sharey=True,
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 1.15]},
    )
    data_axes = axes[:3]
    annotation_ax = axes[3]
    ax_by_band = dict(zip(BANDS_ORDER, data_axes))
    model_overlay_ok = False

    boundary_bands = set(hits_for_event["band"].astype(str))
    for band in BANDS_ORDER:
        # Plot observed magnitudes first; bands with boundary hits are drawn a
        # little more strongly to make them visually stand out.
        ax = ax_by_band[band]
        band_df = phot[phot["band"] == band].copy()
        ax.set_title(f"{BAND_LABEL.get(band, band)} band", fontsize=10)
        ax.grid(alpha=0.3)

        if band_df.empty:
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                alpha=0.7,
            )
            continue

        band_df = band_df.sort_values(x_column)
        x = pd.to_numeric(band_df[x_column], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(band_df["mag"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if not finite.any():
            continue

        yerr = None
        yerr_values = np.full_like(y, np.nan, dtype=float)
        if "mag_err" in band_df.columns:
            yerr_values = pd.to_numeric(band_df["mag_err"], errors="coerce").to_numpy(dtype=float)
            yerr = np.where(np.isfinite(yerr_values), yerr_values, np.nan)[finite]

        is_boundary_band = band in boundary_bands
        ax.errorbar(
            x[finite],
            y[finite],
            yerr=yerr,
            fmt=BAND_MARKER.get(band, "o"),
            ms=6 if is_boundary_band else 5,
            color=BAND_COLOR.get(band),
            alpha=0.75 if is_boundary_band else 0.5,
            linestyle="none",
            label=f"{BAND_LABEL.get(band, band)} data",
        )

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=9, loc="best")

        if phase_grid is None or eval_grid is None or "phase" not in band_df.columns:
            continue

        # Overlay the fitted model in magnitude space. The saved evaluator
        # returns normalized flux, so the curve is scaled by the data peak flux.
        fit_dict = fit_by_band.get(band)
        if fit_dict is None:
            continue

        phase = pd.to_numeric(band_df["phase"], errors="coerce").to_numpy(dtype=float)
        mag_err = np.where(np.isfinite(yerr_values), yerr_values, np.nan)
        data_finite = np.isfinite(phase) & np.isfinite(y) & np.isfinite(mag_err)
        if not data_finite.any():
            continue

        flux, flux_err = mag_and_err_to_flux_and_err(y[data_finite], mag_err[data_finite], zp=zp)
        good_flux = np.isfinite(flux) & (flux > 0.0)
        if not np.any(good_flux):
            continue

        peak_flux = float(np.max(flux[good_flux]))
        if not np.isfinite(peak_flux) or peak_flux <= 0.0:
            continue

        raw_flux_norm = try_eval_model_curve(fit_dict, eval_grid, param_source)
        if raw_flux_norm is None:
            continue

        flux_norm = apply_metzgerkn_pso_rescale(
            fit_dict=fit_dict,
            model_flux_norm=raw_flux_norm,
            data_eval_times=phase[data_finite] - phase_min,
            data_flux=flux,
            data_flux_err=flux_err,
            param_source=param_source,
        )
        model_mag = flux_to_mag(flux_norm * peak_flux, zp=zp)
        ok = np.isfinite(phase_grid) & np.isfinite(model_mag)
        if not np.any(ok):
            continue

        model_name = fit_dict.get("model", fit_dict.get("model_name", "model"))
        ax.plot(
            phase_grid[ok],
            model_mag[ok],
            "--",
            lw=2.0,
            color=BAND_COLOR.get(band),
            label=f"{model_name} fit",
        )
        model_overlay_ok = True

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=9, loc="best")

    for ax in data_axes:
        if x_column == "phase":
            ax.set_xlabel("Days since first detection")
        else:
            ax.set_xlabel(x_column)

    data_axes[0].set_ylabel("Apparent magnitude")
    data_axes[0].invert_yaxis()

    # Show a compact boundary-hit summary alongside the light curves instead
    # of forcing the reader to cross-reference the CSV.
    annotation_ax.axis("off")
    annotation_ax.set_title("Boundary hits", loc="left")
    annotation_ax.text(
        0.0,
        0.98,
        format_annotation_text(hits_for_event),
        transform=annotation_ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.95},
    )

    fig.suptitle(
        f"Parametric lightcurve_fitting result - magnitude space\n"
        f"Boundary-hit event_id={event_id}"
        + ("" if model_overlay_ok else "\n(model overlay unavailable from returned JSON)"),
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    save_figure_pickle(fig, output_path.with_suffix(".pickle"))
    plt.close(fig)


def main() -> None:
    """Parse CLI options, find boundary-hit events, and save annotated plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary-hits-csv", dest="boundary_hits_csv_flag", type=Path, help="Path to metzgerkn_boundary_hits.csv.")
    parser.add_argument(
        "--photometry-dir",
        type=Path,
        default=None,
        help="Directory containing *_used_photometry.csv files. Defaults to the boundary CSV directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for annotated plots. Defaults to <boundary CSV dir>/boundary_hit_lightcurves.",
    )
    parser.add_argument(
        "--x-column",
        default="phase",
        help="Column to use on the x-axis. The default phase is recomputed as time minus first detection when time is available.",
    )
    parser.add_argument(
        "--param-source",
        choices=["auto", "pso_params", "svi_mu"],
        default="auto",
        help="Parameter vector from the result JSON used for the model overlay. auto reads param_source from the boundary CSV when present.",
    )
    parser.add_argument("--n-grid", type=int, default=400)
    parser.add_argument("--zp", type=float, default=DEFAULT_ZP)
    parser.add_argument("--max-events", type=int, default=None, help="Optional maximum number of events to plot.")
    parser.add_argument("--progress-every", type=int, default=25, help="Print progress every N events.")
    parser.add_argument("--verbose", action="store_true", help="Print one line for every event.")
    args = parser.parse_args()

    boundary_hits_path = args.boundary_hits_csv_flag or args.boundary_hits_csv
    if boundary_hits_path is None:
        parser.error("provide metzgerkn_boundary_hits.csv as a positional argument or with --boundary-hits-csv")

    if lcf is None:
        print("Warning: lightcurve_fitting is not importable; annotated light curves will be saved without model overlays.")

    photometry_dir = args.photometry_dir or boundary_hits_path.parent
    output_dir = args.output_dir or boundary_hits_path.parent / "boundary_hit_lightcurves"
    output_dir.mkdir(parents=True, exist_ok=True)

    hits = pd.read_csv(boundary_hits_path)
    required_cols = {"event_id", "band", "parameter", "value", "boundary", "boundary_value", "distance_to_boundary"}
    missing_cols = required_cols - set(hits.columns)
    if missing_cols:
        raise ValueError(f"{boundary_hits_path} is missing columns: {sorted(missing_cols)}")

    hits["event_id"] = hits["event_id"].astype(int)
    hits["band"] = hits["band"].astype(str)

    param_source = args.param_source
    if param_source == "auto":
        if "param_source" in hits.columns:
            sources = sorted(
                str(source)
                for source in hits["param_source"].dropna().unique().tolist()
                if str(source) in {"pso_params", "svi_mu"}
            )
            if len(sources) == 1:
                param_source = sources[0]
            elif len(sources) > 1:
                param_source = sources[0]
                print(
                    "Warning: boundary CSV contains multiple param_source values "
                    f"{sources}; using {param_source!r}. Pass --param-source explicitly to override."
                )
            else:
                param_source = "pso_params"
                print("Warning: boundary CSV has no usable param_source values; using 'pso_params'.")
        else:
            param_source = "pso_params"
            print("Warning: boundary CSV has no param_source column; using 'pso_params'.")
    print(f"Model overlay parameter source: {param_source}")

    photometry_index = build_photometry_index(photometry_dir)
    event_ids = sorted(hits["event_id"].unique().tolist())
    if args.max_events is not None:
        event_ids = event_ids[: args.max_events]

    print(f"Boundary-hit events to process: {len(event_ids)}")
    print(f"Photometry files indexed: {len(photometry_index)}")

    n_plotted = 0
    missing_photometry = []
    for i, event_id in enumerate(event_ids, start=1):
        if args.verbose or (args.progress_every > 0 and (i == 1 or i % args.progress_every == 0 or i == len(event_ids))):
            print(f"[{i}/{len(event_ids)}] event_id={event_id}")

        photometry_path = photometry_index.get(event_id)
        if photometry_path is None:
            missing_photometry.append(event_id)
            if args.verbose:
                print("  missing photometry; skipping")
            continue

        hits_for_event = hits[hits["event_id"] == event_id].copy()
        output_path = output_dir / f"event_{event_id:04d}_boundary_lightcurve.png"
        result_json_path = result_json_for_photometry(photometry_path)
        if args.verbose:
            print(f"  photometry: {photometry_path}")
            print(f"  result JSON: {result_json_path}")
            print(f"  boundary hits: {len(hits_for_event)}")
        elif not result_json_path.exists():
            print(f"  event_id={event_id}: result JSON missing; plotting data without fit overlay")

        plot_event_lightcurve(
            photometry_path=photometry_path,
            result_json_path=result_json_path,
            hits_for_event=hits_for_event,
            output_path=output_path,
            event_id=event_id,
            x_column=args.x_column,
            param_source=param_source,
            n_grid=args.n_grid,
            zp=args.zp,
        )
        n_plotted += 1
        if args.verbose:
            print(f"  saved: {output_path}")

    print(f"Boundary-hit events in CSV: {hits['event_id'].nunique()}")
    print(f"Annotated light curves saved: {n_plotted}")
    print(f"Output directory: {output_dir}")
    if missing_photometry:
        preview = ", ".join(str(event_id) for event_id in missing_photometry[:20])
        suffix = " ..." if len(missing_photometry) > 20 else ""
        print(f"Missing photometry for {len(missing_photometry)} event(s): {preview}{suffix}")


if __name__ == "__main__":
    main()
