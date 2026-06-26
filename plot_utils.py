'''
Script for reproducing the analysis from Noah Jamsin, assessing how
well the t0 parameter is recovered by the parametric fitter, and how
this recovery depends on the number of detections and the chi2 of the fit.
'''

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize


METZGERKN_T0_INDEX = 3
DEFAULT_ZP = 23.90
DEFAULT_BANDS = ("g", "r", "i")
PARAM_SOURCE_PRIORITY = ("pso_params", "svi_mu")
BAND_COLORS = {"g": "green", "r": "red", "i": "purple"}
DEFAULT_MAX_POINTS_BY_BAND = {"i": 10000, "r": 10000, "g": 10000}
# DEFAULT_MAX_POINTS_BY_BAND = {"i": 10, "r": 10, "g": 10}

def save_figure_pickle(fig: plt.Figure, out_path: Path) -> None:
    """Save a Matplotlib figure object so it can be reopened and edited."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(fig, f)


def safe_filename(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))
    return safe.strip("_") or "plot"


def normalize_band(band: Any) -> str:
    """Normalize names such as ps1::r or lsst::r to r."""
    band_str = str(band)
    if "::" in band_str:
        band_str = band_str.split("::")[-1]
    return band_str.strip()


def infer_event_id_from_name(path: Path) -> int | None:
    """Infer the event id from names like lightcurve_LSSTlike_0969_parametric_result.json."""
    match = re.search(r"(\d+)(?=_parametric_result\.json$)", path.name)
    if match is None:
        matches = re.findall(r"\d+", path.stem)
        if not matches:
            return None
        return int(matches[-1])
    return int(match.group(1))


def iter_fit_dicts(obj: Any) -> Iterable[dict[str, Any]]:
    """Yield per-band fit dictionaries from the parametric Rust/Python result object."""
    if isinstance(obj, list):
        for item in obj:
            yield from iter_fit_dicts(item)
    elif isinstance(obj, dict):
        if "band" in obj and any(key in obj for key in PARAM_SOURCE_PRIORITY):
            yield obj
        else:
            for value in obj.values():
                if isinstance(value, (dict, list)):
                    yield from iter_fit_dicts(value)


def choose_params(fit: dict[str, Any], param_source: str) -> tuple[list[float] | None, str | None]:
    """Return the selected parameter vector and the source key used."""
    if param_source != "auto":
        values = fit.get(param_source)
        return (values, param_source) if isinstance(values, list) else (None, None)

    explicit_source = fit.get("param_source")
    if isinstance(explicit_source, str) and isinstance(fit.get(explicit_source), list):
        return fit[explicit_source], explicit_source

    for key in PARAM_SOURCE_PRIORITY:
        values = fit.get(key)
        if isinstance(values, list):
            return values, key

    return None, None


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def mag_and_err_to_flux_and_err(mag: Any, mag_err: Any, zp: float = DEFAULT_ZP) -> tuple[np.ndarray, np.ndarray]:
    mag = np.asarray(mag, dtype=float)
    mag_err = np.asarray(mag_err, dtype=float)

    flux = 10.0 ** ((zp - mag) / 2.5)
    flux_err = (np.log(10.0) / 2.5) * flux * mag_err
    return flux, flux_err


def normalized_flux_errors_for_band(phot_df: pd.DataFrame | None, band: str, zp: float = DEFAULT_ZP) -> np.ndarray:
    if phot_df is None or phot_df.empty or not {"band", "mag", "mag_err"}.issubset(phot_df.columns):
        return np.asarray([], dtype=float)

    normalized_bands = phot_df["band"].map(normalize_band)
    sub = phot_df[normalized_bands == normalize_band(band)].copy()
    if sub.empty:
        return np.asarray([], dtype=float)

    mag = sub["mag"].to_numpy(dtype=float)
    mag_err = sub["mag_err"].to_numpy(dtype=float)
    flux, flux_err = mag_and_err_to_flux_and_err(mag, mag_err, zp=zp)

    good_flux = np.isfinite(flux) & (flux > 0.0)
    peak_flux = float(np.max(flux[good_flux])) if np.any(good_flux) else float("nan")
    if not np.isfinite(peak_flux) or peak_flux <= 0.0:
        return np.full_like(flux_err, np.nan, dtype=float)
    return flux_err / peak_flux


def reduced_chi2_from_fit(
    fit: dict[str, Any],
    params: list[float] | None,
    norm_flux_err: np.ndarray,
) -> float:
    """Same reduced chi2 calculation used in fit_parametric_lightcurve.py."""
    pso_chi2 = finite_float(fit.get("pso_chi2"))
    if params is None or len(params) == 0 or not np.isfinite(pso_chi2):
        return float("nan")

    n_obs = int(fit.get("n_obs", len(norm_flux_err)))
    extra_sigma = math.exp(finite_float(params[-1]))
    if not np.isfinite(extra_sigma):
        extra_sigma = 0.0

    errs = np.asarray(norm_flux_err, dtype=float)
    if errs.size == 0:
        return float("nan")

    log_var = float(np.sum(np.log(errs**2 + extra_sigma**2)))
    dofs = n_obs - len(params)
    if dofs <= 0:
        return float("nan")
    return (n_obs * pso_chi2 - log_var) / dofs


def load_truth_t0(truth_csv: str | Path, event_col: str = "event_id", true_t0_col: str = "t0") -> pd.Series:
    """Load known t0 values indexed by event id."""
    truth_df = pd.read_csv(truth_csv, usecols=[event_col, true_t0_col])
    truth_df = truth_df.dropna(subset=[event_col])
    truth_df[event_col] = truth_df[event_col].astype(int)
    truth_df[true_t0_col] = pd.to_numeric(truth_df[true_t0_col], errors="coerce")
    truth_df = truth_df.dropna(subset=[true_t0_col])
    truth_df = truth_df.drop_duplicates(subset=[event_col], keep="first")
    return truth_df.set_index(event_col)[true_t0_col]


def coverage_deviation_from_truth(
    fitted_t0: float,
    true_t0: float,
    t0_sigma: float | None = None,
    n_sig: float = 3.0,
) -> float:
    """Distance between truth and the fit interval.

    If t0_sigma is not provided, this is simply abs(fitted_t0 - true_t0).
    If t0_sigma is provided:
    return zero when truth is inside fitted_t0 +/- n_sig * sigma, otherwise
    return the distance from truth to the nearest interval edge.
    """
    if not np.isfinite(fitted_t0) or not np.isfinite(true_t0):
        return float("nan")

    if t0_sigma is None or not np.isfinite(t0_sigma) or t0_sigma <= 0:
        return abs(fitted_t0 - true_t0)

    lower = fitted_t0 - n_sig * t0_sigma
    upper = fitted_t0 + n_sig * t0_sigma
    if lower <= true_t0 <= upper:
        return 0.0
    if true_t0 < lower:
        return lower - true_t0
    return true_t0 - upper


def aggregate_values(values: Iterable[float], how: str) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan")
    if how == "sum":
        return float(np.sum(arr))
    if how == "mean":
        return float(np.mean(arr))
    if how == "median":
        return float(np.median(arr))
    if how == "min":
        return float(np.min(arr))
    if how == "max":
        return float(np.max(arr))
    raise ValueError(f"Unknown aggregation method: {how}")


def collect_parametric_coverage_data(
    output_dir: str | Path,
    truth_csv: str | Path,
    *,
    result_pattern: str = "*_parametric_result.json",
    event_col: str = "event_id",
    true_t0_col: str = "t0",
    param_source: str = "pso_params",
    chi2_key: str = "red_chi2",
    use_svi_interval: bool = False,
    n_sig: float = 3.0,
    zp: float = DEFAULT_ZP,
) -> pd.DataFrame:
    """Collect one row per fitted band from parametric fitter outputs.

    The default t0 diagnostic is abs(fitted_t0 - known_t0). When use_svi_interval
    is True and SVI uncertainty is available, the diagnostic becomes the
    distance from known_t0 to the fitted Gaussian interval.
    """
    output_dir = Path(output_dir)
    truth_t0 = load_truth_t0(truth_csv, event_col=event_col, true_t0_col=true_t0_col)

    rows: list[dict[str, Any]] = []
    for result_path in sorted(output_dir.glob(result_pattern)):
        event_id = infer_event_id_from_name(result_path)
        base_name = result_path.name.replace("_parametric_result.json", "")
        phot_path = result_path.with_name(f"{base_name}_used_photometry.csv")

        phot_df = None
        band_counts: dict[str, int] = {}
        total_detections = float("nan")
        if phot_path.exists():
            phot_df = pd.read_csv(phot_path)
            if event_id is None and event_col in phot_df.columns and not phot_df.empty:
                event_id = int(phot_df[event_col].iloc[0])
            if "band" in phot_df.columns:
                normalized_bands = phot_df["band"].map(normalize_band)
                band_counts = normalized_bands.value_counts().to_dict()
            total_detections = int(len(phot_df))

        if event_id is None or event_id not in truth_t0.index:
            continue

        true_t0 = finite_float(truth_t0.at[event_id])
        if not np.isfinite(true_t0):
            continue

        with result_path.open() as f:
            result = json.load(f)

        for fit in iter_fit_dicts(result):
            band = normalize_band(fit.get("band", "unknown"))
            params, used_source = choose_params(fit, param_source)
            if params is None or len(params) <= METZGERKN_T0_INDEX:
                continue

            fitted_t0 = finite_float(params[METZGERKN_T0_INDEX])
            t0_sigma = float("nan")
            if use_svi_interval and used_source == "svi_mu":
                log_sigma = fit.get("svi_log_sigma")
                if isinstance(log_sigma, list) and len(log_sigma) > METZGERKN_T0_INDEX:
                    t0_sigma = math.exp(finite_float(log_sigma[METZGERKN_T0_INDEX]))

            y_val = coverage_deviation_from_truth(
                fitted_t0=fitted_t0,
                true_t0=true_t0,
                t0_sigma=t0_sigma if use_svi_interval else None,
                n_sig=n_sig,
            )

            n_obs = int(fit.get("n_obs", band_counts.get(band, 0)))
            band_n_det = int(band_counts.get(band, n_obs))
            norm_flux_err = normalized_flux_errors_for_band(phot_df, band, zp=zp)
            red_chi2 = reduced_chi2_from_fit(fit, params, norm_flux_err)
            if chi2_key == "red_chi2":
                chi2_value = red_chi2
            else:
                chi2_value = finite_float(fit.get(chi2_key))

            rows.append(
                {
                    "event_id": int(event_id),
                    "source_name": base_name,
                    "band": band,
                    "param_source": used_source,
                    "fitted_t0": fitted_t0,
                    "true_t0": true_t0,
                    "t0_sigma": t0_sigma,
                    "t0_deviation": y_val,
                    "chi2": chi2_value,
                    "chi2_key": chi2_key,
                    "red_chi2": red_chi2,
                    "mag_chi2": finite_float(fit.get("mag_chi2")),
                    "pso_chi2": finite_float(fit.get("pso_chi2")),
                    "n_obs": n_obs,
                    "band_n_det": band_n_det,
                    "total_n_det": total_detections,
                    "result_path": str(result_path),
                    "photometry_path": str(phot_path) if phot_path.exists() else "",
                }
            )

    return pd.DataFrame(rows)


def make_total_rows(
    df: pd.DataFrame,
    *,
    y_aggregate: str = "mean",
    chi2_aggregate: str = "sum",
) -> pd.DataFrame:
    """Build one 'total' row per event from per-band rows."""
    if df.empty:
        return df.copy()

    rows: list[dict[str, Any]] = []
    for event_id, group in df.groupby("event_id", sort=True):
        first = group.iloc[0].to_dict()
        first["band"] = "total"
        first["fitted_t0"] = aggregate_values(group["fitted_t0"], y_aggregate)
        first["t0_deviation"] = aggregate_values(group["t0_deviation"], y_aggregate)
        first["chi2"] = aggregate_values(group["chi2"], chi2_aggregate)
        first["n_obs"] = int(np.nansum(group["n_obs"].to_numpy(dtype=float)))
        if "total_n_det" in group:
            first["band_n_det"] = finite_float(group["total_n_det"].iloc[0])
        else:
            first["band_n_det"] = first["n_obs"]
        rows.append(first)

    return pd.DataFrame(rows)


def make_total_overlay_rows(df: pd.DataFrame, bands: Iterable[str]) -> pd.DataFrame:
    """Copy per-band rows into a combined 'total' panel while preserving their band identity."""
    if df.empty:
        return df.copy()

    normalized_bands = [normalize_band(b) for b in bands]
    total_df = df[df["band"].isin(normalized_bands)].copy()
    if total_df.empty:
        return total_df
    total_df["point_band"] = total_df["band"]
    total_df["band"] = "total"
    return total_df


def filter_by_band_detection_limits(
    df: pd.DataFrame,
    max_points_by_band: dict[str, int] | None,
) -> pd.DataFrame:
    """Drop band light curves above the requested per-band detection-count limits."""
    if df.empty or not max_points_by_band:
        return df.copy()

    limits = {normalize_band(band): int(limit) for band, limit in max_points_by_band.items()}
    keep = np.ones(len(df), dtype=bool)
    for band, limit in limits.items():
        band_mask = df["band"].map(normalize_band).to_numpy() == band
        count_mask = df["band_n_det"].to_numpy(dtype=float) > limit
        keep &= ~(band_mask & count_mask)
    return df.loc[keep].copy()


def event_color_mapping(event_ids: Iterable[int]) -> tuple[Any, Normalize]:
    unique_ids = sorted({int(eid) for eid in event_ids})
    if not unique_ids:
        unique_ids = [0]
    vmin = min(unique_ids)
    vmax = max(unique_ids)
    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5
    return plt.cm.viridis, Normalize(vmin=vmin, vmax=vmax)


def add_event_colorbar(fig: plt.Figure, scatter: Any, event_ids: Iterable[int]) -> None:
    unique_ids = sorted({int(eid) for eid in event_ids})
    if not unique_ids:
        return

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    if len(unique_ids) <= 12:
        cbar.set_ticks(unique_ids)
    else:
        tick_values = np.linspace(min(unique_ids), max(unique_ids), 8)
        cbar.set_ticks(tick_values)
    cbar.set_label("Event id", fontsize=12)


def plot_panel_grid(
    plot_df: pd.DataFrame,
    *,
    panel_bands: list[str],
    x_col: str,
    x_label: str,
    y_label: str,
    title: str,
    out_png: Path,
    log_x: bool = False,
    jitter_x: bool = False,
    color_total_by_band: bool = False,
) -> pd.DataFrame:
    """Shared panel plotting helper."""
    plot_df = plot_df.copy()
    plot_df = plot_df[np.isfinite(plot_df[x_col]) & np.isfinite(plot_df["t0_deviation"])]
    if plot_df.empty:
        print(f"No finite data to plot for {out_png.name}")
        return plot_df

    n_panels = len(panel_bands)
    if n_panels <= 4:
        nrows, ncols = 1, n_panels
    else:
        ncols = 3
        nrows = int(math.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.2 * ncols, 4.3 * nrows), sharey=True)
    axes_arr = np.atleast_1d(axes).ravel()

    unique_ids = sorted(plot_df["event_id"].dropna().astype(int).unique().tolist())
    cmap, norm = event_color_mapping(unique_ids)
    event_scatter = None

    rng = np.random.default_rng(12345)
    for ax, band in zip(axes_arr, panel_bands):
        sub = plot_df[plot_df["band"] == normalize_band(band)].copy()
        if not sub.empty:
            x = sub[x_col].to_numpy(dtype=float)
            if jitter_x:
                x = x + rng.uniform(-0.15, 0.15, size=len(x))
            if color_total_by_band and normalize_band(band) == "total":
                sub = sub.copy()
                sub["_plot_x"] = x
                point_bands = sub.get("point_band", sub["band"]).map(normalize_band)
                for point_band, band_sub in sub.groupby(point_bands):
                    ax.scatter(
                        band_sub["_plot_x"].to_numpy(dtype=float),
                        band_sub["t0_deviation"].to_numpy(dtype=float),
                        color=BAND_COLORS.get(point_band, "gray"),
                        edgecolors="k",
                        alpha=0.85,
                        s=55,
                        label=point_band,
                    )
                ax.legend(title="Band", frameon=False)
            else:
                color_values = sub["event_id"].astype(int).to_numpy(dtype=float)
                event_scatter = ax.scatter(
                    x,
                    sub["t0_deviation"].to_numpy(dtype=float),
                    c=color_values,
                    cmap=cmap,
                    norm=norm,
                    edgecolors="k",
                    alpha=0.85,
                    s=55,
                )

        ax.set_title(str(band))
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        if log_x and not sub.empty and np.all(sub[x_col].to_numpy(dtype=float) > 0):
            ax.set_xscale("log")
        if jitter_x:
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xlabel(x_label)

    for ax in axes_arr[n_panels:]:
        ax.set_visible(False)

    axes_arr[0].set_ylabel(y_label)
    fig.suptitle(title, fontsize=15, y=0.98)
    plt.subplots_adjust(left=0.06, bottom=0.10, right=0.90, top=0.88, wspace=0.12, hspace=0.20)

    if event_scatter is not None:
        add_event_colorbar(fig, event_scatter, unique_ids)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    save_figure_pickle(fig, out_png.with_suffix(".pickle"))
    plt.close(fig)
    return plot_df


def plot_parametric_chi2_vs_ts(
    output_dir: str | Path,
    truth_csv: str | Path,
    *,
    save_dir: str | Path | None = None,
    bands: Iterable[str] = DEFAULT_BANDS,
    include_total: bool = True,
    param_source: str = "pso_params",
    chi2_key: str = "red_chi2",
    use_svi_interval: bool = False,
    n_sig: float = 3.0,
    total_y_aggregate: str = "mean",
    total_chi2_aggregate: str = "sum",
    result_pattern: str = "*_parametric_result.json",
    zp: float = DEFAULT_ZP,
) -> pd.DataFrame:
    """Plot fitted t0 coverage/deviation versus chi2 for fitter outputs.

    Returns the table used for plotting. The saved files are:
    parametric_coverage_chi2_vs_t0_<chi2_key>.png and .pickle.
    """
    output_dir = Path(output_dir)
    save_dir = Path(save_dir) if save_dir is not None else output_dir / "t0_recovery" / "chi2"

    df = collect_parametric_coverage_data(
        output_dir,
        truth_csv,
        result_pattern=result_pattern,
        param_source=param_source,
        chi2_key=chi2_key,
        use_svi_interval=use_svi_interval,
        n_sig=n_sig,
        zp=zp,
    )

    panel_bands = [normalize_band(b) for b in bands]
    plot_df = df[df["band"].isin(panel_bands)].copy() if not df.empty else df
    if not plot_df.empty:
        plot_df["point_band"] = plot_df["band"]
    if include_total and not df.empty:
        total_df = make_total_overlay_rows(df, panel_bands)
        plot_df = pd.concat([plot_df, total_df], ignore_index=True)
        panel_bands.append("total")

    interval_note = f"outside {n_sig:g} sigma interval" if use_svi_interval else "absolute fit error"
    out_png = save_dir / f"parametric_coverage_chi2_vs_t0_{safe_filename(chi2_key)}.png"
    return plot_panel_grid(
        plot_df,
        panel_bands=panel_bands,
        x_col="chi2",
        x_label=chi2_key,
        y_label=f"t0 deviation from truth [days] ({interval_note})",
        title=f"Parametric fitter: t0 deviation vs {chi2_key}",
        out_png=out_png,
        log_x=False,
        jitter_x=False,
        color_total_by_band=True,
    )


def plot_parametric_detections_vs_ts(
    output_dir: str | Path,
    truth_csv: str | Path,
    *,
    save_dir: str | Path | None = None,
    bands: Iterable[str] = DEFAULT_BANDS,
    include_total: bool = True,
    param_source: str = "pso_params",
    use_svi_interval: bool = False,
    n_sig: float = 3.0,
    total_y_aggregate: str = "mean",
    result_pattern: str = "*_parametric_result.json",
    max_points_by_band: dict[str, int] | None = None,
    zp: float = DEFAULT_ZP,
) -> pd.DataFrame:
    """Plot fitted t0 coverage/deviation versus number of detections."""
    output_dir = Path(output_dir)
    save_dir = Path(save_dir) if save_dir is not None else output_dir / "t0_recovery" / "detections"

    df = collect_parametric_coverage_data(
        output_dir,
        truth_csv,
        result_pattern=result_pattern,
        param_source=param_source,
        use_svi_interval=use_svi_interval,
        n_sig=n_sig,
        zp=zp,
    )

    panel_bands = [normalize_band(b) for b in bands]
    if max_points_by_band is None:
        max_points_by_band = DEFAULT_MAX_POINTS_BY_BAND
    df = filter_by_band_detection_limits(df, max_points_by_band)
    plot_df = df[df["band"].isin(panel_bands)].copy() if not df.empty else df
    if not plot_df.empty:
        plot_df["point_band"] = plot_df["band"]
    if include_total and not df.empty:
        total_df = make_total_overlay_rows(df, panel_bands)
        plot_df = pd.concat([plot_df, total_df], ignore_index=True)
        panel_bands.append("total")

    interval_note = f"outside {n_sig:g} sigma interval" if use_svi_interval else "absolute fit error"
    out_png = save_dir / "parametric_coverage_detections_vs_t0.png"
    return plot_panel_grid(
        plot_df,
        panel_bands=panel_bands,
        x_col="band_n_det",
        x_label="Number of detections",
        y_label=f"t0 deviation from truth [days] ({interval_note})",
        title="Parametric fitter: t0 deviation vs detections",
        out_png=out_png,
        log_x=True,
        jitter_x=False,
        color_total_by_band=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnostic plots for parametric lightcurve fitter outputs.")
    parser.add_argument("--output-dir", help="Directory containing *_parametric_result.json and *_used_photometry.csv files.")
    parser.add_argument("--truth-csv", help="CSV containing event_id and known t0 columns.")
    parser.add_argument("--save-dir", default=None, help="Optional base directory for generated diagnostic plots.")
    parser.add_argument("--bands", nargs="+", default=list(DEFAULT_BANDS), help="Bands to show before the total panel.")
    parser.add_argument("--param-source", choices=["pso_params", "svi_mu", "auto"], default="pso_params")
    parser.add_argument("--chi2-key", default="red_chi2", help="Fit-result key to use on the chi2 plot x-axis.")
    parser.add_argument("--use-svi-interval", action="store_true", help="Use t0 +/- n_sig * exp(svi_log_sigma[t0]) when available.")
    parser.add_argument("--n-sig", type=float, default=3.0)
    parser.add_argument("--no-total", action="store_true", help="Do not add the total panel.")
    parser.add_argument("--max-g-points", type=int, default=DEFAULT_MAX_POINTS_BY_BAND["g"])
    parser.add_argument("--max-r-points", type=int, default=DEFAULT_MAX_POINTS_BY_BAND["r"])
    parser.add_argument("--max-i-points", type=int, default=DEFAULT_MAX_POINTS_BY_BAND["i"])
    parser.add_argument("--zp", type=float, default=DEFAULT_ZP)
    args = parser.parse_args()

    max_points_by_band = {
        "g": args.max_g_points,
        "r": args.max_r_points,
        "i": args.max_i_points,
    }
    chi2_save_dir = Path(args.save_dir) / "chi2" if args.save_dir else None
    detections_save_dir = Path(args.save_dir) / "detections" if args.save_dir else None

    plot_parametric_chi2_vs_ts(
        args.output_dir,
        args.truth_csv,
        save_dir=chi2_save_dir,
        bands=args.bands,
        include_total=not args.no_total,
        param_source=args.param_source,
        chi2_key=args.chi2_key,
        use_svi_interval=args.use_svi_interval,
        n_sig=args.n_sig,
        zp=args.zp,
    )
    plot_parametric_detections_vs_ts(
        args.output_dir,
        args.truth_csv,
        save_dir=detections_save_dir,
        bands=args.bands,
        include_total=not args.no_total,
        param_source=args.param_source,
        use_svi_interval=args.use_svi_interval,
        n_sig=args.n_sig,
        max_points_by_band=max_points_by_band,
        zp=args.zp,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
