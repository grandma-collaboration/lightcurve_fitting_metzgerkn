"""
Fit light curves with the `lightcurve_fitting` Rust/PyO3 parametric fitter, using the 'MetzgerKN' model, and produce plots and JSON summaries.
This script is 'MetzgerKN'-specific, but still requires using --model MetzgerKN when running it.

This uses the bindings:
    flux_bands = lightcurve_fitting.build_flux_bands(times, mags, mag_errs, bands)
    result = lightcurve_fitting.fit_parametric(flux_bands, method="laplace")

Supported inputs
----------------
1. Whitespace .dat files with columns:
       datetime band mag mag_err

   Example:
       2017-08-18T00:00:00 ps1::r 17.3 0.05

2. CSV files with configurable columns, defaulting to:
       time,band,mag,mag_err

   If --id-col is supplied, one fit is run per object/source.

Outputs
-------
For each fitted light curve, the script writes:
    - *_parametric_result.json      full raw returned fit object
    - *_used_photometry.csv         cleaned photometry actually fitted
    - *_parametric_data.png         observed magnitude plot
    - *_parametric_data.pickle      editable Matplotlib figure object
    - optionally model evaluations if lcf.eval_model works for the returned result

Examples
--------
# DAT files like the previous AT2017gfo script
python fit_parametric_lightcurve_fitting.py --pattern "At2017gfo*.dat" --input-format dat --model MetzgerKN

# One CSV containing one light curve
python fit_parametric_lightcurve_fitting.py --input my_lightcurve.csv --input-format csv --model MetzgerKN

# One CSV containing many objects
python fit_parametric_lightcurve_fitting.py --input all_lightcurves.csv --input-format csv --id-col object_id --model MetzgerKN

# Use SVI uncertainties instead of Laplace
python fit_parametric_lightcurve_fitting.py --input my_lightcurve.csv --input-format csv --method svi --model MetzgerKN

# Choose which returned parameter vector drives evaluations, printed params, and summaries
python fit_parametric_lightcurve_fitting.py --input my_lightcurve.csv --input-format csv --param-source svi_mu --model MetzgerKN
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
import sys
from pathlib import Path
import re
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

import lightcurve_fitting as lcf
print("lightcurve_fitting loaded from:", getattr(lcf, "__file__", None))


DEBUG_ENABLED = False

def safe_filename(text: str) -> str:
    text = str(text)
    text = text.replace("/", "_per_")
    text = re.sub(r"[^\w.\-+]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")

def infer_event_id(name: str) -> int | str:
    digit_groups = re.findall(r"\d+", str(name))
    if digit_groups:
        return int(digit_groups[-1])
    return str(name)

def format_runtime(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)} min {sec:.1f} s"
    hours, rem = divmod(minutes, 60)
    return f"{int(hours)} h {int(rem)} min {sec:.1f} s"

def debug_print(*args: Any, **kwargs: Any) -> None:
    """Print debug messages to stderr when --debug is enabled."""
    if DEBUG_ENABLED:
        print("[DEBUG]", *args, file=sys.stderr, **kwargs)

def debug_preview(value: Any, max_items: int = 5) -> Any:
    """Return a compact preview for debug output without dumping huge objects."""
    try:
        if isinstance(value, pd.DataFrame):
            return {
                "shape": value.shape,
                "columns": list(value.columns),
                "head": value.head(max_items).to_dict(orient="records"),
            }
        if isinstance(value, pd.Series):
            return {"len": len(value), "head": value.head(max_items).tolist()}
        if isinstance(value, np.ndarray):
            return {"shape": value.shape, "dtype": str(value.dtype), "head": value.ravel()[:max_items].tolist()}
        if isinstance(value, (list, tuple)):
            return {"len": len(value), "head": list(value[:max_items])}
        if isinstance(value, dict):
            return {"keys": list(value.keys())[:max_items], "len": len(value)}
    except Exception as exc:
        return f"<debug preview failed: {type(exc).__name__}: {exc}>"
    return value


# The Rust helper build_flux_bands converts magnitudes to flux internally.
# This zero-point is only used here for plotting model fluxes back in magnitude
# if eval_model returns positive fluxes.
DEFAULT_ZP = 23.90

# Accept common aliases and convert to the band labels expected by the library.
# The final labels are "g", "r", "i".
BAND_MAP = {
    # PS1 labels
    "ps1::g": "g",
    "ps1::r": "r",
    "ps1::i": "i",
    "ps1:g": "g",
    "ps1:r": "r",
    "ps1:i": "i",
    # ZTF labels
    "ztfg": "g",
    "ztfr": "r",
    "ztfi": "i",
    "ztf_g": "g",
    "ztf_r": "r",
    "ztf_i": "i",
    "ztf-g": "g",
    "ztf-r": "r",
    "ztf-i": "i",
    # LSST labels
    "lsst::r": "r",
    "lsst::i": "i",
    "lsst::g": "g",
    "lssti": "i",
    "lsstr": "r",
    "lsstg": "g",
    # Plain labels
    "g": "g",
    "r": "r",
    "i": "i",
    # Some survey numeric conventions; edit if your data differs.
    "1": "g",
    "2": "r",
    "3": "i",
}

PLOT_LABEL = {"g": "g", "r": "r", "i": "i"}
PLOT_MARKER = {"g": "x", "r": "x", "i": "x"}
PLOT_COLOR = {"g": "green", "r": "red", "i": "purple"}

PARAM_NAMES = {
    "MetzgerKN": [
        "log10(M_ej / M_sun)",
        "log10(v_ej / c)",
        "log10(kappa)",
        "t0",
        "ln(sigma_extra)",
    ],
}

PARAM_BOUNDS = {
    "MetzgerKN": {
        "log10(M_ej / M_sun)": (-4.0, -1.3),
        "log10(v_ej / c)": (-0.92, -0.45),
        "log10(kappa)": (-1.2, 1.2),
        "t0": (-1.4, -0.4),
        "ln(sigma_extra)": (-7.0, 0.0),
    },
}

class SkipLightCurve(Exception):
    """Raised when a light curve should be skipped without crashing the batch."""


def normalize_band(value: Any) -> str:
    key = str(value).strip().lower()
    normalized = BAND_MAP.get(key, key)
    debug_print(f"normalize_band: raw={value!r}, key={key!r}, normalized={normalized!r}")
    return normalized


def parse_time_to_mjd(values: Iterable[Any]) -> np.ndarray:
    """
    Convert either numeric times or ISO-like datetimes to MJD.

    Numeric values are assumed to already be MJD/JD-like. If all numeric values
    are > 2,400,000, they are treated as JD and converted to MJD.
    """
    series = pd.Series(list(values))
    debug_print("parse_time_to_mjd: input", debug_preview(series))

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        arr = numeric.to_numpy(dtype=float)
        debug_print("parse_time_to_mjd: all values numeric", debug_preview(arr), "median=", float(np.nanmedian(arr)))
        if np.nanmedian(arr) > 2_400_000:
            debug_print("parse_time_to_mjd: detected JD-like values; converting JD to MJD")
            arr = arr - 2_400_000.5
        debug_print("parse_time_to_mjd: output MJD", debug_preview(arr))
        return arr

    debug_print("parse_time_to_mjd: falling back to datetime parsing")

    try:
        dt = pd.to_datetime(series.astype(str).str.strip(), utc=True, errors="coerce", format="mixed")
    except TypeError:
        # For older pandas versions without format="mixed"
        dt = series.astype(str).str.strip().map(
            lambda x: pd.to_datetime(x, utc=True, errors="coerce")
        )

    if pd.isna(dt).any():
        bad = series[pd.isna(dt)].head(3).tolist()
        raise SkipLightCurve(f"Could not parse some time values, e.g. {bad}")

    
    # dt = pd.to_datetime(series, utc=True, errors="coerce")
    # if dt.isna().any():
    #     bad = series[dt.isna()].head(3).tolist()
    #     raise SkipLightCurve(f"Could not parse some time values, e.g. {bad}")
    
    # print(dt.astype("int64").to_numpy(dtype=float)/1e9)
    # unix_seconds = dt.astype("int64").to_numpy(dtype=float) / 1e9
    # mjd = unix_seconds / 86400.0 + 40587.0

    mjd = (dt - pd.Timestamp("1858-11-17T00:00:00Z")).dt.total_seconds().to_numpy() / 86400.0

    debug_print("parse_time_to_mjd: parsed datetime output MJD", debug_preview(mjd))
    return mjd


def flux_to_mag(flux: np.ndarray, zp: float = DEFAULT_ZP) -> np.ndarray:
    debug_print("flux_to_mag: input", debug_preview(flux), "zp=", zp)
    flux = np.asarray(flux, dtype=float)
    out = np.full(flux.shape, np.nan, dtype=float)
    pos = flux > 0
    out[pos] = zp - 2.5 * np.log10(flux[pos])
    debug_print("flux_to_mag: positive_count=", int(pos.sum()), "output", debug_preview(out))
    return out


def mag_to_flux(mag: np.ndarray, zp: float = DEFAULT_ZP) -> np.ndarray:
    """Convert magnitudes to physical flux using the same ZP as Rust build_flux_bands."""
    debug_print("mag_to_flux: input", debug_preview(mag), "zp=", zp)
    mag = np.asarray(mag, dtype=float)
    flux = 10.0 ** ((zp - mag) / 2.5)
    debug_print("mag_to_flux: output", debug_preview(flux))
    return flux

def mag_and_err_to_flux_and_err(mag, mag_err, zp=DEFAULT_ZP):
    mag = np.asarray(mag, dtype=float)
    mag_err = np.asarray(mag_err, dtype=float)

    flux = 10.0 ** ((zp - mag) / 2.5)
    flux_err = (np.log(10.0) / 2.5) * flux * mag_err

    return flux, flux_err


def flux_and_err_to_mag_and_err(flux, flux_err, zp=DEFAULT_ZP):
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)

    mag = np.full_like(flux, np.nan, dtype=float)
    mag_err = np.full_like(flux, np.nan, dtype=float)

    good = flux > 0

    mag[good] = zp - 2.5 * np.log10(flux[good])
    mag_err[good] = (2.5 / np.log(10.0)) * flux_err[good] / flux[good]

    return mag, mag_err


def save_figure_pickle(fig: plt.Figure, out_path: Path) -> None:
    """Save a Matplotlib figure object so it can be reopened and edited."""
    with open(out_path, "wb") as f:
        pickle.dump(fig, f)


def peak_flux_for_band(df: pd.DataFrame, band: str, zp: float = DEFAULT_ZP) -> float:
    """Return the per-band peak physical flux used for normalization by the Rust fitter."""
    debug_print("peak_flux_for_band: band=", band, "zp=", zp, "input", debug_preview(df))
    sub = df[df["band"].astype(str).str.lower() == str(band).lower()]
    if sub.empty:
        debug_print("peak_flux_for_band: no rows for band", band)
        return float("nan")
    flux = mag_to_flux(sub["mag"].to_numpy(dtype=float), zp=zp)
    flux = flux[np.isfinite(flux) & (flux > 0)]
    if len(flux) == 0:
        debug_print("peak_flux_for_band: no positive finite fluxes for band", band)
        return float("nan")
    peak = float(np.max(flux))
    debug_print("peak_flux_for_band: peak=", peak, "n_flux=", len(flux))
    return peak


def make_jsonable(obj: Any) -> Any:
    """Convert PyO3/Rust/Python objects to JSON-serializable containers."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        return obj

    if isinstance(obj, np.generic):
        return make_jsonable(obj.item())

    if isinstance(obj, np.ndarray):
        return make_jsonable(obj.tolist())

    if isinstance(obj, dict):
        return {str(k): make_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [make_jsonable(v) for v in obj]

    if hasattr(obj, "__dict__"):
        return make_jsonable(vars(obj))

    return str(obj)


def read_data(path: Path, fmt: str = "dat") -> pd.DataFrame:
    """Read whitespace files with datetime band mag mag_err columns."""

    if fmt == "dat":
        debug_print(f"read_data: reading {path}")
        df = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=["time_raw", "band_raw", "mag", "mag_err"],
            comment="#",
        )

        debug_print("read_data: raw dataframe", debug_preview(df))

        df["time"] = parse_time_to_mjd(df["time_raw"])
        df["band"] = df["band_raw"].map(normalize_band)

        out = df[["time", "band", "mag", "mag_err"]].copy()

        debug_print("read_data: output dataframe", debug_preview(out))

    elif fmt == "csv":
        debug_print(f"read_data: reading {path}")
        
        df = pd.read_csv(
            path,
            usecols=["observationStartMJD", "band", "mag_obs", "mag_err"],
            comment="#",
        )

        # print(f"\n ===================== PRINTING ===================== \n")
        # print(path)
        # print(df.columns)
        # print(f"\n ===================== STOPPING PRINTING ===================== \n")
        # raise NotImplementedError("CSV reading with auto-detection is not implemented yet; use read_csv_input for now.")

        debug_print("read_data: raw dataframe", debug_preview(df))

        df = df.rename(columns={
            "observationStartMJD": "time",
            "band": "band_raw",
            "mag_obs": "mag",
            "mag_err": "mag_err",
        })

        df["band"] = df["band_raw"].map(normalize_band)
        out = df[["time", "band", "mag", "mag_err"]].copy()

        debug_print("read_data: output dataframe", debug_preview(out))
    else:
        raise ValueError(f"Unsupported format {fmt!r}")
    
    return out


def read_csv_input(path: Path, args: argparse.Namespace) -> pd.DataFrame:
    debug_print(f"read_csv_input: reading {path}")
    df = pd.read_csv(path)
    debug_print("read_csv_input: raw dataframe", debug_preview(df))

    required_cols = [args.time_col, args.band_col, args.mag_col, args.mag_err_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required CSV column(s): {missing}. Present: {list(df.columns)}")

    out = pd.DataFrame(
        {
            "time": parse_time_to_mjd(df[args.time_col]),
            "band": df[args.band_col].map(normalize_band),
            "mag": pd.to_numeric(df[args.mag_col], errors="coerce"),
            "mag_err": pd.to_numeric(df[args.mag_err_col], errors="coerce"),
        }
    )

    debug_print("read_csv_input: normalized dataframe before id handling", debug_preview(out))

    if args.id_col:
        debug_print(f"read_csv_input: using id column {args.id_col!r}")
        if args.id_col not in df.columns:
            raise ValueError(f"ID column {args.id_col!r} not found. Present: {list(df.columns)}")
        out["object_id"] = df[args.id_col].astype(str)

    debug_print("read_csv_input: output dataframe", debug_preview(out))
    return out


def clean_photometry(df: pd.DataFrame, min_points_per_band: int) -> pd.DataFrame:

    debug_print("clean_photometry: input", debug_preview(df), "min_points_per_band=", min_points_per_band)

    df = df.copy()
    df = df.dropna(subset=["time", "band", "mag", "mag_err"])
    df = df[np.isfinite(df["time"]) & np.isfinite(df["mag"]) & np.isfinite(df["mag_err"])]
    df = df[df["mag_err"] > 0]
    df = df[df["band"].isin(["g", "r", "i"])]
    df = df.sort_values(["band", "time"]).reset_index(drop=True)

    debug_print("clean_photometry: after basic filtering", debug_preview(df))

    if df.empty:
        debug_print("clean_photometry: no valid rows after filtering")
        raise SkipLightCurve("No valid g/r/i photometry after filtering.")

    counts = df["band"].value_counts().to_dict()
    debug_print("clean_photometry: counts before min-points cut", counts)
    kept_bands = [b for b, n in counts.items() if n >= min_points_per_band]
    debug_print("clean_photometry: kept_bands", kept_bands)
    df = df[df["band"].isin(kept_bands)].copy()

    if df.empty:
        raise SkipLightCurve(
            f"No band has at least {min_points_per_band} valid points. Counts before cut: {counts}"
        )

    debug_print("clean_photometry: output", debug_preview(df))
    return df


def prepare_relative_time(
    df: pd.DataFrame,
    min_phase: float | None,
    max_phase: float | None,
) -> tuple[pd.DataFrame, float]:
    """Add days since the first valid observation and apply optional cuts."""
    debug_print(
        "prepare_relative_time: min_phase=",
        min_phase,
        "max_phase=",
        max_phase,
        "input",
        debug_preview(df),
    )
    df = df.copy()
    finite_times = df["time"].to_numpy(dtype=float)
    finite_times = finite_times[np.isfinite(finite_times)]
    if len(finite_times) == 0:
        debug_print("prepare_relative_time: no finite times")
        raise SkipLightCurve("No finite observation times.")

    time_origin = float(np.min(finite_times))
    df["phase"] = df["time"].astype(float) - time_origin

    if min_phase is not None:
        df = df[df["phase"] >= min_phase]
    if max_phase is not None:
        df = df[df["phase"] <= max_phase]

    debug_print(
        "prepare_relative_time: rows after cuts=",
        len(df),
        "time_origin=",
        time_origin,
        "phase_range=",
        (
            float(df["phase"].min()) if not df.empty else None,
            float(df["phase"].max()) if not df.empty else None,
        ),
    )
    if df.empty:
        debug_print("prepare_relative_time: no rows remain")
        raise SkipLightCurve("No points left after time cut.")

    out = df.reset_index(drop=True)
    debug_print("prepare_relative_time: output", debug_preview(out))
    return out, time_origin

def is_metzgerkn_fit(fit_dict: dict[str, Any]) -> bool:
    """Return True if this fit dictionary corresponds to the MetzgerKN model."""
    model_name = str(fit_dict.get("model"))
    return model_name.lower() == "metzgerkn"

def metzgerkn_pso_scale(
    fit_dict: dict[str, Any],
    data_eval_times: np.ndarray,
    data_flux: np.ndarray,
    data_flux_err: np.ndarray,
    param_source: str,
) -> float:
    """
    Reproduce the extra MetzgerKN normalization used inside the Rust PSO cost.

    Rust logic:
        pred_at_data = eval_model(MetzgerKN, params, data_times)
        max_pred = max(pred_at_data over non-upper-limit points)
        scale = clamp(1 / max_pred, 0.1, 10.0)

    Returns
    -------
    scale : float
        Multiplicative scale to apply to the raw normalized MetzgerKN model.
        Returns 1.0 for non-MetzgerKN models or if the scale cannot be computed.
    """

    if not is_metzgerkn_fit(fit_dict):
        return 1.0

    data_eval_times = np.asarray(data_eval_times, dtype=float)
    data_flux = np.asarray(data_flux, dtype=float)
    data_flux_err = np.asarray(data_flux_err, dtype=float)

    pred_at_data = try_eval_model_curve(
        fit_dict,
        data_eval_times,
        param_source,
    )

    if pred_at_data is None:
        return 1.0

    pred_at_data = np.asarray(pred_at_data, dtype=float)

    if pred_at_data.shape != data_eval_times.shape:
        return 1.0

    # Rust definition:
    # is_upper = flux_err > 0 && flux / flux_err < 3
    snr = np.full_like(data_flux, np.inf, dtype=float)
    good_err = data_flux_err > 0.0
    snr[good_err] = data_flux[good_err] / data_flux_err[good_err]

    is_upper = good_err & (snr < 3.0)

    valid = (
        ~is_upper
        & np.isfinite(pred_at_data)
        & np.isfinite(data_eval_times)
    )

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
) -> tuple[np.ndarray, float]:
    """
    Apply the Rust PSO-only MetzgerKN normalization to a model curve.

    For non-MetzgerKN models, this returns the input curve unchanged.
    """

    model_flux_norm = np.asarray(model_flux_norm, dtype=float)

    scale = metzgerkn_pso_scale(
        fit_dict=fit_dict,
        data_eval_times=data_eval_times,
        data_flux=data_flux,
        data_flux_err=data_flux_err,
        param_source=param_source,
    )

    return model_flux_norm * scale, scale

def fit_parametric(df: pd.DataFrame, args: argparse.Namespace) -> Any:

    debug_print("fit_parametric: input dataframe", debug_preview(df))

    # Use phase if present so the fit and the plot share the same coordinate.
    # build_flux_bands subtracts the minimum supplied time internally, so the
    # model evaluator later receives phase - min(phase), matching Rust.

    time_col = "phase" if "phase" in df.columns else "time"
    times = df[time_col].astype(float).tolist()
    mags = df["mag"].astype(float).tolist()
    mag_errs = df["mag_err"].astype(float).tolist()
    bands = df["band"].astype(str).tolist()

    debug_print("fit_parametric: time_col=", time_col, "n_points=", len(times), "bands=", sorted(set(bands)))
    debug_print("fit_parametric: times", debug_preview(times), "mags", debug_preview(mags), "mag_errs", debug_preview(mag_errs))

    debug_print("fit_parametric: calling lcf.build_flux_bands")

    flux_bands = lcf.build_flux_bands(times, mags, mag_errs, bands)

    debug_print("fit_parametric: flux_bands built", debug_preview(make_jsonable(flux_bands)))

    if args.model:

        debug_print("fit_parametric: calling lcf.fit_parametric_model", "model=", args.model, "method=", args.method, "fit_all_models=", args.fit_all_models)

        t0 = time.perf_counter()
        result = lcf.fit_parametric_model(flux_bands, args.model, fit_all_models=args.fit_all_models, method=args.method)
        runtime = time.perf_counter() - t0

        print(f"  Rust/PyO3 fitter runtime: {format_runtime(runtime)}")

        debug_print("fit_parametric: Rust/PyO3 fitter runtime seconds", runtime)

        return result
    
    if args.multiband:
        debug_print("fit_parametric: calling lcf.fit_parametric_multiband", "method=", args.method)

        t0 = time.perf_counter()
        result = lcf.fit_parametric_multiband(flux_bands, method=args.method)
        runtime = time.perf_counter() - t0

        print(f"  Rust/PyO3 fitter runtime: {format_runtime(runtime)}")

        return result

    debug_print("fit_parametric: calling lcf.fit_parametric", "method=", args.method, "fit_all_models=", args.fit_all_models)

    t0 = time.perf_counter()
    result = lcf.fit_parametric(flux_bands, fit_all_models=args.fit_all_models, method=args.method)
    runtime = time.perf_counter() - t0

    print(f"  Rust/PyO3 fitter runtime: {format_runtime(runtime)}")

    return result


def iter_fit_dicts(obj: Any) -> list[dict[str, Any]]:
    debug_print("iter_fit_dicts: input preview", debug_preview(make_jsonable(obj)))
    """
    Recursively collect dicts that look like parametric fit results.

    The confirmed Python binding returns a list of one dict per fitted band,
    with keys such as: band, model, pso_params, svi_mu, mag_chi2.
    """
    obj = make_jsonable(obj)
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
    debug_print("iter_fit_dicts: found", len(found), "fit dict(s)")
    return found


def collect_summary_numbers(obj: Any) -> dict[str, Any]:
    """Pull common scalar metrics from a nested result for concise logging."""
    debug_print("collect_summary_numbers: input preview", debug_preview(make_jsonable(obj)))
    obj = make_jsonable(obj)
    wanted = {
        "model",
        "model_name",
        "reduced_chi2",
        "chi2",
        "bic",
        "aic",
        "band",
        "uncertainty_method",
    }
    found: dict[str, Any] = {}

    def rec_walk(x: Any) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                if k in wanted and k not in found and isinstance(v, (str, int, float, bool)):
                    found[k] = v
                    # print(f"[BUILDING SUMMARY] {found}")
            for v in x.values():
                if len(found) < len(wanted):
                    rec_walk(v)
        elif isinstance(x, list):
            for v in x:
                if len(found) < len(wanted):
                    rec_walk(v)

    rec_walk(obj)

    chi2_band_dict: dict[str, float] = {}
    for band_dict in obj:
        band = band_dict.get("band", "unknown")
        mag_chi2 = band_dict.get("mag_chi2", float("nan"))
        pso_chi2 = band_dict.get("pso_chi2", float("nan"))

        chi2_band_dict[band] = {"mag_chi2": mag_chi2, "pso_chi2": pso_chi2}

    debug_print("collect_summary_numbers: found", found)
    return found, chi2_band_dict


def choose_params(fit_dict: dict[str, Any], param_source: str) -> list[float] | None:
    """Select the parameter vector requested for plotting and reporting."""
    params, _ = choose_params_with_source(fit_dict, param_source)
    return params


def choose_params_with_source(fit_dict: dict[str, Any], param_source: str) -> tuple[list[float] | None, str | None]:
    """Select the parameter vector and return the actual source key used."""
    debug_print("choose_params: requested", param_source, "available_keys=", list(fit_dict.keys()))
    if param_source in fit_dict:
        debug_print("choose_params: using requested source", param_source, debug_preview(fit_dict[param_source]))
        return fit_dict[param_source], param_source

    # Robust fallbacks for possible alternative result shapes.
    for key in ("pso_params", "svi_mu", "params", "posterior_mean", "means"):
        if key in fit_dict:
            debug_print("choose_params: using fallback source", key, debug_preview(fit_dict[key]))
            return fit_dict[key], key

    debug_print("choose_params: no usable parameter vector found")
    return None, None

def get_red_chi2(band: str, result: list[dict[str, Any]], phases, errs, param_source: str) -> float:
    """Extract the reduced chi2 for a given band from the result dicts."""

    # print(f"[GET REDUCED CHI2]  Starting...")

    for fit_dict in result:
        # print(f"[GET REDUCED CHI2]  [CHECK BAND] {fit_dict.get('band', '')}")
        if fit_dict.get("band", "") == band:

            N = fit_dict.get("n_obs", len(phases))
            pso_chi2 = fit_dict.get("pso_chi2")
            params = choose_params(fit_dict, param_source) or []
            extra_sigma = np.exp(params[-1]) if params else 0.0

            # print(f"[GET REDUCED CHI2]  [EXTRACTED VALUES] N={N}, pso_chi2={pso_chi2}, extra_sigma={extra_sigma}")
            log_var = 0
            for err in errs:
                log_var += np.log(err**2 + extra_sigma**2)

            # print(f"[GET REDUCED CHI2]  [LOG VAR] {log_var}")
            dofs = N - len(params)
            # print(f"[GET REDUCED CHI2]  [DEGREES OF FREEDOM] {dofs}")
            if dofs <= 0:
                debug_print("get_red_chi2: non-positive degrees of freedom; returning NaN")
                # print(f"[GET REDUCED CHI2]  [REDUCED CHI2] nan (dofs < 1)\n")
                return float("nan")
            else:
                red_chi2 = (N*pso_chi2 - log_var) / dofs

            # print(f"[GET REDUCED CHI2]  [REDUCED CHI2] {red_chi2}\n")
            return red_chi2
    
    # print(f"[GET REDUCED CHI2]  [REDUCED CHI2] nan (no matching band found)\n")
    return float("nan")

def try_eval_model_curve(
    fit_dict: dict[str, Any],
    phase_grid: np.ndarray,
    param_source: str,
) -> np.ndarray | None:
    """
    Evaluate one fitted model curve.

    The confirmed binding signature is:
        lcf.eval_model(model: str, params: list[float], times: list[float])

    The times must be relative days, not absolute MJD.
    """
    debug_print("try_eval_model_curve: fit_dict keys=", list(fit_dict.keys()), "param_source=", param_source)
    model = fit_dict.get("model", fit_dict.get("model_name"))
    params = choose_params(fit_dict, param_source)
    debug_print("try_eval_model_curve: model=", model, "params_preview=", debug_preview(params), "phase_grid=", debug_preview(phase_grid))

    if model is None or params is None or not hasattr(lcf, "eval_model"):
        return None

    try:
        y = np.asarray(lcf.eval_model(model, params, phase_grid.astype(float).tolist()), dtype=float)
    except Exception as exc:
        debug_print("try_eval_model_curve: eval_model failed", type(exc).__name__, str(exc))
        return None

    if y.shape == phase_grid.shape and np.isfinite(y).any():
        debug_print("try_eval_model_curve: output", debug_preview(y))
        return y
    debug_print("try_eval_model_curve: invalid output shape/finite values", debug_preview(y))
    return None


def plot_fit(df: pd.DataFrame, result: Any, out_png: Path, args: argparse.Namespace) -> dict[str, float]:
    """
    Plot fitted light curves band-by-band.

    Important conventions:
    - Data are plotted either in magnitude space or flux space.
    - The Rust fitter normalizes each band's data flux by that band's peak flux.
    - lcf.eval_model(...) returns model values in the Rust model's normalized/model space.
    - For the MetzgerKN model only, the Rust PSO cost applies an additional shape normalization:
          pred *= clip(1 / max(pred_at_data_times), 0.1, 10.0)
      so we reproduce that normalization before plotting the model.
    """

    debug_print("plot_fit: out_png=", out_png, "input df", debug_preview(df))

    summary, chi2_dict = collect_summary_numbers(result)
    fit_dicts = iter_fit_dicts(result) if args.plot_model else []

    bands_order = ["g", "r", "i"]
    flux_plot = bool(args.flux_plot)

    # ------------------------------------------------------------------
    # Normalize dataframe and validate required columns
    # ------------------------------------------------------------------
    df = df.copy()
    df["band"] = df["band"].astype(str).str.lower()

    required_cols = {"phase", "band", "mag", "mag_err"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"plot_fit requires columns {sorted(required_cols)}; missing {sorted(missing)}")

    finite_phase = df["phase"].to_numpy(dtype=float)
    finite_phase = finite_phase[np.isfinite(finite_phase)]

    if len(finite_phase) == 0:
        raise ValueError("plot_fit received no finite phase values.")

    fit_phase_min = float(np.min(finite_phase))
    plot_phase_min = fit_phase_min - float(args.plot_pre_phase)
    phase_max = float(np.max(finite_phase))
    phase_grid = np.linspace(plot_phase_min, phase_max, int(args.n_grid))
    eval_grid = phase_grid - fit_phase_min

    # ------------------------------------------------------------------
    # Small local helpers
    # ------------------------------------------------------------------
    def finite_or_nan(value: Any) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return value if np.isfinite(value) else float("nan")

    def fmt_chi2(value: Any) -> str:
        value = finite_or_nan(value)
        return f"{value:.2f}" if np.isfinite(value) else "nan"

    def band_chi2_values(band: str) -> dict[str, Any]:
        values = chi2_dict.get(band, {})
        return values if isinstance(values, dict) else {}

    # ------------------------------------------------------------------
    # Precompute per-band data
    # ------------------------------------------------------------------
    band_data: dict[str, dict[str, Any]] = {}
    reduced_chi2_by_band: dict[str, float] = {}

    for band in bands_order:
        sub = df[df["band"] == band].sort_values("phase").copy()

        if sub.empty:
            band_data[band] = {
                "sub": sub,
                "has_data": False,
            }
            reduced_chi2_by_band[band] = float("nan")
            continue

        phase = sub["phase"].to_numpy(dtype=float)
        mag = sub["mag"].to_numpy(dtype=float)
        mag_err = sub["mag_err"].to_numpy(dtype=float)

        flux, flux_err = mag_and_err_to_flux_and_err(mag, mag_err, zp=args.zp)

        good_flux = np.isfinite(flux) & (flux > 0.0)
        peak_flux = float(np.max(flux[good_flux])) if np.any(good_flux) else float("nan")
        norm_flux = flux / peak_flux if np.isfinite(peak_flux) and peak_flux > 0.0 else np.full_like(flux, np.nan, dtype=float)

        if np.isfinite(peak_flux) and peak_flux > 0.0:
            norm_flux_err = flux_err / peak_flux
        else:
            norm_flux_err = np.full_like(flux_err, np.nan, dtype=float)

        red_chi2 = get_red_chi2(
            band=band,
            result=result,
            phases=phase,
            errs=norm_flux_err,
            param_source=args.param_source,
        )

        reduced_chi2_by_band[band] = red_chi2

        band_data[band] = {
            "sub": sub,
            "has_data": True,
            "phase": phase,
            "eval_times": phase - fit_phase_min,
            "mag": mag,
            "mag_err": mag_err,
            "flux": flux,
            "flux_err": flux_err,
            "norm_flux": norm_flux,
            "norm_flux_err": norm_flux_err,
            "peak_flux": peak_flux,
            "red_chi2": red_chi2,
        }

        debug_print(f"[BAND DATA] for band {band}: \n", f"    {band_data[band]}\n\n\n")

    # ------------------------------------------------------------------
    # Create figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5),
        sharex=True,
        sharey=True,
    )

    ax_by_band = dict(zip(bands_order, axes))

    # ------------------------------------------------------------------
    # Plot observed data
    # ------------------------------------------------------------------
    for band in bands_order:
        
        fit_dict = {}
        for fit_dict_it in fit_dicts:
            if fit_dict_it.get("band", "") == band:
                fit_dict = fit_dict_it
                break
        
        params = choose_params(fit_dict, args.param_source)

        ax = ax_by_band[band]
        data = band_data[band]
        chi2_values = band_chi2_values(band)

        mag_chi2 = chi2_values.get("mag_chi2", float("nan"))
        pso_chi2 = chi2_values.get("pso_chi2", float("nan"))
        red_chi2 = reduced_chi2_by_band.get(band, float("nan"))

        param_values = []

        if fit_dict and params is not None:
            for i in range(4):
                param_values.append(params[i])
        else:
            param_values = [float("nan")] * 4

        t0_internal = finite_or_nan(param_values[3])
        t0_on_axis = t0_internal + fit_phase_min if np.isfinite(t0_internal) else float("nan")

        ax.set_title(
            (
                f"{PLOT_LABEL.get(band, band)} band\n"
                f"mag χ²={fmt_chi2(mag_chi2)}, "
                f"PSO cost={fmt_chi2(pso_chi2)}, "
                f"red. flux χ²={fmt_chi2(red_chi2)}\n"
                f"$M_{{ej}}$/$M_\\odot$={10**param_values[0]:.2e}, "
                f"$v_{{ej}}$/c={10**param_values[1]:.3f}, "
                f"$\\kappa$={10**param_values[2]:.3f} $cm^2$/g, "
                f"$t_0$={t0_on_axis:.3f} d since first obs"
            ),
            fontsize=10,
        )
        ax.grid(alpha=0.3)

        if not data["has_data"]:
            debug_print(f"plot_fit: no data for band {band}")
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

        debug_print(f"plot_fit: plotting data for band {band}", debug_preview(data["sub"]))

        if flux_plot:
            ax.errorbar(
                data["phase"],
                data["norm_flux"],
                yerr=data["norm_flux_err"],
                fmt=PLOT_MARKER.get(band, "o"),
                ms=5,
                alpha=0.5,
                color=PLOT_COLOR.get(band),
                label=f"{PLOT_LABEL.get(band, band)} data",
            )
        else:
            ax.errorbar(
                data["phase"],
                data["mag"],
                yerr=data["mag_err"],
                fmt=PLOT_MARKER.get(band, "o"),
                ms=5,
                alpha=0.5,
                color=PLOT_COLOR.get(band),
                label=f"{PLOT_LABEL.get(band, band)} data",
            )

    # ------------------------------------------------------------------
    # Plot model evaluations
    # ------------------------------------------------------------------
    model_overlay_ok = False

    if args.plot_model:
        debug_print("plot_fit: model overlay requested; fit_dict count=", len(fit_dicts))

        for fit_dict in fit_dicts:
            band = str(fit_dict.get("band", "")).lower()

            if band not in bands_order:
                debug_print("plot_fit: skipping unsupported fit band", band)
                continue

            data = band_data.get(band)
            if not data or not data["has_data"]:
                debug_print("plot_fit: skipping unobserved fit band", band)
                continue

            ax = ax_by_band[band]

            raw_flux_norm = try_eval_model_curve(
                fit_dict,
                eval_grid,
                args.param_source,
            )

            if raw_flux_norm is None:
                debug_print("plot_fit: no evaluable model curve for band", band)
                continue

            raw_flux_norm = np.asarray(raw_flux_norm, dtype=float)

            flux_norm, metzger_scale = apply_metzgerkn_pso_rescale(
                fit_dict=fit_dict,
                model_flux_norm=raw_flux_norm,
                data_eval_times=data["eval_times"],
                data_flux=data["flux"],
                data_flux_err=data["flux_err"],
                param_source=args.param_source,
            )

            debug_print(
                "plot_fit: model normalization",
                {
                    "band": band,
                    "model": fit_dict.get("model", fit_dict.get("model_name", "model")),
                    "metzgerkn_scale": metzger_scale,
                    "peak_flux": data["peak_flux"],
                },
            )

            peak_flux = data["peak_flux"]

            if not np.isfinite(peak_flux) or peak_flux <= 0.0:
                debug_print("plot_fit: invalid peak flux; skipping band", band)
                continue

            flux_physical = flux_norm * peak_flux
            model_mag = flux_to_mag(flux_physical, zp=args.zp)

            if flux_plot:
                ok = np.isfinite(phase_grid) & np.isfinite(flux_physical)
                y_model = flux_norm
            else:
                finite_mag_values = [
                    np.asarray(data["mag"], dtype=float),
                    np.asarray(model_mag, dtype=float),
                ]
                finite_mag_values = np.concatenate(
                    [values[np.isfinite(values)] for values in finite_mag_values]
                )
                mag_floor = float(np.max(finite_mag_values) + 0.5) if len(finite_mag_values) else 30.0
                y_model = np.asarray(model_mag, dtype=float).copy()
                zero_flux = np.isfinite(phase_grid) & np.isfinite(flux_physical) & (flux_physical <= 0.0)
                y_model[zero_flux] = mag_floor
                ok = np.isfinite(phase_grid) & np.isfinite(model_mag)
                ok = ok | zero_flux

            debug_print(
                "plot_fit: finite model points",
                {"band": band, "n_finite": int(np.sum(ok))},
            )

            if not np.any(ok):
                continue

            model_name = fit_dict.get("model", fit_dict.get("model_name", "model"))

            ax.plot(
                phase_grid[ok],
                y_model[ok],
                "--",
                lw=2.0,
                color=PLOT_COLOR.get(band),
                label=f"{model_name} fit",
            )

            model_overlay_ok = True

    # ------------------------------------------------------------------
    # Axis formatting
    # ------------------------------------------------------------------
    for band in bands_order:
        ax = ax_by_band[band]
        ax.set_xlabel("Days since first valid observation")
        ax.set_xlim(plot_phase_min, phase_max)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=9, loc="best")

    if flux_plot:
        axes[0].set_ylabel("Normalized flux")
    else:
        axes[0].set_ylabel("Apparent magnitude")
        axes[0].invert_yaxis()

    # ------------------------------------------------------------------
    # Global title
    # ------------------------------------------------------------------
    model_label = None
    for key in ("model", "model_name"):
        if key in summary:
            model_label = str(summary[key])
            break

    if flux_plot:
        title = "Parametric lightcurve_fitting result — flux space"
    else:
        title = "Parametric lightcurve_fitting result — magnitude space"

    if model_label:
        title += f"\nmodel={model_label}"

    if args.plot_model and not model_overlay_ok:
        title += "\n(model overlay unavailable from returned result; JSON was saved)"

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    debug_print("plot_fit: title=", title)

    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    save_figure_pickle(fig, out_png.with_suffix(".pickle"))
    plt.close(fig)

    debug_print("plot_fit: saved", out_png)

    return reduced_chi2_by_band


def fit_one(name: str, df: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> tuple[bool, str, dict[str, Any]]:
    debug_print("fit_one: start", {"name": name, "output_dir": str(output_dir), "args": vars(args)})
    print(f"\n=== {name} ===")

    event_id = infer_event_id(name)

    debug_print("fit_one: preparing outputs in", output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    phot_path = output_dir / f"{name}_used_photometry.csv"
    json_path = output_dir / f"{name}_parametric_result.json"
    png_path = output_dir / f"{name}_parametric_data.png"

    debug_print("fit_one: output paths", {"phot_path": str(phot_path), "json_path": str(json_path), "png_path": str(png_path)})

    df = clean_photometry(df, min_points_per_band=args.min_points_per_band)
    counts = df["band"].value_counts().sort_index().to_dict()
    print("\n  valid points:", ", ".join(f"{b}={n}" for b, n in counts.items()))

    df, time_origin = prepare_relative_time(
        df,
        min_phase=args.min_phase,
        max_phase=args.max_phase,
    )
    args.current_time_origin = time_origin  # used only for debug/plot metadata
    fitter_time_origin = time_origin + float(df["phase"].min())
    print(f"\n  first valid observation MJD: {time_origin:.5f}")
    print(f"  fitter time origin MJD:     {fitter_time_origin:.5f}")
    print(f"\n  fitting method={args.method}, fit_all_models={args.fit_all_models}, model={args.model}")

    debug_print("fit_one: calling fitter")
    result = fit_parametric(df, args)
    debug_print("fit_one: raw fit result preview", debug_preview(make_jsonable(result)))
    result_jsonable = make_jsonable(result)

    cleaned_result = []

    for elem in result_jsonable:
        cleaned_result_elem = {}
        if isinstance(elem, dict) and "multi_bazin" in elem.keys():
            for key in elem.keys():
                if key != "multi_bazin":
                    cleaned_result_elem[key] = elem[key]
            cleaned_result.append(cleaned_result_elem)
        else:
            cleaned_result.append(elem)

    dict_red_chi2 = plot_fit(df, cleaned_result, png_path, args)

    print("\n  Fit results:")

    fit_dicts = iter_fit_dicts(cleaned_result)

    dict_params = {"event_id": event_id}

    if not fit_dicts:
        print("    No fit dictionaries found in returned result.")
    else:
        for fit in fit_dicts:
            band = fit.get("band", "?")
            model = fit.get("model", fit.get("model_name", "?"))

            print(f"\n    Band: {band}")
            # print(f"      Model: {model}")

            # if "n_obs" in fit:
            #     print(f"      N obs: {fit['n_obs']}")

            if "mag_chi2" in fit:
                print(f"      Mag chi2: {fit['mag_chi2']:.4f}")

            if "pso_chi2" in fit:
                print(f"      PSO chi2 (negative log-likelihood): {fit['pso_chi2']:.4f}")

            print(f"      Reduced chi2 (flux): {dict_red_chi2[band]:.4f}")

            # if "uncertainty_method" in fit:
            #     print(f"      Uncertainty: {fit['uncertainty_method']}")

            param_names = PARAM_NAMES.get(model, [])

            def print_params(label: str, values: list[float]) -> None:
                print(f"      {label}:")
                for i, val in enumerate(values):
                    name = param_names[i] if i < len(param_names) else f"param_{i}"
                    note = ""
                    if model == "MetzgerKN" and i == 4:
                        note = "  # likelihood scatter, not used by physical flux evaluator"
                    print(f"        {name:28s} = {val:.6f}{note}")

            selected_params, selected_label = choose_params_with_source(fit, args.param_source)
            if selected_params is not None:
                print_params(f"{selected_label} params", selected_params)

            dict_params[band] = {}
            dict_params[band]["param_source"] = selected_label or args.param_source
            for i, val in enumerate(selected_params or []):
                name = param_names[i] if i < len(param_names) else f"param_{i}"
                dict_params[band][name] = val
            dict_params[band]["red_chi2"] = dict_red_chi2[band]


            for key in fit.keys():
                if 'svi' in key.lower() and key != selected_label:
                    print(f"      {key}: {fit[key]}")

            # svi = fit.get("svi_mu")
            # if svi is not None:
            #     print_params("SVI mean params", svi)

            # if "multi_bazin" in fit:
            #     mb = fit["multi_bazin"]
            #     print("      Multi-Bazin diagnostics:")
            #     print(f"        best_k = {mb.get('best_k')}")
            #     print(f"        cost   = {mb.get('cost')}")
            #     print(f"        bic    = {mb.get('bic')}")

    df_to_save = df.copy()
    df_to_save["event_id"] = event_id
    df_to_save.to_csv(phot_path, index=False)
    with open(json_path, "w") as f:
        json.dump(result_jsonable, f, indent=2)

    print(f"\n  saved photometry: {phot_path}")
    print(f"  saved result:     {json_path}")
    print(f"  saved plot:       {png_path}")
    print(f"  saved plot pickle: {png_path.with_suffix('.pickle')}")

    return True, str(json_path), dict_params


def discover_paths(pattern: str) -> list[Path]:
    debug_print("discover_paths: pattern=", pattern, "cwd=", os.getcwd())

    paths: list[Path] = []
    seen: set[str] = set()

    for p in glob.glob(pattern):
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            paths.append(Path(p))

    debug_print("discover_paths: found paths", [str(p) for p in paths])
    return paths


def main() -> None:
    global DEBUG_ENABLED

    t_script_start = time.perf_counter()
    
    parser = argparse.ArgumentParser(description=__doc__)

    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", default=None, help="Input CSV or DAT file.")
    source.add_argument("--pattern", default="*AT2017gfo*t0p50d*.dat", help="Glob pattern for many DAT files, e.g. 'At2017gfo*.dat'.")

    parser.add_argument(
        "--input-format",
        choices=["auto", "dat", "csv"],
        default="auto",
        help="Input format. 'dat' expects datetime band mag mag_err whitespace columns.",
    )
    parser.add_argument("--output-dir", default="lightcurve_fitting_parametric_results")
    parser.add_argument("--debug", action="store_true", help="Print detailed debug diagnostics to stderr.")

    # CSV-specific columns
    parser.add_argument("--id-col", default=None, help="Optional object/source ID column for CSV batch fitting.")
    parser.add_argument("--time-col", default="time", help="CSV time column. Numeric MJD/JD or datetime strings.")
    parser.add_argument("--band-col", default="band", help="CSV band/filter column.")
    parser.add_argument("--mag-col", default="mag", help="CSV magnitude column.")
    parser.add_argument("--mag-err-col", default="mag_err", help="CSV magnitude uncertainty column.")

    # Fitting options
    parser.add_argument("--method", choices=["laplace", "svi"], default="svi")
    parser.add_argument("--fit-all-models", action="store_true")
    parser.add_argument("--model", default=None, help="Optional forced model name, e.g. Villar, Bazin, TDE, Arnett, Magnetar, MetzgerKN.")
    parser.add_argument("--multiband", action="store_true", help="Use fit_parametric_multiband instead of fit_parametric.")
    parser.add_argument("--min-points-per-band", type=int, default=2)
    parser.add_argument(
        "--boundary-epsilon",
        type=float,
        default=1e-3,
        help="Tolerance for reporting fitted parameters that reached their optimizer bounds.",
    )

    # Time/plot options
    parser.add_argument(
        "--min-phase",
        type=float,
        default=None,
        help="Minimum days since the first valid observation to keep.",
    )
    parser.add_argument(
        "--max-phase",
        type=float,
        default=None,
        help="Maximum days since the first valid observation to keep.",
    )
    parser.add_argument(
        "--plot-pre-phase",
        type=float,
        default=10.0,
        help="Days before the first fitted data point to include in the model plot.",
    )
    parser.add_argument("--plot-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flux-plot", action="store_true", help="Plot flux instead of magnitude (data and model).")
    parser.add_argument(
        "--param-source",
        choices=["pso_params", "svi_mu"],
        default="pso_params",
        help="Parameter vector used for model overlays, printed parameters, boundary-hit checks, and summary plots.",
    )
    parser.add_argument("--n-grid", type=int, default=400)
    parser.add_argument("--zp", type=float, default=DEFAULT_ZP)

    args = parser.parse_args()
    if args.plot_pre_phase < 0:
        parser.error("--plot-pre-phase must be non-negative.")
    DEBUG_ENABLED = bool(args.debug)
    debug_print("main: parsed args", vars(args))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input is None and args.pattern is None:
        args.pattern = "At2017gfo*.dat"

    n_done = 0
    n_skipped = 0
    n_failed = 0
    result_paths: list[str] = []

    list_params: list[dict[str, Any]] = []

    if args.pattern is not None:
        debug_print("main: using pattern input mode")
        paths = discover_paths(args.pattern)
        if not paths:
            print(f"No files matched pattern {args.pattern!r} in {os.getcwd()}")
            raise SystemExit(1)

        print(f"Found {len(paths)} file(s): {[str(p) for p in paths]}")
        for path in paths:

            t_lc_start = time.perf_counter()

            fmt = args.input_format
            if fmt == "auto":
                fmt = "csv" if path.suffix.lower() == ".csv" else "dat"

            try:

                debug_print("main: processing DAT path", path)
                debug_print("main: reading DAT input", path)

                t_read_start = time.perf_counter()
                df = read_data(path, fmt)
                read_runtime = time.perf_counter() - t_read_start

                t_fit_start = time.perf_counter()
                ok, result_path, dict_params = fit_one(path.stem, df, output_dir, args)
                fit_runtime = time.perf_counter() - t_fit_start

                n_done += int(ok)
                result_paths.append(result_path)
                list_params.append(dict_params)

                total_lc_runtime = time.perf_counter() - t_lc_start

                print(
                    f"  runtime: read={format_runtime(read_runtime)}, "
                    f"fit+plot={format_runtime(fit_runtime)}, "
                    f"total={format_runtime(total_lc_runtime)}"
                )

            except SkipLightCurve as exc:
                n_skipped += 1
                total_lc_runtime = time.perf_counter() - t_lc_start
                print(f"\n=== {path.stem} ===")
                print(f"  skipping light curve: {exc}")
                print(f"  runtime before skip: {format_runtime(total_lc_runtime)}")

            except Exception as exc:
                n_failed += 1
                total_lc_runtime = time.perf_counter() - t_lc_start
                print(f"\n=== {path.stem} ===")
                print("  fit failed unexpectedly; skipping this light curve.")
                print(f"  error: {type(exc).__name__}: {exc}")
                print(f"  runtime before failure: {format_runtime(total_lc_runtime)}")

    else:
        debug_print("main: using single input mode", args.input)
        path = Path(args.input)
        fmt = args.input_format
        if fmt == "auto":
            fmt = "csv" if path.suffix.lower() == ".csv" else "dat"
        debug_print("main: resolved input format", fmt)

        try:
            if fmt == "dat":
                debug_print("main: processing DAT path", path)
                df = read_data(path, fmt)
                ok, result_path, dict_params = fit_one(path.stem, df, output_dir, args)
                n_done += int(ok)
                result_paths.append(result_path)
                list_params.append(dict_params)
            else:
                debug_print("main: reading CSV input", path)
                df_all = read_csv_input(path, args)
                if args.id_col:
                    debug_print("main: CSV grouped by object_id", df_all["object_id"].nunique())
                    for object_id, group in df_all.groupby("object_id"):
                        debug_print("main: processing object group", object_id, debug_preview(group))
                        try:
                            ok, result_path, dict_params = fit_one(str(object_id), group, output_dir, args)
                            n_done += int(ok)
                            result_paths.append(result_path)
                            list_params.append(dict_params)
                        except SkipLightCurve as exc:
                            n_skipped += 1
                            print(f"\n=== {object_id} ===")
                            print(f"  skipping light curve: {exc}")
                        except Exception as exc:
                            n_failed += 1
                            print(f"\n=== {object_id} ===")
                            print("  fit failed unexpectedly; skipping this light curve.")
                            print(f"  error: {type(exc).__name__}: {exc}")
                else:
                    ok, result_path, dict_params = fit_one(path.stem, df_all, output_dir, args)
                    n_done += int(ok)
                    result_paths.append(result_path)
                    list_params.append(dict_params)
        except SkipLightCurve as exc:
            n_skipped += 1
            print(f"\n=== {path.stem} ===")
            print(f"  skipping light curve: {exc}")
        except Exception as exc:
            n_failed += 1
            print(f"\n=== {path.stem} ===")
            print("  fit failed unexpectedly.")
            print(f"  error: {type(exc).__name__}: {exc}")

    total_runtime = time.perf_counter() - t_script_start

    print("\nBatch summary:")
    print(f"  fitted:  {n_done}")
    print(f"  skipped: {n_skipped}")
    print(f"  failed:  {n_failed}")
    print(f"  runtime: {format_runtime(total_runtime)}")

    debug_print("main: final counters", {"fitted": n_done, "skipped": n_skipped, "failed": n_failed, "result_paths": result_paths, "list_params": list_params})

    if result_paths:
        print("\nResult JSON files:")
        for p in result_paths:
            print(f"  {p}")

    boundary_hits = []
    boundary_epsilon = float(args.boundary_epsilon)
    for d in list_params:
        event_id = d.get("event_id")
        for band, band_params in d.items():
            if band == "event_id" or not isinstance(band_params, dict):
                continue
            param_source = band_params.get("param_source", args.param_source)

            for name, (lower, upper) in PARAM_BOUNDS["MetzgerKN"].items():
                if name not in band_params:
                    continue

                value = float(band_params[name])
                if not np.isfinite(value):
                    continue

                if lower <= value <= lower + boundary_epsilon:
                    boundary_hits.append(
                        {
                            "event_id": event_id,
                            "band": band,
                            "param_source": param_source,
                            "parameter": name,
                            "value": value,
                            "boundary": "lower",
                            "boundary_value": lower,
                            "distance_to_boundary": abs(value - lower),
                            "epsilon": boundary_epsilon,
                        }
                    )
                if upper - boundary_epsilon <= value <= upper:
                    boundary_hits.append(
                        {
                            "event_id": event_id,
                            "band": band,
                            "param_source": param_source,
                            "parameter": name,
                            "value": value,
                            "boundary": "upper",
                            "boundary_value": upper,
                            "distance_to_boundary": abs(value - upper),
                            "epsilon": boundary_epsilon,
                        }
                    )

    boundary_hits_path = output_dir / "metzgerkn_boundary_hits.csv"
    if boundary_hits:
        boundary_hits_df = pd.DataFrame(boundary_hits)
        boundary_hits_df = boundary_hits_df.sort_values(["event_id", "band", "parameter", "boundary"])
        boundary_hits_df.to_csv(boundary_hits_path, index=False)
        boundary_event_ids = sorted(boundary_hits_df["event_id"].dropna().unique().tolist())

        print("\nMetzgerKN boundary hits:")
        print(f"  epsilon: {boundary_epsilon:g}")
        print(f"  hits:    {len(boundary_hits_df)}")
        print(f"  events:  {len(boundary_event_ids)}")
        print(f"  saved:   {boundary_hits_path}")
    else:
        print("\nMetzgerKN boundary hits:")
        print(f"  none within epsilon={boundary_epsilon:g}")

    ## Plotting distributions of parameters and red_chi2 from all fits, if any succeeded.
    ## Uses the collected dict_params list built up during fitting.

    N_bins = 30
    known_values_df = None
    known_param_by_inferred_param = {
        "log10(M_ej / M_sun)": "mej_tot/M_sun",
        "log10(v_ej / c)": "vej/c",
        "t0": "t0",
    }

    if args.pattern and "alex" in args.pattern.lower():
        known_values_df = pd.read_csv("features_kn_cut_1000_enriched.csv", usecols=["event_id", "log10_mej_tot", "log10_vej", "t0"])
        known_values_df["mej_tot/M_sun"] = 10**known_values_df["log10_mej_tot"]
        known_values_df["vej/c"] = 10**known_values_df["log10_vej"]
        known_values_df = known_values_df.set_index("event_id")
        known_values_df = known_values_df[~known_values_df.index.duplicated(keep="first")]

    for name in PARAM_NAMES["MetzgerKN"]:
        plt.figure(figsize=(6,4))

        values = []
        known_values = []
        known_name = known_param_by_inferred_param.get(name)
        has_known_overlay = known_values_df is not None and known_name is not None
        if "log10" in name:
            xaxis_name = name.replace("log10(", "").replace(")", "")
        elif "ln" in name:
            xaxis_name = name.replace("ln(", "").replace(")", "")
        elif name == "red_chi2":
            xaxis_name = r"Reduced $\chi^2$ (flux)"
        else:
            xaxis_name = name

        for d in list_params:
            event_id = d.get("event_id")
            for band, band_params in d.items():
                if band == "event_id" or not isinstance(band_params, dict):
                    continue
                if name not in band_params:
                    continue

                fitted_value = float(band_params[name])
                if "log10" in name:
                    fitted_value = 10**fitted_value
                elif "ln" in name:
                    fitted_value = np.exp(fitted_value)
                if not np.isfinite(fitted_value):
                    continue

                if has_known_overlay:
                    if event_id not in known_values_df.index:
                        continue
                    known_value = float(known_values_df.at[event_id, known_name])
                    if not np.isfinite(known_value):
                        continue
                    known_values.append(known_value)

                values.append(fitted_value)

        hist_bins = np.histogram_bin_edges(np.concatenate([np.asarray(values, dtype=float), np.asarray(known_values, dtype=float)]), bins=N_bins) if known_values else N_bins
        counts, edges, patches = plt.hist(values, bins=hist_bins, alpha=0.6, label="Inferred")

        # for i in [0, -1]:
        #     patches[i].set_visible(False)

        if known_values:
            plt.hist(
                known_values,
                bins=edges,
                alpha=0.5,
                histtype="stepfilled",
                label="Known",
            )
            plt.legend()
        
        print("")
        print(f"Distribution of {xaxis_name}:")
        print(len(values))
        print(len(known_values))
        print("")

        plt.xlabel(xaxis_name)
        plt.ylabel("Count")
        plt.title(f"Distribution of {xaxis_name} across lightcurves")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        safe_name = safe_filename(name)
        fig = plt.gcf()
        png_path = output_dir / f"distribution_{safe_name}.png"
        plt.savefig(png_path, dpi=150)
        save_figure_pickle(fig, png_path.with_suffix(".pickle"))
        plt.close()

    plt.figure(figsize=(6,4))

    name = "red_chi2"
    values = [d[band][name] for d in list_params for band in d if (band != "event_id" and name in d[band])]

    values = [v for v in values if np.isfinite(v)]
    values = [v for v in values if ~np.isnan(v)]

    xaxis_name = r"Reduced $\chi^2$ (flux)"

    plt.hist(values, bins=N_bins, alpha=0.7)

    plt.xlabel(xaxis_name)
    plt.ylabel("Count")
    plt.title(f"Distribution of {xaxis_name} across lightcurves")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    safe_name = safe_filename(name)
    fig = plt.gcf()
    png_path = output_dir / f"distribution_{safe_name}.png"
    plt.savefig(png_path, dpi=150)
    save_figure_pickle(fig, png_path.with_suffix(".pickle"))
    plt.close()

    if known_values_df is not None:
        for inferred_name, known_name in known_param_by_inferred_param.items():
            deltas_by_band: dict[str, list[float]] = {}

            for d in list_params:
                event_id = d.get("event_id")
                if event_id not in known_values_df.index:
                    continue

                known_value = float(known_values_df.at[event_id, known_name])
                if not np.isfinite(known_value) or known_value == 0:
                    continue

                for band, band_params in d.items():
                    if band == "event_id" or not isinstance(band_params, dict):
                        continue
                    if inferred_name not in band_params:
                        continue

                    inferred_value = float(band_params[inferred_name])
                    if "log10" in inferred_name:
                        inferred_value = 10**inferred_value
                    if not np.isfinite(inferred_value):
                        continue

                    delta = abs(known_value - inferred_value) / abs(known_value)
                    if np.isfinite(delta):
                        deltas_by_band.setdefault(str(band), []).append(delta)

            if not deltas_by_band:
                continue

            plt.figure(figsize=(6,4))
            all_deltas = []
            for band, deltas in sorted(deltas_by_band.items()):
                sorted_deltas = np.sort(np.asarray(deltas, dtype=float))
                cdf = np.arange(1, len(sorted_deltas) + 1) / len(sorted_deltas)
                all_deltas.extend(sorted_deltas.tolist())
                plt.step(
                    sorted_deltas,
                    cdf,
                    where="post",
                    alpha=0.8,
                    label=PLOT_LABEL.get(band, band),
                    color=PLOT_COLOR.get(band),
                )

            sorted_all_deltas = np.sort(np.asarray(all_deltas, dtype=float))
            all_cdf = np.arange(1, len(sorted_all_deltas) + 1) / len(sorted_all_deltas)
            plt.step(
                sorted_all_deltas,
                all_cdf,
                where="post",
                color="black",
                linestyle="--",
                linewidth=2,
                label="All bands",
            )

            inferred_label = inferred_name
            if "log10" in inferred_label:
                inferred_label = inferred_label.replace("log10(", "").replace(")", "")

            plt.xlabel(r"Relative error $\Delta = |known - inferred| / |known|$")
            plt.ylabel(r"Proportion with relative error $\leq \Delta$")
            plt.title(f"Cumulative relative error for {known_name}")
            plt.legend(title="Band")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            safe_known_name = safe_filename(known_name)
            safe_inferred_name = safe_filename(inferred_label)
            fig = plt.gcf()
            png_path = output_dir / f"cdf_relative_error_{safe_known_name}_vs_{safe_inferred_name}.png"
            plt.savefig(png_path, dpi=150)
            save_figure_pickle(fig, png_path.with_suffix(".pickle"))
            plt.close()

    # Plotting real parameter values VS inferred ones across all fits, reading the real values from features_kn_cut_1000_enriched.csv

    if args.pattern and "alex" in args.pattern.lower():
        real_values_df = pd.read_csv("features_kn_cut_1000_enriched.csv", usecols=["event_id", "log10_mej_tot", "log10_vej", "t0"])
        real_values_df["mej_tot/M_sun"] = 10**real_values_df["log10_mej_tot"]
        real_values_df["vej/c"] = 10**real_values_df["log10_vej"]
        real_values_df = real_values_df.set_index("event_id")
        real_values_df = real_values_df[~real_values_df.index.duplicated(keep="first")]
    
        real_params = ["mej_tot/M_sun", "vej/c", "t0"]
        inferred_params = ["log10(M_ej / M_sun)", "log10(v_ej / c)", "t0"]
    
        for real_name, inferred_name in zip(real_params, inferred_params):
            points_by_band: dict[str, tuple[list[float], list[float]]] = {}
    
            for d in list_params:
                event_id = d.get("event_id")
                if event_id not in real_values_df.index:
                    continue
    
                real_value = float(real_values_df.at[event_id, real_name])
                if not np.isfinite(real_value):
                    continue
    
                for band, band_params in d.items():
                    if band == "event_id" or not isinstance(band_params, dict):
                        continue
                    if inferred_name not in band_params:
                        continue
    
                    inferred_value = float(band_params[inferred_name])
                    if "log10" in inferred_name:
                        inferred_value = 10**inferred_value
                    if not np.isfinite(inferred_value):
                        continue
    
                    real_values, inferred_values = points_by_band.setdefault(str(band), ([], []))
                    real_values.append(real_value)
                    inferred_values.append(inferred_value)
    
            if points_by_band:
                inferred_label = inferred_name
                if "log10" in inferred_label:
                    inferred_label = inferred_label.replace("log10(", "").replace(")", "")
    
                plt.figure(figsize=(6,6))
                for band, (real_values, inferred_values) in sorted(points_by_band.items()):
                    plt.scatter(
                        real_values,
                        inferred_values,
                        alpha=0.7,
                        label=PLOT_LABEL.get(band, band),
                        color=PLOT_COLOR.get(band),
                    )
                plt.xlabel(f"Known {real_name}")
                plt.ylabel(f"Inferred {inferred_label}")
                plt.title(f"Real vs Inferred {real_name}")
                plt.legend(title="Band")
                plt.grid(alpha=0.3)
                plt.tight_layout()
                safe_real_name = safe_filename(real_name)
                safe_inferred_name = safe_filename(inferred_label)
                plt.savefig(output_dir / f"real_vs_inferred_{safe_real_name}_vs_{safe_inferred_name}.png", dpi=150)


if __name__ == "__main__":
    main()
