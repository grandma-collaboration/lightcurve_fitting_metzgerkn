#!/usr/bin/env python3
"""
Convert a BOOM object JSON file into one .dat light-curve file per object.

Expected input JSON format:
    [
      {
        "_id": "170019696009019440",
        "prv_candidates": [
          {"jd": 2461088.59, "band": "r", "magpsf": 23.3, "sigmapsf": 0.1, ...},
          ...
        ],
        "fp_hists": [
          {"jd": 2461099.54, "band": "i", "magpsf": 22.8, "sigmapsf": 0.13, ...},
          ...
        ]
      },
      ...
    ]

Output .dat format, one row per photometry point:
    2026-02-16T02:14:39.831Z lsst::r 23.32315445 0.11702178

Usage:
    python boom_objects_to_dat.py merged_objects.json output_dat_folder

Examples:
    python boom_objects_to_dat.py merged_objects.json dat_lcs
    python boom_objects_to_dat.py merged_objects.json dat_lcs --survey-prefix lsst
    python boom_objects_to_dat.py merged_objects.json dat_lcs --filename-prefix kn_filter30_200
    python boom_objects_to_dat.py merged_objects.json dat_lcs --fields prv_candidates
    python boom_objects_to_dat.py merged_objects.json dat_lcs --no-clobber
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_HISTORY_FIELDS = ("prv_candidates", "fp_hists")


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def is_finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def normalize_id(value: Any) -> Optional[str]:
    """Convert object IDs to safe strings without losing integer precision."""
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        if value.is_integer():
            return str(int(value))
        return repr(value)
    s = str(value).strip()
    if not s or s.lower() in {"none", "null", "nan"}:
        return None
    if re.fullmatch(r"[+-]?\d+\.0", s):
        return s[:-2]
    return s


def object_id_from_record(obj: Dict[str, Any], index: int) -> str:
    """Return the BOOM object ID to use in filenames."""
    candidates = (
        obj.get("objectId"),
        obj.get("objectid"),
        obj.get("diaObjectId"),
        obj.get("_id"),
    )
    for value in candidates:
        oid = normalize_id(value)
        if oid is not None:
            return oid
    return f"missing_object_id_index_{index:06d}"


def safe_filename_component(s: str) -> str:
    """Make a string safe for filenames while preserving object IDs clearly."""
    s = str(s).strip()
    s = re.sub(r"[^A-Za-z0-9_.+-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


# -----------------------------------------------------------------------------
# Time conversion
# -----------------------------------------------------------------------------

def jd_to_datetime_utc(jd: float) -> dt.datetime:
    """
    Convert Julian Date to a UTC-like datetime.

    This treats the input JD numerically and emits an ISO timestamp. BOOM/LSST
    fields may be TAI-derived in some contexts, but for this .dat light-curve
    export the important point is a stable absolute timestamp matching the JD.
    """
    # Unix epoch 1970-01-01T00:00:00 UTC is JD 2440587.5.
    seconds = (float(jd) - 2440587.5) * 86400.0
    return dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(seconds=seconds)


def datetime_to_dat_iso(t: dt.datetime) -> str:
    """Format as e.g. 2026-02-16T02:14:39.831Z, trimming excess zeros."""
    t = t.astimezone(dt.timezone.utc)
    if t.microsecond == 0:
        return t.strftime("%Y-%m-%dT%H:%M:%SZ")
    # Milliseconds are usually enough for alert-level photometry and keep files readable.
    # Use microseconds only when needed after rounding to milliseconds would erase information.
    ms = int(round(t.microsecond / 1000.0))
    if ms == 1000:
        t = t + dt.timedelta(seconds=1)
        ms = 0
    if ms == 0:
        return t.strftime("%Y-%m-%dT%H:%M:%SZ")
    return t.strftime("%Y-%m-%dT%H:%M:%S") + f".{ms:03d}Z"


# -----------------------------------------------------------------------------
# Photometry extraction / deduplication
# -----------------------------------------------------------------------------

def photometry_point_key(point: Dict[str, Any]) -> Tuple[Any, ...]:
    """
    Stable key for deduplicating duplicated photometry points.

    Prefer explicit source IDs if present. Otherwise use the physical/light-curve
    signature with rounded floats to absorb tiny JSON representation differences.
    """
    for id_field in ("_id", "candid", "candidateId", "diaSourceId", "parentDiaSourceId"):
        sid = normalize_id(point.get(id_field))
        if sid is not None:
            return ("id", id_field, sid)

    def r(value: Any, ndigits: int = 10) -> Any:
        if is_finite_number(value):
            return round(float(value), ndigits)
        return value

    return (
        "obs",
        r(point.get("jd")),
        r(point.get("midpointMjdTai")),
        point.get("band"),
        r(point.get("ra")),
        r(point.get("dec")),
        r(point.get("magpsf")),
        r(point.get("sigmapsf")),
        r(point.get("snr")),
        r(point.get("snr_psf")),
        r(point.get("psfFlux")),
        r(point.get("psfFluxErr")),
    )


def extract_history_points(
    obj: Dict[str, Any],
    history_fields: Sequence[str],
) -> Tuple[List[Dict[str, Any]], Counter]:
    """
    Extract valid photometric points from one BOOM object record.

    Returns:
        points: list of dicts with jd, band, magpsf, sigmapsf
        stats: reason counts for skipped/deduplicated points
    """
    stats = Counter()
    seen = set()
    points: List[Dict[str, Any]] = []

    for field in history_fields:
        arr = obj.get(field)
        if arr is None:
            stats[f"missing_field:{field}"] += 1
            continue
        if not isinstance(arr, list):
            stats[f"non_list_field:{field}"] += 1
            continue

        for raw in arr:
            if not isinstance(raw, dict):
                stats[f"non_dict_point:{field}"] += 1
                continue

            jd = raw.get("jd")
            band = raw.get("band")
            mag = raw.get("magpsf")
            magerr = raw.get("sigmapsf")

            if not is_finite_number(jd):
                stats[f"missing_or_bad_jd:{field}"] += 1
                continue
            if band is None or str(band).strip() == "":
                stats[f"missing_band:{field}"] += 1
                continue
            if not is_finite_number(mag):
                stats[f"missing_or_bad_magpsf:{field}"] += 1
                continue
            if not is_finite_number(magerr):
                stats[f"missing_or_bad_sigmapsf:{field}"] += 1
                continue

            key = photometry_point_key(raw)
            if key in seen:
                stats["duplicate_photometry_points_removed"] += 1
                continue
            seen.add(key)

            points.append({
                "jd": float(jd),
                "band": str(band).strip(),
                "magpsf": float(mag),
                "sigmapsf": float(magerr),
                "source_field": field,
            })

    points.sort(key=lambda p: (p["jd"], p["band"], p["magpsf"], p["sigmapsf"]))
    return points, stats


def dat_line(point: Dict[str, Any], survey_prefix: str) -> str:
    time_s = datetime_to_dat_iso(jd_to_datetime_utc(point["jd"]))
    band_s = f"{survey_prefix}::{point['band']}"
    return f"{time_s} {band_s} {point['magpsf']:.8f} {point['sigmapsf']:.8f}"


# -----------------------------------------------------------------------------
# IO / main
# -----------------------------------------------------------------------------

def read_json_objects(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Input JSON must be a list of objects, got {type(data).__name__}.")
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Input JSON item {i} is not an object/dict.")
    return data


def write_summary(path: Path, summary: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, allow_nan=False)


def make_output_filename(prefix: str, object_id: str) -> str:
    prefix = safe_filename_component(prefix)
    oid = safe_filename_component(object_id)
    return f"{prefix}_obj{oid}.dat"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one .dat light-curve file per BOOM object from an object JSON list."
    )
    parser.add_argument(
        "input_json",
        help="Path to BOOM object JSON file, e.g. merged_objects.json.",
    )
    parser.add_argument(
        "output_dir",
        help="Folder where the per-object .dat files will be written. Created if needed.",
    )
    parser.add_argument(
        "--survey-prefix",
        default="lsst",
        help="Prefix written before ::band in the .dat file. Default: lsst, giving lsst::r etc.",
    )
    parser.add_argument(
        "--filename-prefix",
        default="boom_lc",
        help="Common filename prefix. Default: boom_lc, giving boom_lc_obj<OBJECTID>.dat.",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=list(DEFAULT_HISTORY_FIELDS),
        help=(
            "Object history fields to use. Default: prv_candidates fp_hists. "
            "Example: --fields prv_candidates"
        ),
    )
    parser.add_argument(
        "--no-clobber",
        action="store_true",
        help="Do not overwrite existing .dat files; raise an error instead.",
    )
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="Do not create .dat files for objects with zero valid photometry points.",
    )
    parser.add_argument(
        "--summary-name",
        default="dat_generation_summary.json",
        help="Name of the JSON summary written inside the output folder.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    input_path = Path(args.input_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_path.is_file():
        raise SystemExit(f"Input JSON does not exist or is not a file: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    objects = read_json_objects(input_path)

    used_filenames = set()
    global_stats = Counter()
    per_object_summary = []

    for i, obj in enumerate(objects):
        object_id = object_id_from_record(obj, i)
        points, stats = extract_history_points(obj, args.fields)
        global_stats.update(stats)

        if args.skip_empty and not points:
            global_stats["objects_skipped_empty"] += 1
            per_object_summary.append({
                "object_id": object_id,
                "output_file": None,
                "n_points_written": 0,
                "skipped_empty": True,
                "stats": dict(stats),
            })
            continue

        filename = make_output_filename(args.filename_prefix, object_id)

        # Handle pathological duplicate object IDs without changing the visible object ID part.
        if filename in used_filenames:
            stem = filename[:-4] if filename.endswith(".dat") else filename
            filename = f"{stem}_dup{i:06d}.dat"
        used_filenames.add(filename)

        out_path = output_dir / filename
        if args.no_clobber and out_path.exists():
            raise SystemExit(f"Output file already exists and --no-clobber was set: {out_path}")

        lines = [dat_line(p, args.survey_prefix) for p in points]
        out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        global_stats["objects_processed"] += 1
        global_stats["photometry_points_written"] += len(points)
        if not points:
            global_stats["empty_dat_files_written"] += 1

        per_object_summary.append({
            "object_id": object_id,
            "output_file": filename,
            "n_points_written": len(points),
            "first_jd": min((p["jd"] for p in points), default=None),
            "last_jd": max((p["jd"] for p in points), default=None),
            "bands": sorted({p["band"] for p in points}),
            "stats": dict(stats),
        })

    summary = {
        "input_json": str(input_path),
        "output_dir": str(output_dir),
        "survey_prefix": args.survey_prefix,
        "filename_prefix": args.filename_prefix,
        "history_fields": list(args.fields),
        "n_objects_in_input": len(objects),
        "n_dat_files_written": global_stats["objects_processed"],
        "n_photometry_points_written": global_stats["photometry_points_written"],
        "global_stats": dict(global_stats),
        "objects": per_object_summary,
    }
    write_summary(output_dir / args.summary_name, summary)

    print("DAT generation complete.")
    print(f"  Input objects:              {len(objects)}")
    print(f"  DAT files written:          {global_stats['objects_processed']}")
    print(f"  Photometry points written:  {global_stats['photometry_points_written']}")
    print(f"  Output folder:              {output_dir}")
    print(f"  Summary:                    {output_dir / args.summary_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
