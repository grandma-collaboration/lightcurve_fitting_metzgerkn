"""
Clean Alex's FIESTA/Bulla simulation feature CSVs and add derived columns for
the MetzgerKN fitting analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "event_id",
    "summary_log10_mej_dyn",
    "summary_log10_mej_wind",
    "summary_v_ej_dyn",
    "summary_v_ej_wind",
    "summary_Ye_dyn",
    "summary_Ye_wind",
    "summary_t0_mjd",
    "first_det_t",
]


def enrich_features(input_file: str | Path, output_file: str | Path) -> pd.DataFrame:
    """Read the raw feature CSV, add MetzgerKN helper columns, and save it."""
    input_file = Path(input_file)
    output_file = Path(output_file)

    df = pd.read_csv(input_file, usecols=FEATURE_COLUMNS)

    # MetzgerKN currently uses only the dynamical ejecta mass/velocity.
    df["log10_mej_tot"] = df["summary_log10_mej_dyn"]
    df["log10_vej"] = np.log10(df["summary_v_ej_dyn"])

    # Approximate the merger time relative to first detection.
    df["t0"] = -df["first_det_t"]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="features_kn_cut_1000.csv", help="Raw feature CSV to enrich.")
    parser.add_argument("--output", default="features_kn_cut_1000_enriched.csv", help="Output enriched CSV.")
    args = parser.parse_args()

    enrich_features(args.input, args.output)
    print(f"Saved output file to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
