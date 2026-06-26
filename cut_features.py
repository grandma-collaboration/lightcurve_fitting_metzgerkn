'''
Small script to clean up the features CSV from Alex's fiesta/Bulla simulations and add some derived features for
lightcurve fitting using the fit_parametric_lightcurve.py script.
'''

import pandas as pd
import numpy as np

# Placeholder: replace this with your CSV file name
input_file = "features_kn_cut_1000.csv"

# Output file name
output_file = "features_kn_cut_1000_enriched.csv"

# Read only the first X rows of the CSV
df = pd.read_csv(input_file,
                 usecols = ["event_id", "summary_log10_mej_dyn", "summary_log10_mej_wind", "summary_v_ej_dyn", "summary_v_ej_wind", "summary_Ye_dyn", "summary_Ye_wind", "summary_t0_mjd", "first_det_t"]
                 )

df["log10_mej_tot"] = df["summary_log10_mej_dyn"]   # + 10**df["summary_log10_mej_wind"]) => assumes that the ejecta mass
                                                    # in metzger model is only the dynamical ejecta

df["log10_vej"] = np.log10(df["summary_v_ej_dyn"])  # + 10**df["summary_log10_mej_wind"]) => assumes that the ejecta velocity
                                                    # in metzger model is only the dynamical ejecta velocity

df["t0"] = - df["first_det_t"]                   # => assumes t0 is close enough to the merger time in Bulla model

# Save the cut CSV
df.to_csv(output_file, index=False)

print(f"Saved output file to {output_file}")