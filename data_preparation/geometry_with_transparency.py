# Credits to https://www.kaggle.com/code/anjum48/early-sharing-prize-dynedge-1-046

import numpy as np
import pandas as pd
import os

from scipy.interpolate import interp1d
from sklearn.preprocessing import RobustScaler
from scipy.stats import moment

DATA_DIR = os.getenv('ICECUBE_DATA_DIR', './data')
assert os.path.exists(DATA_DIR)  

def prepare_sensors(sensors):
    sensors["string"] = 0
    sensors["qe"] = 1

    for i in range(len(sensors) // 60):
        start, end = i * 60, (i * 60) + 60
        sensors.loc[start:end, "string"] = i

        # High Quantum Efficiency in the lower 50 DOMs - https://arxiv.org/pdf/2209.03042.pdf (Figure 1)
        if i in range(78, 86):
            start_veto, end_veto = i * 60, (i * 60) + 10
            start_core, end_core = end_veto + 1, (i * 60) + 60
            sensors.loc[start_core:end_core, "qe"] = 1.35

    # https://github.com/graphnet-team/graphnet/blob/b2bad25528652587ab0cdb7cf2335ee254cfa2db/src/graphnet/models/detector/icecube.py#L33-L41
    # Assume that "rde" (relative dom efficiency) is equivalent to QE
    sensors["x"] /= 500
    sensors["y"] /= 500
    sensors["z"] /= 500
    sensors["qe"] -= 1.25
    sensors["qe"] /= 0.25

    return sensors

def ice_transparency(DATA_DIR):
    # Data from page 31 of https://arxiv.org/pdf/1301.5361.pdf
    # Datum is from footnote 8 of page 29
    df = pd.read_csv('ice_transparency.txt', delim_whitespace=True)
    datum=1950
    df["z"] = df["depth"] - datum
    df["z_norm"] = df["z"] / 500
    df[["scattering_len_norm", "absorption_len_norm"]] = RobustScaler().fit_transform(
        df[["scattering_len", "absorption_len"]]
    )

    # These are both roughly equivalent after scaling
    f_scattering = interp1d(df["z_norm"], df["scattering_len_norm"])
    f_absorption = interp1d(df["z_norm"], df["absorption_len_norm"])
    return f_scattering, f_absorption

f_scattering, f_absorption = ice_transparency(DATA_DIR)

geometry = pd.read_csv(os.path.join(DATA_DIR, "sensor_geometry.csv"))
geometry = prepare_sensors(geometry)
geometry['scatter'] = f_scattering(geometry['z'])
geometry['absorp'] = f_absorption(geometry['z'])

zmin, zmax = geometry.z.min(), geometry.z.max()
fscat_zmin, fscat_zmax = f_scattering(zmin), f_scattering(zmax)

def f_scattering_with_padding(z_arr):
    bottom = z_arr < zmin
    top = z_arr > zmax

    bottom_idx = np.where(bottom)
    top_idx = np.where(top)
    middle_idx = np.where(~(bottom | top))

    vals = np.empty_like(z_arr)
    vals[bottom_idx] = fscat_zmin
    vals[top_idx] = fscat_zmax
    vals[middle_idx] = f_scattering(z_arr[middle_idx])
    return vals

offset_vals = [-0.05, -0.1, -0.2, -0.5, 0.05, 0.1, 0.2, 0.5]
moment_vals = [2, 3, 4]
col_names = [f'z{z}_m{m}_sc' for z in offset_vals for m in moment_vals]
def scattering_moments(z):
    feats = []
    for offset in offset_vals:
        lspace = np.linspace(z + offset, z, 200)
        sct_dist = f_scattering_with_padding(lspace)
        feats.extend([moment(sct_dist, n) for n in moment_vals])
    return feats

scatter_feats = geometry['z'].apply(scattering_moments)
scatter_feats_arr = np.array([np.array(v) for v in scatter_feats.values])
geometry[col_names] = scatter_feats_arr

geometry.to_csv(os.path.join(DATA_DIR, 'sensor_geometry_transparency_moments.csv'), index=False)
