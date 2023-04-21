import numpy as np
import pandas as pd
import json
import os

num_bins = 128

data_dir = os.getenv('ICECUBE_DATA_DIR', './data')
assert os.path.exists(data_dir)  
train_meta = pd.read_parquet(os.path.join(data_dir, 'train_meta.parquet'), engine='pyarrow')


azimuth_bins_left = train_meta.azimuth.quantile(np.linspace(0, 1-1/num_bins, num_bins)).to_list()
azimuth_bins_left[0] = 0.0 # Avoids floating point tolerance errors in left bound
zenith_bins_left = train_meta.zenith.quantile(np.linspace(0, 1-1/num_bins, num_bins)).to_list()
zenith_bins_left[0] = 0.0 # Avoids floating point tolerance errors in left bound

train_meta['azimuth_bins'] = np.digitize(train_meta.azimuth.values, azimuth_bins_left)
azimuth_bin_centers = train_meta.groupby('azimuth_bins')['azimuth'].apply(np.mean).to_list()

train_meta['zenith_bins'] = np.digitize(train_meta.zenith.values, zenith_bins_left)
zenith_bin_centers = train_meta.groupby('zenith_bins')['zenith'].apply(np.mean).to_list()

train_meta['cz'] = np.cos(train_meta['zenith'])
cz_bins_left = train_meta.cz.quantile(np.linspace(0, 1-1/num_bins, num_bins)).to_list()
cz_bins_left[0] = -1.0 # Avoids floating point tolerance errors in left bound
train_meta['cz_bins'] = np.digitize(train_meta.cz.values, cz_bins_left)
cz_bin_centers = train_meta.groupby('cz_bins')['cz'].apply(np.mean).to_list()

bin_data = {
    "azimuth_bins_left": azimuth_bins_left,
    "azimuth_bin_centers": azimuth_bin_centers,
    "zenith_bins_left": zenith_bins_left,
    "zenith_bin_centers": zenith_bin_centers,
    "cos_zenith_bins_left": cz_bins_left,
    "cos_zenith_bin_centers": cz_bin_centers,
}
with open(os.path.join(data_dir, f'/angle_bins_{num_bins}.json'), 'w') as fp:
    json.dump(bin_data, fp)