import os
import numpy as np
import numba  as nb
import polars as pl
import json
import torch

@nb.njit
def set_seed(seed):
    np.random.seed(seed)

_in_types = nb.float64[:,:], nb.int64[:,:], nb.int64, nb.int64, nb.int64, nb.int64
_out_types = nb.types.Tuple((nb.float32[:,:,:], nb.int32[:]))
@nb.jit( _out_types(*_in_types) )
def sample_and_pad(data, pulse_indexes, max_sequence_length, CHARGE_IDX, AUX_IDX, TIME_IDX):
    data[:, CHARGE_IDX] = np.log10(data[:, CHARGE_IDX]) / 3.0
    data[:, AUX_IDX] = data[:, AUX_IDX] - 0.5
    data_x = np.zeros((len(pulse_indexes), max_sequence_length, data.shape[-1]), dtype=np.float32)
    sequence_lengths = np.zeros(len(pulse_indexes), dtype=np.int32)
    for ii in range(len(pulse_indexes)):
        event_data = data[pulse_indexes[ii, 0] : pulse_indexes[ii, 1] + 1]
            
        if len(event_data) > max_sequence_length:
            naux_idx = np.where(event_data[:, AUX_IDX] == -0.5)[0]
            aux_idx = np.where(event_data[:, AUX_IDX] == 0.5)[0]
            if len(naux_idx) < max_sequence_length:
                max_length_possible = min(max_sequence_length, len(event_data))
                num_to_sample = max_length_possible - len(naux_idx)
                aux_idx_sample = np.random.choice(aux_idx, size=num_to_sample, replace=False)
                selected_idx = np.concatenate((naux_idx, aux_idx_sample))
            else:
                selected_idx = np.random.choice(naux_idx, size=max_sequence_length, replace=False)
            selected_idx = np.sort(selected_idx)
            event_data = event_data[selected_idx]
        event_data[:, TIME_IDX] = ( event_data[:, TIME_IDX] - event_data[:, TIME_IDX].min() ) / 3e4

        assert np.all(np.isfinite(event_data))
        data_x[ii, :len(event_data), :] = event_data
        sequence_lengths[ii] = len(event_data)                       
    return data_x, sequence_lengths

class IcecubePreprocessor:
    def __init__(self, 
                 data_dir,
                 geometry_file_name,
                 bin_files_name,
                 feature_names,
                 max_sequence_length=64, 
                 seed=42,
                 filter_config=None):
        self.data_dir = data_dir
        self.feature_names = feature_names
        self.info_names = ['time', 'charge', 'auxiliary', 'sensor_id'] # NOTE: Sensor idx must be last of info names
        self.CHARGE_IDX = self.info_names.index('charge')
        self.AUX_IDX = self.info_names.index('auxiliary')
        self.TIME_IDX = self.info_names.index('time')
        self.SENSOR_IDX = self.info_names.index('sensor_id')

        self.max_sequence_length = max_sequence_length
        self.seed = seed
        self.filter_config = filter_config

        features = pl.read_csv(geometry_file_name).select(self.feature_names).to_numpy()
        self.features = torch.zeros((features.shape[0]+1, features.shape[1]), dtype=torch.float32)
        self.features[1:] = torch.tensor(features)

 
        with open(bin_files_name) as fp:
            self.bin_data = json.load(fp)

    def get_batch_file_names(self, batch_id):
        batch_file_name = os.path.join(self.data_dir, f'train/batch_{batch_id}.parquet')
        batch_meta_file_name = os.path.join(self.data_dir, f'train_meta/batch_{batch_id}.parquet')
        return batch_file_name, batch_meta_file_name

    def prep_batch(self, batch_id):
        batch_file_name, batch_meta_file_name = self.get_batch_file_names(batch_id)
        batch = pl.scan_parquet(batch_file_name)
        data = batch.select(self.info_names).collect().to_numpy()
        data[:, self.SENSOR_IDX] += 1 # Sensor ID 0 goes for 0 padding

        batch_meta = pl.scan_parquet(batch_meta_file_name)

        max_num_events = self.filter_config['max_events_per_batch']

        pulse_indexes = (batch_meta.select(['first_pulse_index', 'last_pulse_index'])
                         .collect().to_numpy()[:max_num_events])
        y = batch_meta.select(['azimuth', 'zenith']).collect().to_numpy()[:max_num_events]

        min_seq_length = self.filter_config['min_sequence_lentgh']
        event_lengths = pulse_indexes[:, 1] - pulse_indexes[:, 0] + 1
        event_to_keep = np.where(event_lengths >= min_seq_length)[0]
        pulse_indexes = pulse_indexes[event_to_keep]
        y = y[event_to_keep]

        np.random.seed(self.seed + batch_id + int( os.getenv('EPOCH', 0) ) )
        set_seed(self.seed) # Needed when using numba
        x, l = sample_and_pad(data, pulse_indexes, self.max_sequence_length,
                              self.CHARGE_IDX, self.AUX_IDX, self.TIME_IDX)

        y = self.prep_gt(y)

        return x, y, l

    def prep_gt(self, y):
        y_combined = np.empty((len(y), 4), dtype=y.dtype)
        y_combined[:, 0] = np.digitize(y[:, 0], self.bin_data['azimuth_bins_left']) - 1
        y_combined[:, 1] = np.digitize(y[:, 1], self.bin_data['zenith_bins_left']) - 1
        y_combined[:, 2:] = y
        return y_combined

    def join_features(self, x_mb):
        """
        Join features to minibatches - Saves RAM by only joining features prior to minibatch consumption
        """
        self.features = self.features.to(x_mb.device)
        B, T, C_old = x_mb.shape
        C = C_old - 1 + len(self.feature_names) # Remove sensor id add features
        x_feat = torch.empty((B, T, C), dtype=x_mb.dtype).to(x_mb.device)
        x_feat[..., :(C_old-1)] = x_mb[..., :self.SENSOR_IDX] # NOTE: Sensor idx must be last of info names
        x_feat[..., (C_old-1):] = self.features[x_mb[..., self.SENSOR_IDX].type(torch.long)]
        return x_feat

if __name__ == '__main__':
    from tqdm.auto import tqdm
    from config import DatasetConfig

    dc = DatasetConfig(debug_mode=True)

    preprocessor = IcecubePreprocessor(dc.data_dir, 
                                       dc.geometry_file_name, 
                                       dc.bins_file_name, 
                                       dc.feature_names, 
                                       dc.max_sequence_length, 
                                       dc.seed,
                                       dc.train_preprocessing_config.filter_config,
                                       )

    # for batch_id in tqdm(range(401, 661)):
    for batch_id in [651]:
        x, y, l = preprocessor.prep_batch(batch_id)
        print(x.shape, y.shape, l.shape, x.mean(), y[0, :4], l[:3])

