import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import polars as pl
from tqdm.auto import tqdm
from multiprocessing.pool import ThreadPool


class IcecubeDataset(Dataset):
    def __init__(self, 
                 seed,
                 batch_list, 
                 bin_files_name, 
                 data_dir, 
                 inputs_name, 
                 num_batches_per_epoch,
                 num_events_per_batch=None,
                 use_threading=True,
                 device='cpu',
        ):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.batch_list = batch_list

        self.use_dynamic_loading = len(batch_list) > num_batches_per_epoch
        self.num_batches_per_epoch = min(num_batches_per_epoch, len(batch_list))
        self.num_events_per_batch = num_events_per_batch
        self.use_threading = use_threading

        with open(bin_files_name) as fp:
            self.bin_data = json.load(fp)

        self.data_dir = data_dir
        self.inputs_name, self.l3k_name = inputs_name
        self.device = device

        self.next_batch_start = 0
        ping_x, ping_y = self.fetch_data(self.get_next_batch_idx(), threaded=False)
        pong_x, pong_y = None, None

        self.pp_pointer = 0
        self.pp_x = [ping_x, pong_x]
        self.pp_y = [ping_y, pong_y]

        self.x, self.y = self.get_ping_pong_data()
        self.tensorize_and_pin()

        if self.use_dynamic_loading:
            self.fetch_data_async(self.get_next_batch_idx())

    def tensorize_and_pin(self):
        # self.x = torch.Tensor(self.x).pin_memory()
        # self.y = torch.Tensor(self.y).pin_memory()
        self.x = torch.tensor(self.x).to(self.device)
        self.y = torch.tensor(self.y).to(self.device)
        

    def on_epoch_end(self):
        if not self.use_dynamic_loading:
            return
        self.join_previous_fetch()
        self.x, self.y = self.get_ping_pong_data()
        self.tensorize_and_pin()
        self.fetch_data_async(self.get_next_batch_idx())

    def get_next_batch_idx(self):
        next_batches = self.rng.choice(self.batch_list, size=self.num_batches_per_epoch, replace=False)
        return next_batches

    def fetch_data_async(self, batch_ids):
        if self.use_threading:
            pool = ThreadPool(processes=1)
            self.async_fetch = pool.apply_async(self.fetch_data, (batch_ids,))
        else:
            self.sync_fetch_data = lambda: self.fetch_data(batch_ids, threaded=False)

    def get_ping_pong_data(self):
        """ Sets the current data based on the ping pong buffer index """
        x, y = self.pp_x[self.pp_pointer], self.pp_y[self.pp_pointer]
        self.pp_pointer = (self.pp_pointer + 1) % 2
        return x, y

    def join_previous_fetch(self):
        if self.use_threading:
            x, y = self.async_fetch.get()
        else:
            x, y = self.sync_fetch_data()
        self.pp_x[self.pp_pointer], self.pp_y[self.pp_pointer] = x, y

    def fetch_data(self, batch_ids, threaded=True):
        x, ygt = [], []
        batch_ids_maybe_tqdm = batch_ids if threaded else tqdm(batch_ids)
        for batch_id in batch_ids_maybe_tqdm:
            predictions = np.load(self.inputs_name.format(batch_id))
            bx = np.concatenate(
                    [predictions['encoder'], predictions['az_logits'], predictions['zn_logits']], 
                    axis=1
            )

            batch_meta_file = os.path.join(self.data_dir, f'train_meta/batch_{batch_id}.parquet')
            meta = pl.scan_parquet(batch_meta_file)
            ya = meta.select(['azimuth', 'zenith']).collect().to_numpy()

            pidx = meta.select(['first_pulse_index', 'last_pulse_index']).collect().to_numpy()
            v = np.where((pidx[:, 1] - pidx[:, 0] + 1) > 256)[0]

            l3k = np.load(self.l3k_name.format(batch_id))
            bx[v] = np.concatenate(
                    [l3k['encoder'], l3k['az_logits'], l3k['zn_logits']], 
                    axis=1
            )

            azp = predictions['azimuth']
            azp[v] = l3k['azimuth']
            by = np.concatenate( [ya, azp[:, None] ], axis=1)

            if isinstance(self.num_events_per_batch, int):
                bx = bx[:self.num_events_per_batch]
                by = by[:self.num_events_per_batch]

            x.append(bx)
            ygt.append(by)
            
        x = np.concatenate(x)
        ygt = np.concatenate(ygt)

        y = np.empty((len(ygt), 5), dtype=ygt.dtype)
        y[:, 0] = np.digitize(ygt[:, 0], self.bin_data['azimuth_bins_left']) - 1
        y[:, 1] = np.digitize(ygt[:, 1], self.bin_data['zenith_bins_left']) - 1
        y[:, 2:] = ygt

        return x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    
if __name__ == '__main__':
    from config import TrainingConfig

    config = TrainingConfig()
    dc = config.dataset_config
    dataset = IcecubeDataset(seed=43,
                             batch_list=list(range(401, 601)),
                             bin_files_name=dc.bin_files_name,
                             data_dir=dc.data_dir,
                             inputs_name=dc.inputs_name,
                             num_batches_per_epoch=2, 
                             num_events_per_batch=50000,
                             use_threading=True
    )
    x, y = dataset[4]
    print(x.shape, y.shape)
    dataset.on_epoch_end()
    x, y = dataset[10000]
    print(x.shape, y.shape)
    for i in range(100):
        dataset.on_epoch_end()
