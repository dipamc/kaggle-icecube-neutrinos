import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from multiprocessing.pool import ThreadPool
    
class PrefetchDataset(Dataset):
    def __init__(self, 
                 data_processor,
                 batch_list, 
                 num_batches_per_epoch,
                 minibatch_size,
                 use_threading=True):
        super().__init__()
        self.data_processor = data_processor
        self.batch_list = batch_list
        self.use_dynamic_loading = len(batch_list) > num_batches_per_epoch
        self.num_batches_per_epoch = min(num_batches_per_epoch, len(batch_list))
        self.use_threading = use_threading
        self.seed = data_processor.seed
        self.minibatch_size = minibatch_size

        self.next_batch_start = 0
        (ping_x, ping_l), ping_y = self.fetch_data(self.get_next_batch_idx(), use_tqdm=True)
        pong_x, pong_l, pong_y = None, None, None

        self.pp_pointer = 0
        self.pp_x = [ping_x, pong_x]
        self.pp_l = [ping_l, pong_l]
        self.pp_y = [ping_y, pong_y]
        
        (self.x, self.l), self.y = self.get_ping_pong_data()
        self.tensorize_and_pin()
        self.prepare_packing_indexes()
        
        if self.use_dynamic_loading:
            self.fetch_data_async(self.get_next_batch_idx())
    
    def tensorize_and_pin(self):
        self.x = torch.Tensor(self.x).pin_memory()
        self.l = torch.Tensor(self.l).pin_memory()
        self.y = torch.Tensor(self.y).pin_memory()

    def prepare_packing_indexes(self):
        sort_idx = np.argsort(self.l)
        self.n_minibatch = np.ceil(len(self.l) / self.minibatch_size)
        nondiv_idx = len(sort_idx) % self.minibatch_size
        if nondiv_idx == 0:
            self.idx_splits = np.split(sort_idx, self.n_minibatch)
        else:
            self.idx_splits = np.split(sort_idx[:-nondiv_idx], self.n_minibatch - 1)
                              
        np.random.seed(self.seed)
        np.random.shuffle(self.idx_splits)
        if not nondiv_idx == 0: # For the last smallest minibatch
            self.idx_splits.append( sort_idx[-nondiv_idx:] )

    def on_epoch_end(self):
        if not self.use_dynamic_loading:
            return
        self.join_previous_fetch()
        (self.x, self.l), self.y = self.get_ping_pong_data()
        self.tensorize_and_pin()
        self.prepare_packing_indexes()
        self.fetch_data_async(self.get_next_batch_idx())
        
    def get_ping_pong_data(self):
        """ Sets the current data based on the ping pong buffer index """
        x, y = self.pp_x[self.pp_pointer], self.pp_y[self.pp_pointer]
        l = self.pp_l[self.pp_pointer]
        self.pp_pointer = (self.pp_pointer + 1) % 2
        return (x, l), y
    
    def get_next_batch_idx(self):
        batch_idx = range(self.next_batch_start, self.next_batch_start + self.num_batches_per_epoch)
        next_batches = np.take(self.batch_list,  batch_idx, mode='wrap')
        self.next_batch_start = (self.next_batch_start + self.num_batches_per_epoch) % len(self.batch_list)
        return next_batches

    def fetch_data_async(self, batch_ids):
        if self.use_threading:
            pool = ThreadPool(processes=1)
            self.async_fetch = pool.apply_async(self.fetch_data, (batch_ids,))
        else:
            self.sync_fetch_data = lambda: self.fetch_data(batch_ids)

    def join_previous_fetch(self):
        del self.x, self.y, self.l
        if self.use_threading:
            (x, l), y = self.async_fetch.get()
        else:
            (x, l), y = self.sync_fetch_data()
        self.pp_x[self.pp_pointer], self.pp_y[self.pp_pointer] = x, y
        self.pp_l[self.pp_pointer] = l

    def fetch_data(self, batch_ids, use_tqdm=False):
        x, y, l = [], [], []
        batch_ids_maybe_tqdm = batch_ids if not use_tqdm else tqdm(batch_ids)
        for batch_id in batch_ids_maybe_tqdm:
            bx, by, bl = self.data_processor.prep_batch(batch_id)
            x.append(bx)
            y.append(by)
            l.append(bl)

        x = np.concatenate(x, axis=0)
        y = np.concatenate(y, axis=0)
        l = np.concatenate(l, axis=0)

        return (x, l), y

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        si, bi = index // self.minibatch_size, index % self.minibatch_size
        pck_idx = self.idx_splits[si][bi]
        return (self.x[pck_idx], self.l[pck_idx]), (self.y[pck_idx], self.l[pck_idx])

    
if __name__ == '__main__':
    from preprocessing import IcecubePreprocessor
    from config import TrainingConfig

    config = TrainingConfig()
    dc = config.dataset_config
    data_processor = IcecubePreprocessor(dc.data_dir, 
                                         dc.geometry_file_name, 
                                         dc.bins_file_name, 
                                         dc.feature_names, 
                                         dc.max_sequence_length, 
                                         dc.seed,
                                         dc.train_preprocessing_config.filter_config)

    dataset = PrefetchDataset(data_processor, batch_list=[651], num_batches_per_epoch=1, minibatch_size=config.batch_size)
    for _ in range(5):
        for index in range(15):
            val = dataset[index]
            print(f"{index=} {val[1]=}, {val[0][0].shape=}")        
        dataset.on_epoch_end()
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    nl = 0
    for x, y in dataloader:
        inp, l = x
        print("Mean & max", l.mean(), l.max(), "Packing %", l.mean()/l.max())
        nl += 1
        if nl == 10:
            break
