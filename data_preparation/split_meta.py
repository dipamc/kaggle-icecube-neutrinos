import pandas as pd
from tqdm.auto import tqdm
import os

data_dir = os.getenv('ICECUBE_DATA_DIR', './data')
assert os.path.exists(data_dir)  
train_meta = pd.read_parquet(os.path.join(data_dir, 'train_meta.parquet'))
os.makedirs(os.path.join(data_dir, 'train_meta'), exist_ok=True)
for batch_id in tqdm(range(1, 661)):
    batch_meta = train_meta[train_meta['batch_id'] == batch_id]
    batch_meta.to_parquet(os.path.join(data_dir, f'train_meta/batch_{batch_id}.parquet'))
