import numpy as np
import pandas as pd
import polars as pl
import os
import time
from tqdm.auto import tqdm
import numba as nb

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import math

import json

DATA_DIR = os.getenv('ICECUBE_DATA_DIR', './data')
assert os.path.exists(DATA_DIR)  
PREP_DIR = DATA_DIR
TRAIN_META_DIR = os.path.join(DATA_DIR, "train_meta/")

MODEL_DIR = "models/"

def angular_dist_score(az_true, zen_true, az_pred, zen_pred):
    """ https://www.kaggle.com/code/sohier/mean-angular-error """
    if not (np.all(np.isfinite(az_true)) and
            np.all(np.isfinite(zen_true)) and
            np.all(np.isfinite(az_pred)) and
            np.all(np.isfinite(zen_pred))):
        raise ValueError("All arguments must be finite")
    
    # pre-compute all sine and cosine values
    sa1 = np.sin(az_true)
    ca1 = np.cos(az_true)
    sz1 = np.sin(zen_true)
    cz1 = np.cos(zen_true)
    
    sa2 = np.sin(az_pred)
    ca2 = np.cos(az_pred)
    sz2 = np.sin(zen_pred)
    cz2 = np.cos(zen_pred)
    
    # scalar product of the two cartesian vectors (x = sz*ca, y = sz*sa, z = cz)
    scalar_prod = sz1*sz2*(ca1*ca2 + sa1*sa2) + (cz1*cz2)
    
    # scalar product of two unit vectors is always between -1 and 1, this is against nummerical instability
    # that might otherwise occure from the finite precision of the sine and cosine functions
    scalar_prod =  np.clip(scalar_prod, -1, 1)
    
    # convert back to an angle (in radian)
    return np.average(np.abs(np.arccos(scalar_prod)))


####################################################################################################

VALIDATE = True # For checking on validation data

if not VALIDATE:
    PARQUETS_DIR = os.path.join(DATA_DIR + 'test')
    BATCH_LIST = list(sorted(os.listdir(PARQUETS_DIR)))
    metadata = pl.read_parquet(f'{DATA_DIR}/test_meta.parquet')
    CHECK_PREDICTION = False
else:
    PARQUETS_DIR = os.path.join(DATA_DIR + 'train')
    vbatches = list(range(651, 661))
    # vbatches = [652, 653, 654, 657, 658, 659, 660]
    # vbatches = [655]
    BATCH_LIST = [f'batch_{vb}.parquet' for vb in vbatches]
    # META_FILES = [f'train_meta_{vb}.parquet' for vb in vbatches]
    META_FILES = [f'batch_{vb}.parquet' for vb in vbatches]
    def read_metadata():
        meta = []
        for mf, vb in zip(META_FILES, vbatches):
            bmeta = pl.read_parquet(f'{TRAIN_META_DIR}/{mf}')
            # Polars isn't going to get much adoption with stupid syntax like this :facepalm:
            bmeta = bmeta.with_columns(pl.lit(vb).alias('batch_id'))
            meta.append(bmeta)
        return pl.concat(meta)
    metadata = read_metadata()
    CHECK_PREDICTION = True
    
GEOMETRY = os.path.join(PREP_DIR, "sensor_geometry_with_transparency.csv")
geometry = pl.scan_csv(GEOMETRY).with_columns(
                [pl.col('sensor_id').cast(pl.Int16)]
            )
    
NUM_BINS = 128
FEATURE_NAMES = ['time', 'charge', 'auxiliary', 'x', 'y', 'z', 'qe', 'scatter', 'absorp']
CHARGE_IDX = FEATURE_NAMES.index('charge')
TIME_IDX = FEATURE_NAMES.index('time')
AUX_IDX = FEATURE_NAMES.index('auxiliary')
N_FEATURES = len(FEATURE_NAMES)

MIN_SEQUENCE_LENGTH = 0
MAX_SEQUENCE_LENGTH = 256
BATCH_SIZE = 500
FINETUNE_LONG_SEQ = False
if FINETUNE_LONG_SEQ:
    MIN_SEQUENCE_LENGTH = 256 # For long sequence
    MAX_SEQUENCE_LENGTH = 3072
    BATCH_SIZE = 15
PACKED = True

MAX_EVENTS = 200_000 # For testing on subset of data

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open(os.path.join(PREP_DIR, f'angle_bins_{NUM_BINS}.json')) as fp:
    bin_data = json.load(fp)
    
azimuth_bin_centers = torch.tensor(bin_data['azimuth_bin_centers']).type(torch.float32).to(device)

zenith_centers = np.array(bin_data['zenith_bin_centers'])
kernel_length = 15
num_zenith_padding = (kernel_length - 1)//2
padded_zenith_bins = np.concatenate([
        -zenith_centers[num_zenith_padding-1 : : -1],
        zenith_centers,
        2 * np.pi - zenith_centers[-1 : -num_zenith_padding-1 : -1],
])
zenith_bin_centers = torch.tensor(padded_zenith_bins).type(torch.float32).to(device)
ZENITH_NUM_BINS = len(zenith_bin_centers)


class Model_Config:
    name = 'base_model'
    checkpoint_path = 'models/base_model.ckpt'
    n_embd = [512]*18
    n_heads = [8]*18
    bias = False
    dropout = 0.0
    neck_dropout = 0.0
    neck_features = 3072
    unwanted_prefix = 'model'


####################################################################################################

# @nb.njit
def set_seed(value):
    np.random.seed(value)

# @nb.jit( nb.types.Tuple( (nb.float32[:,:,:], nb.int32[:]) )(nb.float64[:,:], nb.int64[:,:]) )
def sample_and_pad(data, pulse_indexes):
    data[:, CHARGE_IDX] = np.log10(data[:, CHARGE_IDX]) / 3.0
    data[:, AUX_IDX] = data[:, AUX_IDX] - 0.5
    data_x = np.zeros((len(pulse_indexes), MAX_SEQUENCE_LENGTH, data.shape[-1]), dtype=np.float32)
    sequence_lengths = np.zeros(len(pulse_indexes), dtype=np.int32)
    for ii in range(len(pulse_indexes)):
        event_data = data[pulse_indexes[ii, 0] : pulse_indexes[ii, 1] + 1]
        if len(event_data) > MAX_SEQUENCE_LENGTH:
            naux_idx = np.where(event_data[:, AUX_IDX] == -0.5)[0]
            aux_idx = np.where(event_data[:, AUX_IDX] == 0.5)[0]
            if len(naux_idx) < MAX_SEQUENCE_LENGTH:
                max_length_possible = min(MAX_SEQUENCE_LENGTH, len(event_data))
                num_to_sample = max_length_possible - len(naux_idx)
                aux_idx_sample = np.random.choice(aux_idx, size=num_to_sample, replace=False)
                selected_idx = np.concatenate((naux_idx, aux_idx_sample))
            else:
                selected_idx = np.random.choice(naux_idx, size=MAX_SEQUENCE_LENGTH, replace=False)
            selected_idx = np.sort(selected_idx)
            event_data = event_data[selected_idx]
        event_data[:, TIME_IDX] = ( event_data[:, TIME_IDX] - event_data[:, TIME_IDX].min() ) / 3e4
        assert np.all(np.isfinite(event_data))
        data_x[ii, :len(event_data), :] = event_data
        sequence_lengths[ii] = len(event_data)                       
    return data_x, sequence_lengths

####################################################################################################

def preprocess_data(bfile):
    set_seed(42)
    batch_id = int(bfile.split('.')[0].split('_')[-1])
    batch = pl.scan_parquet(f'{PARQUETS_DIR}/{bfile}')
    batch = batch.join(geometry, on='sensor_id', how='left')
    batch_meta = metadata.filter(pl.col('batch_id') == batch_id)

    data = batch.select(FEATURE_NAMES).collect().to_numpy()
    pulse_indexes = batch_meta.select(['first_pulse_index', 'last_pulse_index']).to_numpy()

    event_length = pulse_indexes[:, 1] - pulse_indexes[:, 0] + 1
    events_to_keep = (event_length > MIN_SEQUENCE_LENGTH)
    events_to_keep[MAX_EVENTS:] = False
    pulse_indexes = pulse_indexes[events_to_keep]

    data_x, seq_lens = sample_and_pad(data, pulse_indexes)

    return data_x, seq_lens, events_to_keep

####################################################################################################

if VALIDATE:
    def check_packing_fraction():
        dx, dl = preprocess_data(BATCH_LIST[0])
        print(dx.shape)
        # Can get quite low for higher sequence lengths - Around 30% for 256 sequence length
        print("packing fraction without sorting - ", np.mean(dl/dl.max()))
        lensplit = np.split(dl[np.argsort(dl)], len(dl)//BATCH_SIZE)
        pfrac = np.mean([np.mean(ls/max(ls)) for ls in lensplit])
        print(f"packing fraction with sortting - batch size {BATCH_SIZE} - ", pfrac)
    
    # check_packing_fraction()

class IceCubeDataset(Dataset):
    def __init__(self, bfile):
        super().__init__()
        dx, sl, ek = preprocess_data(bfile)
        self.x = torch.Tensor(dx)
        self.l = torch.Tensor(sl)
        self.events_to_keep = ek
        
        self.sort_idx = np.argsort(sl) if PACKED else np.arange(len(sl))
        self.reverse_sort_idx = np.argsort(self.sort_idx)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        si = self.sort_idx[index] # Lazy packing, works well for batch size ~1000
        return self.x[si], self.l[si]

####################################################################################################

# Most of the code is taken from
# https://github.com/karpathy/nanoGPT/blob/master/model.py

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

def mlp(n_embd, bias=False, dropout=0.0, out_embd=None):
    out_embd = n_embd if out_embd is None else out_embd
    return nn.Sequential(
        nn.Linear(n_embd, 4 * n_embd, bias=bias),
        nn.GELU(approximate='tanh'),
        nn.Linear(4 * n_embd, out_embd, bias=bias),
        nn.Dropout(dropout)
    )

class SelfAttention(nn.Module):
    def __init__(self, prev_emdb, n_embd, n_heads, bias=False, dropout=0.0):
        super().__init__()
        self.prev_embd = prev_emdb
        self.n_embd = n_embd
        self.n_heads = n_heads
        
        self.c_attn = nn.Linear(prev_emdb, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask, cross_features=None):
        B, T, _ = x.shape
        C = self.n_embd
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        return y

class AttentionBlock(nn.Module):
    def __init__(self, prev_embd, n_embd, n_heads, bias=False, dropout=0.0):
        super().__init__()
        self.ln_1 = LayerNorm(prev_embd, bias)
        assert n_embd % prev_embd == 0, f"{prev_embd=} {n_embd=} should be divisble"
        self.attn = SelfAttention(prev_embd, n_embd, n_heads, bias, dropout)
        self.ln_2 = LayerNorm(n_embd, bias)
        self.mlp = mlp(n_embd, bias, dropout)

    def forward(self, x, attn_mask, cross_features=None):
        x = x + self.attn(self.ln_1(x), attn_mask, cross_features)
        x = x + self.mlp(self.ln_2(x))
        return x

class AttentionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        dropout = config.dropout
        bias = config.bias
        
        attn_layers = []
        prev_embd = config.n_embd[0]
        for n_embd, n_heads in zip(config.n_embd, config.n_heads):
            attn_layers.append( AttentionBlock(prev_embd, n_embd, n_heads, bias, dropout) )
            prev_embd = n_embd
        self.attn = nn.ModuleList(attn_layers)
    
    def forward(self, x, attn_mask):
        out = x
        for attn_layer in self.attn:
            out = attn_layer(out, attn_mask)
        
        return out
    
class SequencePool(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_pools = 1
    
    def forward(self, x, sequence_lengths, padding_mask):
        sumf = torch.sum(x * padding_mask.unsqueeze(2), dim=1) # Mask padded tokens
        meanf = sumf / sequence_lengths.view(-1, 1) # Normalize avg pool values by seq length
        return meanf
    
class Neck(nn.Module):
    def __init__(self, in_features, out_features, bias, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            LayerNorm(in_features, bias=bias),
            nn.Linear(in_features, 4 * in_features, bias=bias),
            nn.GELU(approximate='tanh'),
            nn.Linear(4 * in_features, out_features, bias=bias),
            nn.Dropout(dropout)
        )
        self.n_repeats = out_features // in_features

    def forward(self, x):
        return x.repeat(1, self.n_repeats) + self.mlp(x)

    
class MultiLabelClassifier(nn.Module):    
    def __init__(self, n_features, max_block_size, num_classes, zenith_num_classes, config):
        super().__init__()

        self.inp = nn.Linear(n_features, config.n_embd[0])
        self.drop_inputs = nn.Dropout(config.dropout)

        self.encoder = AttentionEncoder(config)
        
        self.pool = SequencePool()

        num_out_features = config.n_embd[-1] * self.pool.num_pools

        self.neck_az = Neck(num_out_features, config.neck_features, config.bias, config.neck_dropout)
        self.neck_zn = Neck(num_out_features, config.neck_features, config.bias, config.neck_dropout)
        
        self.azimuth = nn.Linear(config.neck_features, num_classes)
        self.zenith = nn.Linear(config.neck_features, zenith_num_classes)

    def get_masks(self, x, l):
        key_padding_mask = torch.arange(x.shape[1]).view(1, -1).to(l.device) < l.view(-1, 1)
        attn_mask = (key_padding_mask.unsqueeze(1) == key_padding_mask.unsqueeze(2)).unsqueeze(1)  # (B, 1, T, T)
        return key_padding_mask, attn_mask

    def forward(self, x):
        inputs, seq_lengths = x
        out = self.inp(inputs)
        out = self.drop_inputs(out)
        key_padding_mask, attn_mask = self.get_masks(inputs, seq_lengths)
        out = self.encoder(out, attn_mask)
        pool = self.pool(out, seq_lengths, key_padding_mask)
    
        az_out = self.azimuth(self.neck_az(pool))
        zn_out = self.zenith(self.neck_zn(pool))
        return az_out, zn_out, pool
    
####################################################################################################

def topksum(pred, centers, k=3):
    zn_topk, zn_topk_idk = torch.topk(pred, k=k, dim=1)
    zn_topk_smax = torch.softmax(zn_topk, axis=1)
    angle = torch.sum(zn_topk_smax * centers[zn_topk_idk], dim=1)
    return angle

def discrete_to_angle(az_pred,
                      zn_pred,
                      azimuth_bin_centers,
                      zenith_bin_centers):
    
    az_softmax = torch.softmax(az_pred, dim=1)
    azx, azy = torch.cos(azimuth_bin_centers), torch.sin(azimuth_bin_centers)
    azmx, azmy = az_softmax @ azx, az_softmax @ azy
    azn = torch.sqrt(azmx**2 + azmy**2)
    az_pred_center = ( torch.arccos(azmx / azn) * torch.sign(azmy) ) % (np.pi * 2)
    
    # zn_pred_center = zenith_bin_centers[torch.argmax(zn_softmax, dim=1)]
    zn_pred_center = topksum(zn_pred, zenith_bin_centers, k=3)
    
    return az_pred_center, zn_pred_center

def prepare_submission(event_ids, azimuth, zenith, validate_ids=False):
    if validate_ids:
        sample_submission = pd.read_parquet(os.path.join(DATA_DIR, 'sample_submission.parquet'))
        assert np.array_equal(event_ids, sample_submission.event_id.values)
    submission_df = pd.DataFrame(
        {
            'event_id': event_ids,
            'azimuth': azimuth.cpu().numpy(),
            'zenith': zenith.cpu().numpy(),
        }
    ).set_index('event_id')
    return submission_df

####################################################################################################

def load_model(model_config):
    model = MultiLabelClassifier(n_features=N_FEATURES, 
                                max_block_size=MAX_SEQUENCE_LENGTH,
                                num_classes=NUM_BINS,
                                zenith_num_classes=ZENITH_NUM_BINS,
                                config=model_config)
    state_dict = torch.load(model_config.checkpoint_path)['state_dict']
    old_keys = list(state_dict.keys())
    # print("Removing prefix", model_config.unwanted_prefix)
    for key in old_keys:
        if model_config.unwanted_prefix in key:
            new_key = key.split(model_config.unwanted_prefix)[1][1:]
            state_dict[new_key] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model

####################################################################################################

@torch.no_grad()
def predict_on_batch(model, dataset, logits_filename=None):
    az_pred_batch, zn_pred_batch = [], []
    pool_batch = []
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    for x, l in tqdm(dataloader):
        b_max_len = int(l.max()) # Lazy packing, works well for batch size ~1000
        azp, znp, pool = model((x[:, :b_max_len].to(device), l.to(device)))
        az_pred_batch.append(azp)
        zn_pred_batch.append(znp)
        pool_batch.append(pool)

    
    az_pred_batch = torch.cat(az_pred_batch, dim=0)[dataset.reverse_sort_idx]
    zn_pred_batch = torch.cat(zn_pred_batch, dim=0)[dataset.reverse_sort_idx]
    pool_batch = torch.cat(pool_batch, dim=0)[dataset.reverse_sort_idx]

    az, zn = discrete_to_angle(az_pred_batch, zn_pred_batch, azimuth_bin_centers, zenith_bin_centers)

    if logits_filename is not None:
        np.savez(logits_filename,
                az_logits=az_pred_batch.cpu().numpy(),
                zn_logits=zn_pred_batch.cpu().numpy(),
                encoder=pool_batch.cpu().numpy(),
                azimuth=az.cpu().numpy(),
                zenith=zn.cpu().numpy(),
            )

    return az_pred_batch, zn_pred_batch

def check_score(batch_id, az_pred, zn_pred, text, dataset):
    if not VALIDATE:
        return
    events_to_keep = dataset.events_to_keep
    az_gt = metadata.filter(pl.col('batch_id') == batch_id).select('azimuth').to_numpy().squeeze()[events_to_keep]
    zen_gt = metadata.filter(pl.col('batch_id') == batch_id).select('zenith').to_numpy().squeeze()[events_to_keep]
    az, zn = discrete_to_angle(az_pred, zn_pred, azimuth_bin_centers, zenith_bin_centers)
    zn = torch.clip(zn, 0.0, np.pi)
    angular_dist = angular_dist_score(az_gt, zen_gt, az.cpu().numpy(), zn.cpu().numpy())
    print(f"\n\n \t ang_dist {text}", angular_dist, "\n")

torch.backends.cuda.enable_flash_sdp(enabled=True)
az_pred, zn_pred = [], []
model = load_model(Model_Config)
for bfile in BATCH_LIST:
    batch_id = int(bfile.split('.')[0].split('_')[-1])
    dataset = IceCubeDataset(bfile)
    print(len(dataset))

    str_id = f'{batch_id}_{Model_Config.name}'
    os.makedirs(f'encoder_preds/{Model_Config.name}', exist_ok=True)
    logits_filename = f'encoder_preds/{Model_Config.name}/preds_{str_id}.npz' if VALIDATE else None

    az_pred_mb, zn_pred_mb = predict_on_batch(model, dataset, logits_filename)
    check_score(batch_id, az_pred_mb, zn_pred_mb, text=str_id, dataset=dataset)