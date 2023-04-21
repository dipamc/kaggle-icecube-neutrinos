# Most of the code is taken from
# https://github.com/karpathy/nanoGPT/blob/master/model.py

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
from flash_attn.flash_attention import FlashAttention

def init_weights_gpt(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None: torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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

        self.flash_attention = FlashAttention()

    def forward(self, x, attn_mask, cross_features=None):
        B, T, _ = x.shape
        C = self.n_embd
        attn_mask, key_padding_mask = attn_mask
        if torch.is_autocast_enabled() and (torch.get_autocast_gpu_dtype() in [torch.float16, torch.bfloat16]):
            qkv =  self.c_attn(x).view(B, T, 3, self.n_heads, C // self.n_heads) # (B, T, 3, nh, hs)
            y = self.flash_attention(qkv, key_padding_mask=key_padding_mask)[0]
            y = y.contiguous().view(B, T, C)
        else:
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
        out = meanf
        return out
    
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

        self.apply(init_weights_gpt)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/np.sqrt(2 * len(config.n_embd)))

    def get_masks(self, x, l):
        key_padding_mask = torch.arange(x.shape[1]).view(1, -1).to(l.device) < l.view(-1, 1)
        attn_mask = (key_padding_mask.unsqueeze(1) == key_padding_mask.unsqueeze(2)).unsqueeze(1)  # (B, 1, T, T)
        return key_padding_mask, attn_mask

    def forward(self, x):
        inputs, seq_lengths = x
        out = self.inp(inputs)
        out = self.drop_inputs(out)
        key_padding_mask, attn_mask = self.get_masks(inputs, seq_lengths)
        out = self.encoder(out, (attn_mask, key_padding_mask))
        out = self.pool(out, seq_lengths, key_padding_mask)
    
        az_out = self.azimuth(self.neck_az(out))
        zn_out = self.zenith(self.neck_zn(out))
        return az_out, zn_out   
    

def check_model():
    import pytorch_lightning as pl
    from pytorch_lightning.utilities.model_summary import ModelSummary

    from config import ModelConfig

    n_features, max_pulse_count = 9, 11
    num_classes = 128
    class LitModel(pl.LightningModule):
        def __init__(self, model):
            super().__init__()
            self.model = model
            z = torch.zeros(10, max_pulse_count, n_features, device='cuda')
            z[0,:,4] = torch.arange(max_pulse_count)
            z[1,:,4] = torch.arange(max_pulse_count, 0, step=-1)
            self.example_input_array = [(z, torch.randint(1,11, size=(10,), device='cuda'))]

        def forward(self, x):
            out = self.model([x[0].to('cuda'), x[1].to('cuda')])
            # out = self.model(x)
            return out

    model = MultiLabelClassifier(n_features, max_pulse_count, num_classes, num_classes, ModelConfig())
    model = model.to('cuda')
    # model = torch.compile(model)

    pl_model = LitModel(model)
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        print(ModelSummary(pl_model, max_depth=-1))

if __name__ == '__main__':
    check_model()
