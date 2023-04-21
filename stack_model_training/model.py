# Most of the code is taken from
# https://github.com/karpathy/nanoGPT/blob/master/model.py

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math

    
class Neck(nn.Module):
    def __init__(self, in_features, middle_features, out_features, bias, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, middle_features, bias=bias),
            nn.LayerNorm(middle_features),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(middle_features, out_features, bias=bias),
            nn.LayerNorm(out_features),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.mlp(x)

    
class MultiLabelClassifier(nn.Module):    
    def __init__(self, num_classes, zenith_num_classes, config):
        super().__init__()

        self.neck = Neck(config.n_embd, config.middle_features, config.neck_features, config.bias, config.dropout)
        self.xyz = nn.Linear(config.neck_features, 3)

    def forward(self, x):
        return self.xyz(self.neck(x))        
    

def check_model():
    import pytorch_lightning as pl
    from pytorch_lightning.utilities.model_summary import ModelSummary

    from config import ModelConfig

    config = ModelConfig()

    num_classes = 128

    class LitModel(pl.LightningModule):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.example_input_array = torch.zeros(10, config.n_embd)

        def forward(self, x):
            return self.model(x)

    model = MultiLabelClassifier(num_classes, num_classes, config)
    # model = torch.compile(model)

    pl_model = LitModel(model)
    print(ModelSummary(pl_model, max_depth=-1))

if __name__ == '__main__':
    check_model()
