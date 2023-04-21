import torch
import os
import json
import math
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as lit
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from contextlib import nullcontext

from preprocessing import IcecubePreprocessor
from prefetch_dataset import PrefetchDataset

from attn_model import MultiLabelClassifier
from optimizer import configure_optimizers
from loss_metrics import MultiLabelLossMetrics

torch.set_float32_matmul_precision('medium')

def get_amp_context(device_type, datatype):
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[datatype]
    device_type = 'cuda' if 'cuda' in device_type else device_type
    context = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    return context

class LitModel(lit.LightningModule):
    def __init__(self, model, loss_metrics, example_input_array, config, feature_join_function):
        super().__init__()
        self.model = model
        self.loss_metrics = loss_metrics
        self.example_input_array = example_input_array
        self.config = config
        self.feature_join_function = feature_join_function

        self.automatic_optimization = False
        self.current_amp_dtype = None
    
    def setup_amp(self, device_type):
        dtype = self.config.precision_schedule.get(self.current_epoch, None)
        if dtype is not None and self.current_amp_dtype != dtype:
            self.amp_context =  get_amp_context(device_type, dtype)
            print(f"Switched amp context to {dtype}")
            self.scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'),
                                                    growth_interval=600)
        assert hasattr(self, 'amp_context'), "Amp context not set, check precision schedule"
        self.current_amp_dtype = dtype
        os.environ['EPOCH'] = str(self.current_epoch)

    def forward_calc_loss(self, batch, batch_idx, training):
        x, y = batch
        self.setup_amp(str(x[0].device))

        with self.amp_context:
            y_pred = self(x)
            loss_metrics = self.loss_metrics.calculate_loss_and_metrics(y, y_pred,
                is_validation=(not training) and (self.logger is not None)
            )
        if training:
            self.manual_backward( self.scaler.scale(loss_metrics['loss']) )
            self.log("loss", loss_metrics['loss'])
        return loss_metrics

    def training_step(self, batch, batch_idx):
        loss_metrics = self.forward_calc_loss(batch, batch_idx, training=True)
        if (batch_idx + 1) % self.config.model_config.accumulate_grads == 0:
            optimizer = self.optimizers()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_value_(self.parameters(), 0.5)
            self.scaler.step(optimizer)
            self.scaler.update()
            optimizer.zero_grad(set_to_none=True)
            self.lr_schedulers().step()
        return loss_metrics

    def validation_step(self, batch, batch_idx):
        loss_metrics = self.forward_calc_loss(batch, batch_idx, training=False)
        return loss_metrics
    
    def forward(self, x):
        inp, l = x
        if not self.config.compile:
            inp = inp[:, :l.max().int()]
            inp = self.feature_join_function(inp)
        else:
            inp, l = x
            inp = self.feature_join_function(inp)
        out = self.model([inp, l])
        return out

    def configure_optimizers(self):
        optimizer_checkpoint = self.config.pretrained_checkpoint_path \
            if self.config.use_pretrained and self.config.load_optimizer_state \
            else None
        optimizer = configure_optimizers(self, self.config.optimizer_config, 
                                         self.config.model_config.accumulate_grads, optimizer_checkpoint)
        return optimizer
    
    def training_epoch_end(self, outputs):
        mean_loss = torch.mean(torch.tensor([o['loss'] for o in outputs])) 
        self.log("train_loss", mean_loss, prog_bar=True)
        for mname in self.loss_metrics.metric_names:
            mval = torch.mean(torch.tensor([o['metrics'][mname] for o in outputs]))
            self.log(f"train_{mname}", mval, prog_bar=True)
        self.trainer.train_dataloader.dataset.datasets.on_epoch_end() # Dataloader change data
        return super().training_epoch_end(outputs)
    
    def validation_epoch_end(self, outputs):
        mean_loss = torch.mean(torch.tensor([o['loss'] for o in outputs]))
        self.log("val_loss", mean_loss, prog_bar=True)
        for mname in self.loss_metrics.metric_names:
            mval = torch.mean(torch.tensor([o['metrics'][mname] for o in outputs]))
            self.log(f"val_{mname}", mval, prog_bar=True)
        self.log('step', self.trainer.current_epoch)

        if self.logger is not None:
            tensorboard = self.logger.experiment
            edata = [o.pop('extra_data') for o in outputs]
            zenith_vals = np.concatenate([o['zn_pred'].cpu().numpy() for o in edata])
            az_vals = np.concatenate([o['az_pred'].cpu().numpy() for o in edata])
            tensorboard.add_histogram('zenith', zenith_vals, self.current_epoch, bins=np.arange(0, 129)*np.pi/128)
            tensorboard.add_histogram('azimuth', az_vals, self.current_epoch, bins=np.arange(0, 129)*np.pi*2/128)
            
        print() # Keep the previous progbar
        self.trainer.val_dataloaders[0].dataset.on_epoch_end()    # Dataloader change data if needed   
        return super().validation_epoch_end(outputs)

def load_model_weights(checkpoint_path, unwanted_prefix, model):
    ckpt = torch.load(checkpoint_path)
    state_dict = ckpt['state_dict']
    old_keys = list(state_dict.keys())
    for key in old_keys:
        if unwanted_prefix in key:
            new_key = key.split(unwanted_prefix)[1][1:]
            state_dict[new_key] = state_dict.pop(key)
    model.load_state_dict(state_dict)

def train(config):
    if config.DEBUG:
        accelerator, lb, use_logger, device = 'gpu', 10, False, 'cuda'
    else:
        accelerator, lb, use_logger, device = 'gpu', None, config.log, 'cuda'
    
    dc = config.dataset_config
    train_data_processor = IcecubePreprocessor(dc.data_dir,
                                               dc.geometry_file_name, 
                                               dc.bins_file_name, 
                                               dc.feature_names, 
                                               dc.max_sequence_length, 
                                               dc.seed,
                                               dc.train_preprocessing_config.filter_config)
    val_data_processor = IcecubePreprocessor(dc.data_dir,
                                             dc.geometry_file_name, 
                                             dc.bins_file_name, 
                                             dc.feature_names, 
                                             dc.max_sequence_length, 
                                             dc.seed,
                                             dc.val_preprocessing_config.filter_config)

    train_dataset = PrefetchDataset(train_data_processor, 
                                    dc.train_batch_list, 
                                    dc.num_batches_per_epoch,
                                    minibatch_size=config.batch_size,
                                    use_threading=True)
    val_dataset = PrefetchDataset(val_data_processor, 
                                  dc.val_batch_list,
                                  dc.num_batches_per_epoch,
                                  minibatch_size=config.batch_size,
                                  use_threading=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=False) # Don't shuffle else packing will not work
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    attn_model = MultiLabelClassifier(n_features=dc.n_features,
                                      max_block_size=dc.max_sequence_length,
                                      num_classes=dc.num_bins,
                                      zenith_num_classes=dc.zenith_num_bins,
                                      config=config.model_config)

    if config.use_pretrained:
        load_model_weights(config.pretrained_checkpoint_path,
                        config.unwanted_prefix,
                        model=attn_model,
        )

    if config.compile:
        attn_model = torch.compile(attn_model)

    bin_data = train_data_processor.bin_data
    loss_metrics = MultiLabelLossMetrics(bin_data, dc.smoothing_kernel_length, 
                                         device, dc.max_sequence_length, config.filt_postfix)

    example_input_array = [(torch.zeros(10, dc.max_sequence_length, len(train_data_processor.info_names)), 
                            torch.randint(32, size=(10,)))]

    pl_model = LitModel(attn_model, loss_metrics, example_input_array, config, 
                        feature_join_function=train_data_processor.join_features)
    print(ModelSummary(pl_model, max_depth=-1))

    monitor_metric = "val_ang_dist" + config.filt_postfix
    checkpoint_callback = ModelCheckpoint(save_top_k=3, monitor=monitor_metric, mode='min', save_last=True)
    checkpoint_callback2 = ModelCheckpoint(save_top_k=3, monitor='val_loss', mode='min', save_last=False)

    callbacks = [LearningRateMonitor(logging_interval='step'), checkpoint_callback, checkpoint_callback2]
    callbacks = [] if not use_logger else callbacks

    trainer = lit.Trainer(max_epochs=config.epochs, 
                            accelerator=accelerator, 
                            limit_train_batches=lb, limit_val_batches=lb, 
                            logger=use_logger, enable_checkpointing=use_logger,
                            callbacks=callbacks,
    )

    trainer.fit(model=pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == '__main__':
    from config import TrainingConfig
    config = TrainingConfig()
    train(config)
