import torch
import os
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from contextlib import nullcontext

from dataset import IcecubeDataset
from model import MultiLabelClassifier
from optimizer import configure_optimizers
from loss_metrics import MultiLabelLossMetrics, VonMishesFisherLossMetrics, ZenithLossMetrics, AzimuthLossMetrics

from utilites import setup_logger

torch.set_float32_matmul_precision('medium')

def get_amp_context(device_type, datatype):
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[datatype] # Float16 needs gradscaler, add that to use
    device_type = 'cuda' if 'cuda' in device_type else device_type
    context = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    return context

class LitModel(pl.LightningModule):
    def __init__(self, model, loss_metrics, example_input_array, config):
        super().__init__()
        self.model = model
        self.loss_metrics = loss_metrics
        self.example_input_array = example_input_array
        self.config = config

        self.automatic_optimization = False
        self.current_amp_dtype = None
    
    def setup_amp(self, device_type):
        dtype = self.config.precision_schedule.get(self.current_epoch, None)
        if dtype is not None and self.current_amp_dtype != dtype:
            self.amp_context =  get_amp_context(device_type, dtype)
            print(f"Switched amp context to {dtype}")
        assert hasattr(self, 'amp_context'), "Amp context not set, check precision schedule"
        self.current_amp_dtype = dtype

    def training_step(self, batch, batch_idx):
        x, y = batch
        self.setup_amp(str(x[0].device))
        with self.amp_context:
            y_pred = self(x)
            loss_metrics = self.loss_metrics.calculate_loss_and_metrics(y, y_pred)
        self.log("loss", loss_metrics['loss'])

        self.manual_backward( loss_metrics['loss'] )
        if (batch_idx + 1) % self.config.optimizer_config.accumulate_grads == 0:
            # torch.nn.utils.clip_grad_value_(self.parameters(), 0.5)
            optimizer = self.optimizers()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            self.lr_schedulers().step()

        loss_metrics['batch_size'] = len(x[0])
        return loss_metrics

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.setup_amp(str(x[0].device))
        with self.amp_context:
            y_pred = self(x)
            loss_metrics = self.loss_metrics.calculate_loss_and_metrics(y, y_pred, 
                                    is_validation=self.logger is not None)

        loss_metrics['batch_size'] = len(x[0])
        return loss_metrics
    
    def forward(self, x):
        out = self.model(x)
        return out

    def configure_optimizers(self):
        optimizer_checkpoint = self.config.pretrained_checkpoint_path \
            if self.config.use_pretrained and self.config.load_optimizer_state \
            else None
        optimizer = configure_optimizers(self, self.config.optimizer_config, optimizer_checkpoint)
        return optimizer
    
    def training_epoch_end(self, outputs):
        n_data = torch.sum(torch.tensor([o['batch_size'] for o in outputs]))
        mean_loss = torch.sum(torch.tensor([o['loss'] * o['batch_size'] for o in outputs])) / n_data
        self.log("train_loss", mean_loss, prog_bar=True)
        for mname in self.loss_metrics.metric_names:
            mval = torch.sum(torch.tensor([o['metrics'][mname] * o['batch_size'] for o in outputs])) / n_data
            self.log(f"train_{mname}", mval, prog_bar=True)
        self.trainer.train_dataloader.dataset.datasets.on_epoch_end() # Dataloader change data
        return super().training_epoch_end(outputs)
    
    def validation_epoch_end(self, outputs):
        n_data = torch.sum(torch.tensor([o['batch_size'] for o in outputs]))
        mean_loss = torch.sum(torch.tensor([o['loss'] * o['batch_size'] for o in outputs])) / n_data
        self.log("val_loss", mean_loss, prog_bar=True)
        for mname in self.loss_metrics.metric_names:
            mval = torch.sum(torch.tensor([o['metrics'][mname] * o['batch_size'] for o in outputs])) / n_data
            self.log(f"val_{mname}", mval, prog_bar=True)
        self.log('step', self.trainer.current_epoch)

        if self.logger is not None:
            tensorboard = self.logger.experiment
            edata = [o.pop('extra_data') for o in outputs]
            if 'zn_pred' in edata[0]:
                zenith_vals = np.concatenate([o['zn_pred'].cpu().numpy() for o in edata])
                tensorboard.add_histogram('zenith', zenith_vals, self.current_epoch, bins=np.arange(0, 129)*np.pi/128)
            if 'az_pred' in edata[0]:
                az_vals = np.concatenate([o['az_pred'].cpu().numpy() for o in edata])
                tensorboard.add_histogram('azimuth', az_vals, self.current_epoch, bins=np.arange(0, 129)*np.pi*2/128)

        print() # Keep the previous progbar
        self.trainer.val_dataloaders[0].dataset.on_epoch_end()
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
        accelerator, lb, use_logger, device = 'gpu', 100, False, 'cuda'
    else:
        accelerator, lb, use_logger, device = 'gpu', None, config.log, 'cuda'
    
    dc = config.dataset_config
    train_dataset = IcecubeDataset(seed=dc.seed,
                                   batch_list=dc.train_batch_list,
                                   bin_files_name=dc.bin_files_name,
                                   data_dir=dc.data_dir,
                                   inputs_name=dc.inputs_name,
                                   num_batches_per_epoch=dc.num_batches_per_epoch, 
                                   num_events_per_batch=dc.train_num_events_per_batch,
                                   use_threading=True,
                                   device=device,
    )
    val_dataset = IcecubeDataset(seed=dc.seed,
                                 batch_list=dc.val_batch_list,
                                 bin_files_name=dc.bin_files_name,
                                 data_dir=dc.data_dir,
                                 inputs_name=dc.inputs_name,
                                 num_batches_per_epoch=dc.num_batches_per_epoch, 
                                 num_events_per_batch=dc.val_num_events_per_batch,
                                 use_threading=True,
                                 device=device,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=10000)

    model = MultiLabelClassifier(num_classes=dc.num_classes,
                                 zenith_num_classes=dc.zenith_num_bins,
                                 config=config.model_config)

    if config.compile:
        model = torch.compile(model)

    loss_metrics = VonMishesFisherLossMetrics(config.filt_postfix)

    example_input_array = [torch.zeros(10, config.model_config.n_embd)]

    pl_model = LitModel(model, loss_metrics, example_input_array, config)
    print(ModelSummary(pl_model, max_depth=-1))

    monitor_metric = "val_ang_dist" + config.filt_postfix
    
    checkpoint_callback = ModelCheckpoint(save_top_k=3, monitor=monitor_metric, mode='min', save_last=True)
    checkpoint_callback2 = ModelCheckpoint(save_top_k=2, monitor='val_ang_dist_vmf', mode='min', save_last=False)
    checkpoint_callback3 = ModelCheckpoint(save_top_k=2, monitor='val_loss', mode='min', save_last=False)


    callbacks = [LearningRateMonitor(logging_interval='step'), checkpoint_callback, checkpoint_callback2, checkpoint_callback3]
    callbacks = [] if not use_logger else callbacks

    logger, version_name = setup_logger(use_logger, '.', config.run_name, 'v_')

    trainer = pl.Trainer(max_epochs=config.epochs, 
                            accelerator=accelerator, 
                            limit_train_batches=lb, limit_val_batches=lb, 
                            logger=logger, enable_checkpointing=use_logger,
                            callbacks=callbacks,
    )

    if not config.DEBUG and use_logger:
        print(f"Logging to {trainer.logger.log_dir}")
        os.system(f'mkdir -p {trainer.logger.log_dir}; cp *.py {trainer.logger.log_dir}')
        if config.commit:
            os.system(f'git add . ; git commit -m "{version_name}" ')

    trainer.fit(model=pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == '__main__':
    from config import TrainingConfig
    config = TrainingConfig()
    train(config)
