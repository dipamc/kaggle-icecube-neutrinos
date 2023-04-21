# Basically a copy of optimizer part from
# https://github.com/karpathy/nanoGPT/blob/master/model.py

from attn_model import LayerNorm
import torch
import inspect

def gpt_adamw_optimizer(module, weight_decay, learning_rate, betas, device_type, head_weight_decay=None):
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, )
    blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm1d)
    for mn, m in module.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            # random note: because named_modules and named_parameters are recursive
            # we will see the same tensors p many many times. but doing it this way
            # allows us to know which parent module any tensor p belongs to...
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in module.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )
    
    # Separate weight decay for head (after pooling) and encoder
    head_weight_decay = weight_decay if head_weight_decay is None else head_weight_decay
    encoder_decay, head_decay = [], []
    head_keys = ['neck', 'azimuth', 'zenith']
    for pn in decay:
        if any([hk in pn for hk in head_keys]):
            head_decay.append(pn)
        else:
            encoder_decay.append(pn)
    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(encoder_decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(head_decay))], "weight_decay": head_weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    # PyTorch 2.0 has a new 'fused' option for AdamW that is much faster 
    # NOTE: Doesn't seem faster on RTX 3060, and doesn't allow gradient clipping
    # use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
    use_fused = False
    print(f"using fused AdamW: {use_fused}")
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

    return optimizer

def load_optimizer_state(checkpoint_path, optimizer):
    ckpt = torch.load(checkpoint_path)
    optimizer_state_dict = ckpt['optimizer_states'][0]
    optimizer.load_state_dict(optimizer_state_dict)

def load_scheduler_state(checkpoint_path, scheduler):
    ckpt = torch.load(checkpoint_path)
    scheduler_state_dict = ckpt['lr_schedulers'][0]
    scheduler.load_state_dict(scheduler_state_dict)

def configure_optimizers(lightning_module, config, accumulate_grads, optimizer_checkpoint):
    optimizer = gpt_adamw_optimizer(lightning_module, 
                                    weight_decay=config.weight_decay, 
                                    learning_rate=config.max_lr,
                                    betas=config.betas, 
                                    device_type=lightning_module.device.type,
                                    head_weight_decay=config.head_weight_decay)

    if optimizer_checkpoint is not None:
        load_optimizer_state(optimizer_checkpoint, optimizer) 
    
    sched_steps = (lightning_module.trainer.estimated_stepping_batches // accumulate_grads) + 1
    print("Scheduler steps", sched_steps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.max_lr,
        div_factor=config.div_factor,
        final_div_factor=config.final_div_factor, 
        pct_start=config.pct_start,
        total_steps=sched_steps,
    )
    if optimizer_checkpoint is not None:
        load_scheduler_state(optimizer_checkpoint, scheduler) 
        
    lr_scheduler_config = {"scheduler": scheduler, "interval": "step"}
    return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}