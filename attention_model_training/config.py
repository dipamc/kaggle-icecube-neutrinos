import os
import numpy as np

# This mode is for tuning on very long sequences
FINETUNE_LONG_SEQ = False

SEED = 42
rng = np.random.RandomState(SEED)
TRAIN_LIST = []
for _ in range(50):
    TRAIN_LIST.extend(list(rng.choice(range(1, 651), size=650, replace=False)))

class ModelConfig:
    n_embd = [128]*18
    n_heads = [2]*18
    neck_features = 3072

    bias = False
    dropout = 0.0
    neck_dropout = 0.0

    assert len(n_embd) == len(n_heads)
    assert neck_features % n_embd[-1] == 0

    accumulate_grads = 1
    batch_size = 384

    if FINETUNE_LONG_SEQ:
        accumulate_grads = 384
        batch_size = 1

class OptimizerConfig:
    # AdamW
    weight_decay = 1e-5
    head_weight_decay = 1e-5
    betas = (0.9, 0.95)

    # One cycle LR schedule
    max_lr = 5e-5
    div_factor = 2
    final_div_factor = 10
    pct_start = 0.15
    if FINETUNE_LONG_SEQ:
        max_lr = 5e-6
        div_factor = 2
        final_div_factor = 1
        pct_start = 0.0

class PreprocessingConfig:
    min_sequence_lentgh = 0
    if FINETUNE_LONG_SEQ:
        min_sequence_lentgh = 256
    train_filter = dict(min_sequence_lentgh=min_sequence_lentgh)
    val_filter = dict(min_sequence_lentgh=min_sequence_lentgh)

    def __init__(self, mode, max_events_per_batch=None):
        if mode=='train':
            self.filter_config = self.train_filter
        elif mode=='val':
            self.filter_config = self.val_filter
        else:
            raise NotImplementedError
        self.filter_config.update(dict(
            max_events_per_batch = max_events_per_batch
        ))

class DatasetConfig:
    train_batch_list = TRAIN_LIST
    val_batch_list = [651]
    num_batches_per_epoch = 4
    if FINETUNE_LONG_SEQ:
        num_batches_per_epoch = 1
    val_num_events = 200000

    train_preprocessing_config = PreprocessingConfig(mode='train')
    val_preprocessing_config = PreprocessingConfig(mode='val', max_events_per_batch=val_num_events)

    data_dir = os.getenv('ICECUBE_DATA_DIR', './data')
    assert os.path.exists(data_dir)  
    seed = SEED

    max_sequence_length = 256
    if FINETUNE_LONG_SEQ:
        max_sequence_length = 3072

    num_bins = 128
    smoothing_kernel_length = 15
    assert smoothing_kernel_length % 2 == 1
    zenith_num_bins = num_bins + (smoothing_kernel_length - 1)
    bins_file_name = os.path.join(data_dir, f'angle_bins_{num_bins}.json')

    geometry_file_name = os.path.join(data_dir, 'sensor_geometry_with_transparency.csv')
    ## features appended with 'time', 'charge', 'auxiliary'
    feature_names = ['x', 'y', 'z', 'qe', 'scatter', 'absorp']

    n_features = len(feature_names) + 3

    def __init__(self, debug_mode):
        if debug_mode:
            self.train_batch_list = [1, 2, 3]
            self.num_batches_per_epoch = 1
    
class TrainingConfig:
    DEBUG = False 
    log = True
    
    # Unfortunately passing different size sequence batches recompiles everytime 
    # and hence is super slow, can use when not packing sequences
    compile = False

    commit = False
    filt_postfix = ''
    run_name = 'attention_icecube' + filt_postfix

    batch_size = ModelConfig.batch_size
    epochs = 650
    save_ckpt_at = []
    precision_schedule = {0: 'float16', 100: 'float32'} # Specify the starting epoch of the dtype change
    if DEBUG:
        precision_schedule = {0: 'float32'} # Check for OOM after dtype switch
    
    # pretrained_checkpoint_path = 'lightning_logs/version_0/checkpoints/last.ckpt' # Set this for finetuning
    pretrained_checkpoint_path = ''

    load_optimizer_state = True
    unwanted_prefix = 'model'
    use_pretrained = os.path.exists(pretrained_checkpoint_path)
    assert (pretrained_checkpoint_path == '') or use_pretrained, "Specified checkpoint doesn't exist"

    assert (not FINETUNE_LONG_SEQ) or (pretrained_checkpoint_path != ''), "Fine tuning should have pretrained"

    dataset_config = DatasetConfig(debug_mode=DEBUG)
    optimizer_config = OptimizerConfig()
    model_config = ModelConfig() 

