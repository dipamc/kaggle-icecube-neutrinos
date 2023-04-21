import os

class ModelConfig:
    n_embd = 512 + 128 + 128 + 14
    middle_features = 8192
    neck_features = 2048
    bias = True
    dropout = 0.0

class OptimizerConfig:
    # AdamW
    weight_decay = 1e-5
    betas = (0.9, 0.95)

    # One cycle
    max_lr = 5e-5
    accumulate_grads = 1
    div_factor = 5
    final_div_factor = 10
    pct_start = 0.05

class DatasetConfig:
    data_dir = os.getenv('ICECUBE_DATA_DIR', './data')
    assert os.path.exists(data_dir)
    enc_dir = '../inference/encoder_preds/'
    # First run ../inference/inference.py to save the logits
    inputs_name = (enc_dir + 'base_model/preds_{:d}_base_model.npz', 
                    enc_dir + 'long_seq_model/preds_{:d}_long_seq_model.npz')

    seed = 42
    num_classes = 1024
    smoothing_kernel_length = 121
    assert smoothing_kernel_length % 2 == 1
    zenith_num_bins = num_classes + (smoothing_kernel_length - 1)
    bin_files_name = os.path.join(data_dir, f'angle_bins_{num_classes}.json')

    train_batch_list = list(range(551, 651))
    val_batch_list = [651]

    num_batches_per_epoch = 5

    train_num_events_per_batch = None # Use full batch
    val_num_events_per_batch = 200000

    def __init__(self, debug_mode):
        if debug_mode:
            self.train_batch_list = [601, 602]
    
class TrainingConfig:
    # DEBUG = True
    # log = False

    DEBUG = False
    log = True
    
    # Unfortunately passing different size sequence batches recompiles everytime 
    # and hence is super slow, can use when not packing sequences
    compile = False

    commit = False
    filt_postfix = ''
    run_name = 'vmf_stack' + filt_postfix

    batch_size = 1024
    epochs = 700
    precision_schedule = {0: 'float32'} 
    
    pretrained_checkpoint_path = ''

    load_optimizer_state = True
    unwanted_prefix = 'model'
    use_pretrained = os.path.exists(pretrained_checkpoint_path)
    assert (pretrained_checkpoint_path == '') or use_pretrained, "Specified checkpoint doesn't exist"

    dataset_config = DatasetConfig(debug_mode=DEBUG)
    optimizer_config = OptimizerConfig()
    model_config = ModelConfig() # TODO : Auto load network sizes from checkpoint


if __name__ == "__main__":
    TrainingConfig()
