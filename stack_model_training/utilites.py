import os

def get_next_version(save_dir, prefix='v_'):
    ## Taken from lightning_fabric.loggers.tensorboard

    assert os.path.isdir(save_dir), f"No such directory {save_dir}"

    existing_versions = []
    for d in os.listdir(save_dir):
        bn = os.path.basename(d)
        if os.path.isdir(os.path.join(save_dir, d)) and (bn.startswith(prefix)):
            dir_ver = bn.split("_")[1].replace("/", "")
            existing_versions.append(int(dir_ver))

    if len(existing_versions) == 0:
        return 0
    return max(existing_versions) + 1

def setup_logger(use_logger, logs_dir, run_name, prefix='v_'):
    if not use_logger or not os.path.isdir(logs_dir):
        return use_logger, run_name
    
    save_dir = os.path.join(logs_dir, 'lightning_logs')
    next_version = get_next_version(save_dir, prefix)
    versioned_run_name = prefix + f'{next_version}_{run_name}'
    try:
        from pytorch_lightning.loggers import TensorBoardLogger
        logger = TensorBoardLogger(save_dir=logs_dir, version=versioned_run_name)
        return logger, versioned_run_name
    except:
        return use_logger, run_name

if __name__ == '__main__':
    print("version", get_next_version('./lightning_logs/'))
    logger = setup_logger(use_logger=True, logs_dir='.', run_name='test')
    print(logger.log_dir)