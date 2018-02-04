import pdb
import numpy as np
import tensorflow as tf

from trainer import Trainer
from config import get_config
from data_loader import get_loader
from utils import prepare_dirs_and_logger, save_config

def main(config):
    prepare_dirs_and_logger(config)

    rng = np.random.RandomState(config.random_seed)
    tf.set_random_seed(config.random_seed)

    if config.is_train:
        data_path = config.data_path
        batch_size = config.batch_size
        do_shuffle = True
    else:
        setattr(config, 'batch_size', 64)
        if config.test_data_path is None:
            data_path = config.data_path
        else:
            data_path = config.test_data_path
        batch_size = config.sample_per_image
        do_shuffle = False

    #dir_loader = 'train8020'
    #dir_loader = 'train6040'
    #dir_loader = 'train4060'
    #dir_loader = 'train2080'
    #dir_loader = 'train1090'
    #dir_loader = 'train0510'
    #dir_loader = 'trainBig0510'
    dir_loader = 'train_all_1090'
    dir_loader = 'train_small_5050'
    config.pct = [int(dir_loader[-4:][:2]), int(dir_loader[-4:][2:])]

    dir_target = 'train5050'
    data_loader = get_loader(
        data_path, config.batch_size, config.scale_size,
        config.data_format, config.split, target=dir_loader)
    data_loader_target = get_loader(
        data_path, config.batch_size, config.scale_size,
        config.data_format, config.split, target=dir_target)
    trainer = Trainer(config, data_loader, data_loader_target)

    if config.is_train:
        save_config(config)
        trainer.train()
    else:
        if not config.load_path:
            raise Exception("[!] You should specify `load_path` to load a pretrained model")
        trainer.test()

if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)
