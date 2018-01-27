import os
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

    config.pct = [20, 80]
    print('\n\nUsing thinning factor according to {}:{} ratio.\n\n'.format(
        config.pct[0], config.pct[1]))

    data_loader_user = get_loader(
        data_path, config.batch_size, config.scale_size,
        config.data_format, split_name=config.split, target='user', n=100)
    data_loader_train = get_loader(
        data_path, config.batch_size, config.scale_size,
        config.data_format, split_name=config.split, target='train', mix='2080')
    data_loader_test = get_loader(
        data_path, config.batch_size, config.scale_size,
        config.data_format, split_name=config.split, target='test')

    trainer = Trainer(config, data_loader_user, data_loader_train, data_loader_test)

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
