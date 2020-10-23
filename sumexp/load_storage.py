import os
from argparse import ArgumentParser
import pickle
from itertools import product
from importlib import import_module
from multiprocessing import Pool

from tqdm import tqdm

from base import *
from setting import STORAGE, CUSTOM_SCR

custom = import_module(CUSTOM_SCR)


def save_database(root, update, threads):
    log_params = list(product(*custom.param_ranges))

    # gather load logs
    load_logs = []
    for log_param in log_params:
        cache_path = pack_cache_path(root, log_param)
        if update and os.path.isfile(cache_path):
            logger.debug(f'{log_param} is already loaded')
            continue
        else:
            log_path = custom.pack_log_path(log_param)
            if os.path.exists(log_path):
                load_logs.append((log_param, log_path, cache_path))
            else:
                logger.debug(f'{log_path} is not found')

    # save all dataset as pickle
    if threads == 1:
        for load_log in tqdm(load_logs):
            save_dataset(load_log)
    else:
        with Pool(processes=threads) as pool:
            imap = pool.imap(save_dataset, load_logs)
            list(tqdm(imap, total=len(load_logs)))
    logger.info(f'size is {len(load_logs)}')

    # save log_params as pickle
    with open(f'{root}/log_params.pickle', 'wb') as f:
        load_log_params = set(log_param for log_param, _, _ in load_logs)
        pickle.dump(load_log_params, file=f)


def save_dataset(load_log):
    _, log_path, cache_path = load_log
    dataset = Dataset(log_path)
    dataset.save(cache_path)
    logger.debug(dataset.__str__())


def argparser():
    parser = ArgumentParser()
    parser.add_argument(
        '--root',
        default=f'{STORAGE}/cache',
        help='cache directory path'
    )
    parser.add_argument(
        '--update',
        action='store_true',
        help='only load new data and add into pickle data'
    )
    parser.add_argument(
        '-t', '--threads',
        type=int,
        default=1,
        help='number of threads'
    )
    parser.add_argument(
        '--log_level',
        type=int,
        default=10,
        help='debug: 10, info 20, warning: 30, error 40, critical 50'
    )
    return parser


if __name__ == '__main__':
    parser = argparser()
    args = parser.parse_args()

    log_level = args.log_level
    set_log_level(log_level)

    logger = setup_logger(name=__name__)

    save_database(args.root, args.update, args.threads)
