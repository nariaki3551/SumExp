import os
import glob
import time
import pickle
from itertools import product
from multiprocessing import Pool
from importlib import import_module
from argparse import ArgumentParser

from tqdm import tqdm

from base import *
from base.DatasUtility import ParamSet
from setting import STORAGE, CUSTOM_SCR

custom = import_module(CUSTOM_SCR)


def save_database(root, update, processes):
    start_time = time.time()
    log_params = list(product(*custom.param_ranges))

    # gather load logs
    load_logs = []
    loaded_log_params = set()
    for log_param in log_params:
        cache_path = pack_cache_path(root, log_param)
        if update and os.path.isfile(cache_path):
            logger.debug(f"{log_param} is already loaded")
            loaded_log_params.add(log_param)
            continue
        else:
            load_set = custom.get_load_set(log_param)
            if load_set is None:
                logger.debug(f"pass log_param = {log_param}")
            elif load_set.readable():
                load_logs.append((log_param, load_set, cache_path))
            else:
                logger.debug(f"{load_set} is not readable")

    # save all dataset as pickle
    if processes == 1:
        for load_set in tqdm(load_logs):
            save_dataset(load_set)
    else:
        with Pool(processes=processes) as pool:
            imap = pool.imap(save_dataset, load_logs)
            list(tqdm(imap, total=len(load_logs)))
    logger.info(f"size is {len(load_logs)}")

    # save log_params as pickle
    with open(f"{root}/log_params.pickle", "wb") as f:
        loaded_log_params |= set(log_param for log_param, _, _ in load_logs)
        loaded_log_params = set(Param(*log_param) for log_param in loaded_log_params)
        pickle.dump(ParamSet(loaded_log_params), file=f)

    logger.info(f"time: {time.time()-start_time:.4f} sec")


def save_dataset(load_log):
    _, log_set, cache_path = load_log
    dataset = Dataset(log_set)
    dataset.save(cache_path)
    logger.debug(dataset.__str__())
    logger.debug(f"dataset save as {cache_path}")


def argparser():
    parser = ArgumentParser()
    parser.add_argument(
        "--root", default=f"{STORAGE}/cache", help="cache directory path"
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="only load new data and add into pickle data",
    )
    parser.add_argument(
        "-p", "--processes", type=int, default=1, help="number of processes"
    )
    parser.add_argument(
        "--log_level",
        type=int,
        default=20,
        help="debug: 10, info 20, warning: 30, error 40, critical 50",
    )
    return parser


if __name__ == "__main__":
    parser = argparser()
    args = parser.parse_args()

    set_log_level(args.log_level)
    logger = setup_logger(name=__name__)

    if not args.update:
        cache_files = glob.glob(f"{args.root}/*.pickle")
        map(os.remove, cache_files)

    logger.info(f"save cache files in {args.root}")
    save_database(args.root, args.update, args.processes)
