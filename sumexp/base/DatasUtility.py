from itertools import starmap
from multiprocessing import Pool
from collections import namedtuple
from importlib import import_module

from base import Dataset, setup_logger, pack_cache_path
from setting import CUSTOM_SCR

custom = import_module(CUSTOM_SCR)
Param = namedtuple('Param', custom.param_names)


class InteractiveDatas(dict):
    """dataset container loaded interactively
    """
    def __init__(self, root, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root = root


    def addDataset(self, log_param, dataset):
        """
        Parameters
        ----------
        log_param : Param
        dataset : Dataset
        """
        self[log_param] = dataset


    def __getitem__(self, log_param):
        """
        Parameters
        ----------
        log_param : Param

        Returns
        -------
        Dataset
        """
        if log_param not in self:
            _, dataset = load(self.root, log_param)
            dataset.setParam(log_param)
            self[log_param] = dataset
        return super().__getitem__(log_param)


def load(root, log_param):
    """load dataset

    Parameters
    ----------
    root : int
        directory which has cache files
    log_param : Param

    Returns
    -------
    log_param : Param
    dataset : Dataset
    """
    cache_path = pack_cache_path(root, log_param)
    dataset = Dataset().load(cache_path)
    return log_param, dataset


def _load(arg):
    """load dataset

    Parameters
    ----------
    arg : tuple
        (root, log_param)

    Returns
    -------
    log_param : Param
    dataset : Dataset
    """
    return load(*arg)


def load_parallel(root, load_params, processes,
        iter_wrapper=lambda x, *args, **kwargs: x):
    """load datasets with parallel processing

    Parameters
    ----------
    root : int
        directory which has cache files
    log_param : Param
    processes : int
    iter_wrapper : func

    Returns
    -------
    list of tuple
        (log_param, dataset)
    """
    assert isinstance(processes, int) and processes >= 1
    args = [(root, log_param) for log_param in load_params]
    if processes == 1:
        return list(iter_wrapper(starmap(load, args), total=len(args)))
    else:
        with Pool(processes=processes) as pool:
           result = list(
                iter_wrapper(pool.imap_unordered(_load, args), total=len(args))
            )
        return result


