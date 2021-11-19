import collections
import importlib
import itertools
import multiprocessing
import os

from setting import CUSTOM_SCR

from base import Dataset, setup_logger

custom = importlib.import_module(CUSTOM_SCR)
Param = collections.namedtuple("Param", custom.param_names)


class ParamSet(set):
    """Param container"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def unique(self, attr):
        """return unique value of attr"""
        return sorted(list(set(getattr(param, attr) for param in self)))


def pack_cache_path(root, log_param):
    """generate cache file path

    Parameters
    ----------
    root : str
        cache directory
    log_param : Param

    Returns
    -------
    cache_path : str
        cache fiel path of log_param
    """
    str_log_param = "_".join(map(str, log_param))
    cache_path = f"{root}/{str_log_param}.pickle"
    return cache_path


class InteractiveDatas(dict):
    """dataset container loaded interactively"""

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
        dataset.setParam(log_param)
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


def load_parallel(
        root, load_params, processes, iter_wrapper=lambda x, *args, **kwargs: x, chunksize=1
        ):
    """load datasets with parallel processing

    Parameters
    ----------
    root : int
        directory which has cache files
    log_param : Param
    processes : int
    iter_wrapper : func
    chunksize: chunksize of imap_unordered

    Returns
    -------
    list of tuple
        (log_param, dataset)
    """
    assert isinstance(processes, int) and processes >= 1
    args = [(root, log_param) for log_param in load_params]
    args = sorted(args, key=lambda x: os.path.getsize(pack_cache_path(*x)), reverse=True)
    if processes == 1:
        return list(iter_wrapper(itertools.starmap(load, args), total=len(args)))
    else:
        with multiprocessing.Pool(processes=processes) as pool:
            result = list(
                iter_wrapper(pool.imap_unordered(_load, args, chunksize=chunksize), total=len(args))
            )
        return result


def tie(params, n=5):
    """tie parmas to regular expressions form

    Parameters
    ----------
    params : set of tuple
    n : int
        number of shuffle

    Note
    ----
    this is developing function
    """
    import copy
    import random

    bestD = None

    for i in range(n):
        D = {tuple((elm,) for elm in param) for param in params}
        n = len(list(D)[0])
        indexes = list(range(n))
        if i > 0:
            random.shuffle(indexes)
        for j in range(n):
            res = tie_last_column(D, indexes)
            D = {tuple(keys[j] for j in range(n)) for keys in res}
            indexes = indexes[1:] + indexes[:1]
        if bestD is None or len(D) < len(bestD):
            bestD = D

    # display
    print(f'#[{"][".join(custom.param_names)}]')
    for param in sorted(list(D)):
        s = ""
        for elm in param:
            if all(isinstance(key, int) for key in elm):
                elm = list(sorted(elm))
                new_elm = []
                while len(elm) > 1:
                    if elm[-1] == elm[0] + len(elm) - 1:
                        new_elm.append(f"{elm[0]}-{elm[-1]}")
                        elm = []
                    elif elm[1] > elm[0] + 1:
                        new_elm.append(elm[0])
                        elm = elm[1:]
                    else:
                        for i in range(len(elm) - 1):
                            if elm[i + 1] > elm[i] + 1:
                                new_elm.append(f"{elm[0]}-{elm[-1]}")
                                elm = elm[i + 1 :]
                                break
                if len(elm) == 1:
                    new_elm.append(elm[0])
                elm = new_elm
            s += f'[{",".join(map(str, elm))}]'
        print(s)


def tie_last_column(D, indexes):
    """tie parmas to regular expressions form

    Parameters
    ----------
    params : set of tuple
    indexes : list of int

    Returns
    -------
    list of dict
        res[i] = dict(key=index, value=tuple-values)
    """
    res = []
    while D:
        _D = D
        keys = dict()
        for i in indexes[:-1]:
            for d in _D:
                key = d[i]
                break
            _D = {d for d in _D if d[i] == key}
            keys[i] = key
        index = indexes[-1]
        keys[index] = tuple(sorted(list({d[index][0] for d in _D})))
        res.append(keys)
        D -= _D
    return res
