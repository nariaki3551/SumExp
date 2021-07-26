import os
import copy
import pickle
from importlib import import_module
from collections.abc import Iterable

import attrdict

from setting import CUSTOM_SCR
from base import setup_logger

logger = setup_logger(name=__name__)
try:
    custom = import_module(CUSTOM_SCR)
except ModuleNotFoundError:
    message = 'check setting.py'
    logger.error(message)
    exit(1)


class Dataset:
    """
    Data Manager

    Parameters
    ----------
    load_set : LoadSet or None
        log file path and load function
    datas : list of data
        data list
    """
    def __init__(self, load_set=None, datas=None):
        assert load_set is None or datas is None
        self.load_set = load_set
        self.datas = list()
        self.globals = attrdict.AttrDict()
        self.param = None

        if load_set is not None:
            for data_dict in load_set.read_seq():
                self.datas.append(attrdict.AttrDict(data_dict))
            for data_dict in load_set.read_global():
                self.globals.update(data_dict)

        if datas is not None:
            self.datas = datas


    def setParam(self, param):
        """
        Parameters
        ----------
        param : Param
        """
        self.param = param


    def iterItems(self, items, remove_none=True):
        """iterator of item

        Parameters
        ----------
        item : iterator of str
            item name
        remove_none : bool
            if it is true, then any data that has none is not yield

        Yield
        -----
        item of each data
        """
        assert isinstance(items, Iterable)
        for data in self:
            if remove_none and any( item not in data or data[item] is None for item in items ):
                continue
            yield list(data[item] for item in items)


    def clone(self):
        """clone this dataset

        Returns
        -------
        Dataset
        """
        datas = copy.deepcopy(self.datas)
        return Dataset(datas=datas)


    def sort(self, key):
        """sort datas

        Parameters
        ----------
        key: function that argument is data
        """
        self.datas.sort(key=key)
        return self


    def save(self, cache_path):
        directory = os.path.dirname(cache_path)
        os.makedirs(directory, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(self, file=f)


    def load(self, cache_path):
        with open(cache_path, 'rb') as f:
            try:
                self = pickle.load(f)
            except ModuleNotFoundError as e:
                import sys
                sys.exit(
                    f'{e} -- '\
                    +f'You may need to create a dummy module'\
                    +f'that cannot be found.'
                )
            except Exception as e:
                logger.error(e)
        return self


    def __eq__(self, other):
        return self.log_path == other.log_path

    def __hash__(self):
        return hash(self.log_path)

    def __len__(self):
        return len(self.datas)

    def __iter__(self):
        return iter(self.datas)

    def __str__(self):
        if self.datas:
            s  = f'load_set {self.load_set}\n'
            s += f'size     {len(self.datas)}\n'
            return s
        else:
            return 'empty dataset'

    def __repr__(self):
        return f'Dataset("{self.load_set}")'


