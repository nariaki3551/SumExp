import os
import copy
import pickle
from importlib import import_module
from collections.abc import Iterable

import attrdict
import pandas as pd

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
    """Data Manager

    Parameters
    ----------
    load_set : LoadSet or None
        log file path and load function
    datas : list of data
        data list

    Attributes
    ----------
    load_set : LoadSet
    datas : list of attrdict.AttrDict
    _keys : set of str
        keys includes any data in datas
    globals : list of dictionary
    param: Param
    """
    def __init__(self, load_set=None, datas=None):
        assert load_set is None or datas is None
        self.load_set = load_set
        self.datas = list()
        self._keys = set()
        self.globals = attrdict.AttrDict()
        self.param = None

        if load_set is not None:
            for data_dict in load_set.read_seq():
                self.datas.append(attrdict.AttrDict(data_dict))
                self._keys.update(set(data_dict.keys()))
            for data_dict in load_set.read_global():
                self.globals.update(data_dict)

        if datas is not None:
            self.datas = datas
            for data in self.datas:
                self.keys.update(set(data.keys()))


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


    def dataGenerator(self, item, items, extend):
        """
        generate D = (None, None, ..., d1, d2, d3, ..., dN, ..., dN)
        such that Di[item] > items[i]
        where self data is (d1, ..., dN) and length of D is equal to one of items

        Parameters
        ----------
        item : str
            item name
        itmes : list of (int or float)
            list of values of item
        extend : bool
            if it is true, extend and describe the data at the end of the item

        Yield
        -----
        dict

        """
        item_ix = 0
        pre_data = None
        for data in self:
            while data[item] > items[item_ix]:
                yield pre_data
                item_ix += 1
            pre_data = data
        yield self.datas[-1]

        while True:
            yield self.datas[-1] if extend else None


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


    def toDataFrame(self):
        """
        Returns
        -------
        pandas.core.frame.DataFrame
        """
        df = pd.DataFrame(self.datas)
        if self.param is not None:
            for name, value in self.param._asdict().items():
                df[name] = value
        return df


    def keys(self):
        return self._keys

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


