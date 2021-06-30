import os
import pickle
from importlib import import_module

import attrdict

from setting import CUSTOM_SCR
from base import setup_logger

custom = import_module(CUSTOM_SCR)
logger = setup_logger(name=__name__)


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


    def iterItem(self, item):
        """iterator of item

        Parameters
        ----------
        item : str
            item name

        Yield
        -----
        item of each data
        """
        for data in self:
            if item in data:
                yield data[item]


    def dataGenerator(self, item, items):
        """data loader

        generate D = (None, None, ..., d1, d2, d3, ..., dN, ..., dN)
        such that Di[item] > items[i]
        where self data is (d1, ..., dN) and length of D is equal to one of items

        Parameters
        ----------
        item : str
            item name
        itmes : list of (int or float)
            list of values of item

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
        while True:
            yield self.datas[-1]


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


