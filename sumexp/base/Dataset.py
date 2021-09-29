import os
import copy
import pickle
from importlib import import_module
from collections.abc import Iterable

import pandas
import attrdict

from setting import CUSTOM_SCR
from base import setup_logger
from base.Plots import lineplot, scatterplot, histplot


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
            self.datas = list(datas)
            for data in self.datas:
                self._keys.update(set(data.keys()))


    def setParam(self, param):
        """
        Parameters
        ----------
        param : Param
        """
        self.param = param


    def iterItems(self, item_or_items, remove_none=True):
        """iterator of item

        Parameters
        ----------
        item_or_items : iterator of str
            item name
        remove_none : bool
            if it is true, then any data that has none is not yield

        Yield
        -----
        item of each data
        """
        if isinstance(item_or_items, (list, tuple)):
            iter_type = type(item_or_items)
            items = item_or_items
        else:
            iter_type = lambda x: x[0]
            items = [item_or_items]

        for data in self:
            if remove_none and any( item not in data or data[item] is None for item in items ):
                continue
            yield iter_type([ data[item] for item in items ])


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
        key: str or function that argument is data
        """
        assert isinstance(key, str) or callable(key)
        if isinstance(key, str):
            self.datas.sort(key=lambda data: data[key])
        elif callable(key):
            self.datas.sort(key=key)
        return self


    def min(self, item):
        """get minimum value of item
        """
        is_in = lambda item, data: item in data and data[item] is not None
        return min( data[item] for data in self if is_in(item, data) )


    def max(self, item):
        """get maximum value of item
        """
        is_in = lambda item, data: item in data and data[item] is not None
        return max( data[item] for data in self if is_in(item, data) )


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


    def lineplot(self,
            xitem,
            yitem,
            custom_operator_x=lambda x: x,
            custom_operator_y=lambda y: y,
            ci=None,
            ax=None,
            *args, **kwargs
            ):
        """line plot

        See Also
        --------
        lineplot of Plots.py
        """
        return lineplot(
            self, xitem, yitem,
            custom_operator_x, custom_operator_y, ci,
            ax, *args, **kwargs)


    def scatterplot(self, xitem, yitem,
            custom_operator_x=lambda x: x,
            custom_operator_y=lambda y: y,
            ax=None,
            *args, **kwargs
            ):
        """scatter plot

        See Also
        --------
        scatterplot in Plots.py
        """
        return scatterplot(
            self, xitem, yitem,
            custom_operator_x, custom_operator_y,
            ax, *args, **kwargs)


    def histplot(self, item, ax=None, *args, **kwargs):
        """create histgram

        See Also
        --------
        histplot in Plots.py
        """
        return histplot(self, item, ax, *args, **kwargs)


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


    def toDataFrame(self, columns=None, param=True):
        """
        Parameters
        ----------
        columns : None or list of str
        param : bool
            if it is true, then param columns is added

        Returns
        -------
        pandas.core.frame.DataFrame
        """
        dataframe = pandas.DataFrame(self.datas)
        if columns is not None:
            try:
                dataframe = dataframe[columns]
            except:
                import traceback
                traceback.print_exc()
                logger.error(f'param = {self.param}, load_set = {self.load_set}')
                exit(1)

        if param and self.param is not None:
            param_dataframe = None
            for name, value in self.param._asdict().items():
                dataframe[name] = value
                if param_dataframe is None:
                    param_dataframe = dataframe[name].astype(str)
                else:
                    param_dataframe += '_' + dataframe[name].astype(str)
            dataframe['param'] = param_dataframe
        for name, value in self.globals.items():
            dataframe[name] = value
        return dataframe


    def diff(self, item, n=1, m=0, normalize=False, prefix='diff_'):
        """add new column of difference item

        Parameters
        ----------
        item : str
        n : int
            period
        m : int
            period
        normalize : bool
        prefix : str
            new column is named prefix + item
        """
        is_in = lambda item, data: item in data and data[item] is not None
        for data in self.datas[:n]:
            data[f'{prefix}{item}'] = None
        for data in self.datas[-m:]:
            data[f'{prefix}{item}'] = None
        for _data, data, data_ in zip(self.datas, self.datas[n:], self.datas[n+m:]):
            if is_in(item, _data) and is_in(item, data_):
                diff = data_[item] - _data[item]
                if normalize:
                    diff /= (n+m)
            else:
                diff = None
            data[f'{prefix}{item}'] = diff
        self._keys.add(f'{prefix}{item}')
        return self


    def keys(self):
        return self._keys

    def __getitem__(self, item):
        return [ data[item] for data in self ]

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


