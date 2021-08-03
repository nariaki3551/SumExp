import os
import pickle
from math import ceil, floor
from importlib import import_module
from collections import defaultdict
from collections.abc import Iterable

import pandas
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from setting import CUSTOM_SCR
from base import setup_logger
from base.DatasUtility import InteractiveDatas, load_parallel, Param
from base.Dataset import Dataset

custom = import_module(CUSTOM_SCR)
logger = setup_logger(name=__name__)



class Database:
    """Dataset Manager

    Attributes
    ----------
    datas : InteractiveDatas
    _keys : set of str
        keys includes any data in datas
    root : str
    params :
    iter_wrapper : function
    processes : int
    """
    def __init__(self, root=None, dataset=None):
        self.datas  = InteractiveDatas(root)
        self._keys  = set()
        self.root   = root
        with open(f'{root}/log_params.pickle', 'rb') as f:
            self.params = pickle.load(f)
        self.iter_wrapper = lambda x, *args, **kwargs: x
        self.processes = 1


    def setProcesses(self, processes):
        """
        Parameters
        ----------
        processes : int
            number of processes
        """
        assert isinstance(processes, int)
        self.processes = processes
        return self


    def setTqdm(self):
        """set tqdm wrapper of getitem
        """
        self.iter_wrapper = tqdm
        return self


    def unsetTqdm(self):
        """set non-display wrapper of getitem
        """
        self.iter_wrapper = lambda x, *args, **kwargs: x
        return self


    def toDataset(self):
        """
        Returns
        -------
        Dataset
        """
        if len(self) > 1:
            print('donot output Dataset because this includes multi datasets')
            return None
        return list(self.datas.values())[0]


    def set(self, param=None, **kwargs):
        """load specific data
        """
        sub = self.sub(param=param, **kwargs)   # not used
        return self


    def setAll(self):
        """load all data
        """
        self = self.sub()
        return self


    def free(self):
        """relase memory of all loaded data
        """
        self.datas = InteractiveDatas(self.root)
        return self


    def sub(self, param=None, **kwargs):
        """create sub-Database

        Parameters
        ----------
        param : None or Param
        """
        assert param is None or ( isinstance(param, Param) and not kwargs )
        assert len(set(kwargs.keys()) - set(custom.param_names)) == 0,\
            f'invalid param is included in {set(kwargs.keys())}'
        item_list = ['*'] * len(custom.param_names)
        if param is not None:
            for i, value in enumerate(param):
                item_list[i] = value
        else:
            for i, param in enumerate(custom.param_names):
                if param in kwargs:
                    item_list[i] = kwargs[param]
        return self[item_list]


    def clone(self):
        """clone this Database
        """
        item_list = ['*'] * len(custom.param_names)
        return self[item_list]


    def min(self, item):
        """get minimum value of item
        """
        is_in = lambda item, data: item in data and data[item] is not None
        return min( dataset.min(item) for dataset in self )


    def max(self, item):
        """get maximum value of item
        """
        is_in = lambda item, data: item in data and data[item] is not None
        return max( dataset.max(item) for dataset in self )


    def reduce(self,
            key, items=None,
            reduce_func=np.mean,
            lim=None, num=100,
            overwrap=0.0, extend=True,
            ):
        """Reduce all datasets to create a single dataset

        Parameters
        ----------
        key : str
            reduce key
        items : list of str
            items
        reduce_func : func
            argument is list of values -> value
        lim : limit of key
            e.g.) (-1, 4)
        num : None or int
            split number of key
        overwrap : float
            plot only if at least a percentage x of the dataset holds the x-data
        extend : bool
            if it is true, extend and describe the data at the end of the x-axis for each dataset
        """
        assert callable(reduce_func)
        assert 0.0 <= overwrap <= 1.0
        assert lim is None or len(lim) == 2

        if lim is None:
            min_key = self.min(key)
            max_key = self.max(key) + 1e-5
        else:
            min_key = lim[0]
            max_key = lim[1] + 1e-5
        if items is None:
            items = self.keys() - {key}
        key_vals = np.linspace(min_key, max_key, num)
        data_dict = { key_val: {key: key_val} for key_val in key_vals[:-1] }

        is_in = lambda item, data: data is not None and item in data and data[item] is not None

        for item in items:
            data_generators = [
                dataset.sort(key=key).dataGenerator(key, key_vals, extend)
                for dataset in self ]
            for key_min, key_max in zip(key_vals, key_vals[1:]):
                values = list()
                for data_generator in data_generators:
                    data = next(data_generator)
                    if is_in(item, data):
                        values.append(data[item])
                if len(values) >= overwrap * len(data_generators):
                    data_dict[key_min][item] = reduce_func(values)
                else:
                    del data_dict[key_min]

        return Dataset( datas=data_dict.values() ).sort(key=key)


    def lineplot(self, xitem, yitem,
            xlim=None, xnum=100,
            custom_operator_x=lambda x: x,
            custom_operator_y=lambda y: y,
            reduce_func=np.mean,
            overwrap=0.0,
            extend=True,
            data=False,
            ax=None,
            *args, **kwargs
            ):
        """line plot

        Parameters
        ----------
        xitem : str
            item of x-axis
        yitem : str
            item of y-axis
        xlim : tuple of int or float
            limit of x-axis
        xnum : int
            plot interval partition of x-axis
        custom_operator_x : func
            xdata is converted to custome_operator(x)
        custom_operator_y : func
            ydata is converted to custome_operator(y)
        reduce_func : func
            argument is list of values -> value
        overwrap : float
            plot only if at least a percentage x of the dataset holds the x-data
        extend : bool
            if it is true, extend and describe the data at the end of the x-axis for each dataset
        data : bool
            return plot data, too
        ax : matplotlib.axes._subplots.AxesSubplot
        other arguments for matplotlib.plot e.g. linestyle, color, label, linewidth

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
        """
        if ax is None:
            fig, ax = plt.subplots()

        dataset = self.reduce(
            xitem, [yitem], reduce_func,
            xlim, xnum, overwrap, extend )
        X = custom_operator_x( dataset[xitem] )
        Y = custom_operator_y( dataset[yitem] )
        line = ax.plot(X, Y, *args, **kwargs)

        if data:
            return ax, line
        else:
            return ax


    def scatterplot(self, xitem, yitem,
            custom_operator_x=lambda x: x,
            custom_operator_y=lambda y: y,
            ax=None,
            data=False,
            *args, **kwargs
            ):
        """scatter plot

        Parameters
        ----------
        xitem : str
            item of x-axis
        yitem : str
            item of y-axis
        custom_operator_x : func
            xdata is converted to custome_operator(x)
        custom_operator_y : func
            ydata is converted to custome_operator(y)
        data : bool
            return plot data, too
        ax : matplotlib.axes._subplots.AxesSubplot

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
        """
        if ax is None:
            fig, ax = plt.subplots()

        X, Y = list(), list()
        for x, y in self.iterItems([xitem, yitem]):
            X.append(x); Y.append(y)
        X = custom_operator_x(X)
        Y = custom_operator_y(Y)
        paths = ax.scatter(X, Y, *args, **kwargs)

        if data:
            return ax, paths
        else:
            return ax


    def histplot(self, item, ax=None, *args, **kwargs):
        """create histgram

        Parameters
        ----------
        item : str
            item name
        ax : matplotlib.axes._subplots.AxesSubplot

        See Also
        --------
        https://matplotlib.org/3.3.1/api/_as_gen/matplotlib.pyplot.hist.html
        """
        if ax is None:
            fig, ax = plt.subplots()

        items = list(self.iterItems(item))
        ax.hist(items, *args, **kwargs)
        return ax


    def iterItems(self, item_or_items, remove_none=True):
        """iterator of items

        Parameters
        ----------
        item : str or list of str
            item name(s)

        Yield
        -----
        item(s) of each data
        """
        if isinstance(item_or_items, (list, tuple)):
            iter_type = type(item_or_items)
        else:
            item_or_items = [item_or_items]
            iter_type = lambda x: x[0]

        for dataset in self:
            for value in dataset.iterItems(item_or_items, remove_none):
                yield iter_type(value)


    def getLoadedParams(self):
        """
        Returns
        -------
        list of Param
        """
        return list(self.datas)


    def toDataFrame(self):
        """
        Returns
        -------
        pandas.core.frame.DataFrame
        """
        dataframe = None
        for dataset in self:
            if df is None:
                dataframe = dataset.toDataFrame()
            else:
                dataframe = pandas.concat([df, dataset.toDataFrame()])
        return dataframe


    def diff(self, item, n=1, prefix='diff_'):
        """add new column of difference item

        Parameters
        ----------
        item : str
        n : int
            period
        prefix : str
            new column is named prefix + item
        """
        for dataset in self:
            dataset.diff(item, n, prefix)
        return self


    def keys(self):
        return self._keys


    def __add__(self, other):
        new_database = Database(self.root)
        new_database.datas = self.datas.update(other.datas)
        new_database.params = self.datas.params | other.datas.params
        return new_database

    def __iadd__(self, other):
        self.datas.update(other.datas)
        self.params |= other.params
        return self

    def __sub__(self, other):
        new_database = Database(self.root)
        new_database.datas \
            = dict(self.datas.items()-other.datas.items())
        new_database.params = self.params - other.params
        return new_database

    def __isub__(self, other):
        self.datas = dict(self.datas.items()-other.datas.items())
        self.params -= other.params
        return self

    def __getitem__(self, item_iter):
        """
        Parameters
        ----------
        iterm_iter : list of parameters or Param

        Returns
        -------
        Dataset or Database
        if item_iter is Param, then return Dataset whose param is Param.
        if item_iter is list of parameters, then return Database which has all data

        Notes
        -----
        database[paramA, paramB, '*', paramC]
        or database[paramA, paramB, '-', paramC]
        """
        logger.debug(f'item_iter={item_iter}')

        if isinstance(item_iter, Param):
            return self.datas[item_iter]

        # get parameters will be loaded
        fixed_params = dict()
        for ix, item in enumerate(item_iter):
            if item not in {'*', '-', '--', None}:
                fixed_params[ix] = item

        logger.debug(f'fixed_params={fixed_params}')
        load_params = list()
        for log_param in self.params:
            for ix, fix_item in fixed_params.items():
                if log_param[ix] != fix_item:
                    break
            else:
                load_params.append(log_param)

        # load with self.processes
        loaded_params = [
            log_param for log_param in load_params
            if log_param in self.datas
        ]
        not_loaded_params = [
            log_param for log_param in load_params
            if log_param not in self.datas
        ]
        if not_loaded_params:
            load_data_info = load_parallel(
                self.root,
                not_loaded_params,
                self.processes,
                self.iter_wrapper,
            )

        # create new database
        new_database = Database(self.root)
        for log_param in loaded_params:
            new_database.datas[log_param] = self.datas[log_param]
            new_database._keys.update(self.datas[log_param].keys())
        if not_loaded_params:
            for log_param, dataset in load_data_info:
                self.datas.addDataset(log_param, dataset)
                self._keys.update(dataset.keys())
                new_database.datas[log_param] = self.datas[log_param]
        new_database.params = set(new_database.datas.keys())
        logger.debug(f'generate database size {len(new_database)}')
        return new_database

    def __len__(self):
        return len(self.datas)

    def __iter__(self):
        return iter(self.datas.values())

    def __contains__(self, dataset):
        return dataset in self.datas.values()


    def __str__(self):
        s = list()
        for dataset in self:
            s_ = dataset.param._asdict()
            s_['size'] = len(dataset)
            s_['seq_data'] = dataset.load_set.seq_data_file()
            s_['global_data'] = dataset.load_set.global_data_file()
            s.append(s_)
        return pandas.DataFrame(s).__repr__()
