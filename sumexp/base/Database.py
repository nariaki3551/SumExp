import os
import pickle
from math import ceil, floor
from importlib import import_module
from collections import defaultdict
from collections.abc import Iterable

from tqdm import tqdm
import numpy as np
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
        if param is not None:
            assert isinstance(param, Param) and not kwargs
            for name, value in zip(custom.param_names, param):
                kwargs[name] = value
        logger.debug(f'sub params {kwargs}')
        assert len(set(kwargs.keys()) - set(custom.param_names)) == 0,\
            f'invalid param is included in {set(kwargs.keys())}'
        item_list = list()
        for param in custom.param_names:
            if param in kwargs and kwargs[param] is not None:
                item_list.append(kwargs[param])
            else:
                item_list.append('*')
        return self[item_list]


    def clone(self):
        n_param = len(custom.param_names)
        item_list = ['*'] * n_param
        return self[item_list]


    def getMinItem(self, item):
        return min(
                min(data[item] for data in dataset if data[item] is not None )
                for dataset in self )


    def getMaxItem(self, item):
        return max(
                max(data[item] for data in dataset if data[item] is not None )
                for dataset in self )


    def lineplot(self, xitem, yitem,
            xlim=None, xnum=100,
            overwrap=False,
            custom_operator_x=None,
            custom_operator_y=None,
            plot_type='meanplot',
            data=False,
            fig=None, ax=None,
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
        overwrap: bool
            if it is true, plot x when all dataset has x-data
        custom_operator_x : func
            xdata is converted to custome_operator(x)
        custom_operator_y : func
            ydata is converted to custome_operator(y)
        plot_type : {'meanplot', 'maxplot', 'minplot'}
            plot type for multiple data
        data : bool
            return plot data, too
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes._subplots.AxesSubplot
        other arguments for matplotlib.plot e.g. linestyle, color, label, linewidth

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes._subplots.AxesSubplot
        """
        assert plot_type in {'meanplot', 'maxplot', 'minplot'},\
                f'plot_type must be meanplot, maxplot or minplot, but got {plot_type}'
        assert xlim is None or len(xlim) == 2
        if ax is None:
            fig, ax = plt.subplots()

        X, Y = self.getLineplotData(xitem, yitem, xlim, xnum, overwrap)

        # plot
        funcs = {'meanplot': np.mean, 'maxplot': max, 'minplot': min}
        Y = list(map(funcs[plot_type], Y))
        if custom_operator_x is not None:
            X = custom_operator_x(X)
        if custom_operator_y is not None:
            Y = custom_operator_y(Y)
        line = ax.plot(X, Y, *args, **kwargs)

        if data:
            return fig, ax, line
        else:
            return fig, ax


    def getLineplotData(self, xitem, yitem, xlim=None, xnum=100, overwrap=False):
        """
        Parameters
        ----------
        xitem : str
        yitem : str
        xlim : tuple of int or float
        xnum: int
        overwrap: bool
            if it is true, plot x when all dataset has x-data
        Returns
        -------
        X : list of float
        Y : list of float
        """
        if xlim is None:
            min_item = self.getMinItem(xitem)
            max_item = self.getMaxItem(xitem)
        else:
            min_item = xlim[0]
            max_item = xlim[1]
        Xlim = np.linspace(min_item, max_item, xnum).tolist()
        Xlim.append(min_item-1)

        data_generators = [
            dataset.sort(key=lambda data: data[xitem]).dataGenerator(xitem, Xlim)
            for dataset in self
        ]

        is_in = lambda item, data: item in data and data[item] is not None

        X, Y = list(), list()
        for i in self.iter_wrapper(list(range(xnum-1))):
            x_min, x_max = Xlim[i], Xlim[i+1]
            y_vals = list()
            for data_generator in data_generators:
                data = data_generator.__next__()
                if data is not None and is_in(yitem, data):
                    y_vals.append(data[yitem])
                    if not overwrap or len(y_vals) == len(data_generators):
                         X.append(x_max)
                         Y.append(y_vals)
        return X, Y


    def scatterplot(self, xitem, yitem,
            custom_operator_x=None,
            custom_operator_y=None,
            fig=None, ax=None,
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
        custom_operator_y : func
            ydata is converted to custome_operator(y)
        plot_type : {'meanplot', 'maxplot', 'minplot'}
            plot type for multiple data
        data : bool
            return plot data, too
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes._subplots.AxesSubplot
        other_data for matplotlib.plot e.g. linestyle, color, label, linewidth

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes._subplots.AxesSubplot
        """
        if ax is None:
            fig, ax = plt.subplots()

        X, Y = list(), list()
        for x, y in self.iterItems([xitem, yitem]):
            X.append(x); Y.append(y)

        if custom_operator_x is not None:
            X = custom_operator_x(X)
        if custom_operator_y is not None:
            Y = custom_operator_y(Y)
        paths = ax.scatter(X, Y, *args, **kwargs)

        if data:
            return fig, ax, paths
        else:
            return fig, ax


    def histplot(self, item, bins=10, histtype='bar', density=False,
            color=None, label=None, fig=None, ax=None):
        """create histgram

        Parameters
        ----------
        item : str
            item name
        bins : int or sequence or str
        histtype : {'bar', 'barstacked', 'step', 'stepfilled'}
        density : bool
        color : color or array-like of colors or None
        label : str or None
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes._subplots.AxesSubplot

        See Also
        --------
        https://matplotlib.org/3.3.1/api/_as_gen/matplotlib.pyplot.hist.html
        """
        if ax is None:
            fig, ax = plt.subplots()

        items = list(self.iterItems(item))

        ax.hist(items, bins=bins, histtype=histtype,
                color=color, label=label, density=density)
        return fig, ax


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

        Notes
        -----
        database[paramA, paramB, '*', paramC]
        or database[paramA, paramB, '-', paramC]
        returns Database which has all data
        """
        logger.debug(f'item_iter={item_iter}')

        if isinstance(item_iter, Param):
            return self.datas[item_iter]

        # get parameters will be loaded
        fixed_params = dict()
        for ix, item in enumerate(item_iter):
            if item not in {'*', '-', '--'}:
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


    def keys(self):
        return self._keys

    def __len__(self):
        return len(self.datas)

    def __iter__(self):
        return iter(self.datas.values())

    def __contains__(self, dataset):
        return dataset in self.datas.values()

    def __str__(self):
        s = f'{"="*8} load datsets : size {len(self)} {"="*20}\n'
        ls = len(s)
        for ix, dataset in enumerate(self):
            s += f'\ndataset {ix}\n'
            dataset_str = '   ' + dataset.__str__()
            dataset_str = dataset_str.replace('\n', '\n\t')
            s += dataset_str+'\n'
        s += '='*(ls-1)
        if self.processes > 1:
            s += f'\nprocesses: {self.processes}'
        return s
