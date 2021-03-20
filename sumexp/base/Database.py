import os
import pickle
from math import ceil, floor
from collections import namedtuple
from importlib import import_module

from tqdm import tqdm
from numpy import mean
import matplotlib.pyplot as plt

from base import Dataset, setup_logger, pack_cache_path
from setting import CUSTOM_SCR

custom = import_module(CUSTOM_SCR)
logger = setup_logger(name=__name__)


Param = namedtuple('Param', custom.param_names)


class InteractiveDatas(dict):
    """dataset container loaded interactively
    """
    def __init__(self, root, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root = root

    def __getitem__(self, log_param):
        if log_param not in self:
            cache_path = pack_cache_path(self.root, log_param)
            self[Param(*log_param)] = Dataset().load(cache_path)
        return super().__getitem__(log_param)


class Database:
    def __init__(self, root):
        self.datas  = InteractiveDatas(root)
        self.root   = root
        with open(f'{root}/log_params.pickle', 'rb') as f:
            self.params = pickle.load(f)
        self.iter_wrapper = lambda x: x


    def setTqdm(self):
        """set tqdm wrapper of getitem
        """
        self.iter_wrapper = tqdm


    def unsetTqdm(self):
        """set non-display wrapper of getitem
        """
        self.iter_wrapper = lambda x: x


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


    def setAll(self):
        """load all data
        """
        for param in self.params:
            self.datas[param]


    def free(self):
        """relase memory of all loaded data
        """
        self.datas = InteractiveDatas(self.root)


    def sub(self, **kwargs):
        logger.debug(f'sub params {kwargs}')
        assert len(set(kwargs.keys()) - set(custom.param_names)) == 0,\
            f'invalid param is included in {set(kwargs.keys())}'
        item_list = list()
        for param in custom.param_names:
            if param in kwargs:
                item_list.append(kwargs[param])
            else:
                item_list.append('*')
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
            xlim=None,
            xinterval=1, plot_type='meanplot', linestyle='-',
            color=None, label=None, fig=None, ax=None,
            custom_operator=None):
        """line plot

        Parameters
        ----------
        xitem : str
            item of x-axis
        yitem : str
            item of y-axis
        xlim : tuple of (int, float)
        xinterval : int
            plot interval of x-axis
        plot_type : {'meanplot', 'maxplot', 'minplot'}
            plot type for multiple data
        linestyle : str
        color :
        label : str
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes._subplots.AxesSubplot
        custom_operator : func
            ydata is converted to custome_operator(y)

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

        X, Y = self.getLineplotDaat(xitem, yitem, xlim, xinterval)

        # plot
        funcs = {'meanplot': mean, 'maxplot': max, 'minplot': min}
        pY = list(map(funcs[plot_type], Y))
        if custom_operator is not None:
            pY = custom_operator(pY)
        line = ax.plot(X, pY, linestyle=linestyle, label=label, color=color, linewidth=2)

        return fig, ax


    def getLineplotDaat(self, xitem, yitem, xlim, xinterval=1):
        """
        Parametrs
        ---------
        xitem : str
        yitem : str
        xlim : tuple of (int, float)
        xinterval : int

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
        Xlim = list(range(ceil(min_item-xinterval), floor(max_item+xinterval), int(xinterval)))
        data_generators = set(dataset.dataGenerator(xitem, Xlim) for dataset in self)

        X, Y = list(), list()
        for x in self.iter_wrapper(Xlim):
            y_vals = list()
            for data_generator in data_generators:
                data = data_generator.__next__()
                if data is not None and data[yitem] is not None:
                    y_val = data[yitem]
                    y_vals.append(y_val)
            # if not y_vals:
            if len(y_vals) == len(data_generators):
                X.append(x)
                Y.append(y_vals)
        return X, Y


    def hist(self, item, bins=10, histtype='bar', density=False,
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

        items = list(self.iterItem(item))

        ax.hist(items, bins=bins, histtype=histtype,
                color=color, label=label, density=density)
        return fig, ax


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
        for dataset in self:
            for value in dataset.iterItem(item):
                yield value


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

    def __getitem__(self, item_iter):
        """
        database[paramA, paramB, '*', paramC]
        or database[paramA, paramB, '-', paramC]
        returns Database which has all data
        """
        logger.debug(f'item_iter={item_iter}')
        new_database = Database(self.root)
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

        for log_param in self.iter_wrapper(load_params):
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
        s = f'{"="*8} load datsets : size {len(self)} {"="*20}\n'
        ls = len(s)
        for ix, dataset in enumerate(self):
            s += f'\ndataset {ix}\n'
            dataset_str = '   ' + dataset.__str__()
            dataset_str = dataset_str.replace('\n', '\n\t')
            s += dataset_str+'\n'
        s += '='*(ls-1)
        return s
