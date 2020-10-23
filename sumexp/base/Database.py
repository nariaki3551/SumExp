import os
import pickle
from math import ceil, floor
from importlib import import_module

from tqdm import tqdm
from numpy import mean
import matplotlib.pyplot as plt

from base import Dataset, setup_logger
from setting import CUSTOM_SCR

custom = import_module(CUSTOM_SCR)
logger = setup_logger(name=__name__)


class Database:
    def __init__(self):
        self.datas = dict()

    def constructor(self, log_params, update=False):
        load_log_paths = []
        for log_param in log_params:
            if update and log_param in self.datas:
                logger.debug(f'{log_param} is already loaded')
                continue
            else:
                log_path = custom.pack_log_path(log_param)
                if os.path.exists(log_path):
                    load_log_paths.append((log_param, log_path))
                else:
                    logger.debug(f'{log_path} is not found')
        for log_param, log_path in tqdm(load_log_paths):
            dataset = Dataset(log_path)
            self.datas[log_param] = dataset
            logger.debug(dataset.__str__())
        logger.info(f'size is {len(self)}')

    def save(self, save_pickle_file):
        with open(save_pickle_file, 'wb') as pf:
            pickle.dump(self.datas, file=pf)
        logger.info('save ' + save_pickle_file)
        logger.debug(f'size {len(self)}')

    def load(self, load_pickle_file):
        with open(load_pickle_file, 'rb') as pf:
            self.datas = pickle.load(pf)
        logger.info('load ' + load_pickle_file)
        logger.debug(f'size {len(self)}')

    def sub(self, **kwargs):
        logger.debug(f'sub params {kwargs}')
        item_list = list()
        for param in custom.param_names:
            if param in kwargs:
                item_list.append(kwargs[param])
            else:
                item_list.append('*')
        return self[item_list]


    def get_min_item(self, item):
        return min( min(data[item] for data in dataset) for dataset in self )

    def get_max_item(self, item):
        return max( max(data[item] for data in dataset) for dataset in self )


    def lineplot(self, xitem, yitem,
            x_interval=1, plot_type='meanplot', linestyle='-',
            color=None, label=None, fig=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        X, Y = self.get_lineplot_data(xitem, yitem, x_interval)

        # plot
        funcs = {'meanplot': mean, 'maxplot': max, 'minplot': min}
        if plot_type in {'meanplot', 'maxplot', 'minplot'}:
            pY = list(map(funcs[plot_type], Y))
        line = ax.plot(X, pY, linestyle=linestyle, label=label, color=color, linewidth=2)

        return fig, ax


    def get_lineplot_data(self, xitem, yitem, x_interval=1):
        min_item = self.get_min_item(xitem)
        max_item = self.get_max_item(xitem)
        Xlim = list(range(ceil(min_item-x_interval), floor(max_item+x_interval), int(x_interval)))
        data_generators = set(dataset.data_generator(xitem, Xlim) for dataset in self)

        X, Y = list(), list()
        for x in Xlim:
            y_vals = list()
            for data_generator in data_generators:
                data = data_generator.__next__()
                if data is not None:
                    y_val = data[yitem]
                    y_vals.append(y_val)
            # if not y_vals:
            if len(y_vals) == len(data_generators):
                X.append(x)
                Y.append(y_vals)
        return X, Y

    def iter_item(self, item):
        """iterator of item

        Paramters
        ---------
        item : str
            item name

        Yield
        -----
        item of each data
        """
        for dataset in self:
            for value in dataset.iter_item(item):
                yield value

    def __add__(self, other):
        new_database = Database()
        new_database.datas = self.datas.update(other.datas)
        return new_database

    def __iadd__(self, other):
        self.datas.update(other.datas)
        return self

    def __sub__(self, other):
        new_database = Database()
        new_database.datas \
            = dict(self.datas.items()-other.datas.items())
        return new_database

    def __isub__(self, other):
        self.datas = dict(self.datas.items()-other.datas.items())

    def __getitem__(self, item_iter):
        """
        database[paramA, paramB, '*', paramC]
        or database[paramA, paramB, '-', paramC]
        returns Database which has all data
        """
        new_database = Database()
        fixed_params = dict()
        for ix, item in enumerate(item_iter):
            if item not in {'*', '-', '--'}:
                fixed_params[ix] = item

        for log_param in self.datas.keys():
            for ix, fix_item in fixed_params.items():
                if log_param[ix] != fix_item:
                    break
            else:
                new_database.datas[log_param] = self.datas[log_param]
        logger.debug(f'generate database size {len(new_database)}')
        return new_database

    def __len__(self):
        return len(self.datas)

    def __iter__(self):
        return iter(self.datas.values())

    def __contains__(self, dataset):
        return dataset in self.datas.values()

    def __str__(self):
        s = f'{"="*8} datsets : size {len(self)} {"="*20}\n'
        ls = len(s)
        for ix, dataset in enumerate(self):
            s += f'\ndataset {ix}\n'
            dataset_str = '   ' + dataset.__str__()
            dataset_str = dataset_str.replace('\n', '\n\t')
            s += dataset_str+'\n'
        s += '='*(ls-1)
        return s
