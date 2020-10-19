from argparse import ArgumentParser
from itertools import product
from importlib import import_module

from base import *
from setting import STORAGE, CUSTOM_SCR

custom_log_params = import_module(".log_params", CUSTOM_SCR)


def save_database(data_path, update):
    database = Database()
    log_params = list(product(*custom_log_params.param_ranges))

    if update:
        database.load(data_path)
    
    database.constructor(
        log_params=log_params,
        update=update,
    )
    database.save(data_path)


def argparser():
    parser = ArgumentParser()
    parser.add_argument(
        '--path',
        default=f'{STORAGE}/database.pickle',
        help=' '.join(('pickle file path dumpled of database',
                      f'default is {STORAGE}/database.pickle'))
    )
    parser.add_argument(
        '--update',
        action='store_true',
        help='only load new data and add into pickle data'
    )
    parser.add_argument(
        '--log_level',
        type=int,
        default=10,
        help='debug: 10, info 20, warning: 30, error 40, critical 50'
    )
    return parser


if __name__ == '__main__':
    parser = argparser()
    args = parser.parse_args()

    log_level = args.log_level
    set_log_level(log_level)

    save_database(args.path, args.update)
