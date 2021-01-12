from setting import STORAGE

from base.LoadSet import LoadSet
from custom_example.read import read

A = ['a1', 'a2']
B = ['b1', 'b2']

param_names = ['a', 'b']
param_ranges = [A, B]


def get_load_set(log_param):
    """
    create log file path from log parameters
    """
    a, b = log_param
    log_path = f'{STORAGE}/{a}_{b}.txt'
    return LoadSet( log_path, read )
