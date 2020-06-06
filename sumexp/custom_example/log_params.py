from setting import STORAGE

A = ['a1', 'a2']
B = ['b1', 'b2']

param_names = ['a', 'b']
param_ranges = [A, B]


def pack_log_path(log_param):
    """
    create log file path from log parameters
    """
    a, b = log_param
    log_path = f'{STORAGE}/{a}_{b}.txt'
    return log_path