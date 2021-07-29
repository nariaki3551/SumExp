from setting import STORAGE
from base.LoadSet import LoadSet
from custom_example2.read_seq import read as read_seq
from custom_example2.read_global import read as read_global

A = ['a1', 'a2']
B = ['b1', 'b2']

param_names = ['a', 'b']
param_ranges = [A, B]


def get_load_set(log_param):
    """
    create log file path from log parameters
    """
    a, b = log_param
    log_seq_path = f'{STORAGE}/{a}_{b}.txt'
    log_global_path = f'{STORAGE}/{a}_{b}_global.txt'
    load_set = LoadSet(
        seq_data=(log_seq_path, read_seq),
        global_data=(log_global_path, read_global)
    )
    return load_set
