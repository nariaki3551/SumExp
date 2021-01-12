from setting import STORAGE

from base.LoadSet import LoadSet
from custom.read import read

# define param_names, param_ranges, pack_log_path
# Example
# -------
# A = ['a1', 'a2']
# B = ['b1', 'b2']
# param_names = ['a', 'b']
# param_ranges = [A, B]
#
# def pack_log_path(log_param):
#     a, b = log_param
#     file_name = f'{STORAGR}/a{a}_b{b}.txt'
#     return  LoadSet( file_name, read )

param_names = []
param_ranges = []

def get_load_set(log_param):
    """
    create log file path from log parameters
    """
    pass
