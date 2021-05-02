# SumExp

Summary of Experiments



## Install

```
git clone https://github.com/nariaki3551/SumExp.git
```



## Usage

see https://github.com/nariaki3551/SumExp/blob/master/sumexp/example.ipynb

### 1. setting

You set *STORAGE* and *CUSTOM_SCR* (Basically, it is not necessary to change).

### 2. edit custom/files

You have to create two file, `log_params.py` and `read_log_file.py`.

#### log_params.py

You define *param_names* and *param_ranges* list, and *get_load_set* function in **log_params.py**

```python
# Example
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
```

The *log_param* argument to the *pack_log_path* function is a tuple of elements in the product of the *param_ranges*. *pack_log_path* function takes *log_param* as input and returns the corresponding data path.

#### read_log_file.py

You define *read* function in **read_log_file.py**. The *read* function takes log file path and yield **dictionary** of data.

```python
# Example
def read(log_file):
    for line in open(log_file, 'r'):
        line = line.strip()
        if not line:
            continue
        time, value = map(float, line.split())
        data_dict = {'time': time, 'value': value}
        yield data_dict
```

### 3. load data from storage

Run `python load_storage.py`  to load data from setting.STORAGE using setting.CUSTOM_SCR.

```
optional arguments:
  -h, --help            show this help message and exit
  --root ROOT           cache directory path
  --update              only load new data and add into pickle data
  -p PROCESSES, --processes PROCESSES
                        number of processes
  --log_level LOG_LEVEL
                        debug: 10, info 20, warning: 30, error 40, critical 50
```

### 4. analysis and plot

see example.ipynb
