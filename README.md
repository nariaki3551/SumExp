# SumExp

Summary of Experiments



## Install

```
git clone https://github.com/nariaki3551/SumExp.git
```



## Usage

### 1. setting

You set *STORAGE* and *CUSTOM_SCR* (Basically, it is not necessary to change).

### 2. edit custom/files

You have to create two file, `log_params.py` and `read_log_file.py`.

#### log_params.py

You define *param_names* and *param_ranges* list, and *pack_log_path* function in **log_params.py**

```python
# Example
A = ['a1', 'a2']
B = ['b1', 'b2']
param_names = ['a', 'b']
param_ranges = [A, B]

def pack_log_path(log_param):
    a, b = log_param
    return f'{STORAGR}/{a}_{b}.txt'
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
  --path PATH           pickle file path dumpled of database default is
                        ../storage_example/database.pickle
  --update              whether update pickle file
  --log_level LOG_LEVEL
                        debug: 10, info 20, warning: 30, error 40, critical 50
```

### 4. analysis and plot

```python
from base import *

# load datas
data_path = '../storage/database.pickle'
database = Database()
database.load(data_path)

# extruct data
sub_database = database.sub(paramA=valueA, ..., paramZ= valueZ)

# line plot
sub_database.lineplot(
    xitem='time',
    yitem='value'
)
```



## Example

You edit `setting.py` as following. 

```python
STORAGE = '../storage_example'
CUSTOM_SCR = 'custom_example'
```

Then, you run

```bash
python load_storage.py
```

As the result, `database.pickle` file is created in `../storate_example/`.

You can plot data by following code.

#### load database


```python
from base import *

data_path = '../storage_example/database.pickle'
database = Database()
database.load(data_path)

print(database)
```

    ======== datsets : size 4 ====================
    
    dataset 0
       log_path ../storage_example/a1_b1.txt
    	size     4
    	
    dataset 1
       log_path ../storage_example/a1_b2.txt
    	size     3
    	
    dataset 2
       log_path ../storage_example/a2_b1.txt
    	size     2
    	
    dataset 3
       log_path ../storage_example/a2_b2.txt
    	size     3
    	
    ==============================================


​    
#### extruct database and plot


```python
sub_database = database.sub(a='a1', b='b1')
print(sub_database)
```


    ======== datsets : size 1 ====================
    
    dataset 0
       log_path ../storage_example/a1_b1.txt
    	size     4
    	
    ==============================================

```python
sub_database.lineplot(
    xitem='time',
    yitem='value'
)
```

#### plot 2


```python
sub_database = database.sub(a='a1')
print(sub_database)
```


```python
sub_database.lineplot(
    xitem='time',
    yitem='value'
)`
```
