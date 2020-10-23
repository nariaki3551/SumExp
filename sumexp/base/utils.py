import hashlib

def pack_cache_path(root, log_param):
    str_log_param = '_'.join(map(str, log_param))
    m = hashlib.md5(str_log_param.encode()).hexdigest()
    cache_path = f'{root}/{m}.pickle'
    return cache_path
