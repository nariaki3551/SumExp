def pack_cache_path(root, log_param):
    str_log_param = '_'.join(map(str, log_param))
    cache_path = f'{root}/{str_log_param}.pickle'
    return cache_path
