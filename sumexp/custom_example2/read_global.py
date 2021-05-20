def read(log_file):
    for line in open(log_file, 'r'):
        line = line.strip()
        if not line:
            continue
        key, value = line.split()
        yield {key: value}
