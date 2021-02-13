import os
import datetime

class LoadSet:
    def __init__(self, log_path, read_func):
        self.log_path = log_path
        self.read_func = read_func


    def read(self):
        return self.read_func(self.log_path)


    def getLogPath(self):
        return self.log_path


    def getMtimeDatetime(self):
        mtime = os.path.getmtime(self.log_path)
        return datetime.datetime.fromtimestamp(mtime)


    def __str__(self):
        s = f'LoadSet{self.log_path, self.read_func.__name__}'
        return s

