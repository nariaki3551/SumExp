import os
from collections import namedtuple

LoadSetElement = namedtuple("LoadSetElement", "file read_func")


class LoadSet:
    """Manage of (logfile, read_function)

    Parameters
    ----------
    seq_data : None or tuple( str, callable )
        sequential data load set,
        tuple(logfile, readfunc) and readfunc(logfile) is available
    global_data : None or tuple( str, callable )
        global data load set,
        tuple(logfile, readfunc) and readfunc(logfile) is available
    """

    def __init__(self, seq_data=None, global_data=None):
        test(seq_data, "seq_data")
        test(global_data, "global_data")
        self.seq_data = None
        self.global_data = None
        if seq_data is not None:
            self.seq_data = LoadSetElement(*seq_data)
        if global_data is not None:
            self.global_data = LoadSetElement(*global_data)

    def read_seq(self):
        """read sequential data"""
        if self.readable_seq():
            logfile, read_func = self.seq_data
            return read_func(logfile)
        else:
            return iter([])

    def read_global(self):
        """read global data"""
        if self.readable_global():
            logfile, read_func = self.global_data
            return read_func(logfile)
        else:
            return iter([])

    def readable_seq(self):
        """
        Returns
        -------
        bool
            return true when both sequential data file is exist
        """
        return self.seq_data is not None and os.path.exists(self.seq_data.file)

    def readable_global(self):
        """
        Returns
        -------
        bool
            return true when both global data file is exist
        """
        return self.global_data is not None and os.path.exists(self.global_data.file)

    def readable(self):
        """
        Returns
        -------
        bool
            return true when both sequential or globabl
            data file is exist
        """
        return self.readable_seq() or self.readable_global()

    def seq_data_file(self):
        """
        Returns
        -------
        str or None
        """
        if self.seq_data is None:
            return None
        else:
            return self.seq_data.file

    def global_data_file(self):
        """
        Returns
        -------
        str or None
        """
        if self.global_data is None:
            return None
        else:
            return self.global_data.file

    def __str__(self):
        s = f"LoadSet({self.seq_data}, {self.global_data})"
        return s


def test(data, name):
    """test data format

    Parameters
    ----------
    data : None or tuple( str, callable )
        sequential data load set,
        tuple(logfile, readfunc) and readfunc(logfile) is available
    name : str
        for assert message
    """
    assert data is None or (
        isinstance(data, (tuple, list))
        and len(data) == 2
        and isinstance(data[0], str)
        and callable(data[1])
    ), f"{name} has invalid format"
