import logging
from colorlog import ColoredFormatter

def get_root_logger():
    return logging.getLogger()

def set_log_level(log_level):
    from base import logger
    logger.setLevel(log_level)

def setup_logger(name):
    """Return a logger with a default ColoredFormatter."""
    formatter = ColoredFormatter(
        "%(log_color)s%(levelname)s:%(filename)s:%(funcName)s:%(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red',
        }
    )

    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger