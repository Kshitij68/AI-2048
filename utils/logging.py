import logging

__author__ = 'Kshitij Mathur (kshitij@gmail.com)'

NAME = __name__
FORMAT = '%(asctime)s %(name)s %(levelname)s: %(message)s'
DEFAULT_LEVEL = logging.INFO


def get_logger(name=NAME):
    """Return a logger with appropriate settings from config"""
    logging.basicConfig(format=FORMAT, level=DEFAULT_LEVEL, datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(name)
    return logger
