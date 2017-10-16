import logging
import os
from logging.handlers import RotatingFileHandler

SERVER_PATH = os.path.abspath(os.path.dirname(__file__))
LOG_PATH = os.path.join(SERVER_PATH, 'logs')

if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)


def set_logger_options(logger, file_name):
    logger.setLevel(logging.DEBUG)

    path = os.path.join(LOG_PATH, "{}_debug.log".format(file_name))
    fh = logging.handlers.RotatingFileHandler(path, maxBytes=1024 * 1024 * 50, backupCount=100)
    fh.setLevel(logging.DEBUG)

    path = os.path.join(LOG_PATH, "{}_error.log".format(file_name))
    fhe = logging.handlers.RotatingFileHandler(path, maxBytes=1024 * 1024 * 50, backupCount=100)
    fhe.setLevel(logging.ERROR)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt='[%(levelname)s:%(name)s %(asctime)s (%(filename)s:%(lineno)d)] %(message)s ',
        datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(formatter)
    fhe.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(fhe)
    logger.addHandler(ch)


def set_default_options(logger):
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt='[%(levelname)s:%(name)s %(asctime)s (%(filename)s:%(lineno)d)] %(message)s ',
        datefmt="%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)

    logger.addHandler(ch)


def set_default_logger(logger):
    logging.root = logger

