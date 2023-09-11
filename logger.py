'''
Description: 日志
Autor: wiki
Date: 2021-04-26 14:19:20
LastEditors: wyh
LastEditTime: 2021-07-17 21:48:17
'''
# -*- encoding:utf-8 -*-
import os
import sys
import datetime
root_path = os.path.abspath(os.path.dirname(__file__))
import logging

def get_logger(name, log_path = None, log_level = 'DEBUG'):
    # create logger
    logger_name = name
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level.upper())
    
    fh = logging.StreamHandler()
    fh.setLevel(log_level.upper())

    # create formatter
    fmt = "[%(asctime)-15s] %(levelname)s %(filename)s %(lineno)d %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt)

    # add handler and formatter to logger
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
logger = get_logger("log", log_level='DEBUG')
if __name__ == "__main__":
    logger = get_logger("MyLog", log_level='DEBUG')
    logger.debug('debug message')
    logger.info('info message')
    logger.warn('warn message')
    logger.error('error message')
    logger.critical('critical message')