# -*- coding: utf-8 -*-
import os, logging


__all__ = ['LABEL_FILE', 'PACKAGE_DIR', 'DATA_DIR']


PACKAGE_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), '')
DATA_DIR = os.path.join(PACKAGE_DIR, 'data')
LABEL_FILE = os.path.join(DATA_DIR, 'labels.txt')

#logging
logger = logging.getLogger('app')
fh = logging.FileHandler(os.path.join(PACKAGE_DIR, 'application.log'))
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(message)s', '%Y-%m-%d %H:%M:%S')
fh.setFormatter(formatter)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)