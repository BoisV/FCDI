from functools import cmp_to_key
import glob
import os
import logging
import re
import datetime
import torch


def get_logger(verbosity=1, name=None):
    str_date = datetime.datetime.now().strftime('%Y-%m-%d')
    log_path = os.path.join(os.getcwd(), 'log')
    log_name = os.path.join(log_path, (str_date+'.log'))
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(log_name):
        open(log_name, mode='w+')

    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(log_name, "a+")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def findLastCheckpoint(root):
    if not os.path.exists(root):
        os.mkdir(root)
        return 0, None

    models = os.listdir(root)

    if len(models) == 0:
        return 0, None

    models.sort()
    initial_epoch = 0
    initial_epoch = max(initial_epoch,  int(models[-1][-7:-4]))
    state_dict = torch.load(os.path.join(root, models[-1]))
    return initial_epoch+1, state_dict


if __name__ == "__main__":
    findLastCheckpoint(root='../models')
