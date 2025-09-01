from Config.config import CONFIG
CONFIG = CONFIG("MOOCTGN")

from DyGLib.train_link_prediction import train
from DyGLib.utils.DataLoader import get_link_prediction_data
import itertools
import os
from subprocess import Popen, PIPE
import copy
from ray import tune
import ray 
import numpy as np
import signal
from contextlib import contextmanager
import time
from multiprocessing import Process, Queue
from tqdm import tqdm

if __name__ == '__main__':
    data = get_link_prediction_data(val_ratio=CONFIG.train.val_ratio, 
                            test_ratio=CONFIG.train.test_ratio, 
                            node_dim=CONFIG.model.node_dim)

    result = train(CONFIG.model, CONFIG.data, CONFIG.train, *data)