
# Import the packages
from bouter import EmbeddedExperiment
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import json

def get_tail_sum(path, goal_sampling_freq = 2400):
    """
    Returns
    - The tail time series
    """
    exp = EmbeddedExperiment(path)
    #Getting the logs:
    beh_log = exp.behavior_log
    t0plus_mdf = beh_log[beh_log.t > 0]
    test = t0plus_mdf.t.to_numpy()
    down_sampling_freq = int(np.floor(len(test)/goal_sampling_freq))
    first_sample = t0plus_mdf.tail_sum[::down_sampling_freq]
    tail_swim_arr = first_sample[:goal_sampling_freq].to_numpy() # Remove any surplus points
    return tail_swim_arr