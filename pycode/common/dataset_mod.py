import sys
sys.dont_write_bytecode = True

import torch.utils.data as data
from PIL import Image
import numpy as np
import math
import csv

class RNNAttitudeEstimatorDataset(data.DataSet):
    def __init__(self, data_list, transform, phase, index_dict_path, dim_fc_out, timesteps, deg_threshold):
        self.data_list = data_list
        self.transform = transform
        self.phase = phase
        self.index_dict_path = index_dict_path
        self.dim_fc_out = dim_fc_out
        self.timesteps = timesteps
        self.deg_threshold = deg_threshold

        self.index_dict = []

        with open(index_dict_path) as f:
            reader = csv.reader(f)
            for row in reader:
                self.index_dict.append(row)

    def __len__(self):
        return len(self.data_list) - self.timesteps