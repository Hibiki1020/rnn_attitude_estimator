import sys
sys.dont_write_bytecode = True

import torch.utils.data as data
import torch
from PIL import Image
import numpy as np
import math
import csv

class RNNAttitudeEstimatorDataset(data.Dataset):
    def __init__(self, data_list, transform, phase, index_dict_path, dim_fc_out, timesteps, deg_threshold, resize):
        self.data_list = data_list
        self.transform = transform
        self.phase = phase
        self.index_dict_path = index_dict_path
        self.dim_fc_out = dim_fc_out #63
        self.timesteps = timesteps
        self.deg_threshold = deg_threshold #30deg

        self.index_dict = []
        self.dict_len = 0

        self.index_dict.append([-1*int(self.deg_threshold)-1, 0])

        with open(index_dict_path) as f:
            reader = csv.reader(f)
            for row in reader:
                tmp_row = [int(row[0]), int(row[1])+1]
                self.index_dict.append(tmp_row)

        self.index_dict.append([int(self.deg_threshold)+1, int(dim_fc_out)-1])

        self.dict_len = len(self.index_dict)


    def search_index(self, number):
        index = int(1000000000)
        for row in self.index_dict:
            if float(number) == float(row[0]):
                index = int(row[1])
                break
            elif float(number) < float(self.index_dict[0][0]): ##-31度以下は-31度として切り上げ
                index = self.index_dict[0][1]
                break
            elif float(number) > float(self.index_dict[self.dim_fc_out-1][0]): #+31度以上は+31度として切り上げ
                index = self.index_dict[self.dim_fc_out-1][1]
                break
        
        return index

    def float_to_array(self, num_float):
        num_deg = float((num_float/3.141592)*180.0)

        num_upper = 0.0
        num_lower = 0.0

        tmp_deg = float(int(num_deg))
        if tmp_deg < num_deg: # 0 < num_deg
            num_lower = tmp_deg
            num_upper = num_lower + 1.0
        elif num_deg < tmp_deg: # tmp_deg < 0
            num_lower = tmp_deg - 1.0
            num_upper = tmp_deg

        dist_low = math.fabs(num_deg - num_lower)
        dist_high = math.fabs(num_deg - num_upper)

        lower_ind = int(self.search_index(num_lower))
        upper_ind = int(self.search_index(num_upper))

        array = np.zeros(self.dim_fc_out)
        
        if upper_ind == lower_ind:
            array[upper_ind] = 1.0
        else:
            array[lower_ind] = dist_high
            array[upper_ind] = dist_low

        return array

    def __len__(self):
        return len(self.data_list) - self.timesteps

    def __getitem__(self, index):
        count = int(self.data_list[index][0])
        img_path = self.data_list[index][1]
        time = self.data_list[index][2]

        x = float(self.data_list[index][3])
        y = float(self.data_list[index][4])
        z = float(self.data_list[index][5])

        tmp_roll = float(self.data_list[index][6])
        tmp_pitch = float(self.data_list[index][7])

        roll_list = self.float_to_array(tmp_roll)
        pitch_list = self.float_to_array(tmp_pitch)

        img_pil = Image.open(img_path)
        img_pil = img_pil.convert("RGB")

        roll_numpy = np.array(roll_list)
        pitch_numpy = np.array(pitch_list)

        img_trans, roll_trans, pitch_trans = self.transform(img_pil, roll_numpy, pitch_numpy)

        return img_trans, roll_trans, pitch_trans
