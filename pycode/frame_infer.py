import sys, codecs
from tkinter import W
#sys.stdout = codecs.getwriter("utf-8")(sys.stdout)
sys.dont_write_bytecode = True

from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import random
import csv

import argparse
import subprocess
import datetime
import yaml
from shutil import copyfile
import os
import shutil
import __init__ as booger

import torch
from torchvision import models
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from tensorboardX import SummaryWriter

from common import network_mod

class InferenceMod:
    def __init__(self, CFG):
        self.CFG = CFG
        self.infer_dataset_top_directory = CFG["infer_dataset_top_directory"]
        self.csv_name = CFG["csv_name"]
        self.weights_top_directory = CFG["weights_top_directory"]
        self.weights_file_name = CFG["weights_file_name"]
        self.weights_path = os.path.join(self.weights_top_directory, self.weights_file_name)
        self.infer_log_top_directory = CFG["infer_log_top_directory"]
        self.infer_log_file_name = CFG["infer_log_file_name"]
        self.index_dict_name = CFG["index_dict_name"]

        self.index_dict_path = "../index_dict/" + self.index_dict_name

        #Hyperparameter
        self.resize = int(CFG["resize"])
        self.mean_element = float(CFG["mean_element"])
        self.std_element = float(CFG["std_element"])
        self.dim_fc_out = int(CFG["dim_fc_out"])
        self.enable_dropout = bool(CFG["enable_dropout"])
        self.dropout_rate = float(CFG["dropout_rate"])
        self.timesteps = int(CFG["timesteps"])

        self.img_cv = np.empty(0)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device: ", self.device)

        self.img_transform = self.getImageTransform(self.resize, self.mean_element, self.std_element)

        self.net = self.getNetwork(self.weights_path, self.dim_fc_out, self.dropout_rate, self.timesteps)
        print("Load Network")

        if self.enable_dropout == True:
            print("Enable Dropout")
            self.do_mc_dropout = self.enable_mc_dropout()

        self.value_dict = []

        with open(self.index_dict_path) as fd:
            reader = csv.reader(fd)
            for row in reader:
                num = float(row[0])
                self.value_dict.append(num)

        self.img_data_list , self.ground_truth_list = self.getData()

    def getImageTransform(self, resize, mean_element, std_element):

        size = (resize, resize)

        img_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((mean_element,), (std_element,))
        ])

        return img_transform

    def getNetwork(self, weights_path, dim_fc_out, dropout_rate, timesteps):
        net = network_mod.Network(dim_fc_out, norm_layer=nn.BatchNorm2d, pretrained_model=None, dropout_rate=dropout_rate, timesteps=timesteps)

        print(net)
        print("Import Network")

        net.to(self.device)
        net.eval()

        #load
        if torch.cuda.is_available:
            state_dict = torch.load(weights_path, map_location=lambda storage, loc: storage)
            print("Load .pth file")
        else:
            state_dict = torch.load(weights_path, map_location={"cuda:0": "cpu"})
            print("Load to CPU")

        net.load_state_dict(state_dict)
        return net

    def enable_mc_dropout(self):
        for module in self.net.modules():
            if module.__class__.__name__.startswith('Dropout'):
                module.train()
                can_dropout = True

        return True

    def getData(self):
        img_data_list = []
        gt_list = []

        csv_path = os.path.join(self.infer_dataset_top_directory, self.csv_name)

        with open(csv_path) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                img_path = self.infer_dataset_top_directory + "/camera_image/" + row[1]
                
                count = int(row[0])
                time = float(row[2])

                gt_roll = float(row[6])
                gt_pitch = float(row[7])

                img_data_list.append(img_path)

                tmp_list = [count, time, gt_roll, gt_pitch]
                gt_list.append(tmp_list)

        return img_data_list, gt_list

    def frame_infer(self):
        print("Start Inference")
        
        result_csv = []
        infer_count = 0

        diff_total_roll = 0.0
        diff_total_pitch = 0.0

        inference_input = []
        inference_gt = []
        count_array = []

        for i in range(0, len(self.img_data_list)-self.timesteps):
            tmp_inference_input = self.img_data_list[i:i+self.timesteps]
            tmp_inference_gt = self.ground_truth_list[i+self.timesteps]
            tmp_count_array = [i, tmp_inference_gt[0]]

            inference_input.append(tmp_inference_input)
            inference_gt.append(tmp_inference_gt)
            count_array.append(tmp_count_array)


        #print(len(inference_gt))
        #print(len(self.img_data_list))
        #print(inference_gt[0][0])
        #print(self.ground_truth_list[self.timesteps][0])

        

        



if __name__ == "__main__":
    parser = argparse.ArgumentParser("./frame_infer.py")
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=False,
        default='../config/infer_config.yaml',
        help='Frame Infer Config'
    )

    FLAGS, unparsed = parser.parse_known_args()


    try:
        print("Opening train config file %s", FLAGS.config)
        CFG = yaml.safe_load(open(FLAGS.config, 'r'))
    except Exception as e:
        print(e)
        print("Error opening train config file %s", FLAGS.config)
        quit()

    estimator = InferenceMod(CFG)
    result_csv = estimator.frame_infer()