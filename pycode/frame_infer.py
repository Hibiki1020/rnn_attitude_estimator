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
import cv2
import PIL.Image as PILIMAGE

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

        for (img_path, gt_path) in zip(self.img_data_list, self.ground_truth_list):
            print("---------Inference at " + str(infer_count + 1) + "---------")
            infer_count += 1
            start_time = time.time()

            input_image = cv2.imread(img_path)

            result = []

            roll_result_list = []
            pitch_result_list = []

            roll_hist_array = [0.0 for _ in range(self.dim_fc_out)]
            pitch_hist_array = [0.0 for _ in range(self.dim_fc_out)]

            roll_value_array = []
            pitch_value_array = []

            input_image = self.trasformImage(input_image)

            roll_output_array, pitch_output_array = self.prediction(input_image)
            
            roll = self.array_to_value_simple(roll_output_array)
            pitch = self.array_to_value_simple(pitch_output_array)

            roll_hist_array += roll_output_array[0]
            pitch_hist_array += pitch_output_array[0]

            diff_roll = np.abs(roll - gt_path[2])
            diff_pitch = np.abs(pitch - gt_path[3])

            print("Infered Roll:  " + str(roll) +  "[deg]")
            print("GT Roll:       " + str(gt_path[2]) + "[deg]")
            print("Infered Pitch: " + str(pitch) + "[deg]")
            print("GT Pitch:      " + str(gt_path[3]) + "[deg]")
            print("Diff Roll: " + str(diff_roll) + " [deg]")
            print("Diff Pitch: " + str(diff_pitch) + " [deg]")


    def trasformImage(self, input_image):
        ## color
        img_pil = self.cvToPIL(input_image)
        img_tensor = self.img_transform(img_pil)
        inputs = img_tensor.unsqueeze_(0)
        inputs = inputs.to(self.device)
        #print(inputs)
        return inputs

    def cvToPIL(self, img_cv):
        #img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = PILIMAGE.fromarray(img_cv)
        return img_pil

    def prediction(self, input_image):
        logged_roll, logged_pitch, roll, pitch = self.net(input_image)

        output_roll_array = roll.to('cpu').detach().numpy().copy()
        output_pitch_array = pitch.to('cpu').detach().numpy().copy()

        return np.array(output_roll_array), np.array(output_pitch_array)

    def array_to_value_simple(self, output_array):
        max_index = int(np.argmax(output_array))
        plus_index = max_index + 1
        minus_index = max_index - 1
        value = 0.0
        
        for value, label in zip(output_array[0], self.value_dict):
            value += value * label

        if max_index == 0:
            value = -31.0
        elif max_index == 62: #361
            value = 31.0

        return value


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