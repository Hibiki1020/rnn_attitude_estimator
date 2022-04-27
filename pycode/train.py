import sys, codecs
#sys.stdout = codecs.getwriter("utf-8")(sys.stdout)
sys.dont_write_bytecode = True

from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import random

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
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from tensorboardX import SummaryWriter

from common import network_mod
from common import dataset_mod
from common import make_datalist_mod
from common import data_transform_mod

class Trainer:
    def __init__(self,
            save_top_path,
            yaml_path,
            weights_path,
            log_path,
            graph_path,
            csv_name,
            index_csv_path,
            multiGPU,
            pretrained_model,
            train_sequences,
            valid_sequences,
            dim_fc_out,
            deg_threshold,
            resize,
            mean_element,
            std_element,
            batch_size,
            num_epochs,
            optimizer_name,
            lr_resnet,
            lr_roll_fc,
            lr_pitch_fc,
            weight_decay,
            alpha,
            timesteps,
            dropout_rate):

            self.save_top_path = save_top_path
            self.yaml_path = yaml_path
            self.weights_path = weights_path
            self.log_path = log_path
            self.graph_path = graph_path
            self.csv_name = csv_name
            self.index_csv_path = index_csv_path
            self.multiGPU = multiGPU
            self.pretrained_model = pretrained_model
            self.train_sequences = train_sequences
            self.valid_sequences = valid_sequences
            self.dim_fc_out = dim_fc_out
            self.deg_threshold = deg_threshold
            self.resize = resize
            self.mean_element = mean_element
            self.std_element = std_element
            self.batch_size = batch_size
            self.num_epochs = num_epochs
            self.optimizer_name = optimizer_name
            self.lr_resnet = lr_resnet
            self.lr_roll_fc = lr_roll_fc
            self.lr_pitch_fc = lr_pitch_fc
            self.weight_decay = weight_decay
            self.alpha = alpha
            self.timesteps = timesteps
            self.dropout_rate = dropout_rate



if __name__ == '__main__':
    parser = argparse.ArgumentParser("./train.py")

    parser.add_argument(
        '--train_cfg', '-c',
        type=str,
        required=False,
        default='../config/train_config.yaml',
        help='Training configuration file'
    )

    FLAGS, unparsed = parser.parse_known_args()

    print("Load YAML file")

    try:
        print("Opening train config file %s", FLAGS.train_cfg)
        CFG = yaml.safe_load(open(FLAGS.train_cfg, 'r'))
    except Exception as e:
        print(e)
        print("Error opening train config file %s", FLAGS.train_cfg)
        quit()

    save_top_path = CFG["save_top_path"]
    yaml_path = save_top_path + "/train_config.yaml"
    weights_path = CFG["save_top_path"] + CFG["weights_path"]
    log_path = CFG["save_top_path"] + CFG["log_path"]
    graph_path = CFG["save_top_path"] + CFG["graph_path"]
    csv_name = CFG["csv_name"]
    index_csv_path = CFG["index_csv_path"]
    multiGPU = int(CFG["multiGPU"])

    pretrained_model = CFG["pretrained_model"]

    train_sequences = CFG["train"]
    valid_sequences = CFG["valid"]

    dim_fc_out = int(CFG["hyperparameter"]["dim_fc_out"])
    deg_threshold = int(CFG["hyperparameter"]["deg_threshold"])
    resize = int(CFG["hyperparameter"]["resize"])
    mean_element = float(CFG["hyperparameter"]["mean_element"])
    std_element = float(CFG["hyperparameter"]["std_element"])
    batch_size = int(CFG["hyperparameter"]["batch_size"])
    num_epochs = int(CFG["hyperparameter"]["num_epochs"])
    optimizer_name = str(CFG["hyperparameter"]["optimizer_name"])
    lr_resnet = float(CFG["hyperparameter"]["lr_resnet"])
    lr_roll_fc = float(CFG["hyperparameter"]["lr_roll_fc"])
    lr_pitch_fc = float(CFG["hyperparameter"]["lr_pitch_fc"])
    weight_decay = float(CFG["hyperparameter"]["weight_decay"])
    alpha = float(CFG["hyperparameter"]["alpha"])
    timesteps = int(CFG["hyperparameter"]["timesteps"])
    dropout_rate = float(CFG["hyperparameter"]["dropout_rate"])

    shutil.copy(FLAGS.train_cfg, yaml_path)

    print("Load Train Dataset")

    train_dataset = dataset_mod.RNNAttitudeEstimatorDataset(
        data_list = make_datalist_mod.makeMultiDataList(train_sequences, csv_name),
        transform = data_transform_mod.DataTransform(
            resize,
            mean_element,
            std_element
        ),
        phase = "train",
        index_dict_path = index_csv_path,
        dim_fc_out = dim_fc_out,
        timesteps = timesteps,
        deg_threshold = deg_threshold
    )

    print("Load Valid Dataeset")

    valid_dataset = dataset_mod.RNNAttitudeEstimatorDataset(
        data_list = make_datalist_mod.makeMultiDataList(valid_sequences, csv_name),
        transform = data_transform_mod.DataTransform(
            resize,
            mean_element,
            std_element
        ),
        phase = "valid",
        index_dict_path = index_csv_path,
        dim_fc_out = dim_fc_out,
        timesteps = timesteps,
        deg_threshold = deg_threshold
    )

    print("Load Network")
    net = network_mod.Network(dim_fc_out, norm_layer=nn.BatchNorm2d, pretrained_model=pretrained_model, dropout_rate=dropout_rate)
    #print(net)

    trainer = Trainer(
        save_top_path,
        yaml_path,
        weights_path,
        log_path,
        graph_path,
        csv_name,
        index_csv_path,
        multiGPU,
        pretrained_model,
        train_sequences,
        valid_sequences,
        dim_fc_out,
        deg_threshold,
        resize,
        mean_element,
        std_element,
        batch_size,
        num_epochs,
        optimizer_name,
        lr_resnet,
        lr_roll_fc,
        lr_pitch_fc,
        weight_decay,
        alpha,
        timesteps,
        dropout_rate
    )