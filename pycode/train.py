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
            lr_rnn,
            lr_roll_fc,
            lr_pitch_fc,
            weight_decay,
            alpha,
            timesteps,
            dropout_rate,
            train_dataset,
            valid_dataset,
            net):

            self.save_top_path = save_top_path
            self.yaml_path = yaml_path
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
            self.lr_rnn = lr_rnn
            self.lr_roll_fc = lr_roll_fc
            self.lr_pitch_fc = lr_pitch_fc
            self.weight_decay = weight_decay
            self.alpha = alpha
            self.timesteps = timesteps
            self.dropout_rate = dropout_rate

            self.train_dataset = train_dataset
            self.valid_dataset = valid_dataset
            self.net = net

            if self.multiGPU == 0:
                self.device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
            else:
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

            self.setRandomCondition()
            self.dataloaders_dict = self.getDataloaders(train_dataset, valid_dataset, batch_size)
            self.net = self.getNetwork(net)
            self.optimizer = self.getOptimizer(self.optimizer_name, self.lr_resnet, self.lr_rnn, self.lr_roll_fc, self.lr_pitch_fc)

            print("Set Training Parameter, Start Learning.")

    def setRandomCondition(self, keep_reproducibility=False, seed=123456789):
        if keep_reproducibility:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def getDataloaders(self, train_dataset, valid_dataset, batch_size):
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size = batch_size,
            shuffle=False,
            num_workers = 2,
            #pin_memory =True
        )

        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size = batch_size,
            shuffle=False,
            num_workers = 2,
            #pin_memory = True
        )

        dataloaders_dict = {"train":train_dataloader, "valid":valid_dataloader}

        return dataloaders_dict

    def getNetwork(self, net):
        print("Loading RNN Network")
        #print(net)
        net = net.to(self.device)
        if self.multiGPU == 1 and self.devide == "cuda":
            net = nn.DataParallel(net)
            cudnn.benchmark = True
            print("Training on multiGPU Device")
        else:
            cudnn.benchmark = True
            print("Training on Single GPU Device")

        return net

    def getOptimizer(self, optimizer_name, lr_resnet, lr_rnn, lr_roll_fc, lr_pitch_fc):
        if self.multiGPU == 1 and self.device == 'cuda':
            list_resnet_param_value, list_rnn_param_value,list_roll_fc_param_value, list_pitch_fc_param_value = self.net.module.getParamValueList()
        elif self.multiGPU == 0:
            list_resnet_param_value, list_rnn_param_value, list_roll_fc_param_value, list_pitch_fc_param_value = self.net.getParamValueList()

        if optimizer_name == "SGD":
            optimizer = optim.SGD([
                {"params": list_resnet_param_value, "lr": lr_resnet},
                {"params": list_rnn_param_value, "lr": lr_rnn},
                {"params": list_roll_fc_param_value, "lr": lr_roll_fc},
                {"params": list_pitch_fc_param_value, "lr": lr_pitch_fc}
            ], momentum=0.9, 
            weight_decay=self.weight_decay)
        elif optimizer_name == "Adam":
            optimizer = optim.Adam([
                {"params": list_resnet_param_value, "lr": lr_resnet},
                {"params": list_rnn_param_value, "lr": lr_rnn},
                {"params": list_roll_fc_param_value, "lr": lr_roll_fc},
                {"params": list_pitch_fc_param_value, "lr": lr_pitch_fc}
            ], weight_decay=self.weight_decay)

        print("optimizer: {}".format(optimizer_name))
        return optimizer

    def process(self):
        start_clock = time.time()
        
        #Loss recorder
        writer = SummaryWriter(log_dir = self.save_top_path + "/log")

        record_train_loss = []
        record_valid_loss = []

        for epoch in range(self.num_epochs):
            print("--------------------------------")
            print("Epoch: {}/{}".format(epoch+1, self.num_epochs))

            for phase in ["train", "valid"]:
                if phase == "train":
                    self.net.train()
                elif phase == "valid":
                    self.net.eval()
                
                #Data Load
                epoch_loss = 0.0

                for img_input, label_roll, label_pitch in tqdm(self.dataloaders_dict[phase]):
                    img_input = img_input.to(self.device)

                    #print(img_input.size())

                    label_roll = label_roll.to(self.device)
                    label_pitch = label_pitch.to(self.device)

                    #Reset Gradient
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase=="train"):
                        logged_roll_inf, logged_pitch_inf, roll_inf, pitch_inf = self.net(img_input)

                        roll_loss = torch.mean(torch.sum(-label_roll*logged_roll_inf, 1))
                        pitch_loss = torch.mean(torch.sum(-label_pitch*logged_pitch_inf, 1))

                        torch.set_printoptions(edgeitems=1000000)

                        if self.device == 'cpu':
                            l2norm = torch.tensor(0., requires_grad = True).cpu()
                        else:
                            l2norm = torch.tensor(0., requires_grad = True).cuda()

                        for w in self.net.parameters():
                            l2norm = l2norm + torch.norm(w)**2
                        
                        total_loss = roll_loss + pitch_loss + self.alpha*l2norm

                        if phase == "train":
                            total_loss.backward()
                            self.optimizer.step()

                        epoch_loss += total_loss.item() * img_input.size(0)

                epoch_loss = epoch_loss/len(self.dataloaders_dict[phase].dataset)
                print("{} Loss: {:.4f}".format(phase, epoch_loss))

                if phase == "train":
                    record_train_loss.append(epoch_loss)
                    writer.add_scalar("Loss/Train", epoch_loss, epoch)
                else:
                    record_valid_loss.append(epoch_loss)
                    writer.add_scalar("Loss/Valid", epoch_loss, epoch)

            if record_train_loss and record_valid_loss:
                writer.add_scalars("Loss/train_and_val", {"train": record_train_loss[-1], "val": record_valid_loss[-1]}, epoch)

        mins = (time.time() - start_clock) // 60
        secs = (time.time() - start_clock) % 60
        print("Training Time: ", mins, "[min]", secs, "[sec]")
        
        writer.close()
        self.saveParam()
        self.saveGraph(record_train_loss, record_valid_loss)

    def saveParam(self):
        save_path = self.save_top_path + "/weights.pth"
        torch.save(self.net.state_dict(), save_path)
        print("Saved Weight")

    def saveGraph(self, record_loss_train, record_loss_val):
        graph = plt.figure()
        plt.plot(range(len(record_loss_train)), record_loss_train, label="Training")
        plt.plot(range(len(record_loss_val)), record_loss_val, label="Validation")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss [m^2/s^4]")
        plt.title("loss: train=" + str(record_loss_train[-1]) + ", val=" + str(record_loss_val[-1]))
        graph.savefig(self.save_top_path + "/train_log.jpg")
        plt.show()



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
    lr_rnn = float(CFG["hyperparameter"]["lr_rnn"])
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
        deg_threshold = deg_threshold,
        resize = resize,
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
        deg_threshold = deg_threshold,
        resize = resize,
    )

    print("Load Network")
    net = network_mod.Network(dim_fc_out, norm_layer=nn.BatchNorm2d, pretrained_model=pretrained_model, dropout_rate=dropout_rate, timesteps=timesteps)
    #print(net)

    trainer = Trainer(
        save_top_path,
        yaml_path,
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
        lr_rnn,
        lr_roll_fc,
        lr_pitch_fc,
        weight_decay,
        alpha,
        timesteps,
        dropout_rate,
        train_dataset,
        valid_dataset,
        net
    )

    trainer.process()