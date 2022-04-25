#!/bin/bash

cur_dir = %CD%

mkdir -p /home/weights
cd /home/weights
wget https://download.pytorch.org/models/resnet50-0676ba61.pth
cd ${cur_dir}