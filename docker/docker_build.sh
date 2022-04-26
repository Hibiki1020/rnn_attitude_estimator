#!/bin/bash

image_name='rnn_attitude_estimator'
image_tag='docker'

docker build -t $image_name:$image_tag .