import sys
sys.dont_write_bytecode = True

from common import feature_extractor_mod

import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as nn_functional

class Network(nn.Module):
    def __init__(self, model, dim_fc_out, norm_layer, pretrained_model, dropout_rate):
        super(Network, self).__init__()
        
        self.dim_fc_out = dim_fc_out
        self.dropout_rate = dropout_rate

        self.rnn_input_dim = 100352 * 2
        self.rnn_hidden_dim = 150
        self.rnn_layer_num = 5

        self.feature_extractor = feature_extractor_mod.resnet50(pretrained_model, norm_layer=norm_layer, bn_eps=1e-5, bn_momentum=0.1, deep_stem=True, stem_width=64)
        self.rnn = nn.RNN(input_size = self.rnn_input_dim, hidden_size = self.rnn_hidden_dim, num_layers = self.rnn_layer_num, dropout = self.dropout_rate, batch_first = True)

        self.roll_fc = nn.Sequential(
            nn.Linear( self.rnn_hidden_dim, self.dim_fc_out),
            nn.Softmax(dim=1)
        )

        self.pitch_fc = nn.Sequential(
            nn.Linear( self.rnn_hidden_dim, self.dim_fc_out),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x_rnn, h = self.rnn(x, None)
        roll = self.roll_fc(x_rnn[:, -1, :])
        pitch = self.pitch_fc(x_rnn[:, -1, :])

        logged_roll = nn_functional.log_softmax(roll, dim=1)
        logged_pitch = nn_functional.log_softmax(pitch, dim=1)

        torch.set_printoptions(edgeitems=10000)

        return logged_roll, logged_pitch, roll, pitch

    def getParamValueList(self):
        list_resnet_param_value = []
        list_rnn_param_value = []
        list_roll_fc_param_value = []
        list_pitch_fc_param_value = []

        for param_name, param_value in self.named_parameters():
            param_value.requires_grad = True
            if "feature_extractor" in param_name:
                list_resnet_param_value.append(param_value)
            if "rnn" in param_name:
                list_rnn_param_value.append(param_value)
            if "roll_fc" in param_name:
                list_roll_fc_param_value.append(param_value)
            if "pitch_fc" in param_name:
                list_pitch_fc_param_value.append(param_value)

        return list_resnet_param_value, list_rnn_param_value ,list_roll_fc_param_value, list_pitch_fc_param_value
