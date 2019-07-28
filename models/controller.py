# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2019-07-24 22:07:12
# @Last Modified by:   yulidong
# @Last Modified time: 2019-07-29 00:04:57

""" Define controller """
import torch
import torch.nn as nn

class Controller(nn.Module):
    """ Controller """
    def __init__(self, latents, actions):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(latents, 128),nn.Linear(128, 256),nn.Linear(256, 128),nn.Linear(128, actions))

    def forward(self, inputs):
        action= self.fc(inputs)
        #print(action.shape)
        action[:,0]=torch.tanh(action[:,0])
        action[:,1]=torch.sigmoid(action[:,1])
        action[:,2]=torch.sigmoid(action[:,2])
        action[:,3]=action[:,3]
        return action
