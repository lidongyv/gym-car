# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2019-07-24 22:07:12
# @Last Modified by:   yulidong
# @Last Modified time: 2019-08-03 19:48:40

""" Define controller """
import torch
import torch.nn as nn

class Controller_class(nn.Module):
    """ Controller """
    def __init__(self, latents, actions):
        super().__init__()
        self.fc_steer = nn.Sequential(nn.Linear(latents, 128),nn.Linear(128, 256),nn.Linear(256, 128),nn.Linear(128, 3))
        self.fc_gas = nn.Sequential(nn.Linear(latents, 128),nn.Linear(128, 256),nn.Linear(256, 128),nn.Linear(128, 2))
        self.fc_brake = nn.Sequential(nn.Linear(latents, 128),nn.Linear(128, 256),nn.Linear(256, 128),nn.Linear(128, 2))
    def forward(self, inputs):
        #print(action.shape)
        steer=torch.nn.functional.log_softmax(self.fc_steer(inputs),dim=1)
        gas=torch.nn.functional.log_softmax(self.fc_gas(inputs),dim=1)
        brake=torch.nn.functional.log_softmax(self.fc_brake(inputs),dim=1)
        # action[:,3]=action[:,3]
        action=[steer,gas,brake]
        return action
# action_loss=torch.nn.NLLLoss()
# loss_steer=action_loss(action_p[0],action[0])
# loss_gas=action_loss(action_p[1],action[1])
# loss_brake=action_loss(action_p[2],action[2])