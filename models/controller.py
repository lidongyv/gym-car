# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2019-07-24 22:07:12
# @Last Modified by:   yulidong
# @Last Modified time: 2019-07-26 11:26:14

""" Define controller """
import torch
import torch.nn as nn

class Controller(nn.Module):
    """ Controller """
    def __init__(self, latents, recurrents, actions):
        super().__init__()
        self.fc = nn.Linear(latents + recurrents, actions)

    def forward(self, *inputs):
        cat_in = torch.cat(inputs, dim=1)
        return self.fc(cat_in)
