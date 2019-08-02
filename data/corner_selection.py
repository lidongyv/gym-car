# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2019-07-28 23:10:02
# @Last Modified by:   yulidong
# @Last Modified time: 2019-08-02 21:37:34
from bisect import bisect
from os import listdir
from os.path import join, isdir
from tqdm import tqdm
import torch
import torch.utils.data
import numpy as np
import os
import matplotlib.pyplot as plt
inpath='/home/ld/gym-car/datasets/control'
outpath='/home/ld/gym-car/datasets/corner'
files=os.listdir(inpath)
obs=[]
action=[]
speed=[]
pos=[]
length=0
index_i=[]
index_j=[]
info=[]
for file in files:
    raw_data = np.load(os.path.join(inpath, file))
    obs=raw_data['obs']
    action=raw_data['action']
    mu=raw_data['mu']
    logvar=raw_data['logvar']
    reward=raw_data['reward']
    info=raw_data['info']
    obs_s=[]
    action_s=[]
    mu_s=[]
    logvar_s=[]
    reward_s=[]
    info_s=[]    
    for j in range(1,len(obs)):
        image=obs[j]
        mask= np.sum(image,axis=2)
        if np.sum(np.where(mask<=10))>60:
            #print('init',j)
            continue
        mask= (mask<np.mean(mask))
        pos=np.array(np.where(mask==1))
        end_pos=np.array(np.where(mask[0,:]==1))
        start_pos=np.array(np.where(mask[-1,:]==1))
        middle_pos=np.array(np.where(mask[32,:]==1))
        #print(end_pos.shape,start_pos.shape)
        try:
            # if end_pos.shape[1]<1:
            #     continue
            # if start_pos.shape[1]<1:
            #     continue
            if end_pos.shape[1]>0 and start_pos.shape[1]>0 and np.abs(np.abs(end_pos[0,0]-middle_pos[0,0])-np.abs(middle_pos[0,0]-start_pos[0,0]))<6:
                #print('straignt')
                continue
            if end_pos.shape[1]>0:
                continue
        except:
            print('exception')
            
        # if start_pos.shape[1]<1:
        #     continue

        obs_s.append(obs[j])
        action_s.append(action[j])
        mu_s.append(mu[j])
        logvar_s.append(logvar[j])
        reward_s.append(reward[j])
        info_s.append(info[j-1])
    print(len(obs_s),os.path.join(outpath, file))
    np.savez_compressed(os.path.join(outpath, file), obs=obs_s,mu=mu_s, logvar=logvar_s, action=action_s, reward=reward_s,info=info_s)
#     speed.append(speed_t)
#     pos.append(pos_t)
#     info.append(raw_data['info'])
#     length+=len(raw_data['action'])
#     index_i.append(i*np.ones(len(raw_data['action'])))
#     index_j.append(np.arange(0,len(raw_data['action'])))
#     if i%1==0:
#         print("loading file", i + 1,"having length",len(raw_data['action']))
# index_i=np.concatenate(index_i).astype(np.int32)
# index_j=np.concatenate(index_j).astype(np.int32)
# #print(index_i)
# data={'obs':obs,'action':action,'info':info,'index':[index_i,index_j]}
# np.savez_compressed(filename, obs=recording_obs,mu=recording_mu, logvar=recording_logvar, action=recording_action, reward=recording_reward,info=recording_info)
