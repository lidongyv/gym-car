# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2019-07-28 23:10:02
# @Last Modified by:   yulidong
# @Last Modified time: 2019-08-04 14:33:37
from bisect import bisect
from os import listdir
from os.path import join, isdir
from tqdm import tqdm
import torch
import torch.utils.data
import numpy as np
import os
import matplotlib.pyplot as plt
O=2
class _RolloutDataset(torch.utils.data.Dataset): 
    def __init__(self, root, transform, train=True,sample_data=1000,sample_count=0): 
        self._transform = transform
        self._root=root
        self._files = listdir(root)
        self._files.sort()
        if train:
            self._files = self._files[sample_data*sample_count:sample_data*(sample_count+1)]
            self._length,self._data = self._create_dataset(self._files, sample_data)

        else:
            self._files = self._files[-O:]
            self._length,self._data = self._create_dataset(self._files, sample_data)

        #self.length=self.__len__()


    def __len__(self):
        # to have a full sequence, you need self.seq_len + 1 elements, as
        # you must produce both an seq_len obs and seq_len next_obs sequences
        # if not self._cum_size:
        #     self.load_next_buffer()
        return self._length

    def _create_dataset(self,filelist, N=1000):  # N is 10000 episodes, M is number of timesteps
        obs=[]
        action=[]
        speed=[]
        pos=[]
        length=0
        index_i=[]
        index_j=[]
        info=[]
        # print(len(filelist))
        # exit()
        for i in range(N):
            speed_t=[0]
            pos_t=[[0,0]]
            count=0
            filename = filelist[i]
            raw_data = np.load(os.path.join(self._root, filename))
            #obs=recording_obs,mu=recording_mu, logvar=recording_logvar, action=recording_action, reward=recording_reward,info=recording_info
            #{'pos':[x,y],'speed':true_speed}
            obs.append(raw_data['obs'])
            action.append(raw_data['action'])
            #print(len(raw_data['action']),len(raw_data['obs']))
            #print(raw_data['info'][:])
            # for j in range(len(raw_data['action'])-1):
            #     speed_t.append(raw_data['info'][j]['speed'])
            #     pos_t.append(raw_data['info'][j]['pos'])
            # speed.append(speed_t)
            # pos.append(pos_t)
            info.append(raw_data['info'])
            length+=len(raw_data['action'])
            index_i.append(i*np.ones(len(raw_data['action'])))
            index_j.append(np.arange(0,len(raw_data['action'])))
            if i%1==0:
                print("loading file", i + 1,"having length",len(raw_data['action']))
        index_i=np.concatenate(index_i).astype(np.int32)
        index_j=np.concatenate(index_j).astype(np.int32)
        #print(index_i)
        data={'obs':obs,'action':action,'info':info,'index':[index_i,index_j]}

        return length,data
    def __getitem__(self, index):
        if(self._data['index'][1][index]+1>=len(self._data['obs'][self._data['index'][0][index]])):
            index=index-1
            print('reach the end')

        obs=self._data['obs'][self._data['index'][0][index]][self._data['index'][1][index]]
        pre=self._data['obs'][self._data['index'][0][index+1]][self._data['index'][1][index+1]]
        action=np.array(self._data['action'][self._data['index'][0][index]][self._data['index'][1][index]])
        speed=np.array(self._data['info'][self._data['index'][0][index]][self._data['index'][1][index]]['speed'])
        pos=np.array(self._data['info'][self._data['index'][0][index]][self._data['index'][1][index]]['pos'])
        #speed=np.array(self._data['speed'][self._data['index'][0][index]][self._data['index'][1][index]])
        #pos=self._data['pos'][self._data['index'][0][index]][self._data['index'][1][index]]
        #speed=1
        # print(obs.shape,obs.dtype)
        obs=self._get_data(obs.astype(np.uint8)).float()
        pre=self._get_data(pre.astype(np.uint8)).float()
        #print(speed)
        if action[0]>0.5:
            action[0]=2
        elif action[0]<-0.5:
            action[0]=0
        else: 
            action[0]=1
        if action[1]>0.5:
            action[1]=1
        else:
            action[1]=0
        if action[2]>0.5:
            action[2]=1
        else:
            action[2]=0
        action=torch.cat([torch.from_numpy(action).float().view(3,1,1),torch.from_numpy(speed).float().view(1,1,1)],dim=0).view(4,1,1)
        action=action.expand(4,obs.shape[1],obs.shape[2])
        #print(obs.shape,action.shape,pre.shape)
        return obs,action,pre

    def _get_data(self, data):
        pass

    def _data_per_sequence(self, data_length):
        pass


class RolloutObservationDataset(_RolloutDataset):

    def _data_per_sequence(self, data_length):
        return data_length

    def _get_data(self, data):
        return self._transform(data)
