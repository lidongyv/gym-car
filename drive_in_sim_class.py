import numpy as np
import gym
from scipy.misc import imresize as resize
from gym.spaces.box import Box
from gym.envs.box2d.car_racing import CarRacing
import argparse
from os.path import join, exists
from os import mkdir
import matplotlib.pyplot as plt
import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from models.vae import VAE
from models.action_vae import VAE_a
from models.controller_class import Controller_class
import visdom
from utils.misc import save_checkpoint
from utils.misc import LSIZE, RED_SIZE
## WARNING : THIS SHOULD BE REPLACE WITH PYTORCH 0.5
from utils.learning import EarlyStopping
from utils.learning import ReduceLROnPlateau
from data.loaders import RolloutObservationDataset
SCREEN_X = 64
SCREEN_Y = 64

def _process_frame(frame):
  obs = frame[0:84, :, :].astype(np.float)/255.0
  obs = resize(obs, (64, 64))
  obs = ((1.0 - obs) * 255).round().astype(np.uint8)
  return obs

class CarRacingWrapper(CarRacing):
  def __init__(self, full_episode=False):
    super(CarRacingWrapper, self).__init__()
    self.full_episode = full_episode
    self.observation_space = Box(low=0, high=255, shape=(SCREEN_X, SCREEN_Y, 3)) # , dtype=np.uint8

  def _step(self, action):
    obs, reward, done, _ = super(CarRacingWrapper, self)._step(action)
    if self.full_episode:
      return _process_frame(obs), reward, False, {}
    return _process_frame(obs), reward, done, {}

def make_env(env_name, seed=-1, render_mode=False, full_episode=False):
  env = CarRacingWrapper(full_episode=full_episode)
  if (seed >= 0):
    env.seed(seed)
  '''
  print("environment details")
  print("env.action_space", env.action_space)
  print("high, low", env.action_space.high, env.action_space.low)
  print("environment details")
  print("env.observation_space", env.observation_space)
  print("high, low", env.observation_space.high, env.observation_space.low)
  assert False
  '''
  return env
transform = transforms.Compose([
  transforms.ToPILImage(),
  transforms.Resize((64, 64)),
  # transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
])
# from https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py
if __name__=="__main__":
  model=VAE(3, 64)
  model=torch.nn.DataParallel(model,device_ids=range(1))
  model.cuda()
  controller=Controller_class(64,3)
  controller=torch.nn.DataParallel(controller,device_ids=range(1))
  controller=controller.cuda()
  state = torch.load('/home/ld/gym-car/log/class/contorl_checkpoint_10.pkl')
  controller.load_state_dict(state['state_dict'])
  print('contorller load success')
  state = torch.load('/home/ld/gym-car/log/class/vae_checkpoint_10.pkl')
  model.load_state_dict(state['state_dict'])
  print('vae load success')
  # from pyglet.window import key
  action = np.array( [0.0, 0.0, 0.0] )
  # def key_press(k, mod):
  #   global restart
  #   if k==0xff0d: restart = True
  #   if k==key.LEFT:  a[0] = -1.0
  #   if k==key.RIGHT: a[0] = +1.0e
  #   if k==key.UP:    a[1] = +1.0
  #   if k==key.DOWN:  a[2] = +0.8   # set 1.0 for wheels to block to zero rotation
  # def key_release(k, mod):
  #   if k==key.LEFT  and a[0]==-1.0: a[0] = 0
  #   if k==key.RIGHT and a[0]==+1.0: a[0] = 0
  #   if k==key.UP:    a[1] = 0
  #   if k==key.DOWN:  a[2] = 0
  env = CarRacing()
  env.render()
  # env.viewer.window.on_key_press = key_press
  # env.viewer.window.on_key_release = key_release
  while True:
    env.reset()
    total_reward = 0.0
    steps = 0
    restart = False
    angle=[]
    while True:
      obs, reward, done, info = env.step(action)
      total_reward += reward
      obs=transform(obs).view(1,3,64,64)
      recon_c, mu_c, var_c = model(obs)
      mu, sigma = mu_c, var_c
      #sigma = torch.exp(sigma/2.0)
      epsilon = torch.randn_like(sigma)
      z=mu+sigma*epsilon
      z=z.cuda().view(obs.shape[0],-1).detach()
      action_p=controller(z)
      # action=action.view(-1).data.cpu().numpy().astype('float32')
      #print(action_p)
      action[0]=np.argmax(np.reshape(action_p[0].data.cpu().numpy().astype('float32'),-1))-1
      action[1]=np.argmax(np.reshape(action_p[1].data.cpu().numpy().astype('float32'),-1))
      action[2]=np.argmax(np.reshape(action_p[2].data.cpu().numpy().astype('float32'),-1))
      action=np.array(action)
      print(info)
      print(action)
      
      # angle.append(action[0])
      if info['speed']>50:
          action[1]=0
          action[2]=0.1
      elif info['speed']<30:
          action[1]=1
          action[2]=0
      else:
          action[1]=0.1
          action[2]=0
      # if len(angle)>15:
      #     if abs(action[0])>0.5:
      #         # action[0]=action[0]/np.abs(action[0])
      #         if info['speed']>40:
      #           action[1]=0
      #           action[2]=1

      #         # elif info['speed']>40:
      #         #   action[1]=0
      #         #   actieon[2]=0.5
      #     else: 
      #         action[0]=0
      #         action[1]=1
      #         action[2]=0
      # else:
      #     action[0]=0
      #     action[1]=1
      #     action[2]=0



      if action[0]==1:

          if info['speed']>40:
              action[1]=0
              action[2]=0.5
          elif info['speed']<10:
              action[1]=1
              action[2]=0
          else:
              action[1]=0.1
              action[2]=0
      elif action[0]==-1:

          if info['speed']>40:
              action[1]=0
              action[2]=0.5
          elif info['speed']<10:
              action[1]=1
              action[2]=0
          else:
              action[1]=0.1
              action[2]=0
      else: 
          #action[0]=0
          if info['speed']>60:
              action[1]=0
              action[2]=0.1
          elif info['speed']<50:
              action[1]=1
              action[2]=0

          # else:
          #     action[1]=0.1
          #     action[2]=0
      # if action[0]!=0:
      #     if info['speed']>40:
      #       action[1]=0
      #       action[2]=1
      #     else:
      #       action[1]=0.1
      #       action[2]=0
      
      if steps % 1 == 0 or done:
        print("\naction " + str(["{:+0.2f}".format(x) for x in action]))
        print("step {} total_reward {:+0.2f}".format(steps, total_reward))
      steps += 1
      env.render()
      if done or restart: 
        env.monitor.close()
        exit()
