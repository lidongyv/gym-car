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
from models.controller import Controller
import visdom
from utils.misc import save_checkpoint
from utils.misc import LSIZE, RED_SIZE
## WARNING : THIS SHOULD BE REPLACE WITH PYTORCH 0.5
from utils.learning import EarlyStopping
from utils.learning import ReduceLROnPlateau
from data.loaders import RolloutObservationDataset
from gym.envs.classic_control import rendering
import time
SCREEN_X = 64
SCREEN_Y = 64
FACTOR = 8

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
  controller=Controller(64,3)
  controller=torch.nn.DataParallel(controller,device_ids=range(1))
  controller=controller.cuda()
  state = torch.load('/home/ld/gym-car/log/vae/contorl_checkpoint_52.pkl')
  controller.load_state_dict(state['state_dict'])
  print('contorller load success')
  state = torch.load('/home/ld/gym-car/log/vae/vae_checkpoint_52.pkl')
  model.load_state_dict(state['state_dict'])
  print('vae load success')
  model_p=VAE_a(7, 64)
  model_p=torch.nn.DataParallel(model_p,device_ids=range(1))
  model_p.cuda()
  state = torch.load('/home/ld/gym-car/log/vae/pre_checkpoint_52.pkl')
  model_p.load_state_dict(state['state_dict'])
  print('prediction load success')

  with torch.no_grad():
    # from pyglet.window import key
    action = np.array( [0.0, 0.0, 0.0] )
    viewer=None
    env = CarRacing()
    # env.render()
    if viewer is None:
      viewer = rendering.SimpleImageViewer()

    z=torch.randn(64)
    z=z.cuda().view(1,-1).detach()
    obs=model.module.decoder(z)
    img=obs.squeeze().data.cpu().numpy().astype('float32').transpose([1,2,0])
    img=np.array(img)
    img = resize(img, (int(np.round(SCREEN_Y*FACTOR)), int(np.round(SCREEN_X*FACTOR))))
    viewer.imshow(img)
    # time.sleep(10)
    # exit()
    total_reward = 0.0
    steps = 0
    restart = False
    
    while True:
        obs=obs.view(1,3,64,64)
        mu_c, var_c = model.module.encoder(obs)
        mu, sigma = mu_c, var_c
        epsilon = torch.randn_like(sigma)
        z=mu+sigma*epsilon
        z=z.cuda().view(obs.shape[0],-1).detach()
        # z=torch.randn(64)
        # z=z.cuda().view(1,-1).detach()
        action=controller(z)
        action_pr=torch.cat([action.view(action.shape[0],3,1,1),10*torch.ones_like(action.view(1,3,1,1)[:,:1,:,:]).cuda()],dim=1) \
          .expand(action.shape[0],4,64,64)
        recon_pr, mu_pr, var_pr = model_p(torch.cat([obs,action_pr],dim=1))
        obs=recon_pr
        img=obs.squeeze().data.cpu().numpy().astype('float32').transpose([1,2,0])
        img=np.array(img)
        img = resize(img, (int(np.round(SCREEN_Y*FACTOR)), int(np.round(SCREEN_X*FACTOR))))
        viewer.imshow(img)
        print('racing!',steps,print(action.data.cpu().numpy().astype('float32')))
        steps+=1

