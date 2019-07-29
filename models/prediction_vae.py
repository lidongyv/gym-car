# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2019-07-26 11:12:08
# @Last Modified by:   yulidong
# @Last Modified time: 2019-07-29 00:26:41

"""
Variational encoder model, used as a visual model
for our model of the world.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import *
from models.vae import Encoder
class Decoder_p(nn.Module):
    """ VAE decoder """
    def __init__(self, img_channels, latent_size):
        super(Decoder_a, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        self.fc1 = nn.Sequential(Linear(latent_size,256),Linear(256,1024))

        self.up1 = deconv2DBatchNormRelu(1024, 512, 2, stride=1)
        self.up1_1 = conv2DBatchNormRelu(512, 512, 3,padding=1)
        #2
        self.up2 = deconv2DBatchNormRelu(512, 256, 3, stride=1)
        self.up2_1 = conv2DBatchNormRelu(256, 256, 3,padding=1)
        #4
        self.up3 = deconv2DBatchNormRelu(256, 128, 3, stride=2,padding=1,output_padding=1)
        self.up3_1 = conv2DBatchNormRelu(128, 128, 3,padding=1)
        #8
        self.up4 = deconv2DBatchNormRelu(128, 64, 3, stride=2,padding=1,output_padding=1)
        self.up4_1 = conv2DBatchNormRelu(64, 64, 3,padding=1)
        #16
        self.up5 = deconv2DBatchNormRelu(64, 32, 3, stride=2,padding=1,output_padding=1)
        self.up5_1 = conv2DBatchNormRelu(32, 32, 3,padding=1)
        #32
        self.up6 = deconv2DBatchNormRelu(32, 16, 3, stride=2,padding=1,output_padding=1)
        #64
        self.up6_1 = conv2D(16, 8, 3,padding=1)
        self.up6_2 = conv2D(8, 3, 3,padding=1)
        self.relu=nn.LeakyReLU(inplace=False)
        self.upsample=nn.Sequential(self.up1,self.up1_1,self.up2,self.up2_1,self.up3,self.up3_1,self.up4,self.up4_1,self.up5,self.up5_1,self.up6,self.up6_1,self.up6_2,self.relu)
        #58+6=64 14+2
    def forward(self, x): 
        # pylint: disable=arguments-differ
        x = self.fc1(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        reconstruction = self.upsample(x)
        return reconstruction

class Encoder_p(nn.Module): 
    # pylint: disable=too-many-instance-attributes
    """ VAE encoder """
    def __init__(self, img_channels, latent_size):
        super(Encoder_a, self).__init__()
        self.latent_size = latent_size
        #self.img_size = img_size
        self.img_channels = img_channels

        self.conv0 = conv2DBatchNormRelu(img_channels, 32, 3, stride=1,padding=1)
        #32*64*64
        self.conv1 = conv2DBatchNormRelu(32, 64, 3, stride=2,padding=(1,0,1,0))
        #(64+2-4)/2+1=32
        self.conv1_1 = conv2DBatchNormRelu(64, 64, 3,padding=1)
        self.conv2 = conv2DBatchNormRelu(64, 128, 3, stride=2,padding=(1,0,1,0))
        #32+2-4/2+1=16
        self.conv2_1 = conv2DBatchNormRelu(128, 128, 3, padding=1)
        self.conv3 = conv2DBatchNormRelu(128, 256, 3, stride=2,padding=(1,0,1,0))
        #16+2-4/2+1=8
        self.conv3_1 = conv2DBatchNormRelu(256, 256, 3, padding=1)
        self.conv4 = conv2DBatchNormRelu(256, 512, 3, stride=2,padding=(1,0,1,0))
        #8+2-4/2+1=4
        self.conv4_1 = conv2DBatchNormRelu(512, 512, 3, padding=1)
        self.conv5 = conv2DBatchNormRelu(512, 1024, 3, stride=2,padding=(1,0,1,0))
        #4+2-3/2+1=2
        self.conv6 = conv2D(1024, 1024, 2, padding=0)
        # self.conv5_2 = conv2D(1024, 512, 1, padding=0)
        self.feature = nn.Sequential(self.conv0,self.conv1,self.conv1_1,self.conv2,self.conv2_1,self.conv3,self.conv3_1,self.conv4,self.conv4_1,self.conv5,self.conv6)
        self.fc_mu = nn.Sequential(Linear(1024, 256),Linear(256,latent_size))

        self.fc_sigma = nn.Sequential(Linear(1024, 256),Linear(256,latent_size))


    def forward(self, x): # pylint: disable=arguments-differ
        x = self.feature(x)
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)

        return mu, sigma

class VAE_p(nn.Module):
    """ Variational Autoencoder """
    def __init__(self, img_channels=3, latent_size=32):
        super(VAE_a, self).__init__()
        self.encoder = Encoder_a(img_channels, latent_size)
        self.decoder = Decoder_a(img_channels, latent_size)

    def forward(self, x): # pylint: disable=arguments-differ
        mu, sigma = self.encoder(x)
        #sigma = torch.exp(sigma^2.0)
        epsilon = torch.randn_like(sigma)
        #z = eps.mul(sigma).add_(mu)
        z=mu+sigma*epsilon

        recon_x = self.decoder(z)
        return recon_x, mu, sigma
