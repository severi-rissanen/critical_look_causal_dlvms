import os
import math, random
import scipy.stats as stats
import numpy as np
import pandas as pd
import collections
import itertools

import torch
from torch.optim import Adam, SGD
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

class FullyConnected(nn.Sequential):
    """
    Fully connected multi-layer network with ELU activations.
    """
    def __init__(self, sizes, final_activation=None):
        layers = []
        layers.append(nn.Linear(sizes[0],sizes[1]))
        for in_size, out_size in zip(sizes[1:], sizes[2:]):
            layers.append(nn.ELU())
            layers.append(nn.Linear(in_size, out_size))
        if final_activation is not None:
            layers.append(final_activation)
        self.length = len(layers)
        super().__init__(*layers)

    def append(self, layer):
        assert isinstance(layer, nn.Module)
        self.add_module(str(len(self)), layer)
        
    def __len__(self):
        return self.length
        
class Decoder(nn.Module):
    def __init__(
        self,
        x_dim,
        z_dim,
        device,
        p_y_zt_nn_layers,
        p_y_zt_nn_width,
        p_t_z_nn_layers,
        p_t_z_nn_width,
        p_x_z_nn_layers,
        p_x_z_nn_width,
        t_mode,
        y_mode,
        x_mode,
        y_separate_enc,
        common_stds,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.device = device
        self.t_mode = t_mode
        self.y_mode = y_mode
        self.p_y_zt_nn_layers = p_y_zt_nn_layers
        self.p_y_zt_nn_width = p_y_zt_nn_width
        self.p_t_z_nn_layers = p_t_z_nn_layers
        self.p_t_z_nn_width = p_t_z_nn_width
        self.p_x_z_nn_layers = p_x_z_nn_layers
        self.p_x_z_nn_width = p_x_z_nn_width
        self.y_separate_enc = y_separate_enc
        self.common_stds = common_stds
        
        #Can be used as a linear predictor if num_hidden=0
        self.n_x_estimands = sum([1 if m==0 or m==2 else m for m in x_mode])
        #for each x we have the possible std estimator also for simplicity, possibly not used
        self.x_nn = FullyConnected([z_dim] + p_x_z_nn_layers*[p_x_z_nn_width] + [(self.n_x_estimands)*2])
        #These use as many heads as needed
        self.t_heads = 2 if self.t_mode==0 else (1 if self.t_mode==2 else self.t_mode)
        self.t_nn = FullyConnected([z_dim] + p_t_z_nn_layers*[p_t_z_nn_width] + [self.t_heads])
        self.y_heads = 2 if self.y_mode==0 else (1 if self.y_mode==2 else self.y_mode)
        self.y_nn = FullyConnected([z_dim+1] + p_y_zt_nn_layers*[p_y_zt_nn_width] + [self.y_heads])
        self.y0_nn = FullyConnected([z_dim] + p_y_zt_nn_layers*[p_y_zt_nn_width] + [self.y_heads])
        self.y1_nn = FullyConnected([z_dim] + p_y_zt_nn_layers*[p_y_zt_nn_width] + [self.y_heads])
        
        #Variance parameters, if one common value estimated
        self.x_log_std = nn.Parameter(torch.FloatTensor(x_dim*[1.], device=device))
        self.t_log_std = nn.Parameter(torch.FloatTensor([1.], device=device))
        self.y_log_std = nn.Parameter(torch.FloatTensor([1.], device=device))
        
        self.to(device)
        
    def forward(self, z, t):
        
        x_res = self.x_nn(z)
        x_pred = x_res[:,:self.n_x_estimands]
        x_std = torch.exp(x_res[:,self.n_x_estimands:])
        
        t_res = self.t_nn(z)
        if self.t_mode == 0:
            t_pred = t_res[:,:1]
            t_std = torch.exp(t_res[:,1:])
        else:
            t_pred = t_res
            t_std = None
        
        if self.y_separate_enc and self.t_mode==2:
            y_res0 = self.y0_nn(z)
            y_res1 = self.y1_nn(z)
            y_std = 0
            if self.y_mode==0:
                y_pred0 = y_res0[:,:1]
                y_std0 = torch.exp(y_res0[:,1:])
                y_pred1 = y_res1[:,:1]
                y_std1 = torch.exp(y_res1[:,1:])
                y_std = y_std1*t + y_std0*(1-t)
            else:
                y_pred0 = y_res0
                y_std0 = None
                y_pred1 = y_res1
                y_std1 = None
            y_pred = y_pred1*t + y_pred0*(1-t)
        else:
            y_res = self.y_nn(torch.cat([z,t],1))
            if self.y_mode == 0:
                y_pred = y_res[:,:1]
                y_std = torch.exp(y_res[:,1:])
            else:
                y_pred = y_res
                y_std = None
        
        if self.common_stds:
            x_std = torch.exp(self.x_log_std).repeat(t.shape[0],1)
            t_std = torch.exp(self.t_log_std).repeat(t.shape[0],1)
            y_std = torch.exp(self.y_log_std).repeat(t.shape[0],1)
        
        return x_pred,x_std,t_pred,t_std,y_pred,y_std

class Encoder(nn.Module):
    def __init__(
        self, 
        x_dim,
        z_dim,
        device,
        t_mode,
        y_mode,
        q_z_nn_layers,
        q_z_nn_width,
        ty_separate_enc,
        common_stds
    ):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.device = device
        self.q_z_nn_layers = q_z_nn_layers
        self.q_z_nn_width = q_z_nn_width
        self.t_mode=t_mode
        self.y_mode=y_mode
        self.ty_separate_enc=ty_separate_enc
        self.common_stds = common_stds
        
        # q(z|x,t,y)
        self.q_z_nn = FullyConnected([x_dim+2] + q_z_nn_layers*[q_z_nn_width] + [z_dim*2])
        
        # In case t and y are binary: (tries to fit a gaussian in a simple way, 
        # but a Gaussian posterior is wrong in the first place)
        # Let's try to fit the stds for different inputs also, seems reasonable
        self.q_z_nn_t0y0 = FullyConnected([x_dim] + q_z_nn_layers*[q_z_nn_width] + [z_dim*2])
        self.q_z_nn_t0y1 = FullyConnected([x_dim] + q_z_nn_layers*[q_z_nn_width] + [z_dim*2])
        self.q_z_nn_t1y0 = FullyConnected([x_dim] + q_z_nn_layers*[q_z_nn_width] + [z_dim*2])
        self.q_z_nn_t1y1 = FullyConnected([x_dim] + q_z_nn_layers*[q_z_nn_width] + [z_dim*2])
        
        self.z_log_std = nn.Parameter(torch.ones(z_dim, device=device))
        
        self.to(device)
        
    def forward(self, x, t, y):
        if self.ty_separate_enc and self.t_mode == 2 and self.y_mode == 2:
            z_res = self.q_z_nn_t0y0(x)*(1-t)*(1-y) + self.q_z_nn_t0y1(x)*(1-t)*(y) + \
                self.q_z_nn_t1y0(x)*(t)*(1-y) + self.q_z_nn_t1y1(x)*(t)*(y)
            z_pred = z_res[:,:self.z_dim]
            z_std = torch.exp(z_res[:,self.z_dim:])
        else:
            z_res = self.q_z_nn(torch.cat([x, t, y], axis=1))
            z_pred = z_res[:,:self.z_dim]
            z_std = torch.exp(z_res[:,self.z_dim:])
        if self.common_stds:
            z_std = torch.exp(self.z_log_std).repeat(x.shape[0],1)
        return z_pred, z_std
        
class CEVAE(nn.Module):
    #The CEVAE used for real data
    def __init__(
        self, 
        x_dim,
        z_dim,
        device,
        p_y_zt_nn_layers,
        p_y_zt_nn_width,
        p_t_z_nn_layers,
        p_t_z_nn_width,
        p_x_z_nn_layers,
        p_x_z_nn_width,
        q_z_nn_layers,
        q_z_nn_width,
        t_mode,
        y_mode,#0 for continuous (Gaussian), 2 or more for categorical distributions (usually 2 or 0)
        x_mode,#a list, 0 for continuous (Gaussian), 2 or more for categorical distributions (usually 2 or 0)
        ty_separate_enc,
        z_mode,
        common_stds
    ):
        super().__init__()
        
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.device = device
        self.y_mode = y_mode
        self.t_mode = t_mode
        self.x_mode = x_mode
        self.z_mode = z_mode
        
        self.pz_logit = nn.Parameter(torch.FloatTensor([0.0]).to(device))#for binary z
        
        assert t_mode == 0 or t_mode > 1
        assert y_mode == 0 or y_mode > 1
        assert all([x_m == 0 or x_m > 1 for x_m in x_mode])
        assert z_mode == 0 or z_mode == 2
        assert len(x_mode) == x_dim
        
        self.encoder = Encoder(
            x_dim,
            z_dim,
            device,
            t_mode,
            y_mode,
            q_z_nn_layers,
            q_z_nn_width,
            ty_separate_enc,
            common_stds
        )
        self.decoder = Decoder(
            x_dim,
            z_dim,
            device,
            p_y_zt_nn_layers,
            p_y_zt_nn_width,
            p_t_z_nn_layers,
            p_t_z_nn_width,
            p_x_z_nn_layers,
            p_x_z_nn_width,
            t_mode,
            y_mode,
            x_mode,
            ty_separate_enc,#Toggle on/off with the encoder for now
            common_stds
        )
        self.to(device)
        self.float()

    def reparameterize(self, mean, std):
        # samples from unit norm and does reparam trick
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def forward(self, x, t, y):#Should have t, y
        z_pred, z_std = self.encoder(x, t, y)
        if self.z_mode == 0:#if z is gaussian
            z = self.reparameterize(z_pred, z_std)
        if self.z_mode == 2:#if z is binary
            #We need to pass in a large vector to have them match with different t values for y estimation
            #Could optimize by passing x and t networks only two values but easier this way
            z = torch.cat([torch.zeros(x.shape[0],1),torch.ones(x.shape[0],1)],0)
            t = torch.cat([t,t],0)
        x_pred, x_std, t_pred, t_std, y_pred, y_std = self.decoder(z,t)
        
        return z_pred, z_std, x_pred, x_std, t_pred, t_std, y_pred, y_std
    
    def sample(self,n):
        different_modes = list(set(self.x_mode))
        x_same_mode_indices = dict()
        for mode in different_modes:
            x_same_mode_indices[mode] = [i for i,m in enumerate(self.x_mode) if m==mode]

        z_sample = torch.randn(n, self.z_dim).to(self.device)
        t_sample = dist.Bernoulli(logits=self.decoder.t_nn(z_sample)[:,[0]]).sample()#binary t
        x_pred,x_std,t_pred,t_std,y_pred,y_std = self.decoder(z_sample, t_sample)
        y_sample = dist.Normal(loc=y_pred, scale=y_std).sample()#Continuous y
        x_sample = np.zeros((n, self.x_dim))

        pred_i = 0#x_pred is much longer than x if x has categorical variables with more categories than 2
        for i,mode in enumerate(self.x_mode):
            if mode==0:
                x_sample[:,i] = dist.Normal(loc=x_pred[:,pred_i], scale=x_std[:,pred_i]).sample().detach().numpy()
                pred_i += 1
            elif mode==2:
                x_sample[:,i] = dist.Bernoulli(logits=x_pred[:,pred_i]).sample().detach().numpy()
                pred_i += 1
            else:
                x_sample[:,i] = dist.Categorical(logits=x_pred[:,pred_i:pred_i+mode]).sample().detach().numpy()
                pred_i += mode
        
        return z_sample, x_sample, t_sample, y_sample