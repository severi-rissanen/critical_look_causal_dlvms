from lineartoydata import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import itertools

sns.set_style('whitegrid')
PLOT_STYLE='ggplot'

import os
import re
import glob

import torch
from torch.optim import Adam, SGD
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar
import ignite.metrics as ignite_metrics
from ignite.contrib.handlers.param_scheduler import LRScheduler
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ExponentialLR

def safe_sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x,-20,20)))

def p_y_zt_from_model(model, p_y_zt_nn):
    """Wrapper function that returns convenient p(y|z,t) estimates from the model with binary t,y. Assumes z_dim=1"""
    def p_y_zt1_linear(z):
        return safe_sigmoid(model.decoder.y1_nn.weight.detach().numpy()*z + model.decoder.y1_nn.bias.detach().numpy())
    def p_y_zt0_linear(z):
        return safe_sigmoid(model.decoder.y0_nn.weight.detach().numpy()*z + model.decoder.y0_nn.bias.detach().numpy())
    def p_y_zt1_nn(z):
        if type(z) == type(0.1):
            z = np.array([z])
        return safe_sigmoid(model.decoder.y1_nn_real(torch.Tensor(z[:,None])).squeeze().detach().numpy())
    def p_y_zt0_nn(z):
        if type(z) == type(0.1):
            z = np.array([z])
        return safe_sigmoid(model.decoder.y0_nn_real(torch.Tensor(z[:,None])).squeeze().detach().numpy())
    if p_y_zt_nn:
        return p_y_zt1_nn, p_y_zt0_nn
    else:
        return p_y_zt1_linear, p_y_zt0_linear

def p_y_zt_from_true_dist(y_a0, y_b0, y_a1, y_b1):
    def p_y_zt1(z):
        return safe_sigmoid(y_a1*z + y_b1)
    def p_y_zt0(z):
        return safe_sigmoid(y_a0*z + y_b0)
    return p_y_zt1, p_y_zt0

def linear_binary_ty_ate(p_y_zt1_func, p_y_zt0_func):
    """Calculates the ATE assuming the standard normal p(z) distribution and given P(y|z,t).
    Assumes z_dim==1 for the model."""
    p_y_dot1,_ = scipy.integrate.quad(lambda z: scipy.stats.norm.pdf(z)*p_y_zt1_func(z), -np.inf, np.inf)
    p_y_dot0,_ = scipy.integrate.quad(lambda z: scipy.stats.norm.pdf(z)*p_y_zt0_func(z), -np.inf, np.inf)
    return p_y_dot1 - p_y_dot0

def linear_binary_ty_pydot(p_y_zt1_func, p_y_zt0_func):
    """The same as linear_binary_ty_ate but returns the actual p(y|do(t)) values"""
    p_y_dot1,_ = scipy.integrate.quad(lambda z: scipy.stats.norm.pdf(z)*p_y_zt1_func(z), -np.inf, np.inf)
    p_y_dot0,_ = scipy.integrate.quad(lambda z: scipy.stats.norm.pdf(z)*p_y_zt0_func(z), -np.inf, np.inf)
    return p_y_dot1, p_y_dot0

def linear_binary_ty_ate_2(p_y_zt1_func, p_y_zt0_func):
    """Same as linear_binary_ty_ate but uses the trapezoidal rule directly. Probably less issues than with scipy."""
    n = 10000
    box_width = 20/(n-1)
    full_range = np.linspace(-10,10,n)
    p_y_dot1 = ((p_y_zt1_func(full_range[:-1])*scipy.stats.norm.pdf(full_range[:-1]) \
                 + p_y_zt1_func(full_range[1:])*scipy.stats.norm.pdf(full_range[1:]))*box_width/2).sum()
    p_y_dot0 = ((p_y_zt0_func(full_range[:-1])*scipy.stats.norm.pdf(full_range[:-1]) \
                 + p_y_zt0_func(full_range[1:])*scipy.stats.norm.pdf(full_range[1:]))*box_width/2).sum()
    return p_y_dot1 - p_y_dot0

def linear_binary_ty_ate_3(model, z_dim):
    """Handles also higher z dimensions. Uses the basic integration rule"""
    n = 1000
    box_width = 12/n
    z_range = np.linspace(-6,6,n)
    z = torch.Tensor(list(itertools.product(z_range, repeat=z_dim)))
    t1 = torch.ones(n**z_dim,1)
    t0 = torch.zeros(n**z_dim,1)
    ypred1 = torch.sigmoid(model.decoder(z,t1)[2]).detach().numpy().squeeze()
    ypred0 = torch.sigmoid(model.decoder(z,t0)[2]).detach().numpy().squeeze()
    p_y_dot1 = np.sum(ypred1 * np.product(scipy.stats.norm.pdf(z),1))*box_width**z_dim
    p_y_dot0 = np.sum(ypred0 * np.product(scipy.stats.norm.pdf(z),1))*box_width**z_dim
    return p_y_dot1 - p_y_dot0
        
def avg_causal_L1_dist(model, c_yt, c_yz, s_y, c_t, s_t, c_x, p_y_zt_nn, n=100, lim=6):
    """Calculates
    ∫|𝑃(𝑦|𝑑𝑜(𝑡),𝑃𝑡𝑟𝑢𝑒(𝑦|𝑑𝑜(𝑡))|𝐿1𝑃(𝑡)𝑑𝑡
    for the model with continuous t and y. Assumes z_dim=1̈́
    TODO: Maybe this should have some kind error analysis"""
    #First calculate the P(t) function
    t_range = np.linspace(-lim,lim,n)
    z_range = np.linspace(-lim,lim,n)
    y_range = np.linspace(-lim,lim,n)
    z_len = 2*lim/(n-1)
    y_len = 2*lim/(n-1)
    t_len = 2*lim/(n-1)
    #P(t|z)
    pt_z_mean_true = c_t*z_range
    pt_z_std_true = s_t

    #P(t) (integration by the simplest possible rule here, could make better with e.g. trapezoidal)
    pt_true = (scipy.stats.norm.pdf(z_range[:,None])*scipy.stats.norm.pdf(t_range[None,:], pt_z_mean_true[:,None],pt_z_std_true)).sum(axis=0)*z_len#shape (z_range, t_range)

    #Find out whether we should flip z in case the model learned it the wrong way around
    #NOTE: Probably not needed for P(y|do(t)) since integrated out, but otherwise could matter
    z_range_model = z_range if np.sign(model.decoder.x_nns[0].weight.item()) == np.sign(c_x[0]) else np.flip(z_range)

    #P(y|z,t)
    zt_range = np.concatenate([np.repeat(z_range_model[:,None],n,axis=0),np.tile(t_range[:,None],(n,1))],axis=1)#shape (1000*1000, 2)
    if p_y_zt_nn:
        py_zt_mean_model = torch.reshape(model.decoder.y_nn_real(torch.Tensor(zt_range)),(n,n)).detach().numpy()#shape (z_range, t_range)
    else:
        py_zt_mean_model = torch.reshape(model.decoder.y_nn(torch.Tensor(zt_range)),(n,n)).detach().numpy()#shape (z_range, t_range)
    py_zt_std_model = torch.exp(model.decoder.y_log_std).detach().numpy()
    py_zt_mean_true = c_yz*z_range[:,None] + c_yt*t_range[None,:]
    py_zt_std_true = s_y

    #P(y|do(t))
    py_dot_model = np.zeros((n,n))#shape (t_range, y_range)
    py_dot_true = np.zeros((n,n))
    for y_index in range(n):
        py_zt_model = scipy.stats.norm.pdf(y_range[y_index], py_zt_mean_model, py_zt_std_model)#shape (z_range, t_range)
        py_dot_model[:,y_index] = (py_zt_model * scipy.stats.norm.pdf(z_range_model[:,None])).sum(axis=0)*z_len
        py_zt_true = scipy.stats.norm.pdf(y_range[y_index], py_zt_mean_true, py_zt_std_true)#shape (z_range, t_range)
        py_dot_true[:,y_index] = (py_zt_true * scipy.stats.norm.pdf(z_range[:,None])).sum(axis=0)*z_len

    #The average distances between P_model(y|do(t)) and P_true(y|do(t))
    causal_dist = np.abs(py_dot_model - py_dot_true).sum(axis=1)*y_len#shape (t_range)
    avg_causal_dist = (causal_dist * pt_true).sum()*t_len
    return avg_causal_dist, py_dot_model, py_dot_true, y_range, t_range, pt_true

def avg_causal_L1_dist_general(model, c_yt, c_yz, s_y, c_t, s_t, c_x, n=100, lim=6):
    """Calculates
    ∫|𝑃(𝑦|𝑑𝑜(𝑡),𝑃𝑡𝑟𝑢𝑒(𝑦|𝑑𝑜(𝑡))|𝐿1𝑃(𝑡)𝑑𝑡
    for the model with continuous t and y. Doesn't assume z_dim=1.
    TODO: Maybe this should have some kind error analysis"""
    #First calculate the P(t) function
    t_range = np.linspace(-lim,lim,n)
    z_range = np.linspace(-lim,lim,n)
    y_range = np.linspace(-lim,lim,n)
    z_len = 2*lim/n
    y_len = 2*lim/n
    t_len = 2*lim/n
    z_dim = model.z_dim
    
    #P(t|z)
    pt_z_mean_true = c_t*z_range
    pt_z_std_true = s_t
    
    #P(t)
    pt_true = (scipy.stats.norm.pdf(z_range[:,None])*scipy.stats.norm.pdf(t_range[None,:], pt_z_mean_true[:,None],pt_z_std_true)).sum(axis=0)*z_len#shape (t_range,)
    
    #P(y|do(t)) for the model
    py_zt_std_model = torch.exp(model.decoder.y_log_std).detach().numpy()
    py_zt_std_true = s_y
    py_dot_model = np.zeros((n,n))#shape (t_range, y_range)
    for z in itertools.product(z_range, repeat=z_dim):
        zt_range = np.concatenate([np.tile(np.array(z),(n,1)),t_range[:,None]], axis=1)
        if model.decoder.p_y_zt_nn:
            if model.decoder.p_y_zt_std:
                py_zt_res_model = model.decoder.y_nn_real(torch.Tensor(zt_range)).detach().numpy()
                py_zt_mean_model = py_zt_res_model[:,0]
                py_zt_std_model = np.exp(py_zt_res_model[:,1][:,None])#Overwrites the constant std assumed above
            else:
                py_zt_mean_model = torch.reshape(model.decoder.y_nn_real(torch.Tensor(zt_range)),(n,)).detach().numpy()#shape (t_range,)
        else:
            py_zt_mean_model = torch.reshape(model.decoder.y_nn(torch.Tensor(zt_range)),(n,)).detach().numpy()#shape (z_range, 
        py_zt_model = scipy.stats.norm.pdf(y_range[None,:], py_zt_mean_model[:,None], py_zt_std_model)#shape (t_range, y_range)
        py_dot_model += py_zt_model * scipy.stats.norm.pdf(z).prod() * z_len**z_dim
    
    #P(y|do(t)) for the true distribution
    py_zt_mean_true = c_yz*z_range[:,None] + c_yt*t_range[None,:]
    py_zt_std_true = s_y
    py_dot_true = np.zeros((n,n))
    for y_index in range(n):
        py_zt_true = scipy.stats.norm.pdf(y_range[y_index], py_zt_mean_true, py_zt_std_true)#shape (z_range, t_range)
        py_dot_true[:,y_index] = (py_zt_true * scipy.stats.norm.pdf(z_range[:,None])).sum(axis=0)*z_len
        
    #The average distances between P_model(y|do(t)) and P_true(y|do(t))
    causal_dist = np.abs(py_dot_model - py_dot_true).sum(axis=1)*y_len#shape (t_range)
    avg_causal_dist = (causal_dist * pt_true).sum()*t_len
    return avg_causal_dist, py_dot_model, py_dot_true, y_range, t_range, pt_true

def avg_causal_L1_dist_MC(model, c_yt, c_yz, s_y, c_t, s_t, c_x, n=100, lim=6, nsample=10000):
    """Calculates
    ∫|𝑃(𝑦|𝑑𝑜(𝑡),𝑃𝑡𝑟𝑢𝑒(𝑦|𝑑𝑜(𝑡))|𝐿1𝑃(𝑡)𝑑𝑡
    for the model with continuous t and y, using MC integration for p(y|do(t))"""
    t_range = np.linspace(-lim,lim,n)
    y_range = np.linspace(-lim,lim,n)
    z_range = np.linspace(-lim,lim,n)
    z_len = 2*lim/n
    y_len = 2*lim/n
    t_len = 2*lim/n
    #First calculate the true P(t) function
    #P(t|z)
    pt_z_mean_true = c_t*z_range
    pt_z_std_true = s_t
    #P(t)
    pt_true = (scipy.stats.norm.pdf(z_range[:,None])*scipy.stats.norm.pdf(t_range[None,:], pt_z_mean_true[:,None],pt_z_std_true)).sum(axis=0)*z_len#shape (t_range,)
    #P(y|do(t)) for the model
    py_dot_model = np.zeros((n,n))#shape (t_range, y_range)
    py_zt_std_model = torch.exp(model.decoder.y_log_std).detach().numpy()
    for i,t in enumerate(t_range):
        z_sample = dist.Normal(0,1).sample((nsample,model.z_dim))
        zt = torch.cat([z_sample,torch.ones(nsample,1)*t],1)
        if model.decoder.p_y_zt_nn:
            if model.decoder.p_y_zt_std:
                py_zt_res_model = model.decoder.y_nn_real(zt).detach().numpy()
                py_zt_mean_model = py_zt_res_model[:,0]
                py_zt_std_model = np.exp(py_zt_res_model[:,1][:,None])
            else:
                py_zt_mean_model = model.decoder.y_nn_real(zt).detach().numpy().squeeze()
        else:
            py_zt_mean_model = model.decoder.y_nn(zt).detach().numpy().squeeze()
        py_zt_model = scipy.stats.norm.pdf(y_range[None,:], py_zt_mean_model[:,None], py_zt_std_model)
        py_dot_model[i,:] = py_zt_model.mean(0)
        
    #P(y|do(t)) for the true distribution
    py_zt_mean_true = c_yz*z_range[:,None] + c_yt*t_range[None,:]
    py_zt_std_true = s_y
    py_dot_true = np.zeros((n,n))
    for y_index in range(n):
        py_zt_true = scipy.stats.norm.pdf(y_range[y_index], py_zt_mean_true, py_zt_std_true)#shape (z_range, t_range)
        py_dot_true[:,y_index] = (py_zt_true * scipy.stats.norm.pdf(z_range[:,None])).sum(axis=0)*z_len
    
    #The average distances between P_model(y|do(t)) and P_true(y|do(t))
    causal_dist = np.abs(py_dot_model - py_dot_true).sum(axis=1)*y_len#shape (t_range)
    avg_causal_dist = (causal_dist * pt_true).sum()*t_len
    return avg_causal_dist, py_dot_model, py_dot_true, y_range, t_range, pt_true