import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
import torch.distributions as dist
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from CEVAE import *
import scipy
import pandas as pd
import pickle
import os
import glob
import re
import copy
import functools
from collections.abc import Iterable

class CEVAEDataset(Dataset):
    def __init__(self, df):
        x_cols = [c for c in df.columns if c.startswith("x")]
        self.X = torch.Tensor(df[x_cols].to_numpy())
        self.t = torch.Tensor(df["t"].to_numpy()[:,None])
        self.y = torch.Tensor(df["y"].to_numpy()[:,None])
        self.length = len(df)
    
    def __getitem__(self, idx):
        return {
            'X': self.X[idx],
            't': self.t[idx],
            'y': self.y[idx]
        }
    def __len__(self):
        return self.length
    
from scipy.stats import multinomial
def categorical_data_df(num_samples, z_probs, x_probs, t_probs, y_probs):
    """z_probs of shape (z_categories,), x_probs of shape (z_categories, x_dim, x_categories), 
    t_probs of shape (z_categories, t_categories), y_probs of shape (z_categories, t_categories, y_categories)"""
    z_c = z_probs.shape[0]
    _, x_dim, x_c = x_probs.shape
    _, t_c, y_c = y_probs.shape
    z = np.random.choice(z_c,num_samples,p=z_probs)
    x = np.zeros((num_samples,x_dim))
    t = np.zeros(num_samples)
    y = np.zeros(num_samples)
    for z_cat in range(z_c):
        num_z_cat = sum(z==z_cat)
        for x_d in range(x_dim):
            x[z==z_cat,x_d] = np.random.choice(x_c,num_z_cat,p=x_probs[z_cat,x_d,:])
        t[z==z_cat] = np.random.choice(t_c,num_z_cat,p=t_probs[z_cat,:])
    for z_cat in range(z_c):
        for t_cat in range(t_c):
            indices = (z==z_cat) & (t==t_cat)
            y[indices] = np.random.choice(y_c,sum(indices),p=y_probs[z_cat,t_cat,:])
    z = z[:,None]
    t = t[:,None]
    y = y[:,None]
    df = pd.DataFrame(np.concatenate([z,x,t,y],1),columns=['z'] + ['x{}'.format(i) for i in range(x_dim)] + 
                      ['t','y'])
    return df

from scipy.stats import dirichlet
def generate_categorical_dist(z_alpha,x_alpha,t_alpha,y_alpha):
    """z_alpha of shape (z_categories,), x_alpha of shape (z_categories, x_dim, x_categories),
    t_alpha of shape (z_categories, t_categories), y_alpha of shape (z_categories, t,categories, y_categories)"""
    z_c, x_dim, x_c = x_alpha.shape
    _, t_c, y_c = y_alpha.shape
    z_probs = dirichlet.rvs(z_alpha, size=1)[0]
    x_probs = np.zeros(x_alpha.shape)
    t_probs = np.zeros(t_alpha.shape)
    y_probs = np.zeros(y_alpha.shape)
    for z_cat in range(z_c):
        for x_d in range(x_dim):
            x_probs[z_cat,x_d,:] = dirichlet.rvs(x_alpha[z_cat,x_d,:],size=1)[0]
        t_probs[z_cat,:] = dirichlet.rvs(t_alpha[z_cat,:],size=1)[0]
        for t_cat in range(t_c):
            y_probs[z_cat,t_cat,:] = dirichlet.rvs(y_alpha[z_cat,t_cat,:],size=1)[0]
    return z_probs,x_probs,t_probs,y_probs

def generate_dist_and_data(num_samples,z_alpha,x_alpha,t_alpha,y_alpha):
    #Wrapper function e.g. for function run_model_for_data_sets
    z_probs,x_probs,t_probs,y_probs = generate_categorical_dist(z_alpha,x_alpha,t_alpha,y_alpha)
    df = categorical_data_df(num_samples, z_probs, x_probs, t_probs, y_probs)
    return (df,(z_probs,x_probs,t_probs,y_probs))

import torch.nn.functional as F
def estimate_model_py_dot(model,n=10000):
    """Estimates the p(y|do(t)) for the CEVAE model defined in CEVAE.py. If y is Gaussian-distributed,
    the result is actually E[y|do(t)]. Assumes that t is categorical."""
    if model.y_mode != 0:
        py_dot = np.zeros((model.t_mode, model.y_mode))
    else:
        py_dot = np.zeros(model.t_mode)
    for t_idx in range(model.t_mode):
        if model.z_mode == 0:
            z = torch.randn(n,model.z_dim)
            t = torch.ones(n,1)*t_idx
            _,_,_,_,y_pred,y_std = model.decoder(z,t)
            if model.y_mode == 2:#y_pred shape (n,1)
                py_dot[t_idx,:] = np.array([1-torch.sigmoid(y_pred).mean().item(),torch.sigmoid(y_pred).mean().item()])
            elif model.y_mode > 2:#y_pred shape (n,y_mode)
                py_dot[t_idx,:] = F.softmax(y_pred,dim=1).mean(0).detach().numpy()
            elif model.y_mode == 0:
                py_dot[t_idx] = y_pred.mean().item()#Really, E[y|do(t)]
        elif model.z_mode == 2:
            prior = torch.Tensor([[torch.sigmoid(model.pz_logit)],[1-torch.sigmoid(model.pz_logit)]])
            z = torch.Tensor([[1],[0]])
            t = torch.Tensor([[t_idx],[t_idx]])
            _,_,_,_,y_pred,y_std = model.decoder(z,t)# (2 x y_mode)
            if model.y_mode == 2:#y_pred shape (n,1)
                py_dot[t_idx,:] = np.array([((1-torch.sigmoid(y_pred))*prior).sum().detach().numpy(), 
                                            (torch.sigmoid(y_pred)*prior).sum().detach().numpy()])
            elif model.y_mode > 2:#y_pred shape (n,y_mode)
                py_dot[t_idx,:] = (F.softmax(y_pred,dim=1)*prior).sum(0).detach().numpy()
            elif model.y_mode == 0:
                py_dot[t_idx] = (y_pred*prior).sum(0).item()#Really, E[y|do(t)]
    return py_dot

def estimate_true_py_dot(z_probs,y_probs):
    return (y_probs*z_probs[:,None,None]).sum(0)

def estimate_AID_from_py_dot(py_dot,z_probs, t_probs, y_probs):
    #Estimates the AID for a given estimated p(y|do(t)), which has dimension (dim t, dim y)
    t_marginal_probs = (z_probs[:,None]*t_probs).sum(0)
    true_py_dot = estimate_true_py_dot(z_probs,y_probs)
    AID = (np.abs(py_dot - true_py_dot).sum(1)*t_marginal_probs).sum()
    return AID

def estimate_AID(model,z_probs,t_probs,y_probs,n=10000):
    #Estimates the AID for a CEVAE model defined in CEVAE.py where t is categorical
    t_marginal_probs = (z_probs[:,None]*t_probs).sum(0)
    model_py_dot = estimate_model_py_dot(model,n)
    true_py_dot = estimate_true_py_dot(z_probs,y_probs)
    AID = (np.abs(model_py_dot - true_py_dot).sum(1)*t_marginal_probs).sum()
    return AID

def estimate_AID_lineardata(model, c_yt, c_yz, s_y, c_t, s_t, c_x, n=100, lim=6, nsample=10000):
    #Estimates the model AID for linear-Gaussian data
    #Estimates AID for data generated with the linear structural model specified by the parameters
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
    for i,t in enumerate(t_range):
        z_sample = dist.Normal(0,1).sample((nsample,model.z_dim))
        zt = torch.cat([z_sample,torch.ones(nsample,1)*t],1)
        if model.decoder.common_stds:
            py_zt_mean_model = model.decoder.y_nn(zt).detach().numpy()[:,0]
            py_zt_std_model = torch.exp(model.decoder.y_log_std).repeat(nsample).detach()
        else:
            py_zt_res = model.decoder.y_nn(zt).detach().numpy()
            py_zt_mean_model, py_zt_std_model = py_zt_res[:,0], np.exp(py_zt_res[:,1])#std has to be estimated with linear data
        py_zt_model = scipy.stats.norm.pdf(y_range[None,:], py_zt_mean_model[:,None], py_zt_std_model[:,None])
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
    AID = (causal_dist * pt_true).sum()*t_len
    return AID, py_dot_model, py_dot_true, y_range, t_range, pt_true

def kld_loss(mu, std):
    #Note that the sum is over the dimensions of z as well as over the units in the batch here
    var = std.pow(2)
    kld = -0.5 * torch.sum(1 + torch.log(var) - mu.pow(2) - var)
    return kld

def kld_loss_binary(z_pred, pz_logit):
    probs = torch.sigmoid(z_pred)
    prior = torch.sigmoid(pz_logit)
    kld = (probs*torch.log(probs/prior) + (1-probs)*torch.log((1-probs)/(1-prior))).sum()
    return kld

def get_losses(z_mean, z_std, x_pred, x_std, t_pred, t_std, y_pred, y_std,
                  x, t, y, x_mode, t_mode, y_mode, kl_scaling=1):
    kld = kld_loss(z_mean,z_std)*kl_scaling
    x_loss = 0
    t_loss = 0
    y_loss = 0
    pred_i = 0#x_pred is much longer than x if x has categorical variables with more categories than 2
    for i,mode in enumerate(x_mode):
        if mode==0:
            x_loss += -dist.Normal(loc=x_pred[:,pred_i],scale=x_std[:,pred_i]).log_prob(x[:,i]).sum()
            pred_i += 1
        elif mode==2:
            x_loss += -dist.Bernoulli(logits=x_pred[:,pred_i]).log_prob(x[:,i]).sum()
            pred_i += 1
        else:
            x_loss += -dist.Categorical(logits=x_pred[:,pred_i:pred_i+mode]).log_prob(x[:,i]).sum()
            pred_i += mode
    if t_mode==2:
        t_loss = -dist.Bernoulli(logits=t_pred).log_prob(t).sum()
    elif t_mode==0:
        t_loss = -dist.Normal(loc=t_pred,scale=t_std).log_prob(t).sum()
    else:
        t_loss = -dist.Categorical(logits=t_pred).log_prob(t[:,0]).sum()
    if y_mode ==2:
        y_loss = -dist.Bernoulli(logits=y_pred).log_prob(y).sum()
    elif y_mode == 0:
        y_loss = -dist.Normal(loc=y_pred,scale=y_std).log_prob(y).sum()
    else:
        y_loss = -dist.Categorical(logits=y_pred).log_prob(y[:,0]).sum()
    return kld, x_loss, t_loss, y_loss

def get_losses_binary(z_pred, x_pred, x_std, t_pred, t_std, y_pred, y_std, x, t, y, pz_logit, x_mode, t_mode, y_mode):
    kld = kld_loss_binary(z_pred, pz_logit)
    qz_probs = torch.cat([1-torch.sigmoid(z_pred),torch.sigmoid(z_pred)],0).squeeze()
    x = torch.cat([x,x],0)#Purpose is to evaluate for both parts of the expected value, t_pred etc. should be prepared for this
    y = torch.cat([y,y],0)
    t = torch.cat([t,t],0)
    x_loss = 0
    t_loss = 0
    y_loss = 0
    pred_i = 0
    for i,mode in enumerate(x_mode):
        if mode==0:
            x_loss += -(dist.Normal(loc=x_pred[:,pred_i],scale=x_std[:,pred_i]).log_prob(x[:,i])*qz_probs).sum()
            pred_i += 1
        elif mode==2:
            x_loss += -(dist.Bernoulli(logits=x_pred[:,pred_i]).log_prob(x[:,i])*qz_probs).sum()
            pred_i += 1
        else:
            x_loss += -(dist.Categorical(logits=x_pred[:,pred_i:pred_i+mode]).log_prob(x[:,i])*qz_probs).sum()
            pred_i += mode
    if t_mode==2:
        t_loss = -(dist.Bernoulli(logits=t_pred).log_prob(t).squeeze()*qz_probs).sum()
    elif t_mode==0:
        t_loss = -(dist.Normal(loc=t_pred,scale=t_std).log_prob(t).squeeze()*qz_probs).sum()
    else:
        t_loss = -(dist.Categorical(logits=t_pred).log_prob(t[:,0]).squeeze()*qz_probs).sum()
    if y_mode ==2:
        y_loss = -(dist.Bernoulli(logits=y_pred).log_prob(y).squeeze()*qz_probs).sum()
    elif y_mode == 0:
        y_loss = -(dist.Normal(loc=y_pred,scale=y_std).log_prob(y).squeeze()*qz_probs).sum()
    else:
        y_loss = -(dist.Categorical(logits=y_pred).log_prob(y[:,0]).squeeze()*qz_probs).sum()
    return kld, x_loss, t_loss, y_loss
    
def train_model(device, plot_curves, print_logs,
              train_loader, num_epochs, lr_start, lr_end, x_dim, z_dim,
              p_y_zt_nn_layers=3, p_y_zt_nn_width=10, 
              p_t_z_nn_layers=3, p_t_z_nn_width=10,
              p_x_z_nn_layers=3, p_x_z_nn_width=10,
              q_z_nn_layers=3, q_z_nn_width=10,
              t_mode=2, y_mode=2, x_mode=[0], ty_separate_enc=False, 
              z_mode=0, x_loss_scaling=1, common_stds=False, collect_params=False, kl_scaling_schedule=None):
    
    model = CEVAE(x_dim, z_dim, device, p_y_zt_nn_layers,
        p_y_zt_nn_width, p_t_z_nn_layers, p_t_z_nn_width,
        p_x_z_nn_layers, p_x_z_nn_width, 
        q_z_nn_layers, q_z_nn_width,
        t_mode,y_mode,x_mode,ty_separate_enc, z_mode, common_stds)
    optimizer = Adam(model.parameters(), lr=lr_start)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = (lr_end/lr_start)**(1/num_epochs))
    
    losses = {"total": [], "kld": [], "x": [], "t": [], "y": []}
    
    modelparams = []
    
    kl_scalings = []
    if kl_scaling_schedule:
        for i in range(num_epochs):
            index = 0
            for j in range(len(kl_scaling_schedule[0])):
                if i/num_epochs >= kl_scaling_schedule[0][j]:
                    index = j
            kl_scaling = ((kl_scaling_schedule[0][index+1]-i/num_epochs)*kl_scaling_schedule[1][index] + (i/num_epochs-kl_scaling_schedule[0][index])*kl_scaling_schedule[1][index+1])/(kl_scaling_schedule[0][index+1]-kl_scaling_schedule[0][index])
            kl_scalings.append(kl_scaling)
    
    for epoch in range(num_epochs):
        #i = 0
        epoch_loss = 0
        epoch_kld_loss = 0
        epoch_x_loss = 0
        epoch_t_loss = 0
        epoch_y_loss = 0
        if print_logs:
            print("Epoch {}:".format(epoch))
            if kl_scaling_schedule:
                print("KL scaling: {}".format(kl_scalings[epoch]))
        for data in train_loader:
            x = data['X'].to(device)
            t = data['t'].to(device)
            y = data['y'].to(device)
            z_pred, z_std, x_pred, x_std, t_pred, t_std, y_pred, y_std = model(x,t,y)
            if z_mode == 0:
                if kl_scaling_schedule is not None:
                    kld, x_loss, t_loss, y_loss = get_losses(z_pred, z_std, x_pred, x_std, t_pred, t_std, y_pred, y_std, x, t, y,
                                                        x_mode, t_mode, y_mode, kl_scalings[epoch])
                else:
                    kld, x_loss, t_loss, y_loss = get_losses(z_pred, z_std, x_pred, x_std, t_pred, t_std, y_pred, y_std, x, t, y,
                                                        x_mode, t_mode, y_mode)
            elif z_mode == 2:
                kld, x_loss, t_loss, y_loss = get_losses_binary(z_pred, x_pred, x_std, t_pred, t_std, y_pred, y_std, x, t, y,
                                                         model.pz_logit, x_mode, t_mode, y_mode)
            x_loss *= x_loss_scaling
            loss = kld + x_loss + t_loss + y_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #i += 1
            #if i%100 == 0 and print_logs:
            #    print("Sample batch loss: {}".format(loss))
            epoch_loss += loss.item()
            epoch_kld_loss += kld.item()
            epoch_x_loss += x_loss.item()
            epoch_t_loss += t_loss.item()
            epoch_y_loss += y_loss.item()
        
        losses['total'].append(epoch_loss)
        losses['kld'].append(epoch_kld_loss)
        losses['x'].append(epoch_x_loss)
        losses['t'].append(epoch_t_loss)
        losses['y'].append(epoch_y_loss)
        if collect_params:
            if collect_params == 2:
                modelparams.append(estimate_model_py_dot(model))
            else:
                modelparams.append(model.decoder.y_nn[0].weight.detach().numpy().copy())
        
        scheduler.step()
  
        if print_logs:
            #print("Estimated ATE {}, p(y=1|do(t=1)): {}, p(y=1|do(t=0)): {}".format(*estimate_imageCEVAE_ATE(model)))
            print("Epoch loss: {}".format(epoch_loss))
            print("x: {}, t: {}, y: {}, kld: {}".format(epoch_x_loss, epoch_t_loss,
                                                        epoch_y_loss, epoch_kld_loss))
    
    fig, ax = plt.subplots(2,2,figsize=(8,8))
    ax[0,0].plot(losses['x'])
    ax[0,1].plot(losses['t'])
    ax[1,0].plot(losses['y'])
    ax[1,1].plot(losses['kld'])
    ax[0,0].set_title("x loss")
    ax[0,1].set_title("t loss")
    ax[1,0].set_title("y loss")
    ax[1,1].set_title("kld loss")
    plt.show()
    print("Total loss in the end: ", losses['total'][-1])
    
    return model, losses, modelparams


def train_model_starting_from(device, plot_curves, print_logs, starting_model,
              train_loader, num_epochs, lr_start, lr_end, x_loss_scaling=1, collect_params=False):
    model = copy.deepcopy(starting_model)
    optimizer = Adam(model.parameters(), lr=lr_start)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = (lr_end/lr_start)**(1/num_epochs))
    modelparams = []
    
    t_mode, y_mode, x_mode, z_mode = model.t_mode, model.y_mode, model.x_mode, model.z_mode
    
    losses = {"total": [], "kld": [], "x": [], "t": [], "y": []}
    for epoch in range(num_epochs):
        i = 0
        epoch_loss = 0
        epoch_kld_loss = 0
        epoch_x_loss = 0
        epoch_t_loss = 0
        epoch_y_loss = 0
        if print_logs:
            print("Epoch {}:".format(epoch))
        for data in train_loader:
            x = data['X'].to(device)
            t = data['t'].to(device)
            y = data['y'].to(device)
            z_pred, z_std, x_pred, x_std, t_pred, t_std, y_pred, y_std = model(x,t,y)
            if z_mode == 0:
                kld, x_loss, t_loss, y_loss = get_losses(z_pred, z_std, x_pred, x_std, t_pred, t_std, y_pred, y_std, x, t, y,
                                                        x_mode, t_mode, y_mode)
            elif z_mode == 2:
                kld, x_loss, t_loss, y_loss = get_losses_binary(z_pred, x_pred, x_std, t_pred, t_std, y_pred, y_std, x, t, y,
                                                         model.pz_logit, x_mode, t_mode, y_mode)
            x_loss *= x_loss_scaling
            loss = kld + x_loss + t_loss + y_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            i += 1
            if i%100 == 0 and print_logs:
                print("Sample batch loss: {}".format(loss))
            epoch_loss += loss.item()
            epoch_kld_loss += kld.item()
            epoch_x_loss += x_loss.item()
            epoch_t_loss += t_loss.item()
            epoch_y_loss += y_loss.item()
        
        losses['total'].append(epoch_loss)
        losses['kld'].append(epoch_kld_loss)
        losses['x'].append(epoch_x_loss)
        losses['t'].append(epoch_t_loss)
        losses['y'].append(epoch_y_loss)
        if collect_params:
            if collect_params == 2:
                modelparams.append(estimate_model_py_dot(model))
            else:
                modelparams.append(model.decoder.y_nn[0].weight.detach().numpy().copy())
    fig, ax = plt.subplots(2,2,figsize=(8,8))
    ax[0,0].plot(losses['x'])
    ax[0,1].plot(losses['t'])
    ax[1,0].plot(losses['y'])
    ax[1,1].plot(losses['kld'])
    ax[0,0].set_title("x loss")
    ax[0,1].set_title("t loss")
    ax[1,0].set_title("y loss")
    ax[1,1].set_title("kld loss")
    plt.show()
    print("Total loss in the end: ", losses['total'][-1])
    
    return model, losses, modelparams

#---------------------------------functions related to running larger scale experiments----------------------

def expand_parameters(params, iterated):
    """Helper function to get the elements in params to be lists of len(iterated)"""
    new_params = len(params)*[None]
    for i in range(len(params)):
        if not isinstance(params[i], list):
            new_params[i] = len(iterated)*[params[i]]#dim (len(train_arguments), len(iterated))
        else:
            assert len(params[i]) == len(iterated)
            new_params[i] = params[i].copy()
    return new_params

def create_or_empty_folder(main_folder,sub_folder):
    try:
        os.mkdir("./data/{}/".format(main_folder))
    except OSError:
        pass
    try:
        os.mkdir("./data/{}/{}/".format(main_folder,sub_folder))
    except OSError:
        print("Creation of the directory './data/{}/{}/ failed. Trying to empty the same folder.".format(main_folder,sub_folder))
        files = glob.glob('./data/{}/{}/*'.format(main_folder, sub_folder))
        for f in files:
            os.remove(f)

def save_dataparameters(dataparameters, main_folder, sub_folder):
    create_or_empty_folder(main_folder,sub_folder)
    with open("./data/{}/{}/params".format(main_folder,sub_folder), "wb") as file:
        pickle.dump(dataparameters, file)

def load_dataparameters(main_folder, sub_folder):
    with open("./data/{}/{}/params".format(main_folder,sub_folder), "rb") as file:
        return pickle.load(file)

def create_dfs_datasets(generate_df, dataparameters, param_times, repeat, main_folder, sub_folder, labels, overwrite=True):
    #dataparameters has to be a list of lists, param_times is how many times we use one data parameter combination
    #repeat is a boolean that tells whether we should use the same data
    if overwrite:
        create_or_empty_folder(main_folder,sub_folder)
    
    dfs = {label: {} for label in labels}
    datasets = {label: {} for label in labels}
    for i,data_params in enumerate(dataparameters):
        if repeat:
            df = generate_df(*data_params)
            dataset = CEVAEDataset(df)
            #SAVE RESULTS
            with open("./data/{}/{}/df_{}".format(main_folder, sub_folder,labels[i]), "wb") as file:
                pickle.dump(df, file)
            for j in range(param_times):
                dfs[labels[i]][j] = df
                datasets[labels[i]][j] = dataset
        else:
            for j in range(param_times):
                df = generate_df(*data_params)
                dataset = CEVAEDataset(df)
                #SAVE RESULTS
                with open("./data/{}/{}/df_{}_{}".format(main_folder, sub_folder,labels[i],j), "wb") as file:
                    pickle.dump(df, file)
                dfs[labels[i]][j] = df
                datasets[labels[i]][j] = dataset
    return dfs, datasets

def load_dfs(main_folder, sub_folder, param_times=None):
    dfs = {}
    datasets = {}
    for file in os.listdir("data/{}/{}/".format(main_folder, sub_folder)):
        match = re.search(r"df_([^_]*)_(\d*)", file)
        if match:
            if not match.group(1) in dfs:
                dfs[match.group(1)] = {}
                datasets[match.group(1)] = {}
            with open("data/{}/{}/{}".format(main_folder,sub_folder,file), "rb") as file:
                dfs[match.group(1)][int(match.group(2))] = pickle.load(file)
                datasets[match.group(1)][int(match.group(2))] = CEVAEDataset(dfs[match.group(1)][int(match.group(2))])
        else:
            match = re.search(r"df_([^_]*)", file)
            with open("data/{}/{}/{}".format(main_folder,sub_folder,file), "rb") as file:
                dfs[match.group(1)] = {}
                datasets[match.group(1)] = {}
                df =  pickle.load(file)
                for i in range(param_times):
                    dfs[match.group(1)][i] = df
                    datasets[match.group(1)][i] = CEVAEDataset(df)
    return dfs, datasets
        
        
def run_model_for_predef_datasets(datasets, param_times, main_folder, sub_folder, BATCH_SIZE, track_function, true_value,
                                  device, train_arguments, labels, data_labels, overwrite=True):
    #Main folder organizes related experiments with same/similar data. Sub-folder has the results from this experiment
    #datasets can be different data for each label or the same data repeated many times, however we want
    if overwrite:
        create_or_empty_folder(main_folder,sub_folder)
    
    train_arguments = expand_parameters(train_arguments, labels)
    train_arguments = list(map(list,zip(*train_arguments))) #dim (len(iterated, len(train_arguments))
    
    models = {label: {} for label in labels}
    losses = {label: {} for label in labels}
    
    for i in range(len(labels)):
        print("Label ", labels[i])
        for j in range(param_times):
            dataloader = DataLoader(datasets[data_labels[i]][j], batch_size=BATCH_SIZE)
            #Running the model
            model, loss, savedparams = train_model(device, False, False, dataloader, *train_arguments[i])
            torch.save(model.state_dict(), "./data/{}/{}/model_{}_{}".format(main_folder,sub_folder,labels[i],j))
            with open("./data/{}/{}/loss_{}_{}".format(main_folder,sub_folder,labels[i],j), "wb") as file:
                pickle.dump(loss, file)
            with open("./data/{}/{}/savedparams_{}_{}".format(main_folder,sub_folder,labels[i],j), "wb") as file:
                pickle.dump(savedparams, file)
            print("Estimated causal effect: {} true value: {}".format(track_function(model), true_value))
            models[labels[i]][j] = model
            losses[labels[i]][j] = loss
    
    return models, losses

def load_models_losses(main_folder, sub_folder, train_arguments, labels, device):
    train_arguments = expand_parameters(train_arguments, labels)
    train_arguments = list(map(list, zip(*train_arguments)))
    #We see only the labels in the folder, but we want the indices for accessing other arguments (train_arguments)
    labels_to_index = dict(zip(map(str,labels), range(len(labels))))
    models = {}
    losses = {}
    for file in os.listdir("data/{}/{}/".format(main_folder, sub_folder)):
        match = re.search(r"([^_]*)_([^_]*)_(\d*)", file)
        if match.group(1) == "model":
            index = labels_to_index[match.group(2)]
            num_epochs, lr_start, lr_end, x_dim, z_dim, p_y_zt_nn_layers, p_y_zt_nn_width, p_t_z_nn_layers, p_t_z_nn_width, p_x_z_nn_layers, p_x_z_nn_width, q_z_nn_layers, q_z_nn_width, t_mode, y_mode, x_mode, ty_separate_enc, z_mode, x_loss_scaling, common_stds = train_arguments[index]
            model = CEVAE(x_dim, z_dim, device, p_y_zt_nn_layers, p_y_zt_nn_width, p_t_z_nn_layers,
                          p_t_z_nn_width, p_x_z_nn_layers, p_x_z_nn_width, q_z_nn_layers, q_z_nn_width,
                          t_mode, y_mode, x_mode, ty_separate_enc, z_mode, common_stds)
            model.load_state_dict(torch.load("data/{}/{}/{}".format(main_folder, sub_folder,file)))
            model.eval()
            if not match.group(2) in models:
                models[match.group(2)] = {int(match.group(3)): model}
            else:
                models[match.group(2)][int(match.group(3))] = model
        elif match.group(1) == "loss":
            with open("data/{}/{}/{}".format(main_folder, sub_folder, file), "rb") as file:
                if not match.group(2) in losses:
                    losses[match.group(2)] = {}
                losses[match.group(2)][int(match.group(3))] = pickle.load(file)
    return models, losses

def load_saved_params(main_folder, sub_folder, labels):
    saved_params = {}
    for file in os.listdir("data/{}/{}/".format(main_folder, sub_folder)):
        match = re.search(r"([^_]*)_([^_]*)_(\d*)", file)
        if match.group(1) == "savedparams":
            with open("data/{}/{}/{}".format(main_folder, sub_folder, file), "rb") as file:
                if not match.group(2) in saved_params:
                    saved_params[match.group(2)] = {}
                saved_params[match.group(2)][int(match.group(3))] = pickle.load(file)
    return saved_params

def run_starting_from_predef_model(dataset, times, model, main_folder, sub_folder, BATCH_SIZE, track_function, true_value,
                                   device, train_arguments, overwrite=True):
    if overwrite:
        create_or_empty_folder(main_folder,sub_folder)
    models = dict()
    losses = dict()
    for j in range(times):
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
        model_new, loss = train_model_starting_from(device, False, False, model, dataloader, *train_arguments)
        torch.save(model_new.state_dict(), "./data/{}/{}/model_{}".format(main_folder,sub_folder,j))
        with open("./data/{}/{}/loss_{}".format(main_folder,sub_folder,j), "wb") as file:
            pickle.dump(loss, file)
        print("Estimated causal effect: {} true value: {}".format(track_function(model_new), true_value))
        models[j] = model_new
        losses[j] = loss
    return models, losses

def load_models_losses_started_from_predef_model(main_folder, sub_folder, starting_model, device):
    models = {}
    losses = {}
    for file in os.listdir("data/{}/{}/".format(main_folder, sub_folder)):
        match = re.search(r"([^_]*)_(\d*)", file)
        if match.group(1) == "model":
            model = copy.deepcopy(starting_model)
            model.load_state_dict(torch.load("data/{}/{}/{}".format(main_folder, sub_folder,file)))
            model.eval()
            models[match.group(2)] = model
        elif match.group(1) == "loss":
            with open("data/{}/{}/{}".format(main_folder, sub_folder, file), "rb") as file:
                losses[match.group(2)] = pickle.load(file)
    return models, losses

def run_model_for_data_sets(datasize, param_times,
                            folder, name, 
                            BATCH_SIZE, generate_data, dataparameters, track_function, true_value,
                            device, train_arguments, labels,
                           share_data_between_runs=False):
    """train_arguments is a list with the following:
    num_epochs, lr_start, lr_end, x_dim, z_dim,
    p_y_zt_nn, p_y_zt_nn_layers, p_y_zt_nn_width, 
    p_t_z_nn, p_t_z_nn_layers, p_t_z_nn_width,
    p_x_z_nn, p_x_z_nn_layers, p_x_z_nn_width"""
    """Runs the model for a parameter sweep. Saves the results in data/{folder}.
    Currently just empties everything in the folder before starting on new stuff.
    Idea: Some of the arguments in train_arguments are datasize is lists, and 
    we iterate through those and save the results. 'iterated' is the list object which names 
    the results"""
    try:
        os.mkdir("data/{}/".format(folder))
    except OSError:
        print("Creation of the directory data/{}/ failed. Trying to empty the same folder.".format(folder))
        files = glob.glob('data/{}/*'.format(folder))
        for f in files:
            os.remove(f)
    assert not (isinstance(datasize,Iterable) and share_data_between_runs)#ensure that share_data_between_runs makes sense
    
    datasize = expand_parameters([datasize], labels)[0]
    train_arguments = expand_parameters(train_arguments, labels)
    train_arguments = list(map(list, zip(*train_arguments))) #dim (len(labels, len(train_arguments))
    
    datas = {label: {} for label in labels}
    models = {label: {} for label in labels}
    losses = {label: {} for label in labels}
    if share_data_between_runs:
        df = generate_data(datasize[0], *dataparameters)
        dataset = CEVAEDataset(df)
    for i in range(len(labels)):
        for j in range(param_times):
            num_samples = datasize[i]
            print("Training data size {}, run {}".format(num_samples, j+1))
            if not share_data_between_runs:
                df = generate_data(num_samples, *dataparameters)
                dataset = CEVAEDataset(df)
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
            #Running the model
            model, loss = train_model(device, False, False, dataloader, *train_arguments[i])
            
            #datas[labels[i]][j] = data
            #models[labels[i]][j] = model
            #losses[labels[i]][j] = loss

            torch.save(model.state_dict(), "./data/{}/model_{}_{}_{}".format(folder,name,labels[i],j))
            file = open("something.pkl", "wb")
            with open("./data/{}/data_{}_{}_{}".format(folder,name,labels[i],j), "wb") as file:
                pickle.dump(df, file)
            with open("./data/{}/loss_{}_{}_{}".format(folder,name,labels[i],j), "wb") as file:
                pickle.dump(loss, file)
            print("Estimated causal effect: {} true value: {}".format(track_function(model), true_value))
            
    return datas, models, losses


def run_model_for_cat_data_sets(datasize, param_times,
                            folder, name, 
                            BATCH_SIZE, generate_data,
                            device, train_arguments, labels, z_alpha, x_alpha, t_alpha, y_alpha):
    """train_arguments is a list with the following:
    num_epochs, lr_start, lr_end, x_dim, z_dim,
      p_y_zt_nn_layers, p_y_zt_nn_width, 
      p_t_z_nn_layers, p_t_z_nn_width,
      p_x_z_nn_layers, p_x_z_nn_width,
      q_z_nn_layers, q_z_nn_width,
      t_binary, y_binary, x_mode, ty_separate_enc"""
    """Runs the model for a parameter sweep. Saves the results in data/{folder}.
    Currently just empties everything in the folder before starting on new stuff.
    Idea: Some of the arguments in train_arguments are datasize is lists, and 
    we iterate through those and save the results. 'iterated' is the list object which names 
    the results.
    NEW: generates the data generating distribution according to z_alpha, x_alpha, t_alpha and y_alpha"""
    
    try:
        os.mkdir("data/{}/".format(folder))
    except OSError:
        print("Creation of the directory data/{}/ failed. Trying to empty the same folder.".format(folder))
        files = glob.glob('data/{}/*'.format(folder))
        for f in files:
            os.remove(f)
    
    datasize = expand_parameters([datasize], labels)[0]
    train_arguments = expand_parameters(train_arguments, labels)
    train_arguments = list(map(list, zip(*train_arguments))) #dim (len(iterated, len(train_arguments))
    
    datas = {label: {} for label in labels}
    models = {label: {} for label in labels}
    losses = {label: {} for label in labels}
    aux_datas = [0]*param_times
    for j in range(param_times):
        aux_data = generate_categorical_dist(z_alpha,x_alpha,t_alpha,y_alpha)
        aux_datas[j] = aux_data
    for i in range(len(labels)):
        print("Param value {}".format(labels[i]))
        for j in range(param_times):
            num_samples = datasize[i]
            print("Training data size {}, run {}".format(num_samples, j+1))
            df = generate_data(num_samples, *aux_datas[j])
            dataset = CEVAEDataset(df)
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
            #Running the model
            model, loss = train_model(device, False, False, dataloader, *train_arguments[i])
            
            #data = (z, images, x, t, y, dataset)
            datas[labels[i]][j] = df
            models[labels[i]][j] = model
            losses[labels[i]][j] = loss

            torch.save(model.state_dict(), "./data/{}/model_{}_{}_{}".format(folder,name,labels[i],j))
            file = open("something.pkl", "wb")
            with open("./data/{}/data_{}_{}_{}".format(folder,name,labels[i],j), "wb") as file:
                pickle.dump(df, file)
            with open("./data/{}/loss_{}_{}_{}".format(folder,name,labels[i],j), "wb") as file:
                pickle.dump(loss, file)
            print("Estimated causal effect: {} true value: {}".format(estimate_AID(model,aux_datas[j][0],aux_datas[j][2],aux_datas[j][3]), 0))
    
    with open("./data/{}/aux_datas_{}".format(folder,name), "wb") as file:
        pickle.dump(aux_datas, file)
    
    return datas, models, losses, aux_datas


def run_model_for_x_dims(datasize, param_times,
                            folder, name, 
                            BATCH_SIZE,
                            device, train_arguments, alpha, x_dim, n_cat):
    """This assumes that x_dim is a list which we want to iterate over.
    More specific because x_dim is more difficult to mess with using the run_model_for_data_sets function."""
    """TODO: Think about the software engineering side of all this. What would be a better, simpler way to run
    all these experiments? Can I figure out a really generic function?"""
    try:
        os.mkdir("data/{}/".format(folder))
    except OSError:
        print("Creation of the directory data/{}/ failed. Trying to empty the same folder.".format(folder))
        files = glob.glob('data/{}/*'.format(folder))
        for f in files:
            os.remove(f)
            
    datasize = expand_parameters([datasize], x_dim)[0]
    train_arguments = expand_parameters(train_arguments, x_dim)
    train_arguments = list(map(list, zip(*train_arguments))) #dim (len(iterated), len(train_arguments))
    
    datas = {label: {} for label in x_dim}
    models = {label: {} for label in x_dim}
    losses = {label: {} for label in x_dim}
    aux_datas = {label: {} for label in x_dim}
    
    for i in range(len(x_dim)):
        print("Param value {}".format(x_dim[i]))
        for j in range(param_times):
            z_alpha = np.array([2]*n_cat)
            x_alpha = np.array([[[2]*n_cat]*x_dim[i]]*n_cat)#x_dim 10
            t_alpha = np.array([[2]*n_cat]*n_cat)#t_cat 2
            y_alpha = np.array([[[2]*n_cat]*n_cat]*n_cat)#y_cat 2
            z_probs,x_probs,t_probs,y_probs = generate_categorical_dist(z_alpha,x_alpha,t_alpha,y_alpha)
            
            num_samples = datasize[i]
            print("Training data size {}, run {}".format(num_samples, j+1))
            df = categorical_data_df(num_samples,z_probs,x_probs,t_probs,y_probs)
            dataset = CEVAEDataset(df)
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
            #Running the model
            print("Train arguments: ", *train_arguments[i])
            model, loss = train_model(device, False, False, dataloader, *train_arguments[i])
            
            aux_data = (z_probs,x_probs,t_probs,y_probs)
            datas[x_dim[i]][j] = df
            models[x_dim[i]][j] = model
            losses[x_dim[i]][j] = loss
            aux_datas[x_dim[i]][j] = aux_data

            torch.save(model.state_dict(), "./data/{}/model_{}_{}_{}".format(folder,name,x_dim[i],j))
            file = open("something.pkl", "wb")
            with open("./data/{}/data_{}_{}_{}".format(folder,name,x_dim[i],j), "wb") as file:
                pickle.dump(df, file)
            with open("./data/{}/loss_{}_{}_{}".format(folder,name,x_dim[i],j), "wb") as file:
                pickle.dump(loss, file)
            with open("./data/{}/aux_data_{}_{}_{}".format(folder,name,x_dim[i],j), "wb") as file:
                pickle.dump(aux_data, file)
    
    return datas, models, losses, aux_datas


def load_dfs_models(folder, name, train_arguments, datasize, labels, device):
    """Loads dataframes and trained models from data/{folder}/ that match the experiment name"""
    datasize = expand_parameters([datasize], labels)
    train_arguments = expand_parameters(train_arguments, labels)
    train_arguments = list(map(list, zip(*train_arguments)))
    #We see only the labels in the folder, but we want the indices for accessing other arguments (train_arguments)
    labels_to_index = dict(zip(map(str,labels), range(len(labels))))
    datas = {}
    models = {}
    losses = {}
    for file in os.listdir("data/{}".format(folder)):
        #Group 1 data/model/loss identifier, group 2 is the name (unnecessary), group 3 is the experiment setup
        #and group 4 is the number of the try
        match = re.search(r"([^_]*)_([^_]*)_([^_]*)_(\d*)", file)
        if match is not None:
            if match.group(2) == name:
                if match.group(1) == "data":
                    if not match.group(3) in datas:
                        with open("data/{}/{}".format(folder,file), "rb") as file:
                            datas[match.group(3)] = {int(match.group(4)): pickle.load(file)}
                    else:
                        with open("data/{}/{}".format(folder,file), "rb") as file:
                            datas[match.group(3)][int(match.group(4))] = pickle.load(file)
                elif match.group(1) == "loss":
                    if not match.group(3) in losses:
                        with open("data/{}/{}".format(folder,file), "rb") as file:
                            losses[match.group(3)] = {int(match.group(4)): pickle.load(file)}
                    else:
                        with open("data/{}/{}".format(folder,file), "rb") as file:
                            losses[match.group(3)][int(match.group(4))] = pickle.load(file)
                elif match.group(1) == "model":
                    index = labels_to_index[match.group(3)]
                    num_epochs, lr_start, lr_end, x_dim, z_dim, p_y_zt_nn_layers, p_y_zt_nn_width, p_t_z_nn_layers, p_t_z_nn_width, p_x_z_nn_layers, p_x_z_nn_width, q_z_nn_layers, q_z_nn_width, t_mode, y_mode, x_mode, ty_separate_enc, z_mode = train_arguments[index]
                    
                    model = CEVAE(x_dim, z_dim, device, p_y_zt_nn_layers, p_y_zt_nn_width, p_t_z_nn_layers,
                                  p_t_z_nn_width, p_x_z_nn_layers, p_x_z_nn_width, q_z_nn_layers, q_z_nn_width,
                                  t_mode, y_mode, x_mode, ty_separate_enc, z_mode)
                    model.load_state_dict(torch.load("data/{}/{}".format(folder,file)))
                    model.eval()
                    if not match.group(3) in models:
                        models[match.group(3)] = {int(match.group(4)): model}
                    else:
                        models[match.group(3)][int(match.group(4))] = model
    return datas, models, losses