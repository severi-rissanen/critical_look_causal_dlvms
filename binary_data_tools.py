from binarytoydata import *
import pandas as pd
import numpy as np
import sklearn
import itertools

import os
import math, random
import scipy.stats as stats
import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

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

from models import *

def calculate_true_ate(prob_df):
    true_py_do1 = 0#(z_expectation*y_expectations[0][3] + (1-z_expectation)*y_expectations[0][1]).item()
    true_py_do0 = 0#(z_expectation*y_expectations[0][2] + (1-z_expectation)*y_expectations[0][0]).item()
    for z in range(2):
        true_py_do1 += prob_df[(prob_df.yf==1)&(prob_df.z==z)&(prob_df.t==1)].P.sum()\
                   *1/prob_df[(prob_df.z==z)&(prob_df.t==1)].P.sum()\
                   *prob_df[(prob_df.z==z)].P.sum()
        true_py_do0 += prob_df[(prob_df.yf==1)&(prob_df.z==z)&(prob_df.t==0)].P.sum()\
                   *1/prob_df[(prob_df.z==z)&(prob_df.t==0)].P.sum()\
                   *prob_df[(prob_df.z==z)].P.sum()
    true_ate = true_py_do1 - true_py_do0
    return true_ate, true_py_do1, true_py_do0#TODO: returning py_do has breaked some old code

def calculate_proxy_ate(prob_df):
    proxy_py_do1 = 0
    proxy_py_do0 = 0
    x_combinations = list(itertools.product([0,1],repeat=2))
    for comb in x_combinations:
        x0,x1 = comb
        proxy_py_do1 += prob_df[(prob_df.yf==1)&(prob_df.x0==x0)&(prob_df.x1==x1)&(prob_df.t==1)].P.sum() \
                        *1/prob_df[(prob_df.x0==x0)&(prob_df.x1==x1)&(prob_df.t==1)].P.sum() \
                        *prob_df[(prob_df.x0==x0)&(prob_df.x1==x1)].P.sum()
        proxy_py_do0 += prob_df[(prob_df.yf==1)&(prob_df.x0==x0)&(prob_df.x1==x1)&(prob_df.t==0)].P.sum() \
                        *1/prob_df[(prob_df.x0==x0)&(prob_df.x1==x1)&(prob_df.t==0)].P.sum() \
                        *prob_df[(prob_df.x0==x0)&(prob_df.x1==x1)].P.sum()
    proxy_ate = proxy_py_do1 - proxy_py_do0
    return proxy_ate

def generate_calculate_prob(z_expec, x_expec, t_expec, y_expec):
    def calculate_prob(z,x,t,y):
        """Calculates a probability according to the numbers above"""
        """x is a seq, others are numbers"""
        z_expectation = z_expec.clone()
        x_expectations = x_expec.clone()
        t_expectations = t_expec.clone()
        y_expectations = y_expec.clone()
        
        prob = z_expectation*z + (1-z_expectation)*(1-z)
        prob *= (t_expectations[0][0]*(1-z)*t + (1-t_expectations[0][0])*(1-z)*(1-t) + 
                 t_expectations[0][1]*z*t + (1-t_expectations[0][1])*z*(1-t))
        prob *= (y_expectations[0][0]*(1-z)*(1-t)*y + (1-y_expectations[0][0])*(1-z)*(1-t)*(1-y) + 
                 y_expectations[0][1]*(1-z)*t*y + (1-y_expectations[0][1])*(1-z)*t*(1-y) + 
                 y_expectations[0][2]*z*(1-t)*y + (1-y_expectations[0][2])*z*(1-t)*(1-y) + 
                 y_expectations[0][3]*z*t*y + (1-y_expectations[0][3])*z*t*(1-y))
        for i in range(len(x_expectations)):
            prob *= (x_expectations[i][0]*(1-z)*x[i] + (1-x_expectations[i][0])*(1-z)*(1-x[i]) + 
                 x_expectations[i][1]*z*x[i] + (1-x_expectations[i][1])*z*(1-x[i]))
        return prob.item()
    
    return calculate_prob

def generate_p_y_xt(prob_df):
    """Generates a function that gives perfect values of P(y=1|X,t)"""
    combinations = itertools.product([0,1],repeat=2+1)
    prob_y_xt = []
    for comb in combinations:
        t = comb[-1]
        x = comb[:-1]
        prob_y = prob_df[(prob_df.yf==1)&(prob_df.x0==x[0])&(prob_df.x1==x[1])&(prob_df.t==t)].P.sum() \
                / prob_df[(prob_df.x0==x[0])&(prob_df.x1==x[1])&(prob_df.t==t)].P.sum()
        prob_y_xt.append([x[0],x[1],t,prob_y])
    def p_y_xt_f(x0,x1,t):
        prob_y_xt_df = pd.DataFrame(prob_y_xt, columns=['x0','x1','t','P'])
        prob_y_xt_df.set_index(keys=['x0','x1','t'], inplace=True)
        return prob_y_xt_df.loc[x0,x1,t].P
    return p_y_xt_f

def generate_q_y_xt(df):
    """Generates a function that estimates P(y=1|X,t) based on data."""
    combinations = itertools.product([0,1],repeat=2+1)
    prob_y_xt = []
    for comb in combinations:
        t = comb[-1]
        x = comb[:-1]
        prob_y = ((df.yf==1)&(df.x0==x[0])&(df.x1==x[1])&(df.t==t)).sum() \
                   / ((df.x0==x[0])&(df.x1==x[1])&(df.t==t)).sum()
        prob_y_xt.append([x[0],x[1],t,prob_y])
    def q_y_xt_f(x0,x1,t):
        prob_y_xt_df = pd.DataFrame(prob_y_xt, columns=['x0','x1','t','P'])
        prob_y_xt_df.set_index(keys=['x0','x1','t'], inplace=True)
        return prob_y_xt_df.loc[x0,x1,t].P
    return q_y_xt_f

def generate_p_t_x(prob_df):
    """Generates a function that gives perfect values of P(t=1|t)"""
    combinations = itertools.product([0,1],repeat=2)
    prob_t_x = []
    for comb in combinations:
        x = comb
        prob_t = prob_df[(prob_df.x0==x[0])&(prob_df.x1==x[1])&(prob_df.t==1)].P.sum() \
                / prob_df[(prob_df.x0==x[0])&(prob_df.x1==x[1])].P.sum()
        prob_t_x.append([x[0],x[1],prob_t])
    def p_t_x_f(x0,x1):
        prob_t_x_df = pd.DataFrame(prob_t_x, columns=['x0','x1','P'])
        prob_t_x_df.set_index(keys=['x0','x1'], inplace=True)
        return prob_t_x_df.loc[x0,x1].P
    return p_t_x_f

def generate_q_t_x(df):
    """Generates a function that gives perfect values of P(t=1|t)"""
    combinations = itertools.product([0,1],repeat=2)
    prob_t_x = []
    for comb in combinations:
        x = comb
        prob_t = ((df.x0==x[0])&(df.x1==x[1])&(df.t==1)).sum() \
                   / ((df.x0==x[0])&(df.x1==x[1])).sum()
        prob_t_x.append([x[0],x[1],prob_t])
    def q_t_x_f(x0,x1):
        prob_t_x_df = pd.DataFrame(prob_t_x, columns=['x0','x1','P'])
        prob_t_x_df.set_index(keys=['x0','x1'], inplace=True)
        return prob_t_x_df.loc[x0,x1].P
    return q_t_x_f

def modelITE(model, x, p_y_xt_f, p_t_x_f):
    """ITE estimation"""
    """model is the dummyCEVAE model and x is a torch tensor.
    p_y_xt_f and p_t_x_f can be the true values got from the data generating distribution, 
    if we don't want to take in to account the error of estimating those."""
    ones = torch.ones((x.shape[0],1), device=model.device)
    zeros = torch.zeros((x.shape[0],1), device=model.device)
    
    #Set up probabilities q(z|x,t,y) and q(t,y|x)
    q_z_t1y1x = torch.sigmoid(model.encoder(x,t=ones,y=ones)[2])
    q_z_t0y1x = torch.sigmoid(model.encoder(x,t=zeros,y=ones)[2])
    q_z_t1y0x = torch.sigmoid(model.encoder(x,t=ones,y=zeros)[2])
    q_z_t0y0x = torch.sigmoid(model.encoder(x,t=zeros,y=zeros)[2])
    
    #Setting up probabilities q(y|x,t) and q(t|x)
    q_y_xt = [[[0,0],[0,0]],[[0,0],[0,0]]]#order x0, x1, t: q_y_xt[x0][x1][t]
    combinations_3 = itertools.product([0,1],repeat=3)
    for x0,x1,t in combinations_3:
        q_y_xt[x0][x1][t] = torch.ones((x.shape[0], 1)) * p_y_xt_f(x0,x1,t)
    q_t_x = [[0,0],[0,0]]
    combinations_2 =itertools.product([0,1],repeat=2)
    for x0,x1 in combinations_2:
        q_t_x[x0][x1] = torch.ones((x.shape[0], 1)) * p_t_x_f(x0,x1)
    
    #calculating q(z|x)
    q_z_x = torch.zeros((x.shape[0], 1),device=model.device)
    for x0,x1 in itertools.product([0,1],repeat=2):
        q_z_x += ( q_z_t1y1x * q_y_xt[x0][x1][1] * q_t_x[x0][x1] \
                + q_z_t0y1x * q_y_xt[x0][x1][0] * (1-q_t_x[x0][x1]) \
                + q_z_t1y0x * (1-q_y_xt[x0][x1][1]) * q_t_x[x0][x1] \
                + q_z_t0y0x * (1-q_y_xt[x0][x1][0]) * (1-q_t_x[x0][x1]) )\
                * ((x[:,0].unsqueeze(1) == x0)&(x[:,1].unsqueeze(1) == x1)).double()

    #Integrating over z to get P(y=1|X,do(t)) according to the model
    p_y_do1_model = q_z_x * torch.sigmoid(model.decoder.y1_nn(ones)) \
                    + (1-q_z_x) * torch.sigmoid(model.decoder.y1_nn(zeros))
    p_y_do0_model = q_z_x * torch.sigmoid(model.decoder.y0_nn(ones)) \
                    + (1-q_z_x) * torch.sigmoid(model.decoder.y0_nn(zeros))
    model_ITE = p_y_do1_model - p_y_do0_model
    return model_ITE

def proxyITE(x, q_y_xt_f):
    """Estimates the ITE for x given the function q_y_xt_f, which estimates P(y=1|X,t) """
    proxy_ITE = torch.zeros((x.shape[0],1), device=x.device)
    
    #Setting up probabilities q(y|x,t)
    q_y_xt = [[[0,0],[0,0]],[[0,0],[0,0]]]#order x0, x1, t: q_y_xt[x0][x1][t]
    for x0,x1,t in itertools.product([0,1],repeat=3):
        q_y_xt[x0][x1][t] = torch.ones((x.shape[0], 1), device=x.device) * q_y_xt_f(x0,x1,t)
    
    #calculating proxy_ITE
    q_z_x = torch.zeros((x.shape[0], 1))
    for x0,x1 in itertools.product([0,1],repeat=2):
        proxy_ITE += (q_y_xt[x0][x1][1] - q_y_xt[x0][x1][0]) \
                    * ((x[:,0].unsqueeze(1) == x0)&(x[:,1].unsqueeze(1) == x1)).double()
    return proxy_ITE
    

def printCombinations(sample_data,df,prob_df):
    """Prints out the joint distributions of sample_data and df"""
    """Assumes that sample_data is torch tensor with columns as different variables
    and df columns are in the same order"""
    dist_VAE, dist_data, dist_true = getJointDistributions(sample_data, df, prob_df)
    labels = list(dist_VAE.columns)[:-1]
    
    for i in range(len(dist_VAE)):
        print_str = "P("
        for j in range(len(labels)):
            print_str += "{}={},".format(labels[j],dist_VAE.iloc[i,j])
        print_str = print_str[:-1] + ')\t{}(VAE)\t={}(data)\t{:.4f}(true)'\
            .format(dist_VAE.iloc[i].P, dist_data.iloc[i].P, dist_true.iloc[i].P)
        print(print_str)

def getJointDistributions(sample_data, df, prob_df):
    """Returns the joint distributions of sample_data and df, given the columns of df"""
    """Assumes that sample_data is torch tensor with columns as different variables
    and df columns are in the same order"""
    
    labels = list(df.columns)
    combinations = itertools.product([0,1],repeat=len(labels))
    
    dist_VAE = []
    dist_data = []
    dist_true = []
    
    for combination in combinations:
        VAEmatches = torch.ones(len(sample_data),1, device=sample_data.device).bool()
        datamatches = pd.Series(np.ones(len(df))).astype('bool')
        truematches = pd.Series(np.ones(len(prob_df))).astype('bool')
        for i in range(len(labels)):
            VAEmatches = VAEmatches & (sample_data[:,i]==combination[i]).unsqueeze(1)
            datamatches = datamatches & (df[labels[i]]==combination[i])
            truematches = truematches & (prob_df[labels[i]]==combination[i])
        probVAE = (VAEmatches).double().sum()/len(sample_data)
        probdata = datamatches.sum()/len(df)
        probtrue = prob_df[truematches].P.sum()
        dist_VAE.append(list(combination) + [probVAE])
        dist_data.append(list(combination) + [probdata])
        dist_true.append(list(combination) + [probtrue])
    
    dist_VAE = pd.DataFrame(dist_VAE, columns=labels + ['P'])
    dist_data = pd.DataFrame(dist_data, columns=labels + ['P'])
    dist_true = pd.DataFrame(dist_true, columns=labels + ['P'])
    
    return dist_VAE, dist_data, dist_true

def getJointDistFromDataFrame(df):
    labels = list(df.columns)
    combinations = itertools.product([0,1],repeat=len(labels))
    dist_data = []
    for combination in combinations:
        datamatches = pd.Series(np.ones(len(df))).astype('bool')
        for i in range(len(labels)):
            datamatches = datamatches & (df[labels[i]]==combination[i])
        probdata = datamatches.sum()/len(df)
        dist_data.append(list(combination) + [probdata])
    dist_data = pd.DataFrame(dist_data, columns=labels + ['P'])
    return dist_data

def get_x_given_z(df):
    P_xx = np.zeros((2,2))
    Q_zx = np.zeros((2,2))