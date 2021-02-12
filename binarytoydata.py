import pandas as pd
import numpy as np
import math, random
import torch
import torch.distributions as dist
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from scipy import stats
import seaborn as sns

def binary_data_df(num_samples, 
                    z_expectation,
                    x_expectations,
                    t_expectations,
                    y_expectations) -> pd.DataFrame:
    """x_expectations is of dim (x vars)x(2). t_expectations is 1x2, y_expectations is 1x4.
    The columns correspond to different conditioning variable values, e.g. P(x=1|z=0) and P(x=1|z=1).
    """
    x_dim = x_expectations.shape[0]
    
    z_dist = dist.Bernoulli(probs=z_expectation)
    zs = z_dist.sample((num_samples,))
    
    #Column 0 corresponds to P(x=1|z=0) (multiple values) and column 1 to P(x=1|z=1)
    xs_0 = dist.Bernoulli(probs=x_expectations[:,0].T).sample((num_samples,))
    xs_1 = dist.Bernoulli(probs=x_expectations[:,1].T).sample((num_samples,))
    xs = torch.where(zs==1,xs_1,xs_0)
    
    #Column 0 corresponds to P(t=1|z=0) and column 1 to P(t=1|z=1)
    ts_0 = dist.Bernoulli(probs=t_expectations[:,0]).sample((num_samples,))
    ts_1 = dist.Bernoulli(probs=t_expectations[:,1]).sample((num_samples,))
    ts = torch.where(zs==1,ts_1,ts_0)
    
    #Combinations (z=0,t=0), (z=0,t=1), (z=1,t=0), (z=1,t=1). 
    #Columns of y_expectations are in that order e.g. P(y=1|z=0,t=0)
    yf_00 = dist.Bernoulli(probs=y_expectations[:,0]).sample((num_samples,))
    yf_01 = dist.Bernoulli(probs=y_expectations[:,1]).sample((num_samples,))
    yf_10 = dist.Bernoulli(probs=y_expectations[:,2]).sample((num_samples,))
    yf_11 = dist.Bernoulli(probs=y_expectations[:,3]).sample((num_samples,))
    #Actual outcomes
    yf = torch.zeros((num_samples,1))
    yf[(zs==0) & (ts==0)] = yf_00[(zs==0) & (ts==0)]
    yf[(zs==0) & (ts==1)] = yf_01[(zs==0) & (ts==1)]
    yf[(zs==1) & (ts==0)] = yf_10[(zs==1) & (ts==0)]
    yf[(zs==1) & (ts==1)] = yf_11[(zs==1) & (ts==1)]
    #Counterfactuals
    y0 = torch.zeros((num_samples,1))
    y0[(zs==0)] = yf_00[(zs==0)]
    y0[(zs==1)] = yf_10[(zs==1)]
    y1 = torch.zeros((num_samples,1))
    y1[(zs==0)] = yf_01[(zs==0)]
    y1[(zs==1)] = yf_11[(zs==1)]
    
    df = pd.DataFrame(torch.cat([zs,xs,ts,yf,y0,y1], axis=1).numpy(), columns=['z'] + ['x{}'.format(i) for i in range(x_dim)] + ['t','yf','y0','y1'])
    
    return df