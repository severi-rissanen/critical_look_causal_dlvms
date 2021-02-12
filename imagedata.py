import numpy as np
import torch
import torch.distributions as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
class ImageDataset(Dataset):
    def __init__(self, images, x, t, y, z_dim=1):
        self.length = x.shape[0]
        x_dim = x.shape[1]
        self.images = images
        self.t = t
        self.X = x
        self.y = y

    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'X': self.X[idx],
            't': self.t[idx],
            'y': self.y[idx]
        }
    def __len__(self):
        return self.length
    
def sigmoid(x):
    return 1/(1+np.exp(-x))

def generate_image_data(num_samples, zdim, generator, t_a, t_b, y_a0, y_b0, y_a1, y_b1, c_x, s_x):
    z = torch.randn(num_samples, zdim)
    with torch.no_grad():
        image_expectations = (1+generator(z[:,:,None,None]))/2
        images = dist.Bernoulli(image_expectations).sample()#This way binary data
        #image_means = generator(z[:,:,None,None])
        #images = image_means# + torch.randn_like(image_means)*0.05 # <- this way continuous data
    z_temp = z[:,0][:,None].detach().numpy()#Use the first dimension for prediction of ordinary variables
    x = torch.Tensor(np.random.normal(np.tile(c_x, (num_samples,1))*z_temp,
                         np.tile(s_x, (num_samples,1)),(num_samples, 1)))
    t = (np.random.random((num_samples, 1)) < sigmoid(t_a*z_temp + t_b)).astype(int)
    y = torch.Tensor((np.random.random((num_samples, 1)) < sigmoid(y_a1*z_temp + y_b1)).astype(int)*t \
        + (np.random.random((num_samples, 1)) < sigmoid(y_a0*z_temp + y_b0)).astype(int)*(1-t))
    t = torch.Tensor(t)
    dataset = ImageDataset(images, x, t, y, zdim)
    return z, images, x, t, y, dataset