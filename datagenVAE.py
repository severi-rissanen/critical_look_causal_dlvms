import torch
from torch import nn, optim
import torch.distributions as dist
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
import pandas as pd

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
        p_x_z_nn_layers,
        p_x_z_nn_width,
        x_mode
    ):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.device = device
        self.p_x_z_nn_layers = p_x_z_nn_layers
        self.p_x_z_nn_width = p_x_z_nn_width
        
        #Can be used as a linear predictor if num_hidden=0
        self.n_x_estimands = sum([1 if m==0 or m==2 else m for m in x_mode])
        #for each x we have the possible std estimator also for simplicity, possibly not used
        self.x_nn = FullyConnected([z_dim] + p_x_z_nn_layers*[p_x_z_nn_width] + [(self.n_x_estimands)*2])
        
        self.to(device)
        
    def forward(self,z):
        x_res = self.x_nn(z)
        x_pred = x_res[:,:self.n_x_estimands]
        x_std = torch.exp(x_res[:,self.n_x_estimands:])
        return x_pred,x_std

class Encoder(nn.Module):
    def __init__(
        self, 
        x_dim,
        z_dim,
        device,
        q_z_nn_layers,
        q_z_nn_width
    ):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.device = device
        self.q_z_nn_layers = q_z_nn_layers
        self.q_z_nn_width = q_z_nn_width
        
        # q(z|x,t,y)
        self.q_z_nn = FullyConnected([x_dim] + q_z_nn_layers*[q_z_nn_width] + [z_dim*2])
        
        self.to(device)
        
    def forward(self,x):
        z_res = self.q_z_nn(x)
        z_pred = z_res[:,:self.z_dim]
        z_std = torch.exp(z_res[:,self.z_dim:])
        return z_pred, z_std

class dataGeneratorVAE(nn.Module):
    def __init__(
        self, 
        x_dim,
        z_dim,
        device,
        p_x_z_nn_layers,
        p_x_z_nn_width,
        q_z_nn_layers,
        q_z_nn_width,
        x_mode#a list, 0 for continuous (Gaussian), 2 or more for categorical distributions (usually 2 or 0)
    ):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.device = device
        self.x_mode = x_mode
        assert all([x_m == 0 or x_m > 1 for x_m in x_mode])
        assert len(x_mode) == x_dim
        
        self.encoder = Encoder(
            x_dim,
            z_dim,
            device,
            q_z_nn_layers,
            q_z_nn_width
        )
        
        self.decoder = Decoder(
            x_dim,
            z_dim,
            device,
            p_x_z_nn_layers,
            p_x_z_nn_width,
            x_mode
        )
        
        self.to(device)
        self.float()
        
    def reparameterize(self, mean, std):
        # samples from unit norm and does reparam trick
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)
    
    def forward(self, x):
        z_mean, z_std = self.encoder(x)
        #TODO: works at least for z_dim=1, maybe errors if z_dim>1
        z = self.reparameterize(z_mean, z_std)
        x_pred, x_std = self.decoder(z)
        
        return z_mean, z_std, x_pred, x_std
    
    def sample(self,n):
        different_modes = list(set(self.x_mode))
        x_same_mode_indices = dict()
        for mode in different_modes:
            x_same_mode_indices[mode] = [i for i,m in enumerate(self.x_mode) if m==mode]

        z_sample = torch.randn(n, self.z_dim).to(self.device)
        x_pred, x_std = self.decoder(z_sample)
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
        
        return z_sample, x_sample
    
def generatedata(num_samples, VAEmodel, ztotmodel, ATE, zvar_index):
    _, zx = VAEmodel.sample(num_samples)
    z = torch.Tensor(zx[:,[zvar_index]])
    x = torch.Tensor(np.concatenate([zx[:,:zvar_index], zx[:,zvar_index+1:]],1))
    t = (torch.rand(num_samples, 1) < torch.sigmoid(ztotmodel(z))).float()
    y = (z+ATE)*t + z*(1-t) + torch.randn(num_samples,1)
    df = pd.DataFrame(torch.cat([z,x,t,y],1).detach().numpy(),columns=["z"]+["x{}".format(i) for i in range(VAEmodel.x_dim-1)] + ["t", "y"])
    return df

def rejection_sample(df, density_estimate_fun, sample_fun):
    #Samples according to a distribution specified by sample_fun (Gaussian in practise)
    #density_estimate_fun is a density estimate of z, sample_fun has to be below the density estimate at all points
    include = sample_fun(df['z'].to_numpy()) / density_estimate_fun(df['z'].to_numpy()[:,None]) > np.random.random(len(df['z']))
    return df.copy()[include]
    
def savemodel(model, name):
    torch.save(model.state_dict(), "./datageneratormodels/{}".format(name))
    
def loadmodel(name, modeltype, args):
    model = modeltype(*args)
    model.load_state_dict(torch.load("./datageneratormodels/{}".format(name)))
    return model

def trainZtoTmodel(device, z, t, lr_start, lr_end, num_epochs ,layers=3, width=10):
    #Returns a model that is 
    model = FullyConnected([1] + [width]*layers + [1])
    optimizer = Adam(model.parameters(), lr=lr_start)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = (lr_end/lr_start)**(1/num_epochs))
    
    dataset = GenericDataset(np.concatenate([z,t],1))
    dataloader = DataLoader(dataset, shuffle=True, batch_size=32)
    
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for data in dataloader:
            z = data[:,0].unsqueeze(1)
            t = data[:,1].unsqueeze(1)
            
            tpred = model(z)
            loss = -dist.Bernoulli(logits=tpred).log_prob(t).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss)
    plt.plot(losses)
    plt.title("Epoch loss")
    return model

class GenericDataset(Dataset):
    def __init__(self, data):
        self.X = torch.Tensor(data)
        self.length = len(data)
    
    def __getitem__(self, idx):
        return self.X[idx]
        
    def __len__(self):
        return self.length
    
    
def train_datagenerator(device, plot_curves, print_logs,
              train_loader, num_epochs, lr_start, lr_end, x_dim, z_dim,
              p_x_z_nn_layers=3, p_x_z_nn_width=10,
              q_z_nn_layers=3, q_z_nn_width=10, x_mode=[0]):
    model = dataGeneratorVAE(x_dim, z_dim, device, p_x_z_nn_layers, p_x_z_nn_width, 
                q_z_nn_layers, q_z_nn_width, x_mode)
    optimizer = Adam(model.parameters(), lr=lr_start)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = (lr_end/lr_start)**(1/num_epochs))
    losses = {"total": [], "kld": [], "x": []}
    
    def kld_loss(mu, std):
        #Note that the sum is over the dimensions of z as well as over the units in the batch here
        var = std.pow(2)
        kld = -0.5 * torch.sum(1 + torch.log(var) - mu.pow(2) - var)
        return kld
    
    different_modes = list(set(x_mode))
    x_same_mode_indices = dict()
    for mode in different_modes:
        x_same_mode_indices[mode] = [i for i,m in enumerate(x_mode) if m==mode]
    
    def get_losses(z_mean, z_std, x_pred, x_std, x):
        kld = kld_loss(z_mean,z_std)
        x_loss = 0
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
        return kld, x_loss
    
    for epoch in range(num_epochs):
        i = 0
        epoch_loss = []
        epoch_kld_loss = []
        epoch_x_loss = []
        if print_logs:
            print("Epoch {}:".format(epoch))
        for data in train_loader:
            x = data.to(device)
            z_mean, z_std, x_pred, x_std = model(x)
            kld, x_loss = get_losses(z_mean, z_std, x_pred, x_std, x)
            loss = kld + x_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            i += 1
            if i%100 == 0 and print_logs:
                print("Sample batch loss: {}".format(loss))
            epoch_loss.append(loss.item())
            epoch_kld_loss.append(kld.item())
            epoch_x_loss.append(x_loss.item())
        
        losses['total'].append(sum(epoch_loss))
        losses['kld'].append(sum(epoch_kld_loss))
        losses['x'].append(sum(epoch_x_loss))
        
        scheduler.step()
  
        if print_logs:
            #print("Estimated ATE {}, p(y=1|do(t=1)): {}, p(y=1|do(t=0)): {}".format(*estimate_imageCEVAE_ATE(model)))
            print("Epoch loss: {}".format(sum(epoch_loss)))
            print("x: {}, kld: {}".format(sum(epoch_x_loss), sum(epoch_kld_loss)))
            
    fig, ax = plt.subplots(2,2,figsize=(8,8))
    ax[0,0].plot(losses['x'])
    ax[0,1].plot(losses['kld'])
    ax[1,0].plot(losses['total'])
    ax[0,0].set_title("x loss")
    ax[0,1].set_title("kld loss")
    ax[1,0].set_title("total loss")
    plt.show()
    
    return model, losses