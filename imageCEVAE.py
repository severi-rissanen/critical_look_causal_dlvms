import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np

class FullyConnected(nn.Sequential):
    """
    Fully connected multi-layer network with ELU activations.
    """
    def __init__(self, sizes, final_activation=None):
        layers = []
        for in_size, out_size in zip(sizes, sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ELU())
        layers.pop(-1)
        if final_activation is not None:
            layers.append(final_activation)
        super().__init__(*layers)

    def append(self, layer):
        assert isinstance(layer, nn.Module)
        self.add_module(str(len(self)), layer)

class Decoder(nn.Module):
    def __init__(
        self,
        x_dim,
        z_dim,
        device,
        ngf=64,
        nc=1,#Amount of channels, always one here
        p_y_zt_nn=False,
        p_y_zt_nn_layers=3,
        p_y_zt_nn_width=10,
        p_t_z_nn=False,
        p_t_z_nn_layers=3,
        p_t_z_nn_width=10,
        p_x_z_nn=False,
        p_x_z_nn_layers=3,
        p_x_z_nn_width=10
    ):
        super().__init__()
        self.x_dim = x_dim
        self.device = device
        self.p_x_z_nn = p_x_z_nn
        
        #Image generating networks
        self.lin = nn.Sequential(nn.Linear(z_dim, 20), nn.ELU())
        self.ct1 = nn.Sequential(nn.ConvTranspose2d(20,4*ngf,kernel_size=3,stride=1,bias=False),nn.ELU())
        self.ct2 = nn.Sequential(nn.ConvTranspose2d(4*ngf,2*ngf,kernel_size=4,stride=2,padding=1,bias=False), nn.ELU())
        self.ct3 = nn.Sequential(nn.ConvTranspose2d(2*ngf,ngf,kernel_size=6,stride=2,padding=1,bias=False), nn.ELU())
        self.ct4 = nn.Sequential(nn.ConvTranspose2d(ngf,nc,kernel_size=6,stride=2,padding=2,bias=False))
        self.logstd = nn.Parameter(torch.Tensor([1]))
        
        #Proxy networks
        self.x_nn = FullyConnected([z_dim] + p_x_z_nn_layers*[p_x_z_nn_width] + [x_dim]) if p_x_z_nn else nn.ModuleList([nn.Linear(z_dim,1, bias=False) for i in range(x_dim)])
        self.x_log_std = nn.Parameter(torch.FloatTensor(x_dim*[1.]).to(device))
        
        #Treatment network
        self.t_nn = FullyConnected([z_dim] + p_t_z_nn_layers*[p_t_z_nn_width] + [1]) if p_t_z_nn else nn.Linear(z_dim,1, bias=True)
        
        #y network
        self.y0_nn = FullyConnected([z_dim] + p_y_zt_nn_layers*[p_y_zt_nn_width] + [1]) if p_y_zt_nn else nn.Linear(z_dim,1, bias=True)#If t and y binary
        self.y1_nn = FullyConnected([z_dim] + p_y_zt_nn_layers*[p_y_zt_nn_width] + [1]) if p_y_zt_nn else nn.Linear(z_dim,1, bias=True)

    def forward(self,z,t):
        #z is dim (batch_size,z_dim)
        image = self.ct1(self.lin(z)[:,:,None,None])
        image = self.ct2(image)
        image = self.ct3(image)
        image = self.ct4(image)
        if self.p_x_z_nn:
            x_pred = self.x_nn(z)
        else:
            x_pred = torch.zeros(z.shape[0], self.x_dim, device=self.device)
            for i in range(self.x_dim):
                x_pred[:,i] = self.x_nn[i](z)[:,0]
        t_pred = self.t_nn(z)
        y_logits0 = self.y0_nn(z)
        y_logits1 = self.y1_nn(z)
        y_pred = y_logits1*t + y_logits0*(1-t)
        return image, torch.exp(self.logstd), x_pred, torch.exp(self.x_log_std), t_pred, y_pred
    
class Encoder(nn.Module):
    def __init__(
        self,
        x_dim,
        z_dim,
        device,
        ngf=64,
        nc=1,
        separate_ty=False
    ):
        super().__init__()
        self.device = device
        self.separate_ty = separate_ty
        
        self.c1 = nn.Sequential(nn.Conv2d(nc,ngf,kernel_size=6,stride=2,padding=2,bias=False), nn.ELU())
        self.c2 = nn.Sequential(nn.Conv2d(ngf,2*ngf,kernel_size=6,stride=2,padding=1,bias=False), nn.ELU())
        self.c3 = nn.Sequential(nn.Conv2d(2*ngf,4*ngf,kernel_size=4,stride=2,padding=1,bias=False), nn.ELU())
        self.c4 = nn.Sequential(nn.Conv2d(4*ngf,40,kernel_size=3,stride=1,bias=False), nn.ELU())
        self.fc = nn.Sequential(nn.Linear(40+x_dim+2,z_dim+5),nn.ELU())#I guess that this could be optimized
        self.fc00 = nn.Sequential(nn.Linear(40+x_dim,z_dim+5),nn.ELU())
        self.fc01 = nn.Sequential(nn.Linear(40+x_dim,z_dim+5),nn.ELU())
        self.fc10 = nn.Sequential(nn.Linear(40+x_dim,z_dim+5),nn.ELU())
        self.fc11 = nn.Sequential(nn.Linear(40+x_dim,z_dim+5),nn.ELU())
        self.mean = nn.Linear(z_dim+5,z_dim)#z_dim+5 to to avoid bottlenecks and on the other hand instability in optimization
        self.logstd = nn.Linear(z_dim+5,z_dim)
        self.mean00 = nn.Linear(z_dim+5,z_dim)
        self.mean01 = nn.Linear(z_dim+5,z_dim)
        self.mean10 = nn.Linear(z_dim+5,z_dim)
        self.mean11 = nn.Linear(z_dim+5,z_dim)
        self.logstd00 = nn.Linear(z_dim+5,z_dim)
        self.logstd01 = nn.Linear(z_dim+5,z_dim)
        self.logstd10 = nn.Linear(z_dim+5,z_dim)
        self.logstd11 = nn.Linear(z_dim+5,z_dim)
        
    def forward(self,image,x,t,y):
        temp = self.c1(image)
        temp = self.c2(temp)
        temp = self.c3(temp)
        temp = self.c4(temp)
        #if self.separate_ty:
        imx = torch.cat([temp[:,:,0,0],x],1)
        temp = self.fc11(imx)*t*y + self.fc10(imx)*t*(1-y) + self.fc01(imx)*(1-t)*y + self.fc00(imx)*(1-t)*(1-y)
        z_mean = self.mean11(temp)*t*y + self.mean10(temp)*t*(1-y) + self.mean01(temp)*(1-t)*y + self.mean00(temp)*(1-t)*(1-y)
        z_std = torch.exp(self.logstd11(temp)*t*y + self.logstd10(temp)*t*(1-y) + self.logstd01(temp)*(1-t)*y + self.logstd00(temp)*(1-t)*(1-y))
        #else:
        #temp = self.fc(torch.cat([temp[:,:,0,0],x,t,y],1))
        #z_mean = self.mean(temp)
        #z_std = torch.exp(self.logstd(temp))
        return z_mean,z_std#dim (batch_size,z_dim)

class ImageCEVAE(nn.Module):
    def __init__(
        self, 
        x_dim,
        z_dim=1,
        device='cpu',
        p_y_zt_nn=False,
        p_y_zt_nn_layers=3,
        p_y_zt_nn_width=10,
        p_t_z_nn=False,
        p_t_z_nn_layers=3,
        p_t_z_nn_width=10,
        p_x_z_nn=False,
        p_x_z_nn_layers=3,
        p_x_z_nn_width=10,
        separate_ty=False
    ):
        super().__init__()
        
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.device = device
        
        self.encoder = Encoder(
            x_dim,
            z_dim,
            device=device,
            separate_ty=separate_ty
        )
        self.decoder = Decoder(
            x_dim,
            z_dim,
            device=device,
            p_y_zt_nn=p_y_zt_nn,
            p_y_zt_nn_layers=p_y_zt_nn_layers,
            p_y_zt_nn_width=p_y_zt_nn_width,
            p_t_z_nn=p_t_z_nn,
            p_t_z_nn_layers=p_t_z_nn_layers,
            p_t_z_nn_width=p_t_z_nn_width,
            p_x_z_nn=p_x_z_nn,
            p_x_z_nn_layers=p_x_z_nn_layers,
            p_x_z_nn_width=p_x_z_nn_width
        )
        self.to(device)
        self.float()

    def reparameterize(self, mean, std):
        # samples from unit norm and does reparam trick
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def forward(self, image, x, t, y):#Should have t, y
        z_mean, z_std = self.encoder(image, x, t, y)
        #TODO: works at least for z_dim=1, maybe errors if z_dim>1
        z = self.reparameterize(z_mean, z_std)
        image, image_std, x_pred, x_std, t_pred, y_pred = self.decoder(z,t)
        
        return image, image_std, z_mean, z_std, x_pred, x_std, t_pred, y_pred
    
    
class ConvyNet(nn.Module):
    def __init__(self,ngf=64,nc=1,device='cuda'):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv2d(nc,ngf,kernel_size=6,stride=2,padding=2,bias=False), nn.ELU())
        self.c2 = nn.Sequential(nn.Conv2d(ngf,2*ngf,kernel_size=6,stride=2,padding=1,bias=False), nn.ELU())
        self.c3 = nn.Sequential(nn.Conv2d(2*ngf,4*ngf,kernel_size=4,stride=2,padding=1,bias=False), nn.ELU())
        self.c4 = nn.Sequential(nn.Conv2d(4*ngf,40,kernel_size=3,stride=1,bias=False), nn.ELU())
        self.fc0 = nn.Linear(40+1,1)
        self.fc1 = nn.Linear(40+1,1)
        self.to(device)
        
    def forward(self,image,x,t):
        temp = self.c1(image)
        temp = self.c2(temp)
        temp = self.c3(temp)
        temp = self.c4(temp)
        temp = self.fc1(torch.cat([temp[:,:,0,0],x],1))*t + self.fc0(torch.cat([temp[:,:,0,0],x],1))*(1-t)
        return temp