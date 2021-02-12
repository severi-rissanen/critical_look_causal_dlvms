import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
import numpy as np

class Generator(nn.Module):
    def __init__(self, nz=10, ngf=64, nc=1):
        """GAN generator.
        
        Args:
          nz:  Number of elements in the latent code.
          ngf: Base size (number of channels) of the generator layers.
          nc:  Number of channels in the generated images.
        """
        super(Generator, self).__init__()
        # YOUR CODE HERE
        self.ct1 = nn.Sequential(nn.ConvTranspose2d(nz,4*ngf,kernel_size=4,stride=2,bias=False),
                                nn.BatchNorm2d(4*ngf), nn.ReLU())
        self.ct2 = nn.Sequential(nn.ConvTranspose2d(4*ngf,2*ngf,kernel_size=4,stride=2,padding=1,bias=False),
                                nn.BatchNorm2d(2*ngf), nn.ReLU())
        self.ct3 = nn.Sequential(nn.ConvTranspose2d(2*ngf,ngf,kernel_size=4,stride=2,padding=2,bias=False),
                                nn.BatchNorm2d(ngf), nn.ReLU())
        self.ct4 = nn.Sequential(nn.ConvTranspose2d(ngf,nc,kernel_size=4,stride=2,padding=1,bias=False),
                                nn.Tanh())

    def forward(self, z, verbose=False):
        """Generate images by transforming the given noise tensor.
        
        Args:
          z of shape (batch_size, nz, 1, 1): Tensor of noise samples. We use the last two singleton dimensions
                          so that we can feed z to the generator without reshaping.
          verbose (bool): Whether to print intermediate shapes (True) or not (False).
        
        Returns:
          out of shape (batch_size, nc, 28, 28): Generated images.
        """
        # YOUR CODE HERE
        z = self.ct1(z)
        z = self.ct2(z)
        z = self.ct3(z)
        z = self.ct4(z)
        return z
    
class Discriminator(nn.Module):
    def __init__(self, nc=1, ndf=64):
        """GAN discriminator.
        
        Args:
          nc:  Number of channels in images.
          ndf: Base size (number of channels) of the discriminator layers.
        """
        # YOUR CODE HERE
        super(Discriminator, self).__init__()
        self.c1 = nn.Sequential(nn.Conv2d(1,ndf,kernel_size=4, stride=2,padding=1,bias=False), nn.LeakyReLU(0.2))
        self.c2 = nn.Sequential(nn.Conv2d(ndf,2*ndf,kernel_size=4, stride=2,padding=2,bias=False), nn.LeakyReLU(0.2))
        self.c3 = nn.Sequential(nn.Conv2d(2*ndf,4*ndf,kernel_size=4, stride=2,padding=1,bias=False), nn.LeakyReLU(0.2))
        self.c4 = nn.Sequential(nn.Conv2d(4*ndf,nc,kernel_size=4, stride=2,bias=False), nn.Sigmoid())

    def forward(self, x, verbose=False):
        """Classify given images into real/fake.
        
        Args:
          x of shape (batch_size, 1, 28, 28): Images to be classified.
        
        Returns:
          out of shape (batch_size,): Probabilities that images are real. All elements should be between 0 and 1.
        """
        # YOUR CODE HERE
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        return x.view(-1)
    
def generator_loss(D, fake_images):
    """Loss computed to train the GAN generator.

    Args:
      D: The discriminator whose forward function takes inputs of shape (batch_size, nc, 28, 28)
         and produces outputs of shape (batch_size, 1).
      fake_images of shape (batch_size, nc, 28, 28): Fake images produces by the generator.

    Returns:
      loss: The sum of the binary cross-entropy losses computed for all the samples in the batch.

    Notes:
    - Make sure that you process on the device given by `fake_images.device`.
    - Use values of global variables `real_label`, `fake_label` to produce the right targets.
    """
    # YOUR CODE HERE
    device = fake_images.device
    loss = F.binary_cross_entropy(D(fake_images), torch.zeros(fake_images.shape[0], device=device).fill_(real_label))
    return loss

real_label = 1
fake_label = 0
def discriminator_loss(D, real_images, fake_images):
    """Loss computed to train the GAN discriminator.

    Args:
      D: The discriminator.
      real_images of shape (batch_size, nc, 28, 28): Real images.
      fake_images of shape (batch_size, nc, 28, 28): Fake images produces by the generator.

    Returns:
      d_loss_real: The mean of the binary cross-entropy losses computed on the real_images.
      D_real: Mean output of the discriminator for real_images. This is useful for tracking convergence.
      d_loss_fake: The mean of the binary cross-entropy losses computed on the fake_images.
      D_fake: Mean output of the discriminator for fake_images. This is useful for tracking convergence.

    Notes:
    - Make sure that you process on the device given by `fake_images.device`.
    - Use values of global variables `real_label`, `fake_label` to produce the right targets.
    """
    # YOUR CODE HERE
    device = fake_images.device
    real_pred = D(real_images)
    fake_pred = D(fake_images)
    d_loss_real = F.binary_cross_entropy(real_pred, torch.zeros(fake_images.shape[0], device=device).fill_(real_label))
    d_loss_fake = F.binary_cross_entropy(fake_pred, torch.zeros(fake_images.shape[0], device=device).fill_(fake_label))
    return d_loss_real, real_pred.mean(), d_loss_fake, fake_pred.mean()