# -*- coding: utf-8 -*-
"""
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data
from torch import nn, optim
import os
import time
from skimage import io

video = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(' Processor is %s' % (device))
# import the video frames for the training part consisting 30000 frames
nSample = 1895  # number of samples to be loaded for the training
loadPath = '/home/fatih/phd/DeepPBM/Data/bmc_real_352x288/Video_00{}/train_img/'.format(video)
frst = io.imread(loadPath + '1.jpg')
height, width, nCh = frst.shape
imgs = np.empty([nSample, nCh, height, width])
print('loading the video frames ....')
for i in range(nSample):
    imName = loadPath + '%d.jpg' % (i + 1)
    frm = io.imread(imName)
    imgs[i, :, :, :] = np.transpose(frm, (2, 0, 1))

print('frames are loaded')

# VAE model parameters for the encoder
img_size = width * height
h_layer_1 = 32
h_layer_2 = 64
h_layer_3 = 128
h_layer_4 = 256  # 128
h_layer_5 = 128  # 128
h_layer_6 = 600  # 2400
latent_dim = 1
kernel_size = (3, 3)
pool_size = 2
stride = 2

norm_par = 10 * 8  # 67x37

# VAE training parameters
batch_size = 4
epoch_num = 200
learnR = 1e-3
beta = 0.8

# Path parameters
save_PATH = '/home/fatih/phd/DeepPBM/Codes/Result/results'
PATH_vae = save_PATH + '/bmc_vid{}_vanilla_352x288_gtx1080/epoch{}_batch{}_z{}_lr{}'.format(video, epoch_num, batch_size, latent_dim, learnR)
if not os.path.exists(PATH_vae):
    os.makedirs(PATH_vae)


# Restore
Restore = False

# load  Dataset
imgs /= 256
nSample, ch, x, y = imgs.shape
imgs = torch.FloatTensor(imgs)
train_loader = torch.utils.data.DataLoader(imgs, batch_size=batch_size, shuffle=True)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # .unsqueeze(0)
        self.econv1 = nn.Conv2d(3, h_layer_1, kernel_size=kernel_size, stride=stride)
        self.ebn1 = nn.BatchNorm2d(h_layer_1, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.econv2 = nn.Conv2d(h_layer_1, h_layer_2, kernel_size=kernel_size, stride=stride)
        self.ebn2 = nn.BatchNorm2d(h_layer_2, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.econv3 = nn.Conv2d(h_layer_2, h_layer_3, kernel_size=kernel_size, stride=stride)
        self.ebn3 = nn.BatchNorm2d(h_layer_3, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.econv4 = nn.Conv2d(h_layer_3, h_layer_4, kernel_size=kernel_size, stride=stride)
        self.ebn4 = nn.BatchNorm2d(h_layer_4, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.econv5 = nn.Conv2d(h_layer_4, h_layer_5, kernel_size=kernel_size, stride=stride)
        self.ebn5 = nn.BatchNorm2d(h_layer_5, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.efc1 = nn.Linear(h_layer_5 * norm_par, h_layer_6)
        self.edrop1 = nn.Dropout(p=0.3, inplace=False)
        self.mu_z = nn.Linear(h_layer_6, latent_dim)
        self.logvar_z = nn.Linear(h_layer_6, latent_dim)
        #
        self.dfc1 = nn.Linear(latent_dim, h_layer_6)
        self.dfc2 = nn.Linear(h_layer_6, h_layer_5 * norm_par)
        self.ddrop1 = nn.Dropout(p=0.3, inplace=False)
        self.dconv1 = nn.ConvTranspose2d(h_layer_5, h_layer_4, kernel_size=kernel_size, stride=stride, padding=0,
                                         output_padding=0)
        self.dbn1 = nn.BatchNorm2d(h_layer_4, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.dconv2 = nn.ConvTranspose2d(h_layer_4, h_layer_3, kernel_size=kernel_size, stride=stride, padding=0,
                                         output_padding=0)
        self.dbn2 = nn.BatchNorm2d(h_layer_3, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.dconv3 = nn.ConvTranspose2d(h_layer_3, h_layer_2, kernel_size=kernel_size, stride=stride, padding=0,
                                         output_padding=0)
        self.dbn3 = nn.BatchNorm2d(h_layer_2, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.dconv4 = nn.ConvTranspose2d(h_layer_2, h_layer_1, kernel_size=kernel_size, stride=stride, padding=0,
                                         output_padding=0)
        self.dbn4 = nn.BatchNorm2d(h_layer_1, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.dconv5 = nn.ConvTranspose2d(h_layer_1, 3, kernel_size=kernel_size, padding=0, stride=stride,
                                         output_padding=1)

        #
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def Encoder(self, x):
        eh1 = self.relu(self.ebn1(self.econv1(x)))
        eh2 = self.relu(self.ebn2(self.econv2(eh1)))
        eh3 = self.relu(self.ebn3(self.econv3(eh2)))
        eh4 = self.relu(self.ebn4(self.econv4(eh3)))
        eh5 = self.relu(self.ebn5(self.econv5(eh4)))
        eh6 = self.relu(self.edrop1(self.efc1(eh5.view(-1, h_layer_5 * norm_par))))
        mu_z = self.mu_z(eh6)
        logvar_z = self.logvar_z(eh6)
        return mu_z, logvar_z

    def Reparam(self, mu_z, logvar_z):
        std = logvar_z.mul(0.5).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        eps = eps.to(device)
        return eps.mul(std).add_(mu_z)

    def Decoder(self, z):
        dh1 = self.relu(self.dfc1(z))
        dh2 = self.relu(self.ddrop1(self.dfc2(dh1)))
        dh3 = self.relu(self.dbn1(self.dconv1(dh2.view(-1, h_layer_5, 8, 10))))
        dh4 = self.relu(self.dbn2(self.dconv2(dh3)))
        dh5 = self.relu(self.dbn3(self.dconv3(dh4)))
        dh6 = self.relu(self.dbn4(self.dconv4(dh5)))
        x = self.dconv5(dh6).view(-1, 3, img_size)
        return self.sigmoid(x)

    def forward(self, x):
        mu_z, logvar_z = self.Encoder(x)
        z = self.Reparam(mu_z, logvar_z)
        return self.Decoder(z), mu_z, logvar_z, z


# initialize model
vae = VAE()
vae.to(device)
vae_optimizer = optim.Adam(vae.parameters(), lr=learnR)

# loss function
SparsityLoss = nn.L1Loss(reduction='sum')


def elbo_loss(recon_x, x, mu_z, logvar_z):
    L1loss = SparsityLoss(recon_x, x.view(-1, 3, img_size))
    KLD = -0.5 * beta * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())

    return L1loss + KLD


# training
if Restore == False:
    print("Training...")

    for i in range(epoch_num):
        time_start = time.time()
        loss_vae_value = 0.0
        for batch_indx, data in enumerate(train_loader):
            # update VAE
            data = data
            data = Variable(data)
            data_vae = data.to(device)
            # data_vae=data #if using gpu comment this line!
            vae_optimizer.zero_grad()
            recon_x, mu_z, logvar_z, z = vae.forward(data_vae)
            loss_vae = elbo_loss(recon_x, data_vae, mu_z, logvar_z)
            loss_vae.backward()
            loss_vae_value += loss_vae.data

            vae_optimizer.step()

        time_end = time.time()
        print('elapsed time (min) : %0.1f' % ((time_end - time_start) / 60))
        print('====> Epoch: %d elbo_Loss : %0.8f' % ((i + 1), loss_vae_value / len(train_loader.dataset)))

    torch.save(vae.state_dict(), PATH_vae+'/last_model.pth')

if Restore:
    vae.load_state_dict(torch.load(PATH_vae+'/last_model.pth'))


def plot_reconstruction():
    for indx in range(nSample):
        # Select images
        img = imgs[indx]
        img_variable = Variable(torch.FloatTensor(img))
        img_variable = img_variable.unsqueeze(0)
        img_variable = img_variable.to(device)
        imgs_z_mu, imgs_z_logvar = vae.Encoder(img_variable)
        imgs_z = vae.Reparam(imgs_z_mu, imgs_z_logvar)
        imgs_rec = vae.Decoder(imgs_z).cpu()
        imgs_rec = imgs_rec.data.numpy()
        img_i = imgs_rec[0]
        img_i = img_i.transpose(1, 0)
        img_i = img_i.reshape(x, y, 3)
        io.imsave((PATH_vae + '/%d' % (indx + 1,) + '.jpg'), img_i)


plot_reconstruction()
