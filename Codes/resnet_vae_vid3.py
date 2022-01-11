import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import os
from PIL import Image
import natsort
import time
import random
import numpy as np

# ====================== seed ======================
seed = 0
os.environ['PYTHONHASHSEED'] = str(seed)
# Torch RNG
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# Python RNG
np.random.seed(seed)
random.seed(seed)

# ======== 320x240 resnet vae ======================

img_size = 320 * 240


class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class ResNet34Enc(nn.Module):
    def __init__(self, z_dim=32):
        super(ResNet34Enc, self).__init__()
        self.z_dim = z_dim
        self.ResNet34 = models.resnet34(pretrained=True)
        self.num_feature = self.ResNet34.fc.in_features
        self.ResNet34.fc = nn.Linear(self.num_feature, 2 * self.z_dim)

    def forward(self, x):
        x = self.ResNet34(x)
        mu = x[:, :self.z_dim]
        logvar = x[:, self.z_dim:]
        return mu, logvar


class BasicBlockDec(nn.Module):
    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes / stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet34Dec(nn.Module):

    def __init__(self, num_Blocks=[3, 4, 6, 3], z_dim=32, nc=3):
        super().__init__()
        self.in_planes = 512

        self.linear = nn.Linear(z_dim, 512)

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, size=(7, 10), mode='bilinear', align_corners=False)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = F.interpolate(x, size=(120, 160), mode='bilinear', align_corners=False)
        x = torch.sigmoid(self.conv1(x))
        x = x.view(x.size(0), 3, 240, 320)
        return x


class VAE(nn.Module):
    def __init__(self, z_dim):
        super(VAE, self).__init__()
        self.encoder = ResNet34Enc(z_dim=z_dim)
        self.decoder = ResNet34Dec(z_dim=z_dim)

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x = self.decoder(z)
        return x, mean, logvar

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2)  # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std).cuda()
        return epsilon * std + mean


SparsityLoss = nn.L1Loss(reduction='sum')


# SparsityLoss = nn.L1Loss(size_average = False, reduce = True)

def loss_func(recon_x, x, mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    L1loss = SparsityLoss(recon_x, x)
    # L1loss = SparsityLoss(recon_x, x.view(-1, 3, img_size))
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return L1loss + KLD


class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(' Processor is %s' % (device))

save_PATH = './Result/bmc2012_result/bmc_vid3_resnet34_320x240_rtx3090'
if not os.path.exists(save_PATH):
    os.makedirs(save_PATH)

epoch_num = 140
batch_size = 8
latent_dim = 1
learnR = 1e-2

exp_path = save_PATH + '/epoch{}_batch{}_z{}_lr{}'.format(epoch_num, batch_size, latent_dim, learnR)
if not os.path.exists(exp_path):
    os.makedirs(exp_path)

"""
vae = VAE(z_dim=latent_dim)
print(vae)
"""

vae = VAE(z_dim=latent_dim).cuda()
optimizer = optim.Adam(vae.parameters(), lr=learnR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
vae.train()

transform = transforms.Compose([transforms.ToTensor()])
transform2 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])


myDatasetTrain = CustomDataSet('../Data/bmc2012/Video_003/Video_003', transform=transform)
myDatasetTest = CustomDataSet('../Data/bmc2012/Video_003/Video_003', transform=transform)

# Data loader (input pipeline)
train_iter = DataLoader(myDatasetTrain, batch_size=batch_size, num_workers=8, shuffle=True, drop_last=True)
test_iter = DataLoader(myDatasetTest, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False)

min_loss = 0
print('Training starts')
tot_start = time.time()
for epoch in range(0, epoch_num):
    time_start = time.time()
    l_sum = 0
    for x in train_iter:
        # x = torch.sigmoid(x).cuda()
        x = x.cuda()
        # print(x.requires_grad)
        optimizer.zero_grad()
        recon_x, mu, logvar = vae.forward(x)
        loss = loss_func(recon_x, x, mu, logvar)
        l_sum += loss
        loss.backward()
        optimizer.step()
    # scheduler.step()
    if l_sum < min_loss:
        torch.save({'epoch': epoch, 'model_state_dict': vae.state_dict(), 'optimizer_state_dict': optimizer.state_dict()
                       , 'loss': l_sum}, exp_path + '/best_model.pth')
        print('--best model--')
        min_loss = l_sum
    if epoch == 0:
        min_loss = l_sum
    time_end = time.time()
    print('elapsed time (min) : %0.1f' % ((time_end - time_start) / 60))
    print('====> Epoch: %d elbo_Loss : %0.8f' % ((epoch + 1), l_sum / myDatasetTrain.__len__()))

tot_end = time.time()
print('total elapsed time (min) : %0.1f' % ((tot_end - tot_start) / 60))
torch.save(vae.state_dict(), exp_path + '/last_model.pth')

# vae.load_state_dict(torch.load(exp_path + '/best_model.pth'))
checkpoint = torch.load(exp_path + '/best_model.pth')
vae.load_state_dict(checkpoint['model_state_dict'])

print('Inference starts')
vae.eval()
inx = 1
with torch.no_grad():
    for t_img in test_iter:
        t_img = Variable(t_img).cuda()
        result, mu, logvar = vae.forward(t_img)
        for batches in range(batch_size):
            infer_out = result.data[batches]
            infer_out = infer_out.cpu()
            infer_out = transform2(infer_out)
            utils.save_image(infer_out, exp_path + '/' + str(inx).zfill(6) + '.jpg', normalize=True)
            inx += 1
            print(inx)
